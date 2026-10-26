"""
Ввод/вывод.

Выход: 1-битный TIFF (Baseline), запись полосами (strips) — весь битмап
в памяти не держится, поэтому можно выводить полноформатные флексоформы.
Photometric = MinIsWhite: бит 1 = печатающий элемент (как ждут CTP/CDI).
Сжатие: lzw (5), deflate (Adobe ZIP, 8), packbits (32773) или none.
Полосы сжимаются параллельно на всех ядрах. Для файлов > 3,5 ГБ пишется BigTIFF.
"""
from __future__ import annotations
import os
import struct
import subprocess
import tempfile
import zlib
import glob
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import imagecodecs
except Exception:  # pragma: no cover
    imagecodecs = None

COMPRESSION = {"none": 1, "lzw": 5, "deflate": 8, "packbits": 32773}


def _compress(data: bytes, comp: int) -> bytes:
    if comp == 1:
        return data
    if comp == 8:
        return zlib.compress(data, 6)
    if imagecodecs is None:
        raise RuntimeError("Для сжатия LZW/PackBits нужен пакет imagecodecs: python -m pip install imagecodecs")
    if comp == 5:
        return imagecodecs.lzw_encode(data)
    if comp == 32773:
        return imagecodecs.packbits_encode(data)
    raise ValueError(comp)


def decompress(data: bytes, comp: int) -> bytes:
    """Распаковка полосы TIFF (для чтения 1-бит файлов)."""
    if comp == 1:
        return data
    if comp in (8, 32946):
        return zlib.decompress(data)
    if imagecodecs is None:
        raise RuntimeError("Для LZW/PackBits нужен пакет imagecodecs: python -m pip install imagecodecs")
    if comp == 5:
        return imagecodecs.lzw_decode(data)
    if comp == 32773:
        return imagecodecs.packbits_decode(data)
    raise ValueError(f"сжатие {comp} не поддерживается для построчного чтения")


class BilevelTiffWriter:
    def __init__(self, path: str, width: int, height: int, dpi: float,
                 rows_per_strip: int, compression: str = "deflate",
                 software: str = "HybridRIP"):
        self.path, self.w, self.h = path, width, height
        self.dpi, self.rps = dpi, rows_per_strip
        if compression not in COMPRESSION:
            raise ValueError(f"Неизвестное сжатие {compression!r}; варианты: {', '.join(COMPRESSION)}")
        self.comp = COMPRESSION[compression]
        _compress(b"\0" * 8, self.comp)          # сразу проверить, что кодек есть
        self.pool = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2)))
        self.software = software
        raw = (width + 7) // 8 * height
        self.big = raw > 3_500_000_000
        self.f = open(path, "wb")
        if self.big:
            self.f.write(b"II" + struct.pack("<HHHQ", 43, 8, 0, 0))
        else:
            self.f.write(b"II" + struct.pack("<HI", 42, 0))
        self.offsets, self.counts = [], []
        self.row_bytes = (width + 7) // 8
        self._pending = np.zeros((0, self.row_bytes), np.uint8)
        self.rows_written = 0

    def write_rows(self, rows: np.ndarray):
        """Строки как bool-массив (H, W)."""
        self.write_packed(np.packbits(rows, axis=1))

    def write_packed(self, packed: np.ndarray):
        """Строки уже упакованные по 8 пикселей в байт (H, ceil(W/8)), старший бит слева."""
        buf = np.concatenate([self._pending, packed]) if self._pending.size else packed
        n_full = buf.shape[0] // self.rps
        blocks = [buf[i * self.rps:(i + 1) * self.rps].tobytes() for i in range(n_full)]
        # zlib и imagecodecs отпускают GIL -> полосы сжимаются параллельно
        for data, rows in zip(self.pool.map(lambda b: _compress(b, self.comp), blocks),
                              [self.rps] * n_full):
            self._write_strip(data, rows)
        self._pending = buf[n_full * self.rps:].copy()

    def _emit(self, block: np.ndarray):
        self._write_strip(_compress(np.ascontiguousarray(block).tobytes(), self.comp), block.shape[0])

    def _write_strip(self, data: bytes, rows: int):
        self.offsets.append(self.f.tell())
        self.counts.append(len(data))
        self.f.write(data)
        if self.f.tell() % 2:
            self.f.write(b"\0")
        self.rows_written += rows

    def close(self):
        if self._pending.shape[0]:
            self._emit(self._pending)
            self._pending = self._pending[:0]
        self.pool.shutdown()
        assert self.rows_written == self.h, (self.rows_written, self.h)
        f = self.f
        big = self.big
        # внешние данные тегов
        def put(b):
            off = f.tell()
            f.write(b)
            if f.tell() % 2:
                f.write(b"\0")
            return off
        cnt_fmt = "Q" if big else "I"
        T_LONG = 16 if big else 4
        offs = put(struct.pack(f"<{len(self.offsets)}{cnt_fmt}", *self.offsets)) if len(self.offsets) > 1 else None
        cnts = put(struct.pack(f"<{len(self.counts)}{cnt_fmt}", *self.counts)) if len(self.counts) > 1 else None
        num = int(round(self.dpi * 1000))
        res = put(struct.pack("<II", num, 1000))
        sw = (self.software + "\0").encode()
        swo = put(sw)
        tags = [
            (256, 4, 1, self.w), (257, 4, 1, self.h), (258, 3, 1, 1),
            (259, 3, 1, self.comp), (262, 3, 1, 0),
            (273, T_LONG, len(self.offsets), offs if offs is not None else self.offsets[0]),
            (277, 3, 1, 1), (278, 4, 1, self.rps),
            (279, T_LONG, len(self.counts), cnts if cnts is not None else self.counts[0]),
            (282, 5, 1, res), (283, 5, 1, res), (296, 3, 1, 2),
            (305, 2, len(sw), swo),
        ]
        ifd = f.tell()
        if big:
            f.write(struct.pack("<Q", len(tags)))
            for tag, typ, n, val in tags:
                if typ == 3 and n == 1:
                    f.write(struct.pack("<HHQHHI", tag, typ, n, val, 0, 0))
                else:
                    f.write(struct.pack("<HHQQ", tag, typ, n, val))
            f.write(struct.pack("<Q", 0))
            f.seek(8)
            f.write(struct.pack("<Q", ifd))
        else:
            f.write(struct.pack("<H", len(tags)))
            for tag, typ, n, val in tags:
                if typ == 3 and n == 1:
                    f.write(struct.pack("<HHIHH", tag, typ, n, val, 0))
                else:
                    f.write(struct.pack("<HHII", tag, typ, n, val))
            f.write(struct.pack("<I", 0))
            f.seek(4)
            f.write(struct.pack("<I", ifd))
        f.close()


# ----------------------------------------------------------------------------
# Чтение входа. Возвращает список (имя_сепарации, массив_или_функция), ppi.
# Массив краски хранится в исходном типе (uint8/uint16, 0 = нет краски),
# без перевода в float — для больших форматов это экономит память в 2–4 раза.
# Вместо массива может быть функция без аргументов (ленивая загрузка).
# ----------------------------------------------------------------------------
def ink_max(a: np.ndarray) -> float:
    if a.dtype == np.uint8:
        return 255.0
    if a.dtype == np.uint16:
        return 65535.0
    return 1.0


def _prepare(a: np.ndarray) -> np.ndarray:
    if a.dtype in (np.uint8, np.uint16):
        return a
    if a.dtype == bool:
        return a.astype(np.uint8) * 255
    a = a.astype(np.float32)
    return a / a.max() if a.max() > 1 else a


def _invert(a: np.ndarray) -> np.ndarray:
    """Инверсия «на месте», если возможно (без второй копии в памяти)."""
    m = ink_max(a)
    if a.dtype in (np.uint8, np.uint16):
        if not a.flags.writeable:
            a = a.copy()
        np.subtract(a.dtype.type(int(m)), a, out=a)
        return a
    return 1.0 - a


def _read_tiff_array(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        try:
            a = p.asarray()
        except ValueError:
            # сжатие, которое tifffile не умеет без imagecodecs (LZW и т.п.) -> Pillow
            from PIL import Image
            a = np.array(Image.open(path))
        photometric = p.photometric.name if p.photometric is not None else "MINISBLACK"
        ppi = None
        try:
            xr = p.tags["XResolution"].value
            unit = p.tags.get("ResolutionUnit")
            unit = unit.value if unit is not None else 2
            ppi = xr[0] / xr[1]
            if int(unit) == 3:
                ppi *= 2.54
        except Exception:
            pass
        inknames = None
        if "InkNames" in p.tags:
            inknames = [s for s in str(p.tags["InkNames"].value).split("\0") if s]
    return _prepare(a), photometric, ppi, inknames


def read_gray_ink(path):
    a, photometric, _, _ = _read_tiff_array(path)
    if a.ndim == 3:
        a = a[..., 0].copy()
    return a if photometric == "MINISWHITE" else _invert(a)


def read_tiff(path: str, ppi_override=None):
    a, photometric, ppi, inknames = _read_tiff_array(path)
    ppi = ppi_override or ppi or 300.0
    seps = []
    if a.ndim == 2:
        ink = a if photometric == "MINISWHITE" else _invert(a)
        seps.append((os.path.splitext(os.path.basename(path))[0], ink))
    elif photometric == "SEPARATED":
        names = inknames or ["C", "M", "Y", "K", "S5", "S6", "S7", "S8"]
        for i in range(a.shape[2]):
            seps.append((names[i] if i < len(names) else f"S{i+1}",
                         (lambda i=i: np.ascontiguousarray(a[..., i]))))
    else:  # RGB -> упрощённо в серый (для теста)
        g = a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        seps.append(("K", 1.0 - g / ink_max(a)))
    return seps, float(ppi)


def read_any(path: str, ppi_override=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return read_tiff(path, ppi_override)
    if ext in (".pdf", ".ps", ".eps", ".ai"):
        return render_with_ghostscript(path, ppi_override or 600)
    from PIL import Image
    im = Image.open(path)
    ppi = ppi_override or (im.info.get("dpi", (300, 300))[0]) or 300
    if im.mode == "CMYK":
        a = np.asarray(im)
        return [(n, (lambda i=i: np.ascontiguousarray(a[..., i]))) for i, n in enumerate("CMYK")], float(ppi)
    g = np.array(im.convert("L"))
    return [(os.path.splitext(os.path.basename(path))[0], _invert(g))], float(ppi)


def find_ghostscript() -> str:
    """Ищет исполняемый файл Ghostscript.
    Порядок: переменная окружения GS -> PATH (gswin64c / gswin32c / gs) ->
    стандартные папки установки Windows (C:\\Program Files\\gs\\gs*\\bin)."""
    import shutil
    env = os.environ.get("GS")
    if env and os.path.exists(env):
        return env
    for name in ("gswin64c", "gswin32c", "gs"):
        p = shutil.which(name)
        if p:
            return p
    if os.name == "nt":
        roots = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
        roots += [r"C:\Program Files", r"C:\Program Files (x86)"]
        found = []
        for r in filter(None, roots):
            for exe in ("gswin64c.exe", "gswin32c.exe"):
                found += glob.glob(os.path.join(r, "gs", "gs*", "bin", exe))
        if found:
            # самая новая версия
            return sorted(set(found), key=lambda p: [int(x) if x.isdigit() else 0 for x in
                          re.split(r"[^\d]+", os.path.basename(os.path.dirname(os.path.dirname(p))))])[-1]
    raise FileNotFoundError(
        "Ghostscript не найден. Установите его (https://ghostscript.com/releases/gsdnld.html) "
        "или укажите путь к gswin64c.exe в переменной окружения GS / в поле окна программы.")


_PROCESS_ORDER = {"cyan": 0, "magenta": 1, "yellow": 2, "black": 3}


def render_with_ghostscript(path: str, ppi: float, page: int = 1):
    """PDF/PS -> контоновые сепарации (включая плашечные) через Ghostscript tiffsep."""
    gs = find_ghostscript()
    tmp = tempfile.mkdtemp(prefix="hrip_gs_")
    out = os.path.join(tmp, "sep.tif")
    cmd = [gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=tiffsep",
           f"-r{int(ppi)}", f"-dFirstPage={page}", f"-dLastPage={page}",
           "-dOverprint=/simulate", "-sCompression=none", f"-sOutputFile={out}", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Ghostscript завершился с ошибкой ({gs}):\n{r.stderr or r.stdout}")
    files = glob.glob(os.path.join(tmp, "sep(*).tif"))
    if not files:
        raise RuntimeError("Ghostscript не создал сепараций. Проверьте PDF.")

    def key(f):
        n = f.split("(", 1)[1].rsplit(")", 1)[0]
        return (_PROCESS_ORDER.get(n.lower(), 10), n)
    # композитный CMYK-файл не нужен, а он самый большой
    try:
        os.remove(out)
    except OSError:
        pass
    import atexit
    import shutil
    atexit.register(shutil.rmtree, tmp, True)
    seps = []
    for f in sorted(files, key=key):
        name = f.split("(", 1)[1].rsplit(")", 1)[0]
        seps.append((name, (lambda f=f: read_gray_ink(f))))   # читаем по одной
    return seps, float(ppi)
