"""
HybridRIP Viewer — просмотр и наложение 1-битных TIFF (сепараций).

  • каждая сепарация своим цветом, наложение «как на оттиске» (субтрактивно)
  • колесо мыши — масштаб вокруг курсора, левая кнопка — перетаскивание
  • выбранную сепарацию можно сдвигать стрелками (Shift — ×10, Ctrl — ×100 пикселей)
    для проверки совмещения; сдвиг показан в списке
  • большие файлы (гигапиксели) не грузятся целиком: строится уменьшенная
    пирамида, а при увеличении читаются только нужные полосы TIFF

Запуск:  python viewer.py [файл1.tif файл2.tif ...]
"""
from __future__ import annotations
import os
import re
import sys
import threading
import queue
import zlib
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybridrip.tiffio import decompress  # noqa: E402
try:
    import imagecodecs  # noqa: F401
    STRIP_CODECS = (1, 5, 8, 32946, 32773)
except Exception:
    STRIP_CODECS = (1, 8, 32946)

LEVELS = (8, 32, 128)           # коэффициенты уменьшения пирамиды
RAW_MAX_DOWNSCALE = 6           # до какого уменьшения читать исходные биты

PALETTE = {
    "cyan": (0, 160, 227), "magenta": (230, 0, 126), "yellow": (255, 237, 0),
    "black": (30, 30, 30),
}
SPOT_COLORS = [(255, 120, 0), (0, 150, 70), (120, 60, 170), (200, 30, 30),
               (0, 110, 200), (150, 100, 40)]


def guess_color(name: str, idx: int):
    n = os.path.splitext(name.lower())[0]
    tail = re.split(r"[_\-]", n)[-1].strip()       # "testchart_cmyk_C" -> "c"
    for key, col in PALETTE.items():
        letter = {"cyan": "c", "magenta": "m", "yellow": "y", "black": "k"}[key]
        if re.search(rf"(^|[^a-z]){key}($|[^a-z])", tail) or tail == letter:
            return col
    return SPOT_COLORS[idx % len(SPOT_COLORS)]


# ============================================================== чтение 1-бит TIFF
class BitTiff:
    """Ленивое чтение 1-битного TIFF по полосам с кешем."""

    def __init__(self, path: str):
        import tifffile
        self.path = path
        self.name = os.path.basename(path)
        with tifffile.TiffFile(path) as t:
            p = t.pages[0]
            if p.bitspersample != 1 or p.samplesperpixel != 1:
                raise ValueError(f"{self.name}: не 1-битный файл ({p.bitspersample} бит)")
            self.H, self.W = p.shape[:2]
            self.comp = int(p.compression)
            self.photometric = int(p.photometric)
            self.tiled = p.is_tiled
            self.offsets = list(p.dataoffsets)
            self.counts = list(p.databytecounts)
            self.rps = int(p.rowsperstrip) if not self.tiled else 0
            try:
                xr = p.tags["XResolution"].value
                self.dpi = xr[0] / xr[1]
                unit = p.tags.get("ResolutionUnit")
                if unit is not None and int(unit.value) == 3:
                    self.dpi *= 2.54
            except Exception:
                self.dpi = 2400.0
            fo = p.tags.get("FillOrder")
            self.lsb = fo is not None and int(fo.value) == 2
        self.row_bytes = (self.W + 7) // 8
        self.invert = self.photometric == 1      # MinIsBlack: 0 = чёрный = краска
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_limit = 256 * 1024 * 1024
        self.cache_size = 0
        self.full = None
        self.lock = threading.Lock()
        if self.tiled or self.comp not in STRIP_CODECS:
            self._load_full()

    def _load_full(self):
        """Неподдерживаемое построчное чтение (плитки, LZW, G4...) — грузим целиком."""
        if self.W * self.H > 3_000_000_000:
            raise ValueError(f"{self.name}: сжатие {self.comp} поддерживается только для файлов до 3 Гпикс")
        try:
            import tifffile
            a = tifffile.imread(self.path)
        except Exception:
            im = Image.open(self.path)
            a = np.asarray(im.convert("1"))
            # PIL уже учёл photometric: True = белый
            a = ~a
            self.full = np.packbits(a.astype(bool), axis=1)
            self.invert = False
            return
        a = a.astype(bool)
        self.full = np.packbits(a, axis=1)

    def _strip(self, i: int) -> np.ndarray:
        with self.lock:
            if i in self.cache:
                self.cache.move_to_end(i)
                return self.cache[i]
        with open(self.path, "rb") as f:
            f.seek(self.offsets[i])
            raw = f.read(self.counts[i])
        raw = decompress(raw, self.comp)
        rows = min(self.rps, self.H - i * self.rps)
        a = np.frombuffer(raw, np.uint8)[: rows * self.row_bytes].reshape(rows, self.row_bytes)
        if self.lsb:
            a = np.unpackbits(a, axis=1, bitorder="little")
            a = np.packbits(a, axis=1)
        if self.invert:
            a = ~a
        with self.lock:
            self.cache[i] = a
            self.cache_size += a.nbytes
            while self.cache_size > self.cache_limit and len(self.cache) > 1:
                _, old = self.cache.popitem(last=False)
                self.cache_size -= old.nbytes
        return a

    def packed_rows(self, y0: int, y1: int) -> np.ndarray:
        """Упакованные строки [y0, y1) (1 = краска)."""
        y0, y1 = max(0, y0), min(self.H, y1)
        if y1 <= y0:
            return np.zeros((0, self.row_bytes), np.uint8)
        if self.full is not None:
            a = self.full[y0:y1]
            return ~a if self.invert else a
        s0, s1 = y0 // self.rps, (y1 - 1) // self.rps
        parts = [self._strip(i) for i in range(s0, s1 + 1)]
        a = parts[0] if len(parts) == 1 else np.concatenate(parts)
        off = y0 - s0 * self.rps
        return a[off: off + (y1 - y0)]

    def bits(self, y0, y1, x0, x1) -> np.ndarray:
        """bool-массив области (вне файла — нули)."""
        out = np.zeros((y1 - y0, x1 - x0), bool)
        ry0, ry1 = max(0, y0), min(self.H, y1)
        rx0, rx1 = max(0, x0), min(self.W, x1)
        if ry1 <= ry0 or rx1 <= rx0:
            return out
        p = self.packed_rows(ry0, ry1)
        b0, b1 = rx0 // 8, (rx1 + 7) // 8
        u = np.unpackbits(p[:, b0:b1], axis=1)
        u = u[:, rx0 - b0 * 8: rx0 - b0 * 8 + (rx1 - rx0)]
        out[ry0 - y0: ry1 - y0, rx0 - x0: rx1 - x0] = u.astype(bool)
        return out


def build_pyramid(bt: BitTiff, progress=None):
    """Уровни покрытия (uint8, 0..255) для коэффициентов LEVELS."""
    f0 = LEVELS[0]
    assert f0 == 8, "первый уровень пирамиды считается по байтам и должен быть 8"
    Hl, Wl = (bt.H + f0 - 1) // f0, (bt.W + f0 - 1) // f0
    lvl = np.zeros((Hl, Wl), np.uint8)
    wpad = Wl * f0
    rows = f0 * 64
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)
    for y in range(0, bt.H, rows):
        y1 = min(bt.H, y + rows)
        p = bt.packed_rows(y, y1)
        n = p.shape[0]
        nb = (n + f0 - 1) // f0
        if n < nb * f0:
            p = np.concatenate([p, np.zeros((nb * f0 - n, p.shape[1]), np.uint8)])
        # f0 = 8 -> один байт = 8 пикселей по горизонтали: считаем биты через таблицу
        cnt = lut[p]                                   # (rows, row_bytes)
        need = wpad // 8
        if cnt.shape[1] < need:
            cnt = np.pad(cnt, ((0, 0), (0, need - cnt.shape[1])))
        cnt = cnt[:, :need].reshape(nb, f0, need).sum(1)          # (nb, Wl) при f0 = 8
        lvl[y // f0: y // f0 + nb] = np.minimum(255, (cnt * 255 + 32) // 64).astype(np.uint8)
        if progress:
            progress(y1 / bt.H)
    pyr = {f0: lvl}
    prev, pf = lvl, f0
    for f in LEVELS[1:]:
        k = f // pf
        h, w = prev.shape[0] // k, prev.shape[1] // k
        if h < 1 or w < 1:
            break
        nxt = prev[: h * k, : w * k].reshape(h, k, w, k).mean((1, 3)).astype(np.uint8)
        pyr[f] = nxt
        prev, pf = nxt, f
    return pyr


# ============================================================== слой
class Layer:
    def __init__(self, bt: BitTiff, color, idx):
        self.bt = bt
        self.color = color
        self.visible = tk.BooleanVar(value=True)
        self.dx = 0
        self.dy = 0
        self.pyr = None
        self.progress = 0.0

    def coverage(self, x0, y0, z, Wv, Hv):
        """Покрытие 0..1 для окна просмотра. x0, y0 — устройство (с учётом сдвига слоя)."""
        bt = self.bt
        inv = 1.0 / z
        lx0, ly0 = x0 - self.dx, y0 - self.dy
        if inv <= RAW_MAX_DOWNSCALE or self.pyr is None:
            if inv <= 1.0:                                   # увеличение: ближайший
                xs = np.floor(lx0 + (np.arange(Wv) + 0.5) * inv).astype(np.int64)
                ys = np.floor(ly0 + (np.arange(Hv) + 0.5) * inv).astype(np.int64)
                rx0, rx1 = int(xs.min()), int(xs.max()) + 1
                ry0, ry1 = int(ys.min()), int(ys.max()) + 1
                b = bt.bits(ry0, ry1, rx0, rx1)
                return b[(ys - ry0)[:, None], (xs - rx0)[None, :]].astype(np.float32)
            if self.pyr is None and inv > RAW_MAX_DOWNSCALE:
                return None
            k = max(1, int(round(inv)))                        # уменьшение: среднее k×k
            rx0, ry0 = int(np.floor(lx0)), int(np.floor(ly0))
            nx, ny = int(np.ceil(Wv * inv / k)) + 1, int(np.ceil(Hv * inv / k)) + 1
            b = bt.bits(ry0, ry0 + ny * k, rx0, rx0 + nx * k)
            cov = b.reshape(ny, k, nx, k).mean((1, 3), dtype=np.float32)
            xs = np.clip(((np.arange(Wv) + 0.5) * inv / k).astype(np.int64), 0, nx - 1)
            ys = np.clip(((np.arange(Hv) + 0.5) * inv / k).astype(np.int64), 0, ny - 1)
            return cov[ys[:, None], xs[None, :]]
        f = max([l for l in self.pyr if l <= inv] or [min(self.pyr)])
        a = self.pyr[f]
        xs = np.floor((lx0 + (np.arange(Wv) + 0.5) * inv) / f).astype(np.int64)
        ys = np.floor((ly0 + (np.arange(Hv) + 0.5) * inv) / f).astype(np.int64)
        okx = (xs >= 0) & (xs < a.shape[1])
        oky = (ys >= 0) & (ys < a.shape[0])
        c = a[np.clip(ys, 0, a.shape[0] - 1)[:, None], np.clip(xs, 0, a.shape[1] - 1)[None, :]]
        c = c.astype(np.float32) / 255.0
        c *= oky[:, None] & okx[None, :]
        return c


# ============================================================== окно
class ViewerApp:
    def __init__(self, root: tk.Misc, files=()):
        self.root = root
        root.title("HybridRIP Viewer — наложение 1-битных сепараций")
        root.geometry("1280x820")
        self.layers: list[Layer] = []
        self.z = 0.05
        self.x0 = 0.0
        self.y0 = 0.0
        self.q = queue.Queue()
        self._drag = None
        self._pending = False
        self._photo = None
        self._build()
        root.after(100, self._poll)
        for f in files:
            self.add_file(f)

    # ---------------------------------------------------------- интерфейс
    def _build(self):
        r = self.root
        left = ttk.Frame(r, padding=6)
        left.pack(side="left", fill="y")
        bar = ttk.Frame(left)
        bar.pack(fill="x")
        ttk.Button(bar, text="Открыть TIFF…", command=self.open_files).pack(side="left")
        ttk.Button(bar, text="Убрать", command=self.remove_selected).pack(side="left", padx=4)

        cols = ("vis", "name", "shift")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=14, selectmode="browse")
        self.tree.heading("vis", text="👁")
        self.tree.heading("name", text="Сепарация")
        self.tree.heading("shift", text="Сдвиг, px")
        self.tree.column("vis", width=30, anchor="center")
        self.tree.column("name", width=210)
        self.tree.column("shift", width=80, anchor="center")
        self.tree.pack(fill="y", expand=True, pady=6)
        self.tree.bind("<Double-1>", self._toggle_vis)
        self.tree.bind("<space>", self._toggle_vis)

        b2 = ttk.Frame(left)
        b2.pack(fill="x")
        ttk.Button(b2, text="Цвет…", command=self.pick_color).pack(side="left")
        ttk.Button(b2, text="Показать/скрыть", command=self._toggle_vis).pack(side="left", padx=4)
        ttk.Button(b2, text="Сброс сдвига", command=self.reset_shift).pack(side="left")
        b3 = ttk.Frame(left)
        b3.pack(fill="x", pady=6)
        ttk.Button(b3, text="Весь лист", command=self.fit).pack(side="left")
        ttk.Button(b3, text="1:1", command=lambda: self.set_zoom(1.0)).pack(side="left", padx=4)
        ttk.Button(b3, text="4:1", command=lambda: self.set_zoom(4.0)).pack(side="left")
        self.solo = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Только выбранная", variable=self.solo,
                        command=self.redraw).pack(anchor="w")
        ttk.Label(left, foreground="#666", justify="left", text=(
            "Колесо — масштаб, левая кнопка — двигать лист.\n"
            "Стрелки — сдвиг выбранной сепарации\n(Shift ×10, Ctrl ×100 px).\n"
            "PageUp / PageDown — выбрать соседнюю сепарацию.\n"
            "Двойной щелчок — показать/скрыть.")).pack(anchor="w", pady=8)

        right = ttk.Frame(r)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, bg="#f4f4f4", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.status = ttk.Label(right, text="Откройте 1-битные TIFF", anchor="w", padding=(6, 2))
        self.status.pack(fill="x")

        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<MouseWheel>", self._wheel)                       # Windows / macOS
        c.bind("<Button-4>", lambda e: self._wheel(e, 120))        # Linux
        c.bind("<Button-5>", lambda e: self._wheel(e, -120))
        c.bind("<Motion>", self._hover)
        # Стрелки = сдвиг сепарации. Привязка и к окну, и прямо к списку: у списка
        # есть свои стрелки (переключение строки), а привязка виджета срабатывает
        # раньше встроенной и возвращает "break", так что строка не переключается.
        for w in (self.root, self.tree, self.canvas):
            for key in ("Left", "Right", "Up", "Down"):
                for mod in ("", "Shift-", "Control-"):
                    w.bind(f"<{mod}{key}>", self._nudge)
            w.bind("<Prior>", lambda e: self._select_step(-1))   # PageUp
            w.bind("<Next>", lambda e: self._select_step(1))     # PageDown

    # ---------------------------------------------------------- файлы
    def open_files(self):
        fs = filedialog.askopenfilenames(filetypes=[("TIFF", "*.tif *.tiff"), ("Все", "*.*")])
        for f in fs:
            self.add_file(f)

    def add_file(self, path):
        try:
            bt = BitTiff(path)
        except Exception as e:
            messagebox.showerror("Viewer", f"{os.path.basename(path)}:\n{e}")
            return
        if self.layers and abs(bt.dpi - self.layers[0].bt.dpi) > 0.5:
            messagebox.showwarning("Viewer", f"{bt.name}: {bt.dpi:g} dpi, у первого файла "
                                             f"{self.layers[0].bt.dpi:g} dpi — масштаб будет разный.")
        layer = Layer(bt, guess_color(bt.name, len(self.layers)), len(self.layers))
        self.layers.append(layer)
        iid = str(id(layer))
        self.tree.insert("", "end", iid=iid, values=("●", bt.name, "0, 0"))
        self._tag_color(layer)
        self.tree.selection_set(iid)
        if len(self.layers) == 1:
            self.fit()
        self.status.config(text=f"{bt.name}: {bt.W}×{bt.H} px, {bt.dpi:g} dpi "
                                f"({bt.W * 25.4 / bt.dpi:.0f}×{bt.H * 25.4 / bt.dpi:.0f} мм)")

        def work():
            try:
                pyr = build_pyramid(bt, lambda p: self.q.put(("prog", layer, p)))
                self.q.put(("pyr", layer, pyr))
            except Exception as e:
                self.q.put(("err", layer, str(e)))
        threading.Thread(target=work, daemon=True).start()

    def _tag_color(self, layer):
        iid = str(id(layer))
        hx = "#%02x%02x%02x" % layer.color
        self.tree.tag_configure(iid, foreground=hx if sum(layer.color) < 600 else "#b59f00")
        self.tree.item(iid, tags=(iid,))

    def selected(self):
        s = self.tree.selection()
        if not s:
            return None
        for l in self.layers:
            if str(id(l)) == s[0]:
                return l
        return None

    def remove_selected(self):
        l = self.selected()
        if l:
            self.tree.delete(str(id(l)))
            self.layers.remove(l)
            self.redraw()

    def pick_color(self):
        l = self.selected()
        if not l:
            return
        c = colorchooser.askcolor(color="#%02x%02x%02x" % l.color, title=l.bt.name)
        if c and c[0]:
            l.color = tuple(int(v) for v in c[0])
            self._tag_color(l)
            self.redraw()

    def _toggle_vis(self, e=None):
        l = self.selected()
        if l:
            l.visible.set(not l.visible.get())
            self.tree.set(str(id(l)), "vis", "●" if l.visible.get() else "○")
            self.redraw()

    def reset_shift(self):
        l = self.selected()
        if l:
            l.dx = l.dy = 0
            self.tree.set(str(id(l)), "shift", "0, 0")
            self.redraw()

    def _nudge(self, e):
        l = self.selected()
        if not l:
            return
        step = 100 if e.state & 0x4 else 10 if e.state & 0x1 else 1
        d = {"Left": (-step, 0), "Right": (step, 0), "Up": (0, -step), "Down": (0, step)}[e.keysym]
        l.dx += d[0]
        l.dy += d[1]
        mm = 25.4 / l.bt.dpi
        self.tree.set(str(id(l)), "shift", f"{l.dx}, {l.dy}")
        self.status.config(text=f"{l.bt.name}: сдвиг {l.dx}, {l.dy} px "
                                f"({l.dx * mm:.3f}, {l.dy * mm:.3f} мм)")
        self.redraw()
        return "break"

    # ---------------------------------------------------------- навигация
    def extent(self):
        if not self.layers:
            return 1000, 1000
        return max(l.bt.W for l in self.layers), max(l.bt.H for l in self.layers)

    def fit(self):
        W, H = self.extent()
        cw, ch = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        self.z = min(cw / W, ch / H) * 0.97
        self.x0 = W / 2 - cw / 2 / self.z
        self.y0 = H / 2 - ch / 2 / self.z
        self.redraw()

    def set_zoom(self, z, cx=None, cy=None):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        cx = cw / 2 if cx is None else cx
        cy = ch / 2 if cy is None else cy
        dxp = self.x0 + cx / self.z
        dyp = self.y0 + cy / self.z
        self.z = float(np.clip(z, 1e-4, 32.0))
        self.x0 = dxp - cx / self.z
        self.y0 = dyp - cy / self.z
        self.redraw()

    def _wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        self.set_zoom(self.z * (1.25 if d > 0 else 0.8), e.x, e.y)

    def _press(self, e):
        self.canvas.focus_set()
        self._drag = (e.x, e.y, self.x0, self.y0)

    def _select_step(self, step):
        items = self.tree.get_children()
        if not items:
            return "break"
        cur = self.tree.selection()
        i = items.index(cur[0]) if cur else -1
        i = (i + step) % len(items)
        self.tree.selection_set(items[i])
        self.tree.see(items[i])
        self.redraw()
        return "break"

    def _motion(self, e):
        if self._drag:
            sx, sy, x0, y0 = self._drag
            self.x0 = x0 - (e.x - sx) / self.z
            self.y0 = y0 - (e.y - sy) / self.z
            self.redraw()

    def _release(self, e):
        self._drag = None

    def _hover(self, e):
        if not self.layers:
            return
        dpi = self.layers[0].bt.dpi
        x = self.x0 + e.x / self.z
        y = self.y0 + e.y / self.z
        self.status.config(text=f"x {x * 25.4 / dpi:8.2f} мм   y {y * 25.4 / dpi:8.2f} мм   "
                                f"({int(x)}, {int(y)} px)   масштаб {self.z * 100:.1f} %")

    # ---------------------------------------------------------- отрисовка
    def redraw(self):
        if not self._pending:
            self._pending = True
            self.root.after(15, self._render)

    def _render(self):
        self._pending = False
        c = self.canvas
        Wv, Hv = max(c.winfo_width(), 2), max(c.winfo_height(), 2)
        img = np.ones((Hv, Wv, 3), np.float32)
        sel = self.selected()
        waiting = []
        for l in self.layers:
            if not l.visible.get() or (self.solo.get() and l is not sel):
                continue
            cov = l.coverage(self.x0, self.y0, self.z, Wv, Hv)
            if cov is None:
                waiting.append(f"{l.bt.name} {l.progress * 100:.0f} %")
                continue
            col = np.array(l.color, np.float32) / 255.0
            img *= 1.0 - cov[..., None] * (1.0 - col)
        # граница листа
        W, H = self.extent()
        out = (img * 255).astype(np.uint8)
        self._photo = ImageTk.PhotoImage(Image.fromarray(out))
        c.delete("all")
        c.create_image(0, 0, image=self._photo, anchor="nw")
        x1, y1 = (0 - self.x0) * self.z, (0 - self.y0) * self.z
        x2, y2 = (W - self.x0) * self.z, (H - self.y0) * self.z
        c.create_rectangle(x1, y1, x2, y2, outline="#999")
        if waiting:
            c.create_text(10, 10, anchor="nw", fill="#a00",
                          text="Строится обзор: " + ", ".join(waiting) + " — приблизьте или подождите")

    def _poll(self):
        try:
            while True:
                kind, layer, val = self.q.get_nowait()
                if kind == "prog":
                    layer.progress = val
                    if int(val * 100) % 10 == 0:
                        self.redraw()
                elif kind == "pyr":
                    layer.pyr = val
                    self.redraw()
                elif kind == "err":
                    messagebox.showerror("Viewer", f"{layer.bt.name}: {val}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


def open_viewer(files=(), master=None):
    root = tk.Toplevel(master) if master else tk.Tk()
    app = ViewerApp(root, files)
    return root, app


if __name__ == "__main__":
    root, _ = open_viewer(sys.argv[1:])
    root.mainloop()
