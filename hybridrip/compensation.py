"""
Компенсация растискивания по файлам промеров (по одному файлу на полимер / процесс).

Файл промеров — текст CSV (разделитель ; или табуляция или запятая, десятичная
запятая допускается), значения в процентах:

    # name: Полимер A 1.14, 2400 dpi
    # dpi: 2400
    # lpi: 150
    # min_dot_am: 2
    # min_dot_fm: 0.5
    target;am;fm;line
    0;0;0;0
    2;4.1;6.0;3.5
    ...
    100;100;100;100

  target — задано в файле (шкала печатается БЕЗ компенсации),
  am / fm / line — измерено на форме или оттиске для каждого вида растра.
  Колонка measured (или «замер») — одна кривая на все растры.
  Колонка aim (необязательно) — желаемый результат; по умолчанию линейный.
  Пустая ячейка — не измерено. Метаданные min_dot_* перекрывают мин. точку.

Компенсация c(t) — обратная функция к измеренной: подаём на форму c(t),
чтобы на ней получилось aim(t).
"""
from __future__ import annotations
import json
import os
import re
import numpy as np

SCREENS = ("am", "line", "fm")
EXTS = (".csv", ".txt", ".tsv")
NONE_LABEL = "— без компенсации —"

_ALIASES = {
    "target": ("target", "input", "file", "задано", "цель", "файл", "%", "tone", "тон", "вход"),
    "am": ("am", "ам", "амплитудный", "regular", "регулярный", "amplitude"),
    "fm": ("fm", "фм", "stoch", "stochastic", "стохастика", "стохастический"),
    "line": ("line", "lines", "линейка", "линейчатый", "lin"),
    "measured": ("measured", "all", "value", "замер", "измерено", "все"),
    "aim": ("aim", "цель_результат", "желаемое", "target_out"),
}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(HERE, "settings.json")


# ============================================================== настройки
def load_settings() -> dict:
    s = {"compensation_dir": "compensation", "default_compensation": ""}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s


def save_settings(s: dict):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def comp_dir(settings: dict | None = None) -> str:
    s = settings or load_settings()
    d = s.get("compensation_dir") or "compensation"
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def list_files(folder: str | None = None) -> list[str]:
    """Имена (без расширения) файлов промеров в папке, по алфавиту."""
    folder = folder or comp_dir()
    if not os.path.isdir(folder):
        return []
    names = [os.path.splitext(f)[0] for f in os.listdir(folder)
             if f.lower().endswith(EXTS) and not f.startswith(("_", "~"))]
    return sorted(set(names), key=str.lower)


def resolve(name_or_path: str | None, folder: str | None = None) -> str | None:
    """Имя из списка или путь -> путь к файлу; пусто / «без» -> None."""
    if not name_or_path or name_or_path in (NONE_LABEL, "none", "-"):
        return None
    if os.path.isfile(name_or_path):
        return name_or_path
    folder = folder or comp_dir()
    for ext in ("",) + EXTS:
        p = os.path.join(folder, name_or_path + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Файл компенсации «{name_or_path}» не найден в {folder}")


# ============================================================== разбор файла
def _num(s: str, comma_decimal: bool):
    s = s.strip().replace("%", "").replace(" ", "").replace(" ", "")
    if not s:
        return None
    if comma_decimal:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _key(h: str):
    h = h.strip().lower().replace(" ", "_")
    for k, al in _ALIASES.items():
        if h in al:
            return k
    return None


def parse(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    meta, rows, header, delim = {}, [], None, None
    for line in text.splitlines():
        st = line.strip()
        if not st:
            continue
        if st.startswith("#"):
            m = re.match(r"#\s*([\w\-]+)\s*[:=]\s*(.*)$", st)
            if m:
                meta[m.group(1).strip().lower()] = m.group(2).strip()
            continue
        if header is None:
            delim = ";" if ";" in st else "\t" if "\t" in st else ","
            header = [_key(h) for h in st.split(delim)]
            if "target" not in header:
                raise ValueError(f"{os.path.basename(path)}: в заголовке нет колонки target (задано)")
            continue
        rows.append(st.split(delim))
    if header is None:
        raise ValueError(f"{os.path.basename(path)}: нет строки заголовка")
    comma_decimal = delim != ","
    cols: dict[str, list] = {k: [] for k in header if k}
    for r in rows:
        vals = [_num(v, comma_decimal) for v in r] + [None] * len(header)
        for k, v in zip(header, vals):
            if k:
                cols[k].append(v)
    target = np.array([np.nan if v is None else v for v in cols["target"]], float)
    scale = 100.0 if np.nanmax(target) > 1.0 + 1e-9 else 1.0
    out = {"meta": meta, "name": meta.get("name") or os.path.splitext(os.path.basename(path))[0],
           "path": path, "curves": {}, "measured": {}, "aim": None, "min_dots": {}}

    def series(k):
        y = np.array([np.nan if v is None else v for v in cols[k]], float)
        ok = ~np.isnan(target) & ~np.isnan(y)
        return target[ok] / scale, y[ok] / scale

    if "aim" in cols:
        out["aim"] = series("aim")
    for s in SCREENS:
        if s in cols and len(series(s)[0]) >= 2:
            out["measured"][s] = series(s)
    if "measured" in cols and len(series("measured")[0]) >= 2:
        for s in SCREENS:
            out["measured"].setdefault(s, series("measured"))
    if not out["measured"]:
        raise ValueError(f"{os.path.basename(path)}: нет ни одной колонки с измерениями (am / fm / line / measured)")
    for s in SCREENS:
        v = meta.get(f"min_dot_{s}")
        if v not in (None, ""):
            out["min_dots"][s] = float(v.replace(",", ".")) / 100.0
    for s, (t, m) in out["measured"].items():
        out["curves"][s] = invert(t, m, out["aim"])
    return out


# ============================================================== обращение кривой
def _isotonic(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Монотонная (неубывающая) аппроксимация — сглаживает шум замеров."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = v
            vals.pop(); wts.pop(); cnt.pop()
    return np.repeat(vals, cnt)


def invert(t: np.ndarray, m: np.ndarray, aim=None, n: int = 101) -> list:
    """Измерено m(t) -> точки кривой компенсации [[вход, выход], ...] (0..1)."""
    order = np.argsort(t)
    t, m = np.asarray(t, float)[order], np.asarray(m, float)[order]
    if t[0] > 0:
        t, m = np.r_[0.0, t], np.r_[0.0, m]
    if t[-1] < 1:
        t, m = np.r_[t, 1.0], np.r_[m, 1.0]
    m = np.clip(_isotonic(m), 0.0, 1.0)
    # строго возрастающая для обращения (плато -> крошечный наклон)
    m = m + np.linspace(0.0, 1e-6, m.size)
    g = np.linspace(0.0, 1.0, n)
    if aim is not None:
        at, av = aim
        o = np.argsort(at)
        desired = np.interp(g, np.r_[0.0, at[o], 1.0], np.r_[0.0, av[o], 1.0])
    else:
        desired = g
    c = np.interp(desired, m, t)
    c = np.maximum.accumulate(np.clip(c, 0.0, 1.0))
    c[0], c[-1] = 0.0, 1.0
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(g, c)]


# ============================================================== применение
def apply(cfg: dict, log=print) -> dict:
    """Если в cfg задан compensation (имя или путь) — подставить кривые и мин. точки."""
    name = cfg.get("compensation")
    path = resolve(name) if name else None
    if not path:
        return cfg
    c = json.loads(json.dumps(cfg))
    comp = parse(path)
    sc = c.setdefault("screens", {})
    have = list(comp["curves"])
    for s in SCREENS:
        src = s if s in comp["curves"] else have[0]
        if src != s:
            log(f"  ВНИМАНИЕ: в «{comp['name']}» нет промера для {s.upper()}, взята кривая {src.upper()}")
        sc.setdefault(s, {})["curve"] = comp["curves"][src]
        if s in comp["min_dots"]:
            sc[s]["min_dot"] = comp["min_dots"][s]
    meta = comp["meta"]
    for key, cur in (("dpi", c.get("dpi")), ("lpi", sc.get("am", {}).get("lpi")),
                     ("fm_element_px", sc.get("fm", {}).get("element_px"))):
        if key in meta and cur is not None:
            try:
                if abs(float(meta[key].replace(",", ".")) - float(cur)) > 1e-6:
                    log(f"  ВНИМАНИЕ: промер сделан при {key}={meta[key]}, сейчас {cur} — кривая может не подходить")
            except ValueError:
                pass
    mins = ", ".join(f"{s.upper()} {100 * sc[s].get('min_dot', 0):g} %" for s in SCREENS)
    log(f"Компенсация: {comp['name']} ({os.path.basename(path)}); мин. точка: {mins}")
    c["compensation_applied"] = {"name": comp["name"], "file": path}
    return c


# ============================================================== шаблон промеров
DEFAULT_STEPS = (0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 97, 100)


def write_template(path: str, steps=DEFAULT_STEPS, meta: dict | None = None):
    lines = ["# Шаблон промеров HybridRIP. Заполните колонки am / fm / line значениями,",
             "# измеренными на форме (или оттиске) для шкалы, напечатанной этой же программой",
             "# БЕЗ компенсации. Переименуйте файл по названию полимера и положите в папку compensation.",
             "# Пустая ячейка = не измерено. Десятичная запятая допускается."]
    for k, v in (meta or {}).items():
        lines.append(f"# {k}: {v}")
    lines.append("target;am;fm;line")
    for s in steps:
        lines.append(f"{s};;;")
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
