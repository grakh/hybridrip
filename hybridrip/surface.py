"""
Микроструктура плашек (surface screening) со сплошным ободком.

После растрирования для каждого печатающего пикселя считается расстояние до
ближайшего непечатающего (distance transform). Если оно больше ширины ободка
`rim_px`, в пикселе может быть «прокол» по узору микроструктуры; плотность
проколов плавно нарастает на протяжении `fade_px`, чтобы не было видимой
границы между ободком и структурой.

Результат: плашки и крупные слившиеся элементы (тени) получают внутри
структуру, которая лучше удерживает и переносит краску, а край остаётся
сплошным и чётким. Элементы уже `min_size_px` (тонкие линии, мелкий текст,
перемычки между растровыми точками в полутонах) не затрагиваются вовсе.

Узоры (pattern):
  stoch — стохастический (blue noise), проколы ячейкой cell_px
  grid  — регулярная сетка круглых проколов с шагом period_px (angle 0° или 45°)
  lines — тонкие линии с шагом period_px (angle 0° или 45°)
  Регулярные узоры строятся точно по сетке пикселей (иначе неровные проколы и муар).
  bayer — упорядоченный узор 8×8 (ячейка cell_px)
"""
from __future__ import annotations
import math
import numpy as np
import cv2

from . import kernel as K
from .screens import load_bluenoise

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = K.njit
    prange = range

PATTERNS = {"stoch": 0, "grid": 1, "lines": 2, "bayer": 3}
DEFAULTS = {"enabled": False, "pattern": "stoch", "rim_px": 4, "fade_px": 4, "min_size_px": 24,
            "density": 0.15, "cell_px": 1, "period_px": 4, "angle": 45.0}


def _rank_uniform(primary: np.ndarray, *secondary: np.ndarray) -> np.ndarray:
    """Значения узора -> равномерные пороги (по рангу). Ничьи разбиваются вторичными
    ключами так, чтобы одинаковые элементы узора (все проколы / все линии) росли равномерно."""
    keys = [k.ravel() for k in reversed((primary,) + secondary)]
    keys.insert(0, np.arange(primary.size))           # последний ключ — индекс
    order = np.lexsort(tuple(keys))
    r = np.empty(primary.size, np.float64)
    r[order] = (np.arange(primary.size) + 0.5) / primary.size
    return r.reshape(primary.shape).astype(np.float32)


def _bitrev(v: np.ndarray, n: int) -> np.ndarray:
    """Перестановка «бит-реверс»: равномерно разносит порядок заполнения вдоль линии."""
    bits = max(1, int(np.ceil(np.log2(max(n, 2)))))
    out = np.zeros_like(v)
    for i in range(bits):
        out |= ((v >> i) & 1) << (bits - 1 - i)
    return out


def pixel_tile(kind: str, period: int, angle: float) -> np.ndarray:
    """Точная по пикселям плитка порогов для grid / lines. 45° — шахматный сдвиг узлов
    (grid) или диагональные линии (lines); иначе 0°."""
    p = max(2, int(round(period)))
    diag = abs((angle % 90) - 45) < 22.5
    if kind == "grid":
        n = 2 * p if diag else p
        y, x = np.mgrid[0:n, 0:n].astype(np.float64) + 0.5
        nodes = [(p / 2, p / 2), (p / 2 + p, p / 2 + p)] if diag else [(p / 2, p / 2)]
        d = np.full(x.shape, np.inf)
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        for (cx, cy) in nodes:
            for ox in (-n, 0, n):
                for oy in (-n, 0, n):
                    ex, ey = x - cx - ox, y - cy - oy
                    dd = ex * ex + ey * ey
                    m = dd < d
                    d[m], dx[m], dy[m] = dd[m], ex[m], ey[m]
        ang = np.round(np.arctan2(dy, dx), 6)           # одинаков для всех узлов
        return _rank_uniform(np.round(d, 6), ang)
    # lines
    y, x = np.mgrid[0:p, 0:p]
    u = ((x + y) % p) if diag else (x % p)
    d = np.abs(u + 0.5 - p / 2.0)
    along = _bitrev(x if diag else y, p)
    return _rank_uniform(d, u.astype(np.float64), along.astype(np.float64))


def _bayer(n=8):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return ((m + 0.5) / m.size).astype(np.float32)


@njit(parallel=True, cache=True, fastmath=True)
def _surface_pass(u8, dist, ya, rim, fade, density, ptype,
                  mask, cell, period, ca, sa, shape, lut, smin, sinv, removed):
    rows, W = u8.shape
    th = mask.shape[0]
    tw = mask.shape[1]
    for r in prange(rows):
        y = ya + r
        cnt = 0
        for x in range(W):
            if u8[r, x] == 0:
                continue
            d = dist[r, x]
            if d <= rim:
                continue
            f = 1.0
            if fade > 0.0:
                f = (d - rim) / fade
                if f > 1.0:
                    f = 1.0
            dens = density * f
            if ptype == 0 or ptype == 3:
                T = mask[(y // cell) % th, (x // cell) % tw]
            else:
                T = K._periodic(x, y, period, ca, sa, shape, lut, smin, sinv)
            if T < dens:
                u8[r, x] = 0
                cnt += 1
        removed[r] = cnt


class SurfaceScreen:
    def __init__(self, cfg: dict | None):
        p = dict(DEFAULTS)
        p.update(cfg or {})
        self.p = p
        self.enabled = bool(p.get("enabled")) and float(p.get("density", 0)) > 0
        self.rim = float(p["rim_px"])
        self.fade = float(p["fade_px"])
        self.R = max(self.rim, float(p.get("min_size_px", 0)) / 2.0)   # мин. «полутолщина» элемента
        self.margin = int(math.ceil(max(self.rim + self.fade, 2 * self.R))) + 2
        self.ptype = PATTERNS.get(p["pattern"], 0)
        self.cell = max(1, int(p["cell_px"]))
        if self.ptype == 3:
            self.mask = _bayer(8)
        elif self.ptype in (1, 2):
            self.mask = pixel_tile("grid" if self.ptype == 1 else "lines",
                                   int(p["period_px"]), float(p["angle"]))
            self.cell = 1
            self.ptype = 0          # дальше — тот же путь, что и для плитки-маски
        else:
            self.mask = load_bluenoise(128, seed=11).astype(np.float32)
        shape = "line" if self.ptype == 2 else "round"
        self.lut, self.smin, self.sinv = K.spot_lut(shape)
        self.shape = K.SHAPES[shape]
        a = math.radians(float(p["angle"]))
        self.ca, self.sa = math.cos(a), math.sin(a)
        self.period = max(2.0, float(p["period_px"]))
        self.density = float(p["density"])

    def describe(self) -> str:
        p = self.p
        geo = (f"шаг {p['period_px']} px, угол {p['angle']}°" if p["pattern"] in ("grid", "lines")
               else f"ячейка {self.cell} px")
        return (f"микроструктура {p['pattern']}: проколы {100 * self.density:g} %, {geo}, "
                f"ободок {p['rim_px']} px, нарастание {p['fade_px']} px, мин. элемент {p['min_size_px']} px")

    def apply_unpacked(self, u8: np.ndarray, ya: int) -> int:
        """u8 — (rows, W) uint8 0/1, изменяется на месте. ya — номер первой строки на листе."""
        if not self.enabled or not u8.any():
            return 0
        dist = cv2.distanceTransform(u8, cv2.DIST_L2, 5)
        if self.R > self.rim:
            # только элементы, где есть «ядро» толщиной >= min_size: пиксель допускается,
            # если до ядра можно дойти вглубь элемента (dist + d2 <= R)
            core = (dist > self.R).astype(np.uint8)
            if not core.any():
                return 0
            d2 = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
            dist[((d2 + dist) > self.R + 1.5) & (core == 0)] = 0.0
        removed = np.zeros(u8.shape[0], np.int64)
        _surface_pass(u8, dist, ya, self.rim, self.fade, self.density, self.ptype,
                      self.mask, self.cell, self.period, self.ca, self.sa, self.shape,
                      self.lut, self.smin, self.sinv, removed)
        return int(removed.sum())
