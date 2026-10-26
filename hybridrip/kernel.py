"""
Быстрое попиксельное ядро растрирования (Numba, все ядра процессора).

Для каждого пикселя устройства за один проход:
  тон (билинейно из входа) -> веса растров (билинейно из грубой сетки)
  -> смешанная тоновая кривая и мин. точка -> пороги AM / линейка / FM
  -> равномерное смешивание порогов -> бит -> упаковка в байты.
Память: только входная сепарация (uint8/uint16) + одна полоса выхода.
"""
from __future__ import annotations
import math
import numpy as np

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
    NUMBA_ERROR = ""
except Exception as _e:  # pragma: no cover
    HAVE_NUMBA = False
    NUMBA_ERROR = f"{type(_e).__name__}: {_e}"

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]
    prange = range

SHAPES = {"euclid": 0, "round": 1, "ellipse": 2, "square": 3, "line": 4}
LUT_N = 16384
F_N = 4096


@njit(cache=True, fastmath=True, inline="always")
def _spot(shape, u, v):
    tau = 6.283185307179586
    if shape == 0:
        return -(math.cos(tau * u) + math.cos(tau * v))
    if shape == 1:
        return u * u + v * v
    if shape == 2:
        return -(math.cos(tau * u) + 0.75 * math.cos(tau * v))
    if shape == 3:
        return max(abs(u), abs(v))
    return -math.cos(tau * u)


@njit(cache=True, fastmath=True, inline="always")
def _periodic(x, y, period, ca, sa, shape, lut, smin, sinv):
    xc = x + 0.5
    yc = y + 0.5
    u = (xc * ca + yc * sa) / period
    v = (-xc * sa + yc * ca) / period
    u = u - math.floor(u + 0.5)
    v = v - math.floor(v + 0.5)
    s = _spot(shape, u, v)
    p = (s - smin) * sinv
    if p <= 0.0:
        return lut[0]
    n = lut.shape[0] - 1
    if p >= n:
        return lut[n]
    i = int(p)
    f = p - i
    return lut[i] * (1.0 - f) + lut[i + 1] * f


@njit(cache=True, fastmath=True, inline="always")
def _eq_mix(t1, t2, a):
    """Аналитическое равномерное смешивание двух независимых порогов."""
    b = 1.0 - a
    z = a * t1 + b * t2
    lo = min(a, b)
    hi = max(a, b)
    if lo < 1e-4:
        return z
    ab2 = 2.0 * lo * hi
    if z < lo:
        out = z * z / ab2
    elif z <= hi:
        out = (z - 0.5 * lo) / hi
    else:
        out = 1.0 - (1.0 - z) * (1.0 - z) / ab2
    return min(max(out, 0.0), 0.999999)


@njit(cache=True, fastmath=True, inline="always")
def _emp_mix(t1, t2, a, F):
    """Табличное равномерное смешивание для коррелированных растров (AM+линейка)."""
    z = a * t1 + (1.0 - a) * t2
    nb = F.shape[0]
    pos = a * (nb - 1)
    j = int(pos)
    if j > nb - 2:
        j = nb - 2
    fa = pos - j
    nz = F.shape[1] - 1
    pz = z * nz
    k = int(pz)
    if k > nz - 1:
        k = nz - 1
    if k < 0:
        k = 0
    fz = pz - k
    f0 = F[j, k] * (1.0 - fz) + F[j, k + 1] * fz
    f1 = F[j + 1, k] * (1.0 - fz) + F[j + 1, k + 1] * fz
    out = f0 * (1.0 - fa) + f1 * fa
    return min(max(out, 0.0), 0.999999)


@njit(cache=True, fastmath=True, inline="always")
def _bilinear(img, fx, fy):
    h, w = img.shape
    if fx < 0.0:
        fx = 0.0
    if fy < 0.0:
        fy = 0.0
    if fx > w - 1:
        fx = w - 1.0
    if fy > h - 1:
        fy = h - 1.0
    x0 = int(fx)
    y0 = int(fy)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    ax = fx - x0
    ay = fy - y0
    top = img[y0, x0] * (1.0 - ax) + img[y0, x1] * ax
    bot = img[y1, x0] * (1.0 - ax) + img[y1, x1] * ax
    return top * (1.0 - ay) + bot * ay


@njit(cache=True, fastmath=True, inline="always")
def _lut(curve, t):
    n = curve.shape[0] - 1
    p = t * n
    i = int(p)
    if i >= n:
        return curve[n]
    f = p - i
    return curve[i] * (1.0 - f) + curve[i + 1] * f


@njit(parallel=True, cache=True, fastmath=True)
def render_strip(y0, Wreal, out, ink, ink_inv_max, inv_scale,
                 wgrid, inv_scale_c,
                 curves, min_dots,
                 am_p, am_ca, am_sa, am_shape, am_lut, am_smin, am_sinv,
                 ln_p, ln_ca, ln_sa, ln_shape, ln_lut, ln_smin, ln_sinv,
                 fm_mask, fm_e,
                 emp_F, use_emp, mode, selector,
                 counts):
    nrows, nbytes = out.shape
    W = nbytes * 8
    fm_tile = fm_mask.shape[0]
    sel_tile = selector.shape[0]
    zero_tol = 0.5 * ink_inv_max
    for r in prange(nrows):
        y = y0 + r
        fy = (y + 0.5) * inv_scale - 0.5
        fyc = (y + 0.5) * inv_scale_c - 0.5
        cnt = 0
        for bx in range(nbytes):
            byte = 0
            for bit in range(8):
                x = bx * 8 + bit
                if x >= Wreal:
                    break
                fx = (x + 0.5) * inv_scale - 0.5
                t = _bilinear(ink, fx, fy) * ink_inv_max
                if t <= zero_tol:
                    continue
                # веса растров
                fxc = (x + 0.5) * inv_scale_c - 0.5
                wa = _bilinear(wgrid[0], fxc, fyc)
                wl = _bilinear(wgrid[1], fxc, fyc)
                wf = _bilinear(wgrid[2], fxc, fyc)
                if wa < 0.0: wa = 0.0
                if wl < 0.0: wl = 0.0
                if wf < 0.0: wf = 0.0
                s = wa + wl + wf
                if s <= 0.0:
                    wa = 1.0; s = 1.0
                wa /= s; wl /= s; wf /= s
                # смешанная тоновая кривая и минимальная точка
                if t >= 0.999:
                    te = 1.0
                else:
                    te = wa * _lut(curves[0], t) + wl * _lut(curves[1], t) + wf * _lut(curves[2], t)
                    md = wa * min_dots[0] + wl * min_dots[1] + wf * min_dots[2]
                    if te < md:
                        te = md
                # пороги и смешивание
                ta = 0.0
                tl = 0.0
                tf = 0.0
                if wa > 1e-4:
                    ta = _periodic(x, y, am_p, am_ca, am_sa, am_shape, am_lut, am_smin, am_sinv)
                if wl > 1e-4:
                    tl = _periodic(x, y, ln_p, ln_ca, ln_sa, ln_shape, ln_lut, ln_smin, ln_sinv)
                if wf > 1e-4:
                    tf = fm_mask[(y // fm_e) % fm_tile, (x // fm_e) % fm_tile]
                if mode == 1:  # dither: попиксельный выбор растра
                    sv = selector[y % sel_tile, x % sel_tile]
                    if sv < wa:
                        T = ta
                    elif sv < wa + wl:
                        T = tl
                    else:
                        T = tf
                else:          # morph: равномерное смешивание порогов
                    wal = wa + wl
                    if wa > 1e-4 and wl > 1e-4:
                        a = wa / wal
                        if use_emp:
                            t12 = _emp_mix(ta, tl, a, emp_F)
                        else:
                            t12 = _eq_mix(ta, tl, a)
                    elif wa > 1e-4:
                        t12 = ta
                    else:
                        t12 = tl
                    if wf > 1e-4 and wal > 1e-4:
                        T = _eq_mix(t12, tf, wal)
                    elif wf > 1e-4:
                        T = tf
                    else:
                        T = t12
                if te > T:
                    byte |= 128 >> bit
                    cnt += 1
            out[r, bx] = byte
        counts[r] = cnt


@njit(parallel=True, cache=True)
def block_coverage(packed, W, pf):
    """Превью: доля печатающих пикселей в блоках pf×pf (0..255)."""
    nrows = packed.shape[0] // pf
    ncols = W // pf
    res = np.zeros((nrows, ncols), np.uint8)
    for br in prange(nrows):
        for bc in range(ncols):
            c = 0
            for yy in range(br * pf, br * pf + pf):
                for xx in range(bc * pf, bc * pf + pf):
                    if packed[yy, xx >> 3] & (128 >> (xx & 7)):
                        c += 1
            res[br, bc] = 255 - (255 * c) // (pf * pf)
    return res


# ---------------------------------------------------------------- подготовка таблиц
def spot_lut(shape_name: str, n_samples: int = 768):
    """LUT: значение spot-функции -> равномерный порог (через эмпирическую CDF)."""
    from .screens import _spot as spot_np
    g = (np.arange(n_samples) + 0.5) / n_samples - 0.5
    u, v = np.meshgrid(g, g)
    s = np.sort(spot_np(shape_name, u, v).ravel())
    smin, smax = float(s[0]), float(s[-1])
    grid = np.linspace(smin, smax, LUT_N)
    # доля значений <= grid (середина диапазона одинаковых значений)
    lo = np.searchsorted(s, grid, side="left")
    hi = np.searchsorted(s, grid, side="right")
    cdf = ((lo + hi) * 0.5 / s.size).astype(np.float32)
    return cdf, smin, (LUT_N - 1) / (smax - smin)


def empirical_F(s1: np.ndarray, s2: np.ndarray, nbins: int = 33):
    """Таблицы F_j(z) для сетки весов a_j: функция распределения смеси a*T1+(1-a)*T2."""
    s1 = s1.ravel().astype(np.float32)
    s2 = s2.ravel().astype(np.float32)
    z = np.linspace(0.0, 1.0, F_N).astype(np.float32)
    F = np.empty((nbins, F_N), np.float32)
    for j, a in enumerate(np.linspace(0.0, 1.0, nbins)):
        m = np.sort(a * s1 + (1 - a) * s2)
        F[j] = np.searchsorted(m, z, side="right") / m.size
    return F


def curve_lut(pts, n: int = 1024):
    t = np.linspace(0, 1, n, dtype=np.float32)
    if not pts:
        return t
    p = np.array(sorted(pts), np.float32)
    return np.interp(t, p[:, 0], p[:, 1]).astype(np.float32)
