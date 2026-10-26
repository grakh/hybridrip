"""
Ядро растрирования: контоновая сепарация -> 1-битный TIFF с гибридным растром.

Конвейер для каждой сепарации:
  1. карта весов растров по критериям на ГРУБОЙ управляющей сетке
     (control_ppi, по умолчанию 150 ppi) + сглаживание — веса всё равно
     плавные, а память на формате 1×0,6 м падает с гигабайт до десятков МБ
  2. полосами по выходу, попиксельно (Numba, все ядра):
     тон -> смешанная кривая растискивания и мин. точка -> пороги AM / линейка / FM
     -> равномерное смешивание -> бит
  3. упакованные биты сразу пишутся в TIFF; превью считается из них же
"""
from __future__ import annotations
import math
import os
import time
import json
import sys
import numpy as np
import cv2

from .screens import periodic_threshold, FMScreen, load_bluenoise
from .criteria import build_weights, SCREENS
from .tiffio import BilevelTiffWriter, ink_max
from . import kernel as K
from .surface import SurfaceScreen

DEFAULT_ANGLES = {"C": 67.5, "M": 37.5, "Y": 82.5, "K": 7.5}


def apply_curve(t: np.ndarray, pts) -> np.ndarray:
    if not pts:
        return t
    p = np.array(sorted(pts), np.float32)
    return np.interp(t, p[:, 0], p[:, 1]).astype(np.float32)


def channel_angle(name: str, idx: int, cfg: dict) -> float:
    ang = cfg.get("angles", {})
    if name in ang:
        return float(ang[name])
    alias = {"c": "C", "cyan": "C", "m": "M", "magenta": "M",
             "y": "Y", "yellow": "Y", "k": "K", "black": "K"}
    if name.lower() in alias:
        a = alias[name.lower()]
        return float(ang.get(a, DEFAULT_ANGLES[a]))
    return [22.5, 52.5, 82.5, 7.5, 37.5, 67.5][idx % 6]


def _load_mask_factory(cfg_dir):
    def load(path, size):
        p = path if os.path.isabs(path) else os.path.join(cfg_dir, path)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(p)
        m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
        return m.astype(np.float32) / 255.0
    return load


def _downsample(ink: np.ndarray, size) -> np.ndarray:
    """INTER_AREA по полосам: cv2 не принимает стороны > 32767 px."""
    W_in = ink.shape[1]
    Wc, Hc = size
    if ink.shape[0] < 30000 and W_in < 30000:
        return cv2.resize(ink, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    # грубый, но безопасный путь для гигантских входов
    ys = np.linspace(0, ink.shape[0], Hc + 1).astype(int)
    xs = np.linspace(0, W_in, Wc + 1).astype(int)
    out = np.empty((Hc, Wc), np.float32)
    for i in range(Hc):
        row = ink[ys[i]:ys[i + 1]].astype(np.float32).mean(0)
        out[i] = np.add.reduceat(row, xs[:-1]) / np.diff(xs)
    return out


def _numba_missing():
    return RuntimeError(
        "Не загружается пакет numba (нужен для быстрого растрирования).\n"
        f"Причина: {K.NUMBA_ERROR}\n"
        "Установите его в ЭТОТ Python командой:\n"
        f'"{sys.executable}" -m pip install --user numba')


def edge_mask(ink: np.ndarray, imax: float, radius: int = 2) -> np.ndarray:
    """Маска контуров на входном разрешении: рядом (в радиусе radius пикселей) есть и
    пусто (<= 2 %), и сплошная (>= 98 %). Там вместо растрирования — резкий край по 50 %,
    серые пиксели сглаживания дают субпиксельное положение края. Плавные переходы
    (фото, растяжки) такого скачка не имеют и растрируются как обычно. Считается полосами."""
    H, W = ink.shape
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    out = np.zeros((H, W), np.uint8)
    lo_t, hi_t = 0.02 * imax, 0.98 * imax
    step = 4096
    for y in range(0, H, step):
        a, b = max(0, y - radius), min(H, y + step + radius)
        blk = ink[a:b]
        lo = (blk <= lo_t).astype(np.uint8)
        hi = (blk >= hi_t).astype(np.uint8)
        if not lo.any() or not hi.any():
            continue
        for x in range(0, W, 30000):                  # cv2: стороны до 32767 px
            xa, xb = max(0, x - radius), min(W, x + 30000 + radius)
            e = cv2.dilate(lo[:, xa:xb], k) & cv2.dilate(hi[:, xa:xb], k)
            out[y:min(H, y + step), x:min(W, x + 30000)] = \
                e[y - a:y - a + min(step, H - y), x - xa:x - xa + min(30000, W - x)]
    return out


def screen_separation(name: str, ink: np.ndarray, ppi: float, cfg: dict,
                      out_path: str, idx: int = 0, cfg_dir: str = ".",
                      preview_dir: str | None = None, log=print) -> dict:
    if not K.HAVE_NUMBA:
        raise _numba_missing()
    t0 = time.time()
    dpi = float(cfg.get("dpi", 2400))
    scale = dpi / ppi
    H_in, W_in = ink.shape
    imax = ink_max(ink)

    # --- 1. веса на грубой сетке
    cppi = min(float(cfg.get("control_ppi", 150)), ppi)
    Wc, Hc = max(2, int(round(W_in * cppi / ppi))), max(2, int(round(H_in * cppi / ppi)))
    coarse = _downsample(ink, (Wc, Hc)) / imax
    w, info = build_weights(coarse, cfg, cppi / 25.4, _load_mask_factory(cfg_dir))
    w = np.ascontiguousarray(w, np.float32)

    sc = cfg.get("screens", {})
    am, ln, fm = sc.get("am", {}), sc.get("line", {}), sc.get("fm", {})
    curves = np.stack([K.curve_lut(sc.get(s, {}).get("curve")) for s in SCREENS])
    min_dots = np.array([float(sc.get(s, {}).get("min_dot", 0.0)) for s in SCREENS], np.float64)

    # ожидаемое покрытие (по грубой сетке, для контроля)
    eff = sum(w[k] * apply_curve(coarse, sc.get(s, {}).get("curve")) for k, s in enumerate(SCREENS))
    md = sum(w[k] * min_dots[k] for k in range(3))
    nz = coarse > 0.5 / 255
    eff = np.where(nz & (eff < md), md, eff) * nz
    expected = float(np.where(coarse >= 0.999, 1.0, eff).mean())

    # --- 2. параметры растров
    angle = channel_angle(name, idx, cfg)
    am_lpi = float(am.get("lpi", 150))
    ln_lpi = float(ln.get("lpi", am_lpi))
    am_shape = am.get("shape", "euclid")
    ln_shape = ln.get("shape", "line")
    ln_angle = angle + float(ln.get("angle_offset", 0.0))
    am_lut, am_smin, am_sinv = K.spot_lut(am_shape)
    ln_lut, ln_smin, ln_sinv = K.spot_lut(ln_shape)
    a1, a2 = math.radians(angle), math.radians(ln_angle)
    fms = FMScreen(int(fm.get("element_px", 2)), seed=int(fm.get("seed", 1)), channel_index=idx)
    mode = 1 if cfg.get("blend_mode", "morph") == "dither" else 0
    selector = load_bluenoise(128, seed=7).astype(np.float32)

    use_emp = bool(mode == 0 and float((w[0] * w[1]).max()) > 1e-8)
    if use_emp:
        Ys, Xs = np.mgrid[0:1024, 0:1024].astype(np.float32)
        emp_F = K.empirical_F(
            periodic_threshold(Xs, Ys, dpi=dpi, lpi=am_lpi, angle_deg=angle, shape=am_shape),
            periodic_threshold(Xs, Ys, dpi=dpi, lpi=ln_lpi, angle_deg=ln_angle, shape=ln_shape))
    else:
        emp_F = np.zeros((2, 2), np.float32)

    # --- 3. полосы
    W = int(round(W_in * scale))
    H = int(round(H_in * scale))
    W8 = (W + 7) // 8
    pf = int(cfg.get("preview_factor", 0)) or max(1, math.ceil(max(W, H) / int(cfg.get("preview_max_px", 3000))))
    rows = max(pf, (int(cfg.get("strip_pixels", 16_000_000)) // max(W, 1)) // pf * pf)
    use_edge = bool(cfg.get("hard_edges", True))
    edge = edge_mask(ink, imax) if use_edge else np.zeros((1, 1), np.uint8)
    surf = SurfaceScreen(cfg.get("surface"))
    mg = surf.margin if surf.enabled else 0          # запас строк сверху/снизу для distance transform
    buf = np.empty((rows + 2 * mg, W8), np.uint8)
    counts = np.empty(rows + 2 * mg, np.int64)
    removed_total = 0
    writer = BilevelTiffWriter(out_path, W, H, dpi, rows_per_strip=64,
                               compression=cfg.get("compression", "lzw"))
    log(f"  выход {W}×{H} px ({W / dpi * 25.4:.0f}×{H / dpi * 25.4:.0f} мм @ {dpi:g} dpi), "
        f"угол {angle:g}°, сетка весов {Wc}×{Hc}")
    if surf.enabled:
        log(f"  {surf.describe()}")
    prev_rows = []
    crops = cfg.get("preview_crops_mm", [])
    crop_bufs = [[] for _ in crops]
    covered = 0
    pxmm = dpi / 25.4
    last = 0.0
    try:
        for y0 in range(0, H, rows):
            n = min(rows, H - y0)
            ya, yb = max(0, y0 - mg), min(H, y0 + n + mg)
            big = buf[:yb - ya]
            K.render_strip(ya, W, big, ink, 1.0 / imax, 1.0 / scale,
                           w, cppi / dpi, curves, min_dots,
                           dpi / am_lpi, math.cos(a1), math.sin(a1), K.SHAPES[am_shape], am_lut, am_smin, am_sinv,
                           dpi / ln_lpi, math.cos(a2), math.sin(a2), K.SHAPES[ln_shape], ln_lut, ln_smin, ln_sinv,
                           fms.mask, fms.e, emp_F, use_emp, mode, selector,
                           edge, use_edge, counts[:yb - ya])
            if surf.enabled:
                u8 = np.unpackbits(big, axis=1)[:, :W]
                u8 = np.ascontiguousarray(u8)
                surf.apply_unpacked(u8, ya)
                core = u8[y0 - ya:y0 - ya + n]
                before = int(counts[y0 - ya:y0 - ya + n].sum())
                out = np.packbits(core, axis=1)
                c_after = int(core.sum())
                removed_total += before - c_after
                covered += c_after
            else:
                out = big
                covered += int(counts[:n].sum())
            writer.write_packed(out)
            if preview_dir and n >= pf:
                prev_rows.append(K.block_coverage(out[: (n // pf) * pf], W, pf))
            for ci, (cx, cy, cw, ch) in enumerate(crops):
                cx0, cy0 = int(cx * pxmm), int(cy * pxmm)
                cx1, cy1 = min(W, int((cx + cw) * pxmm)), int((cy + ch) * pxmm)
                a, b = max(y0, cy0), min(y0 + n, cy1)
                if a < b and cx0 < cx1:
                    bits = np.unpackbits(out[a - y0:b - y0], axis=1)[:, cx0:cx1]
                    crop_bufs[ci].append(bits)
            now = time.time()
            if now - last > 0.5 or y0 + n >= H:
                el = now - t0
                done = (y0 + n) / H
                eta = el / done * (1 - done) if done > 0 else 0
                log(f"  {name}: {y0 + n}/{H} строк, осталось ~{eta:.0f} c", end="\r")
                last = now
    finally:
        writer.close()
    log("")

    res = {
        "separation": name, "file": out_path, "width_px": W, "height_px": H,
        "dpi": dpi, "angle": angle, "coverage_out": covered / (W * H),
        "coverage_in": expected, "seconds": round(time.time() - t0, 1), **info,
        "compensation": (cfg.get("compensation_applied") or {}).get("name", ""),
        "surface_removed": removed_total / (W * H),
    }
    if surf.enabled:
        log(f"  микроструктура: проколото {100 * removed_total / (W * H):.2f} % площади листа")
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name)
        if prev_rows:
            cv2.imwrite(os.path.join(preview_dir, f"{safe}_preview.png"), np.concatenate(prev_rows))
        rgb = np.stack([w[2], w[1], w[0]], -1)  # BGR: FM синий, линейка зелёный, AM красный
        cv2.imwrite(os.path.join(preview_dir, f"{safe}_weights.png"), (255 * rgb).astype(np.uint8))
        for ci, cb in enumerate(crop_bufs):
            if cb:
                cv2.imwrite(os.path.join(preview_dir, f"{safe}_crop{ci + 1}.png"),
                            np.where(np.concatenate(cb), 0, 255).astype(np.uint8))
    return res


def run(input_path: str, cfg: dict, out_dir: str, cfg_dir: str = ".", log=print):
    from .tiffio import read_any
    if not K.HAVE_NUMBA:
        raise _numba_missing()
    from .compensation import apply as apply_compensation
    cfg = apply_compensation(cfg, log)
    log("Чтение входа…")
    seps, ppi = read_any(input_path, cfg.get("input_ppi"))
    only = cfg.get("separations")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    results = []
    for i, (name, ink) in enumerate(seps):
        if only and name not in only:
            continue
        if callable(ink):
            ink = ink()
        if ink.max() <= 0:
            log(f"  {name}: пустая сепарация, пропуск")
            del ink
            continue
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name)
        out = os.path.join(out_dir, f"{base}_{safe}.tif")
        log(f"Растрирование {name}: вход {ink.shape[1]}×{ink.shape[0]} @ {ppi:g} ppi")
        r = screen_separation(name, ink, ppi, cfg, out, idx=i, cfg_dir=cfg_dir,
                              preview_dir=os.path.join(out_dir, "preview")
                              if cfg.get("previews", True) else None, log=log)
        del ink
        results.append(r)
        log(f"  -> {out}  ({r['width_px']}×{r['height_px']}, "
            f"покрытие {100*r['coverage_out']:.2f} % / ожидалось ≈{100*r['coverage_in']:.2f} %, "
            f"{r['seconds']} c)")
    with open(os.path.join(out_dir, f"{base}_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results
