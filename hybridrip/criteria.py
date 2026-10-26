"""
Карта весов растров: для каждого пикселя (на входном разрешении) веса
w_am, w_line, w_fm, сумма = 1. Карта строится по критериям, затем
сглаживается по площади (ширина перехода в мм) — это и убирает резкие
границы между зонами разного растра.

Критерии (применяются по порядку, каждый следующий может перекрыть предыдущий):
  tone    — по тону (типично: FM в светах, AM в средних тонах, линейка/AM в тенях)
  detail  — по детальности/текстуре изображения (мелкие детали -> FM)
  masks   — по маскам-файлам (светлое = применить) и прямоугольникам в мм
"""
from __future__ import annotations
import numpy as np
import cv2

SCREENS = ("am", "line", "fm")


def _as_weights(spec) -> np.ndarray:
    if isinstance(spec, str):
        w = np.array([1.0 if s == spec else 0.0 for s in SCREENS])
    else:
        w = np.array([float(spec.get(s, 0.0)) for s in SCREENS])
    if w.sum() <= 0:
        raise ValueError(f"Пустой набор весов: {spec}")
    return w / w.sum()


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def tone_weights(tone: np.ndarray, anchors) -> np.ndarray:
    """anchors: [[тон, "fm" | {"am":0.5,"fm":0.5}], ...], тон 0..1.
    Между опорными точками веса интерполируются плавно (smoothstep)."""
    pts = sorted(((float(t), _as_weights(s)) for t, s in anchors), key=lambda p: p[0])
    ts = np.array([p[0] for p in pts])
    # таблица на 4096 значений тона (кусочно-smoothstep), затем поиск по таблице:
    # память не зависит от сложности кривой
    g = np.linspace(0.0, 1.0, 4096)
    idx = np.clip(np.searchsorted(ts, g, side="right") - 1, 0, len(ts) - 1)
    idx2 = np.clip(idx + 1, 0, len(ts) - 1)
    t0, t1 = ts[idx], ts[idx2]
    f = np.where(t1 > t0, smoothstep(0, 1, (g - t0) / np.maximum(t1 - t0, 1e-9)), 0.0)
    out = np.empty((len(SCREENS),) + tone.shape, np.float32)
    for k in range(len(SCREENS)):
        vals = np.array([p[1][k] for p in pts])
        table = (vals[idx] * (1 - f) + vals[idx2] * f).astype(np.float32)
        out[k] = np.interp(tone, g, table)
    return out


def detail_measure(tone: np.ndarray, px_per_mm: float, radius_mm: float) -> np.ndarray:
    """Локальная «детальность»: средняя величина градиента в окрестности."""
    gx = cv2.Sobel(tone, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(tone, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.sqrt(gx * gx + gy * gy)
    # нормируем на мм, чтобы порог не зависел от входного разрешения
    g = g * px_per_mm
    sig = max(radius_mm * px_per_mm, 0.5)
    return cv2.GaussianBlur(g, (0, 0), sig)


def _apply_override(w: np.ndarray, strength: np.ndarray, screen) -> np.ndarray:
    target = _as_weights(screen)[:, None, None].astype(np.float32)
    s = np.clip(strength, 0, 1)[None]
    return w * (1 - s) + target * s


def build_weights(tone: np.ndarray, cfg: dict, px_per_mm: float,
                  load_mask=None) -> tuple[np.ndarray, dict]:
    crit = cfg.get("criteria", {})
    info = {}
    # 1) тон
    tc = crit.get("tone", {"anchors": [[0, "am"], [1, "am"]]})
    w = tone_weights(tone, tc["anchors"])

    # 2) детальность
    dc = crit.get("detail")
    if dc and dc.get("enabled", True):
        d = detail_measure(tone, px_per_mm, dc.get("radius_mm", 0.4))
        s = smoothstep(dc["low"], dc["high"], d) * dc.get("strength", 1.0)
        info["detail_p50_p95"] = (float(np.percentile(d, 50)), float(np.percentile(d, 95)))
        w = _apply_override(w, s, dc.get("screen", "fm"))

    # 3) маски и области
    h, wd = tone.shape
    for m in crit.get("masks", []):
        if "file" in m:
            mk = load_mask(m["file"], (wd, h))
        elif "rect_mm" in m:
            x, y, rw, rh = [v * px_per_mm for v in m["rect_mm"]]
            mk = np.zeros((h, wd), np.float32)
            mk[int(round(y)):int(round(y + rh)), int(round(x)):int(round(x + rw))] = 1.0
        else:
            continue
        w = _apply_override(w, mk * m.get("strength", 1.0), m["screen"])

    # 4) сглаживание: ширина перехода
    tr = cfg.get("transition_mm", 1.5)
    if tr > 0:
        sig = tr * px_per_mm / 2.56         # transition_mm = ширина перехода по уровню 10–90 %
        for k in range(w.shape[0]):
            w[k] = cv2.GaussianBlur(w[k], (0, 0), sig, borderType=cv2.BORDER_REFLECT)
    w = np.clip(w, 0, None)
    w /= np.maximum(w.sum(0, keepdims=True), 1e-6)
    return w.astype(np.float32), info
