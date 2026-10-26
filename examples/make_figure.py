"""Наглядная иллюстрация: фрагменты 1-битного результата + профиль тона
в зоне перехода (равномерное смешивание против «наивного»)."""
import sys
import numpy as np
import cv2
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from hybridrip.screens import periodic_threshold, FMScreen
from hybridrip.blend import equalized_mix

out = sys.argv[1] if len(sys.argv) > 1 else "figure.png"
a = tifffile.imread("out/testchart_cmyk_K.tif")
px = 2400 / 25.4

def crop(x_mm, y_mm, w_mm, h_mm):
    c = a[int(y_mm * px):int((y_mm + h_mm) * px), int(x_mm * px):int((x_mm + w_mm) * px)]
    return np.where(c, 0, 255).astype(np.uint8)

# синтетический переход AM -> FM на 40 %: наивное vs равномерное смешивание
W, H = 3000, 600
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
t_am = periodic_threshold(X, Y, dpi=2400, lpi=150, angle_deg=7.5, shape="euclid")
t_fm = FMScreen(3).threshold(X, Y)
wgt = np.clip((X - 900) / 1200, 0, 1)
wgt = wgt * wgt * (3 - 2 * wgt)
naive = 0.40 > ((1 - wgt) * t_am + wgt * t_fm)
good = 0.40 > equalized_mix(t_am, t_fm, 1 - wgt)
k = int(px)
prof = lambda b: np.convolve(b.mean(0), np.ones(k) / k, "valid") * 100
xmm = (np.arange(W - k + 1) + k / 2) / px

fig = plt.figure(figsize=(14, 10), dpi=110)
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.2], hspace=0.45)
panels = [
    (crop(18, 24, 20, 3), "Поле 40 %: линейчатый → AM (переход ≈ 2 мм, без скачка тона)"),
    (crop(43, 24, 20, 3), "Поле 40 %: AM → стохастика FM"),
    (crop(1, 6, 24, 3), "Клин в светах: FM (0–4 %) → AM (с 10 %)"),
]
for i, (img, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img, cmap="gray", interpolation="nearest", aspect="auto")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(gs[3])
ax.plot(xmm, prof(naive), color="#c0392b", lw=1.5, label="наивное смешивание порогов (T = a·T1 + b·T2)")
ax.plot(xmm, prof(good), color="#1f6f8b", lw=2, label="равномерное смешивание (HybridRIP)")
ax.axhline(40, color="#888", lw=0.8, ls="--")
ax.axvspan(900 / px, 2100 / px, color="#eee", zorder=-1)
ax.set_ylim(25, 45)
ax.set_xlabel("мм поперёк зоны перехода AM → FM (серым — зона перехода)")
ax.set_ylabel("покрытие на форме, %")
ax.set_title("Почему нет видимой границы: тон в зоне перехода сохраняется", loc="left", fontsize=11)
ax.legend(loc="lower left", fontsize=9, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(out, bbox_inches="tight")
print(out)
