"""Синтетическая тестовая шкала (CMYK TIFF, 300 ppi, 80×60 мм).

K: верх — плавный клин 0–100 %; середина — ровное поле 40 % (на нём маски
   переключают растр: проверка невидимой границы при одинаковом тоне);
   низ — «фото»-текстура и резкие фигуры (критерий детальности).
C: радиальная растяжка.
"""
import numpy as np
import tifffile
import cv2

PPI = 300
W_MM, H_MM = 80, 60
px = PPI / 25.4
W, H = int(W_MM * px), int(H_MM * px)
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
xm, ym = x / px, y / px

K = np.zeros((H, W), np.float32)
# 1) клин 0–100 % (0–15 мм по высоте)
band1 = ym < 15
K[band1] = np.clip((xm[band1] - 2) / 76, 0, 1)
# 2) ровное поле 40 % (17–35 мм)
band2 = (ym >= 17) & (ym < 35)
K[band2] = 0.40
# 3) текстура + резкие фигуры (37–60 мм)
band3 = ym >= 37
rng = np.random.default_rng(3)
noise = sum(cv2.resize(rng.random((max(2, H // s), max(2, W // s))).astype(np.float32),
                       (W, H), interpolation=cv2.INTER_CUBIC) / (i + 1)
            for i, s in enumerate((60, 20, 6, 2)))
noise = (noise - noise.min()) / (noise.max() - noise.min())
tex = 0.15 + 0.6 * noise * np.clip((xm - 40) / 10, 0, 1) + 0.25 * (xm < 40) * (0.5 + 0.5 * np.sin(xm * 0.6))
K[band3] = np.clip(tex[band3], 0, 1)
# резкие фигуры: «буквы»-прямоугольники и круг
for i in range(6):
    x0 = 4 + i * 5.5
    K[(ym > 52) & (ym < 57) & (xm > x0) & (xm < x0 + 0.8 + i * 0.4)] = 1.0
K[((xm - 65) ** 2 + (ym - 52) ** 2) < 16] = 0.05

C = np.clip(1 - np.hypot(xm - 40, ym - 30) / 45, 0, 1) * 0.9

cmyk = np.stack([C, np.zeros_like(C), np.zeros_like(C), K], -1)
img = (cmyk * 255 + 0.5).astype(np.uint8)
tifffile.imwrite("testchart_cmyk.tif", img, photometric="separated",
                 resolution=(PPI, PPI), resolutionunit="INCH")
print("testchart_cmyk.tif", img.shape)
