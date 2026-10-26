"""
Шкала для промера растискивания: по ряду плашек для AM, FM и линейки,
растрированных этой же программой БЕЗ компенсации и без мин. точки,
плюс пустой шаблон файла промеров с теми же ступенями.

Порядок работы:
  1. «Шкала для промера» -> calibration_<...>.tif + template_<...>.csv
  2. вывести шкалу на форму нужного полимера, изготовить форму (и/или оттиск)
  3. промерить плашки (прибор для форм / денситометр), вписать значения в шаблон
  4. переименовать шаблон по названию полимера и положить в папку compensation
"""
from __future__ import annotations
import datetime as _dt
import json
import os
import numpy as np
import cv2

from .compensation import DEFAULT_STEPS, write_template

ROWS = (("am", "AM"), ("fm", "FM"), ("line", "LINE"))


MARGIN, LABEL_W, TITLE_H, NUM_H, ROW_GAP = 5.0, 26.0, 9.0, 6.0, 4.0


def patch_rects(steps=DEFAULT_STEPS, patch_mm=10.0, gap_mm=2.0):
    """{растр: [(x, y, w, h) мм для каждой ступени]} — общая разметка шкалы."""
    out = {}
    y = MARGIN + TITLE_H + NUM_H
    for s, _ in ROWS:
        out[s] = [(MARGIN + LABEL_W + i * (patch_mm + gap_mm), y, patch_mm, patch_mm)
                  for i in range(len(steps))]
        y += patch_mm + ROW_GAP
    return out


def measure_bitmap(tif: str, steps=DEFAULT_STEPS, patch_mm=10.0, gap_mm=2.0, inner_mm=6.0):
    """Доля печатающих пикселей в центре каждой плашки 1-бит шкалы (то, что ушло на форму)."""
    import tifffile
    with tifffile.TiffFile(tif) as t:
        p = t.pages[0]
        xr = p.tags["XResolution"].value
        dpi = xr[0] / xr[1]
        a = p.asarray()
    px = dpi / 25.4
    res = {}
    for s, rects in patch_rects(steps, patch_mm, gap_mm).items():
        vals = []
        for (x, y, w, h) in rects:
            cx, cy, r = x + w / 2, y + h / 2, inner_mm / 2
            blk = a[int((cy - r) * px):int((cy + r) * px), int((cx - r) * px):int((cx + r) * px)]
            vals.append(float(blk.mean()))
        res[s] = np.array(vals)
    return res


def make_chart(out_dir: str, cfg: dict, steps=DEFAULT_STEPS, patch_mm: float = 10.0,
               gap_mm: float = 2.0, log=print, compensated: bool = False) -> tuple[str, str]:
    """compensated=False — шкала для промера (без компенсации и мин. точки);
    compensated=True — контрольная шкала с выбранной компенсацией (проверка после калибровки)."""
    from .engine import screen_separation
    from .compensation import apply as apply_compensation

    c = json.loads(json.dumps(cfg))
    dpi = float(c.get("dpi", 2400))
    if compensated:
        c = apply_compensation(c, log)
    sc = c.setdefault("screens", {})
    for s, _ in ROWS:
        sc.setdefault(s, {})
        if not compensated:
            sc[s]["curve"] = [[0, 0], [1, 1]]      # без компенсации
            sc[s]["min_dot"] = 0.0                  # и без мин. точки — меряем «как есть»
    comp_name = (c.get("compensation_applied") or {}).get("name", "")
    c["compensation"] = None
    c["transition_mm"] = 0.0
    c["previews"] = False
    c["criteria"] = {"tone": {"anchors": [[0, "am"], [1, "am"]]},
                     "detail": {"enabled": False}, "masks": []}

    ppi = 600.0
    px = ppi / 25.4
    margin, label_w, title_h, num_h, row_gap = MARGIN, LABEL_W, TITLE_H, NUM_H, ROW_GAP
    n = len(steps)
    W_mm = margin * 2 + label_w + n * (patch_mm + gap_mm)
    H_mm = margin * 2 + title_h + num_h + len(ROWS) * (patch_mm + row_gap)
    W, H = int(round(W_mm * px)), int(round(H_mm * px))
    ink = np.zeros((H, W), np.uint16)               # 16 бит: ступени точно, без округления до 1/255
    full = 65535

    def text(s, x_mm, y_mm, size_mm):
        scale = size_mm * px / 30.0
        img = np.zeros((H, W), np.uint8)
        cv2.putText(img, s, (int(x_mm * px), int(y_mm * px)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 255, max(1, int(scale * 2.2)), cv2.LINE_AA)
        np.maximum(ink, (img > 127).astype(np.uint16) * full, out=ink)

    am, fm, ln = sc["am"], sc["fm"], sc["line"]
    date = _dt.date.today().isoformat()
    kind = f"CHECK [{comp_name}]" if compensated else "calibration"
    title = (f"HybridRIP {kind}  {dpi:g} dpi  AM {am.get('lpi', 150):g} lpi {am.get('shape', 'euclid')}"
             f"  FM {fm.get('element_px', 2)} px  LINE {ln.get('lpi', am.get('lpi', 150)):g} lpi  {date}")
    text(title, margin, margin + title_h * 0.65, 4.0)
    y = margin + title_h
    x0 = margin + label_w
    for i, st in enumerate(steps):
        text(str(st), x0 + i * (patch_mm + gap_mm) + 0.5, y + num_h * 0.7, 3.2)
    y += num_h
    masks = []
    for s, label in ROWS:
        text(label, margin, y + patch_mm * 0.7, 5.0)
        for i, st in enumerate(steps):
            xa = x0 + i * (patch_mm + gap_mm)
            ink[int(round(y * px)):int(round((y + patch_mm) * px)),
                int(round(xa * px)):int(round((xa + patch_mm) * px))] = int(round(st / 100 * full))
        masks.append({"rect_mm": [x0 - gap_mm / 2, y - row_gap / 2,
                                  n * (patch_mm + gap_mm), patch_mm + row_gap], "screen": s})
        y += patch_mm + row_gap
    c["criteria"]["masks"] = masks

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{dpi:g}dpi_{am.get('lpi', 150):g}lpi_fm{fm.get('element_px', 2)}px"
    if compensated:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in comp_name)
        tif = os.path.join(out_dir, f"check_{safe}_{tag}.tif")
        log(f"Контрольная шкала с компенсацией «{comp_name}»: {W_mm:.0f}×{H_mm:.0f} мм")
        screen_separation("K", ink, ppi, c, tif, idx=3, log=log)
        log(f"  -> {tif}")
        return tif, ""
    tif = os.path.join(out_dir, f"calibration_{tag}.tif")
    csv = os.path.join(out_dir, f"template_{tag}.csv")
    log(f"Шкала для промера: {W_mm:.0f}×{H_mm:.0f} мм, {n} ступеней × AM / FM / линейка")
    screen_separation("K", ink, ppi, c, tif, idx=3, log=log)
    write_template(csv, steps, {
        "name": "", "date": date, "dpi": f"{dpi:g}",
        "lpi": f"{am.get('lpi', 150):g}", "am_shape": am.get("shape", "euclid"),
        "fm_element_px": f"{fm.get('element_px', 2)}",
        "line_lpi": f"{ln.get('lpi', am.get('lpi', 150)):g}",
        "polymer": "", "measured_on": "форма / оттиск", "device": "",
        "min_dot_am": "", "min_dot_fm": "", "min_dot_line": "",
    })
    log(f"  -> {tif}\n  -> {csv}")
    return tif, csv
