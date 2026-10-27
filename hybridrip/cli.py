"""Командная строка:  python -m hybridrip вход.tif -c config.json -o out/"""
from __future__ import annotations
import argparse
import json
import os
import sys

from .engine import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hybridrip",
                                 description="Гибридное растрирование AM / линейчатый / FM с плавными переходами")
    ap.add_argument("input", nargs="?", help="TIFF (CMYK/grayscale, 8/16 бит), PNG/JPG или PDF/PS (нужен Ghostscript)")
    ap.add_argument("-c", "--config", default=None, help="JSON-конфигурация")
    ap.add_argument("-o", "--out", default=None,
                    help="папка для результата (по умолчанию — output_dir из settings.json, иначе ./out)")
    ap.add_argument("--dpi", type=float, help="разрешение устройства (перекрывает конфиг)")
    ap.add_argument("--ppi", type=float, help="разрешение входа, если в файле не указано")
    ap.add_argument("--mode", choices=["morph", "dither"], help="способ смешивания растров")
    ap.add_argument("--transition", type=float, help="ширина перехода, мм")
    ap.add_argument("--comp", help="компенсация: имя файла из папки compensation, путь или none "
                                   "(по умолчанию — из settings.json)")
    ap.add_argument("--list-comp", action="store_true", help="показать файлы компенсаций и выйти")
    ap.add_argument("--angle", action="append", default=[], metavar="ИМЯ=ГРАДУСЫ",
                    help='угол сепарации, можно несколько раз: --angle "PANTONE 485*=22.5" --angle C=15')
    ap.add_argument("--list-seps", action="store_true", help="показать сепарации файла и их углы, без растрирования")
    a = ap.parse_args(argv)
    from . import compensation as comp
    if a.list_comp:
        st = comp.load_settings()
        for n in comp.list_files():
            print(("* " if n == st.get("default_compensation") else "  ") + n)
        return 0
    if not a.input:
        ap.error("укажите входной файл")

    cfg_path = a.config or os.path.join(os.path.dirname(__file__), "..", "configs", "flexo_default.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    for spec in a.angle:
        if "=" not in spec:
            ap.error(f"--angle {spec}: нужно ИМЯ=ГРАДУСЫ")
        k, v = spec.rsplit("=", 1)
        cfg.setdefault("angles", {})[k.strip()] = float(v.replace(",", "."))
    if a.list_seps:
        from .tiffio import list_separations
        from .angles import angle_for
        lib = comp.load_settings().get("spot_angles", {})
        for i, n in enumerate(list_separations(a.input)):
            ang, src = angle_for(n, i, cfg, lib)
            print(f"  {n:30s} {ang:6g}°  ({src})")
        return 0
    if a.dpi: cfg["dpi"] = a.dpi
    if a.ppi: cfg["input_ppi"] = a.ppi
    if a.mode: cfg["blend_mode"] = a.mode
    if a.transition is not None: cfg["transition_mm"] = a.transition
    if a.comp is not None:
        cfg["compensation"] = None if a.comp.lower() in ("none", "-") else a.comp
    elif "compensation" not in cfg:
        cfg["compensation"] = comp.load_settings().get("default_compensation") or None
    out = a.out or comp.load_settings().get("output_dir") or "out"
    run(a.input, cfg, out, cfg_dir=os.path.dirname(os.path.abspath(cfg_path)))


if __name__ == "__main__":
    sys.exit(main())
