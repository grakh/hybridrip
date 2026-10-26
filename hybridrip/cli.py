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
    ap.add_argument("-o", "--out", default="out", help="папка для результата")
    ap.add_argument("--dpi", type=float, help="разрешение устройства (перекрывает конфиг)")
    ap.add_argument("--ppi", type=float, help="разрешение входа, если в файле не указано")
    ap.add_argument("--mode", choices=["morph", "dither"], help="способ смешивания растров")
    ap.add_argument("--transition", type=float, help="ширина перехода, мм")
    ap.add_argument("--comp", help="компенсация: имя файла из папки compensation, путь или none "
                                   "(по умолчанию — из settings.json)")
    ap.add_argument("--list-comp", action="store_true", help="показать файлы компенсаций и выйти")
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
    if a.dpi: cfg["dpi"] = a.dpi
    if a.ppi: cfg["input_ppi"] = a.ppi
    if a.mode: cfg["blend_mode"] = a.mode
    if a.transition is not None: cfg["transition_mm"] = a.transition
    if a.comp is not None:
        cfg["compensation"] = None if a.comp.lower() in ("none", "-") else a.comp
    elif "compensation" not in cfg:
        cfg["compensation"] = comp.load_settings().get("default_compensation") or None
    run(a.input, cfg, a.out, cfg_dir=os.path.dirname(os.path.abspath(cfg_path)))


if __name__ == "__main__":
    sys.exit(main())
