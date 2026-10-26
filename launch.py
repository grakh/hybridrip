"""Запуск HybridRIP: проверка Python и библиотек, при необходимости установка, затем окно."""
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# модуль импорта -> пакет pip
NEEDED = {
    "numpy": "numpy", "scipy": "scipy", "cv2": "opencv-python-headless",
    "tifffile": "tifffile", "PIL": "pillow", "numba": "numba", "imagecodecs": "imagecodecs",
}


def fail(msg):
    print("\n" + msg + "\n")
    sys.exit(1)


print(f"Python {sys.version.split()[0]}  ({sys.executable})")
if sys.version_info < (3, 9):
    fail("Нужен Python 3.9 или новее.")
try:
    import tkinter  # noqa: F401
except Exception:
    fail("В этом Python нет tkinter. Установите Python с python.org (компонент tcl/tk включён по умолчанию).")

if "--check" in sys.argv:
    # Проверка после install_libs.bat: версии всех библиотек + компиляция ядра
    ok = True
    for mod, pkg in NEEDED.items():
        try:
            m = importlib.import_module(mod)
            print(f"  OK    {pkg:24s} {getattr(m, '__version__', '')}")
        except Exception as e:
            ok = False
            print(f"  НЕТ   {pkg:24s} {type(e).__name__}: {e}")
    if ok:
        print("\nПроверка быстрого ядра (первый раз компилируется 10–30 с)…")
        try:
            sys.path.insert(0, HERE)
            import numpy as np
            from hybridrip.engine import screen_separation
            import tempfile
            ink = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (32, 1))
            with tempfile.TemporaryDirectory() as td:
                r = screen_separation("K", ink, 300.0, {"dpi": 1200, "previews": False},
                                      os.path.join(td, "t.tif"), log=lambda *a, **k: None)
            print(f"  OK    ядро работает (покрытие {100 * r['coverage_out']:.1f} %, "
                  f"ожидалось ≈{100 * r['coverage_in']:.1f} %)")
        except Exception as e:
            ok = False
            print(f"  ОШИБКА ядра: {type(e).__name__}: {e}")
    print("\nВсё установлено, можно запускать start_windows.bat" if ok
          else "\nЕсть проблемы — пришлите этот текст.")
    sys.exit(0 if ok else 1)

missing = []
for mod, pkg in NEEDED.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)

if missing:
    print("Устанавливаю библиотеки:", ", ".join(missing), "(один раз, 1–3 минуты)…")
    cmd = [sys.executable, "-m", "pip", "install", "--user", "--disable-pip-version-check", *missing]
    if subprocess.call(cmd) != 0:
        fail("Не удалось установить библиотеки. Проверьте интернет и запустите ещё раз.")
    import site
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)   # папка могла появиться только сейчас
    importlib.invalidate_caches()

print("Запуск окна HybridRIP…")
sys.path.insert(0, HERE)
import gui  # noqa: E402

gui.App().mainloop()
