"""
HybridRIP — простое окно управления (tkinter).
Запуск:  python gui.py
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hybridrip.engine import run  # noqa: E402
from hybridrip import compensation as comp  # noqa: E402

DEFAULT_CFG = os.path.join(HERE, "configs", "flexo_default.json")


def open_folder(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HybridRIP — гибридный растр AM / линейка / FM")
        self.minsize(780, 760)
        self.q = queue.Queue()
        self.worker = None
        self.v = {}
        self._build()
        self.load_config(DEFAULT_CFG)
        self.after(100, self._poll)

    # ------------------------------------------------------------- интерфейс
    def _var(self, key, value="", kind=tk.StringVar):
        self.v[key] = kind(value=value)
        return self.v[key]

    def _build(self):
        pad = {"padx": 6, "pady": 3}
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # --- файлы
        f = ttk.LabelFrame(root, text="Файлы", padding=8)
        f.pack(fill="x")
        rows = [("Входной файл (TIFF/PNG/PDF):", "input", self.pick_input),
                ("Конфигурация (JSON):", "config", self.pick_config),
                ("Папка результата:", "out", self.pick_out),
                ("Ghostscript (для PDF, пусто = авто):", "gs", self.pick_gs)]
        for r, (label, key, cmd) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(f, textvariable=self._var(key), width=60).grid(row=r, column=1, sticky="we", **pad)
            ttk.Button(f, text="Выбрать…", command=cmd).grid(row=r, column=2, **pad)
        f.columnconfigure(1, weight=1)

        # --- параметры
        p = ttk.LabelFrame(root, text="Параметры", padding=8)
        p.pack(fill="x", pady=8)

        def field(r, c, label, key, width=8, kind=None, values=None):
            ttk.Label(p, text=label).grid(row=r, column=c, sticky="w", **pad)
            var = self._var(key)
            if values:
                w = ttk.Combobox(p, textvariable=var, values=values, width=width, state="readonly")
            else:
                w = ttk.Entry(p, textvariable=var, width=width)
            w.grid(row=r, column=c + 1, sticky="w", **pad)

        field(0, 0, "Разрешение устройства, dpi", "dpi", values=["2400", "2540", "4000", "4800", "5080"])
        field(0, 2, "Разрешение входа, ppi (пусто = из файла)", "ppi")
        field(1, 0, "Линиатура AM / линейки, lpi", "lpi")
        field(1, 2, "Форма точки AM", "shape", width=10,
              values=["euclid", "round", "ellipse", "square"])
        field(2, 0, "FM-элемент, пикс.", "fm_px")
        field(2, 2, "Ширина перехода, мм", "transition")
        field(3, 0, "Смешивание", "mode", width=10, values=["morph", "dither"])
        field(3, 2, "Мин. точка AM, %", "min_am")
        field(4, 0, "Сжатие TIFF", "compression", width=10, values=["lzw", "deflate", "packbits", "none"])
        ttk.Checkbutton(p, text="Резкие контуры (плашки и текст без растровой бахромы)",
                        variable=self._var("hard_edges", True, tk.BooleanVar)).grid(row=4, column=2, columnspan=2, sticky="w", **pad)

        # --- выбор растра по тону
        t = ttk.LabelFrame(root, text="Растр по тону (%)", padding=8)
        t.pack(fill="x")
        ttk.Label(t, text="FM до").grid(row=0, column=0, **pad)
        ttk.Entry(t, textvariable=self._var("fm_to"), width=6).grid(row=0, column=1, **pad)
        ttk.Label(t, text="→ AM с").grid(row=0, column=2, **pad)
        ttk.Entry(t, textvariable=self._var("am_from"), width=6).grid(row=0, column=3, **pad)
        ttk.Label(t, text="AM до").grid(row=0, column=4, **pad)
        ttk.Entry(t, textvariable=self._var("am_to"), width=6).grid(row=0, column=5, **pad)
        ttk.Label(t, text="→ линейка с").grid(row=0, column=6, **pad)
        ttk.Entry(t, textvariable=self._var("line_from"), width=6).grid(row=0, column=7, **pad)
        ttk.Checkbutton(t, text="FM на мелких деталях",
                        variable=self._var("detail", True, tk.BooleanVar)).grid(row=1, column=0, columnspan=4, sticky="w", **pad)
        ttk.Checkbutton(t, text="Без линейки в тенях (только AM)",
                        variable=self._var("no_line", False, tk.BooleanVar)).grid(row=1, column=4, columnspan=4, sticky="w", **pad)

        # --- компенсация растискивания
        k = ttk.LabelFrame(root, text="Компенсация растискивания (полимер)", padding=8)
        k.pack(fill="x", pady=(8, 0))
        self.cb_comp = ttk.Combobox(k, textvariable=self._var("comp"), width=42, state="readonly")
        self.cb_comp.grid(row=0, column=0, sticky="w", **pad)
        ttk.Button(k, text="★ По умолчанию", command=self.set_default_comp).grid(row=0, column=1, **pad)
        ttk.Button(k, text="График", command=self.show_comp_graph).grid(row=0, column=2, **pad)
        ttk.Button(k, text="Обновить", command=self.refresh_comp).grid(row=0, column=3, **pad)
        ttk.Button(k, text="Папка промеров", command=self.open_comp_dir).grid(row=0, column=4, **pad)
        self.lbl_default = ttk.Label(k, foreground="#666")
        self.lbl_default.grid(row=1, column=0, sticky="w", **pad)
        ttk.Button(k, text="Шкала для промера", command=lambda: self.make_chart(False)).grid(row=1, column=1, **pad)
        ttk.Button(k, text="Контрольная шкала", command=lambda: self.make_chart(True)).grid(row=1, column=2, **pad)
        ttk.Button(k, text="Добавить промер…", command=self.import_comp).grid(row=1, column=3, columnspan=2, sticky="w", **pad)
        self.refresh_comp(select_default=True)

        # --- микроструктура плашек
        m = ttk.LabelFrame(root, text="Микроструктура плашек (сплошной ободок по контуру)", padding=8)
        m.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(m, text="Включить", variable=self._var("surf_on", False, tk.BooleanVar)).grid(row=0, column=0, sticky="w", **pad)
        def mf(r, c, label, key, values=None, width=6):
            ttk.Label(m, text=label).grid(row=r, column=c, sticky="e", **pad)
            var = self._var(key)
            w = (ttk.Combobox(m, textvariable=var, values=values, width=width, state="readonly")
                 if values else ttk.Entry(m, textvariable=var, width=width))
            w.grid(row=r, column=c + 1, sticky="w", **pad)
        mf(0, 1, "Узор", "surf_pattern", ["stoch", "grid", "lines", "bayer"], 7)
        mf(0, 3, "Проколы, %", "surf_density")
        mf(0, 5, "Ободок, px", "surf_rim")
        mf(1, 1, "Шаг, px (grid/lines)", "surf_period")
        mf(1, 3, "Угол", "surf_angle", ["45", "0"], 4)
        mf(1, 5, "Мин. элемент, px", "surf_min")
        ttk.Button(m, text="Шкала микроструктур", command=self.make_surface_chart).grid(row=0, column=7, rowspan=2, **pad)

        # --- кнопки
        b = ttk.Frame(root)
        b.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(b, text="▶  Растрировать", command=self.start)
        self.btn_run.pack(side="left")
        ttk.Button(b, text="Открыть папку результата", command=self.open_out).pack(side="left", padx=6)
        ttk.Button(b, text="Просмотр TIFF", command=self.open_viewer).pack(side="left", padx=6)
        ttk.Button(b, text="Сохранить конфиг как…", command=self.save_config).pack(side="left")
        ttk.Button(b, text="Тестовая шкала", command=self.use_testchart).pack(side="right")

        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status = ttk.Label(root, text="Готово к работе")
        self.status.pack(anchor="w", pady=(2, 4))

        # --- лог
        lf = ttk.LabelFrame(root, text="Журнал", padding=4)
        lf.pack(fill="both", expand=True)
        self.log = tk.Text(lf, height=10, wrap="word", state="disabled")
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ------------------------------------------------------------- файлы
    def pick_input(self):
        p = filedialog.askopenfilename(filetypes=[
            ("Изображения", "*.tif *.tiff *.png *.jpg *.jpeg"), ("PDF / PS", "*.pdf *.ps *.eps"), ("Все", "*.*")])
        if p:
            self.v["input"].set(p)
            if not self.v["out"].get():
                self.v["out"].set(os.path.join(os.path.dirname(p), "hybridrip_out"))

    def pick_config(self):
        p = filedialog.askopenfilename(initialdir=os.path.join(HERE, "configs"),
                                       filetypes=[("JSON", "*.json")])
        if p:
            self.load_config(p)

    def pick_gs(self):
        p = filedialog.askopenfilename(title="gswin64c.exe",
                                       filetypes=[("Ghostscript", "gswin64c.exe gswin32c.exe gs gs.exe"), ("Все", "*.*")])
        if p:
            self.v["gs"].set(p)

    def pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.v["out"].set(p)

    def use_testchart(self):
        self.v["input"].set(os.path.join(HERE, "examples", "testchart_cmyk.tif"))
        self.load_config(os.path.join(HERE, "examples", "testchart_config.json"))
        self.v["out"].set(os.path.join(HERE, "examples", "out"))

    def open_viewer(self):
        """Просмотрщик с результатами последнего запуска (или пустой)."""
        import viewer
        files = list(getattr(self, "last_files", []))
        if not files:
            out = self.v["out"].get()
            if out and os.path.isdir(out):
                files = sorted(os.path.join(out, f) for f in os.listdir(out)
                               if f.lower().endswith((".tif", ".tiff")))
        viewer.open_viewer(files, master=self)

    def open_out(self):
        out = self.v["out"].get()
        if out and os.path.isdir(out):
            open_folder(out)
        else:
            messagebox.showinfo("HybridRIP", "Папка результата ещё не создана.")

    # ------------------------------------------------------------- компенсация
    def refresh_comp(self, select_default=False):
        names = comp.list_files()
        self.cb_comp["values"] = [comp.NONE_LABEL] + names
        st = comp.load_settings()
        d = st.get("default_compensation") or ""
        if d and d not in names:
            self.lbl_default.config(text=f"по умолчанию: {d} — ФАЙЛ НЕ НАЙДЕН", foreground="#b00")
        else:
            self.lbl_default.config(text=f"по умолчанию: {d or comp.NONE_LABEL}", foreground="#666")
        cur = self.v["comp"].get()
        if select_default or cur not in self.cb_comp["values"]:
            self.v["comp"].set(d if d in names else comp.NONE_LABEL)

    def selected_comp_path(self):
        name = self.v["comp"].get()
        return None if name in ("", comp.NONE_LABEL) else comp.resolve(name)

    def set_default_comp(self):
        name = self.v["comp"].get()
        st = comp.load_settings()
        st["default_compensation"] = "" if name == comp.NONE_LABEL else name
        comp.save_settings(st)
        self.refresh_comp()
        self._log(f"Компенсация по умолчанию: {name}\n")

    def show_comp_graph(self):
        try:
            path = self.selected_comp_path()
            if not path:
                messagebox.showinfo("HybridRIP", "Выберите файл компенсации в списке.")
                return
            from curve_window import show_curves
            show_curves(self, comp.parse(path))
        except Exception as e:
            messagebox.showerror("HybridRIP", f"Ошибка в файле промеров:\n{e}")

    def open_comp_dir(self):
        d = comp.comp_dir()
        os.makedirs(d, exist_ok=True)
        open_folder(d)

    def import_comp(self):
        fs = filedialog.askopenfilenames(title="Файлы промеров (CSV/TXT)",
                                         filetypes=[("Промеры", "*.csv *.txt *.tsv"), ("Все", "*.*")])
        if not fs:
            return
        import shutil
        d = comp.comp_dir()
        os.makedirs(d, exist_ok=True)
        added = []
        for f in fs:
            try:
                comp.parse(f)                    # проверка до копирования
            except Exception as e:
                messagebox.showerror("HybridRIP", f"{os.path.basename(f)}:\n{e}")
                continue
            dst = os.path.join(d, os.path.basename(f))
            if os.path.exists(dst) and not messagebox.askyesno("HybridRIP", f"{os.path.basename(f)} уже есть. Заменить?"):
                continue
            shutil.copy2(f, dst)
            added.append(os.path.splitext(os.path.basename(f))[0])
        self.refresh_comp()
        if added:
            self.v["comp"].set(added[-1])
            self._log(f"Добавлены промеры: {', '.join(added)}\n")

    def make_chart(self, compensated: bool):
        if self.worker and self.worker.is_alive():
            return
        if not self.ensure_libs():
            return
        try:
            cfg = self.collect()
        except Exception as e:
            messagebox.showerror("HybridRIP", f"Ошибка в параметрах:\n{e}")
            return
        if compensated and not cfg.get("compensation"):
            messagebox.showinfo("HybridRIP", "Для контрольной шкалы выберите компенсацию в списке.")
            return
        out = self.v["out"].get() or os.path.join(HERE, "calibration")
        self.btn_run.state(["disabled"])
        self.progress["value"] = 0

        def work():
            def log(msg="", end="\n", **_):
                self.q.put(("log", msg, end))
            try:
                from hybridrip.calibration import make_chart
                tif, csv = make_chart(out, cfg, log=log, compensated=compensated)
                res = [{"file": tif}]
                if csv:
                    log("Промерьте шкалу на форме, заполните шаблон, переименуйте по названию полимера "
                        "и добавьте кнопкой «Добавить промер…».")
                self.q.put(("done", res, None))
            except Exception:
                self.q.put(("error", traceback.format_exc(), None))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def make_surface_chart(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.ensure_libs():
            return
        try:
            cfg = self.collect()
        except Exception as e:
            messagebox.showerror("HybridRIP", f"Ошибка в параметрах:\n{e}")
            return
        out = self.v["out"].get() or os.path.join(HERE, "calibration")
        self.btn_run.state(["disabled"])

        def work():
            def log(msg="", end="\n", **_):
                self.q.put(("log", msg, end))
            try:
                from hybridrip.calibration import make_surface_chart
                tif = make_surface_chart(out, cfg, log=log)
                log("Напечатайте шкалу, измерьте оптическую плотность плашек и выберите узор и % проколов "
                    "с наибольшей плотностью и ровной плашкой.")
                self.q.put(("done", [{"file": tif}], None))
            except Exception:
                self.q.put(("error", traceback.format_exc(), None))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------- конфиг
    def load_config(self, path):
        try:
            with open(path, encoding="utf-8") as fh:
                self.cfg = json.load(fh)
        except Exception as e:
            messagebox.showerror("HybridRIP", f"Не удалось прочитать конфиг:\n{e}")
            return
        self.cfg_path = path
        c, s = self.cfg, self.cfg.get("screens", {})
        self.v["config"].set(path)
        self.v["dpi"].set(str(int(c.get("dpi", 2400))))
        self.v["ppi"].set("" if not c.get("input_ppi") else str(c["input_ppi"]))
        self.v["lpi"].set(str(s.get("am", {}).get("lpi", 150)))
        self.v["shape"].set(s.get("am", {}).get("shape", "euclid"))
        self.v["fm_px"].set(str(s.get("fm", {}).get("element_px", 3)))
        self.v["transition"].set(str(c.get("transition_mm", 1.5)))
        self.v["mode"].set(c.get("blend_mode", "morph"))
        self.v["compression"].set(c.get("compression", "lzw"))
        self.v["min_am"].set(f'{100 * s.get("am", {}).get("min_dot", 0.02):g}')
        anchors = c.get("criteria", {}).get("tone", {}).get("anchors", [])
        def edge(a, b, default):
            for (t0, s0), (t1, s1) in zip(anchors, anchors[1:]):
                if s0 == a and s1 == b:
                    return t0, t1
            return default
        fm_to, am_from = edge("fm", "am", (0.04, 0.10))
        am_to, line_from = edge("am", "line", (0.80, 0.92))
        for k, val in (("fm_to", fm_to), ("am_from", am_from), ("am_to", am_to), ("line_from", line_from)):
            self.v[k].set(f"{100 * val:g}")
        self.v["no_line"].set(not any(sp == "line" for _, sp in anchors))
        self.v["detail"].set(bool(c.get("criteria", {}).get("detail", {}).get("enabled", False)))
        self.v["hard_edges"].set(bool(c.get("hard_edges", True)))
        from hybridrip.surface import DEFAULTS as SD
        sf = dict(SD); sf.update(c.get("surface") or {})
        self.v["surf_on"].set(bool(sf["enabled"]))
        self.v["surf_pattern"].set(sf["pattern"])
        self.v["surf_density"].set(f"{100 * float(sf['density']):g}")
        self.v["surf_rim"].set(str(sf["rim_px"]))
        self.v["surf_period"].set(str(sf["period_px"]))
        self.v["surf_angle"].set(f"{float(sf['angle']):g}")
        self.v["surf_min"].set(str(sf["min_size_px"]))

    def collect(self):
        """Значения из формы -> словарь конфигурации."""
        c = json.loads(json.dumps(self.cfg))  # копия
        num = lambda k: float(self.v[k].get().replace(",", "."))
        c["dpi"] = num("dpi")
        c["input_ppi"] = num("ppi") if self.v["ppi"].get().strip() else None
        c["transition_mm"] = num("transition")
        c["blend_mode"] = self.v["mode"].get()
        c["compression"] = self.v["compression"].get()
        s = c.setdefault("screens", {})
        for k in ("am", "line", "fm"):
            s.setdefault(k, {})
        s["am"]["lpi"] = s["line"]["lpi"] = num("lpi")
        s["am"]["shape"] = self.v["shape"].get()
        s["am"]["min_dot"] = num("min_am") / 100
        s["fm"]["element_px"] = int(num("fm_px"))
        p = [num(k) / 100 for k in ("fm_to", "am_from", "am_to", "line_from")]
        if not (0 <= p[0] < p[1] < p[2] < p[3] <= 1):
            raise ValueError("Границы тонов должны идти по возрастанию: FM до < AM с < AM до < линейка с")
        anchors = [[0.0, "fm"], [p[0], "fm"], [p[1], "am"], [p[2], "am"]]
        anchors += [[1.0, "am"]] if self.v["no_line"].get() else [[p[3], "line"], [1.0, "line"]]
        crit = c.setdefault("criteria", {})
        crit.setdefault("tone", {})["anchors"] = anchors
        det = crit.setdefault("detail", {"screen": "fm", "radius_mm": 0.4, "low": 0.1, "high": 0.3, "strength": 0.7})
        det["enabled"] = bool(self.v["detail"].get())
        c["compensation"] = self.selected_comp_path()
        c["hard_edges"] = bool(self.v["hard_edges"].get())
        sf = c.setdefault("surface", {})
        sf.update({"enabled": bool(self.v["surf_on"].get()), "pattern": self.v["surf_pattern"].get(),
                   "density": num("surf_density") / 100, "rim_px": num("surf_rim"),
                   "period_px": int(num("surf_period")), "angle": num("surf_angle"),
                   "min_size_px": num("surf_min")})
        if not (0 <= sf["density"] <= 0.6):
            raise ValueError("Проколы микроструктуры: 0–60 %")
        return c

    def save_config(self):
        try:
            c = self.collect()
        except Exception as e:
            messagebox.showerror("HybridRIP", str(e))
            return
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")],
                                         initialdir=os.path.join(HERE, "configs"))
        if p:
            c.pop("compensation", None)        # компенсация выбирается в окне, не в конфиге
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(c, fh, ensure_ascii=False, indent=2)
            self.v["config"].set(p)
            self.cfg, self.cfg_path = c, p

    # ------------------------------------------------------------- запуск
    # ------------------------------------------------------------- установка библиотек
    def ensure_libs(self) -> bool:
        """True, если всё на месте. Иначе предлагает установить и перезапускает окно."""
        from hybridrip import kernel
        missing = [] if kernel.HAVE_NUMBA else ["numba"]
        try:
            import imagecodecs  # noqa: F401
        except Exception:
            missing.append("imagecodecs")
        if not missing:
            return True
        if not messagebox.askyesno(
                "HybridRIP",
                f"Не установлены пакеты: {', '.join(missing)}.\n\n"
                "Установить сейчас? Нужен интернет, 1–2 минуты.\nПосле установки окно перезапустится."):
            return False
        self.btn_run.state(["disabled"])
        self.status.config(text="Установка библиотек…")
        self._log(f"Установка: {' '.join(missing)} в {sys.executable}\n")

        def work():
            cmd = [sys.executable, "-m", "pip", "install", "--user",
                   "--disable-pip-version-check", *missing]
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                for line in p.stdout:
                    self.q.put(("log", line.rstrip(), "\n"))
                ok = p.wait() == 0
            except Exception as e:
                self.q.put(("log", str(e), "\n"))
                ok = False
            self.q.put(("installed", ok, None))
        threading.Thread(target=work, daemon=True).start()
        return False

    def _restart(self):
        subprocess.Popen([sys.executable, os.path.join(HERE, "gui.py")], cwd=HERE)
        self.destroy()

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.ensure_libs():
            return
        inp, out = self.v["input"].get(), self.v["out"].get()
        if not inp or not os.path.exists(inp):
            messagebox.showwarning("HybridRIP", "Выберите входной файл.")
            return
        if not out:
            messagebox.showwarning("HybridRIP", "Укажите папку результата.")
            return
        try:
            cfg = self.collect()
        except Exception as e:
            messagebox.showerror("HybridRIP", f"Ошибка в параметрах:\n{e}")
            return
        gs = self.v["gs"].get().strip()
        if gs:
            os.environ["GS"] = gs
        if inp.lower().endswith((".pdf", ".ps", ".eps")):
            try:
                from hybridrip.tiffio import find_ghostscript
                found = find_ghostscript()
                self.v["gs"].set(found)
                if not self.v["ppi"].get().strip():
                    cfg["input_ppi"] = 600
                self._log(f"Ghostscript: {found}; рендер PDF {cfg['input_ppi']:g} ppi\n")
            except Exception as e:
                messagebox.showerror("HybridRIP", str(e))
                return
        self.btn_run.state(["disabled"])
        self.progress["value"] = 0
        self._log(f"=== {os.path.basename(inp)} → {out}\n")
        cfg_dir = os.path.dirname(os.path.abspath(self.cfg_path))
        self.worker = threading.Thread(target=self._work, args=(inp, cfg, out, cfg_dir), daemon=True)
        self.worker.start()

    def _work(self, inp, cfg, out, cfg_dir):
        def log(msg="", end="\n", **_):
            self.q.put(("log", msg, end))
        try:
            res = run(inp, cfg, out, cfg_dir=cfg_dir, log=log)
            self.q.put(("done", res, None))
        except Exception:
            self.q.put(("error", traceback.format_exc(), None))

    def _poll(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "log":
                    m = re.search(r"^\s*(.+?): (\d+)/(\d+) строк(.*)$", a)
                    if m and b == "\r":
                        self.progress["value"] = 100 * int(m.group(2)) / int(m.group(3))
                        self.status.config(text=f"Сепарация {m.group(1)}: {m.group(2)} из {m.group(3)} строк{m.group(4)}")
                    elif a.strip():
                        self._log(a + "\n")
                elif kind == "installed":
                    self.btn_run.state(["!disabled"])
                    if a:
                        self._log("Установлено. Перезапуск окна…\n")
                        messagebox.showinfo("HybridRIP", "Библиотеки установлены. Окно перезапустится, "
                                                         "затем снова нажмите «Растрировать».")
                        self._restart()
                        return
                    self.status.config(text="Ошибка установки")
                    messagebox.showerror("HybridRIP", "Не удалось установить библиотеки — подробности в журнале.")
                elif kind == "done":
                    self.last_files = [r["file"] for r in a]
                    self.btn_run.state(["!disabled"])
                    self.progress["value"] = 100
                    self.status.config(text=f"Готово: {len(a)} файл(ов)")
                    self._log("Готово.\n\n")
                elif kind == "error":
                    self.btn_run.state(["!disabled"])
                    self.status.config(text="Ошибка")
                    self._log(a + "\n")
                    messagebox.showerror("HybridRIP", a.strip().splitlines()[-1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
