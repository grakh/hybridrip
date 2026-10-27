"""Окно «Углы сепараций»: таблица сепарация → угол, триада, плашки, библиотека Pantone."""
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from hybridrip.angles import angle_for, conflicts, process_key, norm, PRESETS, DEFAULT_PROCESS
from hybridrip import compensation as comp


class AnglesWindow(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Углы сепараций")
        self.geometry("720x540")
        self.transient(app)
        st = comp.load_settings()
        self.library = dict(st.get("spot_angles", {}))
        self.job = dict(app.cfg.get("angles", {}) or {})
        self.rows = {}          # имя -> [угол, источник]
        self.edited = set()     # что пользователь менял в этом окне
        self._editor = None

        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)
        ttk.Label(top, foreground="#555", justify="left", wraplength=690, text=(
            "Двойной щелчок по углу — изменить. Плашкам можно задать шаблон со *: «PANTONE 485*».\n"
            "Углы плашек сохраняются в библиотеку и подставляются во всех следующих заданиях.")
        ).pack(anchor="w")
        cols = ("name", "angle", "src")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=14)
        for c, t, w, a in (("name", "Сепарация", 300, "w"), ("angle", "Угол, °", 90, "center"),
                           ("src", "Откуда", 140, "center")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=a)
        self.tree.pack(fill="both", expand=True, pady=6)
        self.tree.tag_configure("warn", background="#ffe2c4")
        self.tree.tag_configure("proc", foreground="#1f5fa8")
        self.tree.bind("<Double-1>", self._edit)

        b = ttk.Frame(top)
        b.pack(fill="x")
        ttk.Button(b, text="Прочитать из входного файла", command=self.read_input).pack(side="left")
        ttk.Button(b, text="Добавить плашку…", command=self.add).pack(side="left", padx=4)
        ttk.Button(b, text="Удалить", command=self.remove).pack(side="left")
        ttk.Button(b, text="Триада по умолчанию", command=self.reset_process).pack(side="left", padx=4)
        self.remember = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Запоминать углы плашек в библиотеке (settings.json)",
                        variable=self.remember).pack(anchor="w", pady=(6, 0))
        self.warn = ttk.Label(top, foreground="#b04000", wraplength=690, justify="left")
        self.warn.pack(anchor="w", pady=4)
        bb = ttk.Frame(top)
        bb.pack(fill="x")
        ttk.Button(bb, text="OK", command=self.ok).pack(side="right")
        ttk.Button(bb, text="Отмена", command=self.destroy).pack(side="right", padx=6)

        for n in ("C", "M", "Y", "K"):
            self._put(n)
        for n in list(self.library) + [k for k in self.job if process_key(k) is None]:
            self._put(n)
        self._refresh()

    # ------------------------------------------------------------------ данные
    def _cfg(self):
        return {"angles": self.job, "spot_default_angles": self.app.cfg.get("spot_default_angles")}

    def _put(self, name, idx=None):
        key = process_key(name) or name
        if key in self.rows:
            return
        a, src = angle_for(key, idx if idx is not None else len(self.rows), self._cfg(), self.library)
        self.rows[key] = [a, src]

    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        assign = {n: v[0] for n, v in self.rows.items()}
        bad = set()
        for x, y in conflicts(assign):
            bad.update((x, y))
        for n, (a, src) in self.rows.items():
            tags = ["warn"] if n in bad else []
            if process_key(n):
                tags.append("proc")
            self.tree.insert("", "end", iid=n, values=(n, f"{a:g}", src), tags=tags)
        pairs = conflicts(assign)
        self.warn.config(text=("Одинаковый угол (возможен муар при наложении): " +
                               "; ".join(f"{x} и {y}" for x, y in pairs)) if pairs else "")

    # ------------------------------------------------------------------ действия
    def _edit(self, e):
        iid = self.tree.identify_row(e.y)
        col = self.tree.identify_column(e.x)
        if not iid or col != "#2":
            return
        x, y, w, h = self.tree.bbox(iid, col)
        var = tk.StringVar(value=self.tree.set(iid, "angle"))
        cb = ttk.Combobox(self.tree, textvariable=var, values=PRESETS, width=8)
        cb.place(x=x, y=y, width=w, height=h)
        cb.focus_set()
        cb.select_range(0, "end")

        def done(_=None):
            try:
                v = float(var.get().replace(",", ".")) % 180
            except ValueError:
                cb.destroy()
                return
            self.rows[iid] = [v, "задание" if process_key(iid) else
                              ("библиотека" if self.remember.get() else "задание")]
            self.edited.add(iid)
            cb.destroy()
            self._refresh()
        cb.bind("<Return>", done)
        cb.bind("<<ComboboxSelected>>", done)
        cb.bind("<FocusOut>", done)
        cb.bind("<Escape>", lambda _: cb.destroy())

    def add(self):
        name = simpledialog.askstring("Плашка", "Название сепарации (можно шаблон со *):", parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        if process_key(name):
            messagebox.showinfo("Углы", "Это цвет триады — он уже есть в списке.", parent=self)
            return
        ang = simpledialog.askfloat("Угол", f"Угол для «{name}», °:", parent=self, initialvalue=22.5)
        if ang is None:
            return
        self.rows[name] = [ang % 180, "библиотека" if self.remember.get() else "задание"]
        self.edited.add(name)
        self._refresh()

    def remove(self):
        for iid in self.tree.selection():
            if process_key(iid):
                continue                      # триаду не удаляем
            self.rows.pop(iid, None)
            self.library = {k: v for k, v in self.library.items() if norm(k) != norm(iid)}
            self.job = {k: v for k, v in self.job.items() if norm(k) != norm(iid)}
        self._refresh()

    def reset_process(self):
        for k, v in DEFAULT_PROCESS.items():
            self.rows[k] = [v, "триада"]
            self.job.pop(k, None)
        self._refresh()

    def read_input(self):
        path = self.app.v["input"].get()
        if not path or not os.path.exists(path):
            messagebox.showinfo("Углы", "Сначала выберите входной файл в главном окне.", parent=self)
            return
        gs = self.app.v["gs"].get().strip()
        if gs:
            os.environ["GS"] = gs
        self.config(cursor="watch")
        q = queue.Queue()

        def work():                          # в потоке — без обращений к окну
            try:
                from hybridrip.tiffio import list_separations
                q.put(("ok", list_separations(path)))
            except Exception as e:
                q.put(("err", str(e)))

        def poll():                          # в главном потоке
            try:
                kind, val = q.get_nowait()
            except queue.Empty:
                self.after(100, poll)
                return
            self.config(cursor="")
            if kind == "ok":
                self._got(val)
            else:
                messagebox.showerror("Углы", val, parent=self)
        threading.Thread(target=work, daemon=True).start()
        self.after(100, poll)

    def _got(self, names):
        for i, n in enumerate(names):
            self._put(n, i)
        self._refresh()

    def ok(self):
        job = {k: v for k, v in self.job.items() if process_key(k) is None}
        lib = dict(self.library)
        for n, (a, src) in self.rows.items():
            pk = process_key(n)
            if pk:
                job[pk] = a
            elif n not in self.edited:
                continue                      # не менялось — остаётся как было (библиотека / шаблон)
            elif self.remember.get():
                lib = {k: v for k, v in lib.items() if norm(k) != norm(n)}
                lib[n] = a
                job.pop(n, None)
            else:
                job[n] = a
        self.app.cfg["angles"] = job
        comp.set_setting("spot_angles", lib)
        self.app._log("Углы: " + ", ".join(f"{k} {v:g}°" for k, v in job.items()) +
                      (f"; библиотека плашек: {len(lib)}" if lib else "") + "\n")
        self.destroy()
