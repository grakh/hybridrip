"""Окно с графиком: промер (точки, пунктир) и рассчитанная кривая компенсации (сплошная)."""
import tkinter as tk
from tkinter import ttk

COLORS = {"am": "#d03030", "fm": "#2060d0", "line": "#209040"}
LABELS = {"am": "AM", "fm": "FM", "line": "Линейка"}


def show_curves(master, comp: dict):
    w = tk.Toplevel(master)
    w.title(f"Компенсация — {comp['name']}")
    W, H, L, B, T, R = 600, 560, 56, 50, 30, 20
    c = tk.Canvas(w, width=W, height=H, bg="white", highlightthickness=0)
    c.pack(padx=8, pady=8)

    def X(v):
        return L + v * (W - L - R)

    def Y(v):
        return H - B - v * (H - B - T)

    for i in range(0, 11):
        v = i / 10
        c.create_line(X(v), Y(0), X(v), Y(1), fill="#eee")
        c.create_line(X(0), Y(v), X(1), Y(v), fill="#eee")
        c.create_text(X(v), Y(0) + 12, text=f"{i * 10}", fill="#555", font=("Segoe UI", 8))
        c.create_text(X(0) - 14, Y(v), text=f"{i * 10}", fill="#555", font=("Segoe UI", 8))
    c.create_rectangle(X(0), Y(1), X(1), Y(0), outline="#999")
    c.create_line(X(0), Y(0), X(1), Y(1), fill="#bbb", dash=(2, 3))
    c.create_text((X(0) + X(1)) / 2, H - 14, text="задано в файле, %", fill="#333")
    c.create_text(16, (Y(0) + Y(1)) / 2, text="%", fill="#333")
    c.create_text(X(0), 14, anchor="w", fill="#333",
                  text="пунктир — промер (растискивание),  сплошная — компенсация")

    ly = Y(0.2)                      # легенда в правом нижнем углу — там кривых нет
    for s in ("am", "fm", "line"):
        col = COLORS[s]
        if s in comp["measured"]:
            t, m = comp["measured"][s]
            pts = [(X(a), Y(b)) for a, b in zip(t, m)]
            if len(pts) > 1:
                c.create_line(*[v for p in pts for v in p], fill=col, dash=(4, 3))
            for (px, py) in pts:
                c.create_oval(px - 3, py - 3, px + 3, py + 3, outline=col, fill="white")
        if s in comp["curves"]:
            pts = [(X(a), Y(b)) for a, b in comp["curves"][s]]
            c.create_line(*[v for p in pts for v in p], fill=col, width=2)
            md = comp["min_dots"].get(s)
            extra = f", мин. {md * 100:g} %" if md is not None else ""
            m50 = ""
            if s in comp["measured"]:
                import numpy as np
                t, m = comp["measured"][s]
                m50 = f": 50 % → {100 * float(np.interp(0.5, t, m)):.1f} %"
            c.create_line(X(0.42), ly, X(0.48), ly, fill=col, width=2)
            c.create_text(X(0.50), ly, anchor="w", text=f"{LABELS[s]}{m50}{extra}", fill="#333")
            ly += 18

    meta = comp.get("meta", {})
    info = "  ".join(f"{k}: {v}" for k, v in meta.items() if v and k not in ("name",))
    ttk.Label(w, text=info or comp["path"], wraplength=W, foreground="#555").pack(padx=8, pady=(0, 8), anchor="w")
    return w
