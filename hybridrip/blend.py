"""
Смешивание пороговых матриц без сдвига тона.

Проблема: если в зоне перехода просто взять T = a*T1 + (1-a)*T2, то сумма
двух равномерных величин уже НЕ равномерна (распределение трапециевидное),
и в зоне перехода появится полоса «темнее/светлее» — та самая видимая
граница. Поэтому смешанный порог прогоняется через точную функцию
распределения суммы (аналитически, формула для трапеции), и результат
снова равномерен на [0,1) при любом a. Тон в переходе сохраняется,
а геометрия растра плавно «перетекает» из одной структуры в другую:
AM-точки постепенно получают стохастический край и распадаются на
FM-элементы, линии постепенно «рвутся» в точки и т. д.

Режим "dither" — альтернатива: в каждом пикселе выбирается порог одного
из растров, вероятность выбора = вес, выбор управляется blue-noise
маской (тонкое перемешивание структур).
"""
from __future__ import annotations
import numpy as np


def equalized_mix(t1: np.ndarray, t2: np.ndarray, a: np.ndarray) -> np.ndarray:
    """a — вес t1 (0..1). Возвращает равномерно распределённый порог."""
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    b = 1.0 - a
    z = a * t1 + b * t2
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    eps = 1e-4
    safe_lo = np.maximum(lo, eps)
    ab2 = 2.0 * safe_lo * hi
    f_low = z * z / ab2
    f_mid = (z - 0.5 * lo) / hi
    f_high = 1.0 - (1.0 - z) ** 2 / ab2
    out = np.where(z < lo, f_low, np.where(z <= hi, f_mid, f_high))
    out = np.where(lo < eps, z, out)
    return np.clip(out, 0.0, 0.999999).astype(np.float32)


class EmpiricalMixer:
    """Равномерное смешивание для КОРРЕЛИРОВАННЫХ растров (например, AM-точка
    и линейчатый растр под одним углом: их пороги зависимы, аналитическая
    формула не точна). Функция распределения смеси измеряется на образце
    реальных порогов для сетки весов и затем применяется попиксельно."""

    def __init__(self, s1: np.ndarray, s2: np.ndarray, nbins: int = 33, nq: int = 1025):
        self.a_grid = np.linspace(0.0, 1.0, nbins).astype(np.float32)
        self.q = np.linspace(0.0, 1.0, nq).astype(np.float32)
        s1, s2 = s1.ravel(), s2.ravel()
        tabs = []
        for a in self.a_grid:
            z = a * s1 + (1 - a) * s2
            tabs.append(np.quantile(z, self.q).astype(np.float32))
        self.tabs = np.stack(tabs)
        # строго возрастающие таблицы для interp
        self.tabs += np.linspace(0, 1e-6, nq, dtype=np.float32)[None]

    def __call__(self, t1, t2, a):
        a = np.clip(a, 0, 1).astype(np.float32)
        z = a * t1 + (1 - a) * t2
        pos = a * (len(self.a_grid) - 1)
        i0 = np.clip(np.floor(pos).astype(np.int32), 0, len(self.a_grid) - 2)
        f = pos - i0
        out = np.empty_like(z)
        for j in np.unique(i0):
            m = i0 == j
            zj = z[m]
            F0 = np.interp(zj, self.tabs[j], self.q)
            F1 = np.interp(zj, self.tabs[j + 1], self.q)
            out[m] = F0 * (1 - f[m]) + F1 * f[m]
        return np.clip(out, 0.0, 0.999999)


def mix_thresholds(thr: list[np.ndarray], w: np.ndarray, mixers=None) -> np.ndarray:
    """Последовательное равномерное смешивание N порогов.
    thr — список массивов (H,W), w — веса (N,H,W), сумма = 1.
    mixers[i] — необязательная функция смешивания для шага i (иначе аналитическая)."""
    acc = thr[0]
    wacc = w[0].copy()
    for i in range(1, len(thr)):
        wi = w[i]
        tot = wacc + wi
        a = np.where(tot > 1e-6, wacc / np.maximum(tot, 1e-6), 1.0)
        fn = (mixers or {}).get(i, equalized_mix)
        acc = fn(acc, thr[i], a)
        wacc = tot
    return acc


def dither_select(thr: list[np.ndarray], w: np.ndarray, selector: np.ndarray) -> np.ndarray:
    """В каждом пикселе берётся порог одного растра; selector — blue-noise [0,1)."""
    out = thr[-1].copy()
    cum = np.zeros_like(selector)
    chosen = np.zeros(selector.shape, bool)
    for i in range(len(thr) - 1):
        cum = cum + w[i]
        pick = (~chosen) & (selector < cum)
        out = np.where(pick, thr[i], out)
        chosen |= pick
    return out
