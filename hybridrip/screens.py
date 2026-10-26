"""
Генераторы пороговых матриц (threshold arrays) для трёх типов растра.

Все генераторы возвращают пороги T в диапазоне [0, 1) с РАВНОМЕРНЫМ
распределением по площади. Это ключевое свойство: пиксель становится
печатающим, если tone > T, поэтому доля печатающих пикселей = tone.
Именно равномерность порогов позволяет потом смешивать растры без
сдвига тона (см. blend.py).

  AM   — регулярный (амплитудный) растр, иррациональный угол, разные формы точки
  LINE — линейчатый растр
  FM   — стохастика (blue noise, алгоритм void-and-cluster), элемент k×k пикселей
"""
from __future__ import annotations

import os
import hashlib
import numpy as np


# ----------------------------------------------------------------------------
# Функции формы точки (spot functions). u, v — координаты внутри ячейки,
# центр ячейки = 0, диапазон [-0.5, 0.5). Меньшее значение = чернеет раньше.
# ----------------------------------------------------------------------------
def _spot(shape: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    tau = 2.0 * np.pi
    if shape == "euclid":          # круглая в светах, шахматка на 50 %, круглые просветы в тенях
        return -(np.cos(tau * u) + np.cos(tau * v))
    if shape == "round":           # круглая точка во всём диапазоне до смыкания
        return u * u + v * v
    if shape == "ellipse":         # «цепная» точка: смыкание в двух направлениях в разных тонах
        return -(np.cos(tau * u) + 0.75 * np.cos(tau * v))
    if shape == "square":
        return np.maximum(np.abs(u), np.abs(v))
    if shape == "line":            # линейчатый растр (линии вдоль оси v)
        return -np.cos(tau * u)
    raise ValueError(f"Неизвестная форма точки: {shape}")


_CDF_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _spot_cdf(shape: str, n: int = 512):
    """Эмпирическая функция распределения значений spot-функции по ячейке.
    Через неё значения spot-функции переводятся в равномерные пороги."""
    if shape not in _CDF_CACHE:
        g = (np.arange(n) + 0.5) / n - 0.5
        u, v = np.meshgrid(g, g)
        s = np.sort(_spot(shape, u, v).ravel())
        # мелкий детерминированный «разрыв связей» для одинаковых значений
        s = s + np.linspace(0, 1e-9, s.size)
        _CDF_CACHE[shape] = (s, np.linspace(0.0, 1.0, s.size, endpoint=False))
    return _CDF_CACHE[shape]


def periodic_threshold(x: np.ndarray, y: np.ndarray, *, dpi: float, lpi: float,
                       angle_deg: float, shape: str) -> np.ndarray:
    """Порог для AM/линейчатого растра в точках устройства (x, y) (в пикселях).
    Иррациональный растр: координаты поворачиваются аналитически, поэтому
    подходит любой угол и любая линиатура."""
    period = dpi / lpi
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    xc = x + 0.5
    yc = y + 0.5
    u = (xc * ca + yc * sa) / period
    v = (-xc * sa + yc * ca) / period
    u = u - np.floor(u + 0.5)
    v = v - np.floor(v + 0.5)
    s = _spot(shape, u, v)
    xs, ys = _spot_cdf(shape)
    return np.interp(s, xs, ys).astype(np.float32)


# ----------------------------------------------------------------------------
# Стохастика: blue-noise маска методом void-and-cluster (Ulichney, 1993).
# ----------------------------------------------------------------------------
def _gauss_kernel_torus(n: int, sigma: float) -> np.ndarray:
    d = np.minimum(np.arange(n), n - np.arange(n)).astype(np.float64)
    g = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return np.outer(g, g)


def void_and_cluster(n: int = 128, sigma: float = 1.5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    K = _gauss_kernel_torus(n, sigma)
    Kf = np.fft.rfft2(K)

    def energy(b):
        return np.fft.irfft2(np.fft.rfft2(b.astype(np.float64)) * Kf, s=b.shape)

    def kern_at(y, x):
        return np.roll(np.roll(K, y, axis=0), x, axis=1)

    total = n * n
    ones0 = total // 10
    b = np.zeros((n, n), bool)
    b.flat[rng.choice(total, ones0, replace=False)] = True

    # Шаг 0: равномерно распределить начальный паттерн
    E = energy(b)
    for _ in range(total):
        c = np.argmax(np.where(b, E, -np.inf))
        b.flat[c] = False
        E -= kern_at(*divmod(c, n))
        v = np.argmin(np.where(b, np.inf, E))
        b.flat[v] = True
        E += kern_at(*divmod(v, n))
        if v == c:
            break

    rank = np.zeros(total, np.int64)
    proto = b.copy()
    E_proto = E.copy()

    # Фаза 1: убираем «самые тесные кластеры» -> ранги ones0-1 ... 0
    b1, E1 = proto.copy(), E_proto.copy()
    for r in range(ones0 - 1, -1, -1):
        c = np.argmax(np.where(b1, E1, -np.inf))
        b1.flat[c] = False
        E1 -= kern_at(*divmod(c, n))
        rank[c] = r

    # Фазы 2–3: заполняем «самые большие пустоты» -> ранги ones0 ... total-1
    b2, E2 = proto.copy(), E_proto.copy()
    for r in range(ones0, total):
        v = np.argmin(np.where(b2, np.inf, E2))
        b2.flat[v] = True
        E2 += kern_at(*divmod(v, n))
        rank[v] = r

    return ((rank.reshape(n, n) + 0.5) / total).astype(np.float32)


def load_bluenoise(n: int = 128, seed: int = 1, cache_dir: str | None = None) -> np.ndarray:
    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(f"vac-{n}-{seed}".encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"bluenoise_{n}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    m = void_and_cluster(n, seed=seed)
    np.save(path, m)
    return m


class FMScreen:
    """Стохастический растр с элементом element_px × element_px пикселей
    устройства (например, 2–3 px при 2400–2540 dpi ≈ 20–30 мкм — типично
    для флексо, где одиночный пиксель на форме не держится)."""

    def __init__(self, element_px: int = 2, tile: int = 128, seed: int = 1,
                 channel_index: int = 0):
        base = load_bluenoise(tile, seed=seed)
        # разные каналы получают разные варианты маски (поворот/отражение/сдвиг),
        # чтобы сепарации не совпадали точка-в-точку
        k = channel_index % 8
        m = np.rot90(base, k % 4)
        if k >= 4:
            m = m.T
        m = np.roll(m, (channel_index * 37) % tile, axis=0)
        m = np.roll(m, (channel_index * 59) % tile, axis=1)
        self.mask = np.ascontiguousarray(m)
        self.e = max(1, int(element_px))
        self.tile = tile

    def threshold(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = (x // self.e).astype(np.int64) % self.tile
        iy = (y // self.e).astype(np.int64) % self.tile
        return self.mask[iy, ix]
