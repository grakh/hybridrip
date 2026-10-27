"""
Нативное ядро на C++ (hybridrip/bin/hybridrip_core.dll или libhybridrip_core.so), через ctypes.

Одна DLL работает с любой версией Python; компилятор у пользователя не нужен.
Если библиотеки нет или она не загрузилась — движок использует эталонное ядро на Numba.
Принудительный выбор: переменная окружения HYBRIDRIP_BACKEND=native | numba.
"""
from __future__ import annotations
import ctypes as C
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_NAME = "hybridrip_core.dll" if os.name == "nt" else (
    "libhybridrip_core.dylib" if sys.platform == "darwin" else "libhybridrip_core.so")
LIB_PATH = os.path.join(HERE, "bin", LIB_NAME)
EXPECTED_VERSION = 1

i64, f64 = C.c_int64, C.c_double
pf32, pf64, pu8, pi64 = (C.POINTER(C.c_float), C.POINTER(C.c_double),
                         C.POINTER(C.c_uint8), C.POINTER(C.c_int64))


class HRParams(C.Structure):
    # порядок полей = struct HRParams в native/hybridrip_core.cpp
    _fields_ = [
        ("ink", C.c_void_p), ("ink_type", i64), ("ink_h", i64), ("ink_w", i64),
        ("ink_inv_max", f64), ("inv_scale", f64),
        ("wgrid", pf32), ("wg_h", i64), ("wg_w", i64), ("inv_scale_c", f64),
        ("curves", pf32), ("curve_n", i64), ("min_dots", pf64),
        ("am_p", f64), ("am_ca", f64), ("am_sa", f64), ("am_shape", i64),
        ("am_lut", pf32), ("am_lut_n", i64), ("am_smin", f64), ("am_sinv", f64),
        ("ln_p", f64), ("ln_ca", f64), ("ln_sa", f64), ("ln_shape", i64),
        ("ln_lut", pf32), ("ln_lut_n", i64), ("ln_smin", f64), ("ln_sinv", f64),
        ("fm_mask", pf32), ("fm_tile", i64), ("fm_e", i64),
        ("emp_F", pf32), ("emp_nb", i64), ("emp_nz", i64), ("use_emp", i64),
        ("mode", i64), ("selector", pf32), ("sel_tile", i64),
        ("edge", pu8), ("use_edge", i64),
        ("Wreal", i64), ("nbytes", i64), ("nthreads", i64),
    ]


_lib = None
LOAD_ERROR = ""


def _load():
    global _lib, LOAD_ERROR
    if _lib is not None or LOAD_ERROR:
        return _lib
    if os.environ.get("HYBRIDRIP_BACKEND", "").lower() == "numba":
        LOAD_ERROR = "отключено (HYBRIDRIP_BACKEND=numba)"
        return None
    try:
        lib = C.CDLL(LIB_PATH)
        lib.hr_version.restype = C.c_int
        lib.hr_params_size.restype = C.c_int
        lib.hr_max_threads.restype = C.c_int
        if lib.hr_version() != EXPECTED_VERSION:
            raise RuntimeError(f"версия ядра {lib.hr_version()}, ожидалась {EXPECTED_VERSION}")
        if lib.hr_params_size() != C.sizeof(HRParams):
            raise RuntimeError("несовпадение структуры параметров (пересоберите native/)")
        lib.hr_render_strip.argtypes = [C.POINTER(HRParams), i64, i64, pu8, pi64]
        lib.hr_render_strip.restype = C.c_int
        lib.hr_surface_pass.argtypes = [pu8, pf32, i64, i64, i64, f64, f64, f64,
                                        pf32, i64, i64, i64, i64, pi64]
        lib.hr_surface_pass.restype = C.c_int
        lib.hr_block_coverage.argtypes = [pu8, i64, i64, i64, i64, pu8, i64]
        lib.hr_block_coverage.restype = C.c_int
        _lib = lib
    except Exception as e:
        LOAD_ERROR = f"{type(e).__name__}: {e}"
    return _lib


def available() -> bool:
    return _load() is not None


def describe() -> str:
    lib = _load()
    if lib is None:
        return f"C++ ядро недоступно ({LOAD_ERROR})"
    return f"C++ ядро ({os.path.basename(LIB_PATH)}, потоков: {lib.hr_max_threads()})"


def _ptr(a: np.ndarray, t):
    return a.ctypes.data_as(C.POINTER(t))


class Renderer:
    """Готовит параметры один раз на сепарацию; render() вызывается для каждой полосы."""

    def __init__(self, *, ink, ink_inv_max, inv_scale, wgrid, inv_scale_c, curves, min_dots,
                 am, ln, fm_mask, fm_e, emp_F, use_emp, mode, selector, edge, use_edge, Wreal,
                 nthreads=0):
        self.lib = _load()
        # держим ссылки на массивы, пока живёт Renderer (иначе указатели станут висячими)
        ink = np.ascontiguousarray(ink)
        ink_type = {np.dtype(np.uint8): 0, np.dtype(np.uint16): 1}.get(ink.dtype, 2)
        if ink_type == 2 and ink.dtype != np.float32:
            ink = ink.astype(np.float32)
        self._keep = dict(
            ink=ink, wgrid=np.ascontiguousarray(wgrid, np.float32),
            curves=np.ascontiguousarray(curves, np.float32),
            min_dots=np.ascontiguousarray(min_dots, np.float64),
            am_lut=np.ascontiguousarray(am["lut"], np.float32),
            ln_lut=np.ascontiguousarray(ln["lut"], np.float32),
            fm=np.ascontiguousarray(fm_mask, np.float32),
            emp=np.ascontiguousarray(emp_F, np.float32),
            sel=np.ascontiguousarray(selector, np.float32),
            edge=np.ascontiguousarray(edge, np.uint8),
        )
        k = self._keep
        if use_edge and k["edge"].shape != ink.shape:
            raise ValueError("маска контуров должна совпадать по размеру со входом")
        p = HRParams()
        p.ink = k["ink"].ctypes.data
        p.ink_type, p.ink_h, p.ink_w = ink_type, ink.shape[0], ink.shape[1]
        p.ink_inv_max, p.inv_scale = ink_inv_max, inv_scale
        p.wgrid, p.wg_h, p.wg_w = _ptr(k["wgrid"], C.c_float), k["wgrid"].shape[1], k["wgrid"].shape[2]
        p.inv_scale_c = inv_scale_c
        p.curves, p.curve_n = _ptr(k["curves"], C.c_float), k["curves"].shape[1]
        p.min_dots = _ptr(k["min_dots"], C.c_double)
        p.am_p, p.am_ca, p.am_sa, p.am_shape = am["p"], am["ca"], am["sa"], am["shape"]
        p.am_lut, p.am_lut_n, p.am_smin, p.am_sinv = _ptr(k["am_lut"], C.c_float), k["am_lut"].size, am["smin"], am["sinv"]
        p.ln_p, p.ln_ca, p.ln_sa, p.ln_shape = ln["p"], ln["ca"], ln["sa"], ln["shape"]
        p.ln_lut, p.ln_lut_n, p.ln_smin, p.ln_sinv = _ptr(k["ln_lut"], C.c_float), k["ln_lut"].size, ln["smin"], ln["sinv"]
        p.fm_mask, p.fm_tile, p.fm_e = _ptr(k["fm"], C.c_float), k["fm"].shape[0], fm_e
        p.emp_F, p.emp_nb, p.emp_nz, p.use_emp = (_ptr(k["emp"], C.c_float), k["emp"].shape[0],
                                                   k["emp"].shape[1], int(bool(use_emp)))
        p.mode = mode
        p.selector, p.sel_tile = _ptr(k["sel"], C.c_float), k["sel"].shape[0]
        p.edge, p.use_edge = _ptr(k["edge"], C.c_uint8), int(bool(use_edge))
        p.Wreal, p.nbytes, p.nthreads = Wreal, (Wreal + 7) // 8, nthreads
        self.p = p

    def render(self, y0: int, out: np.ndarray, counts: np.ndarray):
        assert out.flags.c_contiguous and out.dtype == np.uint8 and out.shape[1] == self.p.nbytes
        rc = self.lib.hr_render_strip(C.byref(self.p), y0, out.shape[0], _ptr(out, C.c_uint8),
                                      _ptr(counts, C.c_int64))
        if rc != 0:
            raise RuntimeError(f"hr_render_strip вернул {rc}")


def surface_pass(u8, dist, ya, rim, fade, density, mask, cell, nthreads=0) -> int:
    lib = _load()
    rows, W = u8.shape
    removed = np.zeros(rows, np.int64)
    mask = np.ascontiguousarray(mask, np.float32)
    rc = lib.hr_surface_pass(_ptr(u8, C.c_uint8), _ptr(np.ascontiguousarray(dist, np.float32), C.c_float),
                             rows, W, ya, rim, fade, density, _ptr(mask, C.c_float),
                             mask.shape[0], mask.shape[1], cell, nthreads, _ptr(removed, C.c_int64))
    if rc != 0:
        raise RuntimeError(f"hr_surface_pass вернул {rc}")
    return int(removed.sum())


def block_coverage(packed: np.ndarray, W: int, pf: int) -> np.ndarray:
    lib = _load()
    packed = np.ascontiguousarray(packed)
    res = np.empty((packed.shape[0] // pf, W // pf), np.uint8)
    lib.hr_block_coverage(_ptr(packed, C.c_uint8), packed.shape[0], packed.shape[1], W, pf,
                          _ptr(res, C.c_uint8), 0)
    return res
