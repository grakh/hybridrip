// HybridRIP — нативное ядро растрирования (C++17, OpenMP).
//
// Та же математика, что и в hybridrip/kernel.py (эталон на Numba), но:
//   * обычная DLL / .so с C-интерфейсом — вызывается из Python через ctypes,
//     одна сборка работает с любой версией Python;
//   * координаты входа и сетки весов по X считаются один раз на полосу
//     (масштаб постоянен), а не для каждого пикселя;
//   * строки полосы считаются параллельно (OpenMP).
//
// Сборка: native/build.sh (Linux .so и Windows .dll через MinGW),
//         native/build_msvc.bat (Windows, Visual Studio) — по желанию.

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#define HR_API extern "C" __declspec(dllexport)
#else
#define HR_API extern "C" __attribute__((visibility("default")))
#endif

#define HR_VERSION 1

// ---------------------------------------------------------------- параметры
// Порядок и типы полей должны совпадать с ctypes.Structure в hybridrip/native.py
struct HRParams {
    // вход (контон сепарации)
    const void*   ink;          // uint8 / uint16 / float32, (ink_h, ink_w), C-порядок
    int64_t       ink_type;     // 0 = uint8, 1 = uint16, 2 = float32
    int64_t       ink_h, ink_w;
    double        ink_inv_max;  // 1 / максимум типа
    double        inv_scale;    // ppi / dpi
    // веса растров на грубой сетке: float32 (3, wg_h, wg_w)
    const float*  wgrid;
    int64_t       wg_h, wg_w;
    double        inv_scale_c;  // control_ppi / dpi
    // тоновые кривые float32 (3, curve_n) и мин. точки (3)
    const float*  curves;
    int64_t       curve_n;
    const double* min_dots;
    // AM
    double am_p, am_ca, am_sa;
    int64_t am_shape;
    const float* am_lut; int64_t am_lut_n; double am_smin, am_sinv;
    // линейка
    double ln_p, ln_ca, ln_sa;
    int64_t ln_shape;
    const float* ln_lut; int64_t ln_lut_n; double ln_smin, ln_sinv;
    // FM (blue noise)
    const float* fm_mask; int64_t fm_tile; int64_t fm_e;
    // табличное смешивание AM + линейка
    const float* emp_F; int64_t emp_nb, emp_nz; int64_t use_emp;
    // 0 = morph, 1 = dither
    int64_t mode;
    const float* selector; int64_t sel_tile;
    // маска контуров (uint8, размер входа)
    const uint8_t* edge; int64_t use_edge;
    // выход
    int64_t Wreal;              // ширина в пикселях
    int64_t nbytes;             // байт в строке выхода
    int64_t nthreads;           // 0 = все ядра
};

// cos(2πu) для u ∈ [-0.5, 0.5] по таблице с линейной интерполяцией:
// погрешность < 3e-7 — на результат почти не влияет, но в несколько раз быстрее std::cos.
static const int COS_N = 4096;
struct CosTable {
    double v[COS_N + 2];
    CosTable() { for (int i = 0; i <= COS_N + 1; ++i) v[i] = std::cos(6.283185307179586 * ((double)i / COS_N - 0.5)); }
};
static const CosTable g_cos;

static inline double cos2pi(double u) {
    double p = (u + 0.5) * COS_N;
    if (p < 0.0) p = 0.0;
    if (p > (double)COS_N) p = (double)COS_N;
    const int i = (int)p;
    const double f = p - (double)i;
    return g_cos.v[i] + (g_cos.v[i + 1] - g_cos.v[i]) * f;
}

static inline double spot(int64_t shape, double u, double v) {
    switch (shape) {
        case 0: return -(cos2pi(u) + cos2pi(v));
        case 1: return u * u + v * v;
        case 2: return -(cos2pi(u) + 0.75 * cos2pi(v));
        case 3: return std::max(std::fabs(u), std::fabs(v));
        default: return -cos2pi(u);
    }
}

static inline double periodic(double x, double y, double period, double ca, double sa,
                              int64_t shape, const float* lut, int64_t lut_n,
                              double smin, double sinv) {
    const double xc = x + 0.5, yc = y + 0.5;
    double u = (xc * ca + yc * sa) / period;
    double v = (-xc * sa + yc * ca) / period;
    u = u - std::floor(u + 0.5);
    v = v - std::floor(v + 0.5);
    const double s = spot(shape, u, v);
    const double p = (s - smin) * sinv;
    const int64_t n = lut_n - 1;
    if (p <= 0.0) return lut[0];
    if (p >= (double)n) return lut[n];
    const int64_t i = (int64_t)p;
    const double f = p - (double)i;
    return lut[i] * (1.0 - f) + lut[i + 1] * f;
}

static inline double eq_mix(double t1, double t2, double a) {
    const double b = 1.0 - a;
    const double z = a * t1 + b * t2;
    const double lo = std::min(a, b), hi = std::max(a, b);
    if (lo < 1e-4) return z;
    const double ab2 = 2.0 * lo * hi;
    double out;
    if (z < lo)       out = z * z / ab2;
    else if (z <= hi) out = (z - 0.5 * lo) / hi;
    else              out = 1.0 - (1.0 - z) * (1.0 - z) / ab2;
    return std::min(std::max(out, 0.0), 0.999999);
}

static inline double emp_mix(double t1, double t2, double a, const float* F, int64_t nb, int64_t nzp1) {
    const double z = a * t1 + (1.0 - a) * t2;
    const double pos = a * (double)(nb - 1);
    int64_t j = (int64_t)pos;
    if (j > nb - 2) j = nb - 2;
    const double fa = pos - (double)j;
    const int64_t nz = nzp1 - 1;
    const double pz = z * (double)nz;
    int64_t k = (int64_t)pz;
    if (k > nz - 1) k = nz - 1;
    if (k < 0) k = 0;
    const double fz = pz - (double)k;
    const float* r0 = F + j * nzp1;
    const float* r1 = F + (j + 1) * nzp1;
    const double f0 = r0[k] * (1.0 - fz) + r0[k + 1] * fz;
    const double f1 = r1[k] * (1.0 - fz) + r1[k + 1] * fz;
    const double out = f0 * (1.0 - fa) + f1 * fa;
    return std::min(std::max(out, 0.0), 0.999999);
}

static inline double lut_curve(const float* c, int64_t n1, double t) {
    const int64_t n = n1 - 1;
    const double p = t * (double)n;
    const int64_t i = (int64_t)p;
    if (i >= n) return c[n];
    const double f = p - (double)i;
    return c[i] * (1.0 - f) + c[i + 1] * f;
}

// Координаты билинейной выборки по одной оси: индекс0, индекс1, доля.
struct Axis { int64_t i0, i1; double a; };

static inline Axis axis_at(double f, int64_t size) {
    if (f < 0.0) f = 0.0;
    if (f > (double)(size - 1)) f = (double)(size - 1);
    Axis r;
    r.i0 = (int64_t)f;
    r.i1 = std::min(r.i0 + 1, size - 1);
    r.a = f - (double)r.i0;
    return r;
}

template <typename T>
static inline double bilin(const T* img, int64_t w, const Axis& ax, const Axis& ay) {
    const T* r0 = img + ay.i0 * w;
    const T* r1 = img + ay.i1 * w;
    const double top = (double)r0[ax.i0] * (1.0 - ax.a) + (double)r0[ax.i1] * ax.a;
    const double bot = (double)r1[ax.i0] * (1.0 - ax.a) + (double)r1[ax.i1] * ax.a;
    return top * (1.0 - ay.a) + bot * ay.a;
}

template <typename TI>
static void render_rows(const HRParams& P, const TI* ink, int64_t y0, int64_t nrows,
                        uint8_t* out, int64_t* counts) {
    const int64_t W = P.Wreal;
    // таблицы по X — общие для всех строк полосы
    std::vector<Axis> ax_in(W), ax_wg(W);
    for (int64_t x = 0; x < W; ++x) {
        ax_in[x] = axis_at((x + 0.5) * P.inv_scale - 0.5, P.ink_w);
        ax_wg[x] = axis_at((x + 0.5) * P.inv_scale_c - 0.5, P.wg_w);
    }
    const double zero_tol = 0.5 * P.ink_inv_max;
    const int64_t wg_plane = P.wg_h * P.wg_w;
    const float* wg0 = P.wgrid;
    const float* wg1 = P.wgrid + wg_plane;
    const float* wg2 = P.wgrid + 2 * wg_plane;
    const float* c0 = P.curves;
    const float* c1 = P.curves + P.curve_n;
    const float* c2 = P.curves + 2 * P.curve_n;

#ifdef _OPENMP
    if (P.nthreads > 0) omp_set_num_threads((int)P.nthreads);
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int64_t r = 0; r < nrows; ++r) {
        const int64_t y = y0 + r;
        const Axis ay = axis_at((y + 0.5) * P.inv_scale - 0.5, P.ink_h);
        const Axis ayc = axis_at((y + 0.5) * P.inv_scale_c - 0.5, P.wg_h);
        const uint8_t* er0 = P.use_edge ? P.edge + ay.i0 * P.ink_w : nullptr;
        const uint8_t* er1 = P.use_edge ? P.edge + ay.i1 * P.ink_w : nullptr;
        const float* fm_row = P.fm_mask + ((y / P.fm_e) % P.fm_tile) * P.fm_tile;
        const float* sel_row = P.selector + (y % P.sel_tile) * P.sel_tile;
        uint8_t* orow = out + r * P.nbytes;
        int64_t cnt = 0;
        for (int64_t bx = 0; bx < P.nbytes; ++bx) {
            uint8_t byte = 0;
            for (int bit = 0; bit < 8; ++bit) {
                const int64_t x = bx * 8 + bit;
                if (x >= W) break;
                const Axis& ax = ax_in[x];
                const double t = bilin(ink, P.ink_w, ax, ay) * P.ink_inv_max;
                if (P.use_edge && (er0[ax.i0] | er0[ax.i1] | er1[ax.i0] | er1[ax.i1])) {
                    if (t >= 0.5) { byte |= (uint8_t)(128 >> bit); ++cnt; }
                    continue;
                }
                if (t <= zero_tol) continue;
                const Axis& axc = ax_wg[x];
                double wa = bilin(wg0, P.wg_w, axc, ayc);
                double wl = bilin(wg1, P.wg_w, axc, ayc);
                double wf = bilin(wg2, P.wg_w, axc, ayc);
                if (wa < 0.0) wa = 0.0;
                if (wl < 0.0) wl = 0.0;
                if (wf < 0.0) wf = 0.0;
                double s = wa + wl + wf;
                if (s <= 0.0) { wa = 1.0; s = 1.0; }
                wa /= s; wl /= s; wf /= s;
                double te;
                if (t >= 0.999) {
                    te = 1.0;
                } else {
                    te = wa * lut_curve(c0, P.curve_n, t) + wl * lut_curve(c1, P.curve_n, t)
                       + wf * lut_curve(c2, P.curve_n, t);
                    const double md = wa * P.min_dots[0] + wl * P.min_dots[1] + wf * P.min_dots[2];
                    if (te < md) te = md;
                }
                double ta = 0.0, tl = 0.0, tf = 0.0;
                if (wa > 1e-4) ta = periodic((double)x, (double)y, P.am_p, P.am_ca, P.am_sa, P.am_shape,
                                            P.am_lut, P.am_lut_n, P.am_smin, P.am_sinv);
                if (wl > 1e-4) tl = periodic((double)x, (double)y, P.ln_p, P.ln_ca, P.ln_sa, P.ln_shape,
                                            P.ln_lut, P.ln_lut_n, P.ln_smin, P.ln_sinv);
                if (wf > 1e-4) tf = fm_row[(x / P.fm_e) % P.fm_tile];
                double T;
                if (P.mode == 1) {
                    const double sv = sel_row[x % P.sel_tile];
                    T = (sv < wa) ? ta : (sv < wa + wl) ? tl : tf;
                } else {
                    const double wal = wa + wl;
                    double t12;
                    if (wa > 1e-4 && wl > 1e-4) {
                        const double a = wa / wal;
                        t12 = P.use_emp ? emp_mix(ta, tl, a, P.emp_F, P.emp_nb, P.emp_nz)
                                        : eq_mix(ta, tl, a);
                    } else if (wa > 1e-4) {
                        t12 = ta;
                    } else {
                        t12 = tl;
                    }
                    if (wf > 1e-4 && wal > 1e-4) T = eq_mix(t12, tf, wal);
                    else if (wf > 1e-4)          T = tf;
                    else                         T = t12;
                }
                if (te > T) { byte |= (uint8_t)(128 >> bit); ++cnt; }
            }
            orow[bx] = byte;
        }
        counts[r] = cnt;
    }
}

// ---------------------------------------------------------------- экспорт
HR_API int hr_version() { return HR_VERSION; }

HR_API int hr_params_size() { return (int)sizeof(HRParams); }

HR_API int hr_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Растрирует строки [y0, y0 + nrows) в out (nrows × nbytes, упакованные биты, старший слева).
HR_API int hr_render_strip(const HRParams* P, int64_t y0, int64_t nrows, uint8_t* out, int64_t* counts) {
    if (!P || !out || !counts) return -1;
    switch (P->ink_type) {
        case 0: render_rows(*P, (const uint8_t*)P->ink, y0, nrows, out, counts); return 0;
        case 1: render_rows(*P, (const uint16_t*)P->ink, y0, nrows, out, counts); return 0;
        case 2: render_rows(*P, (const float*)P->ink, y0, nrows, out, counts); return 0;
        default: return -2;
    }
}

// Микроструктура: u8 (rows × W, 0/1) изменяется на месте; dist — расстояние до непечатающего.
HR_API int hr_surface_pass(uint8_t* u8, const float* dist, int64_t rows, int64_t W, int64_t ya,
                           double rim, double fade, double density,
                           const float* mask, int64_t th, int64_t tw, int64_t cell,
                           int64_t nthreads, int64_t* removed) {
    if (!u8 || !dist || !mask || !removed) return -1;
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads((int)nthreads);
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t y = ya + r;
        const float* mrow = mask + ((y / cell) % th) * tw;
        uint8_t* row = u8 + r * W;
        const float* drow = dist + r * W;
        int64_t cnt = 0;
        for (int64_t x = 0; x < W; ++x) {
            if (!row[x]) continue;
            const double d = drow[x];
            if (d <= rim) continue;
            double f = 1.0;
            if (fade > 0.0) { f = (d - rim) / fade; if (f > 1.0) f = 1.0; }
            if (mrow[(x / cell) % tw] < density * f) { row[x] = 0; ++cnt; }
        }
        removed[r] = cnt;
    }
    return 0;
}

// Превью: доля печатающих пикселей в блоках pf×pf -> 0..255 (255 = белое).
HR_API int hr_block_coverage(const uint8_t* packed, int64_t rows, int64_t nbytes, int64_t W,
                             int64_t pf, uint8_t* res, int64_t nthreads) {
    const int64_t nr = rows / pf, nc = W / pf;
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads((int)nthreads);
#pragma omp parallel for schedule(static)
#endif
    for (int64_t br = 0; br < nr; ++br) {
        for (int64_t bc = 0; bc < nc; ++bc) {
            int64_t c = 0;
            for (int64_t yy = br * pf; yy < br * pf + pf; ++yy) {
                const uint8_t* row = packed + yy * nbytes;
                for (int64_t xx = bc * pf; xx < bc * pf + pf; ++xx)
                    c += (row[xx >> 3] >> (7 - (xx & 7))) & 1;
            }
            res[br * nc + bc] = (uint8_t)(255 - (255 * c) / (pf * pf));
        }
    }
    return 0;
}
