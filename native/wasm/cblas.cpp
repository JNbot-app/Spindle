// The two BLAS routines the engine uses, written with WASM SIMD. Each output keeps the summation
// order of plain loops (one running sum over k for B rows; four strided sums for dot products),
// so results do not depend on the blocking. Blocking only keeps partial sums in registers.
#include "cblas.h"
#include <cstddef>
#include <cstring>
#include <wasm_simd128.h>

namespace {
// c[r][j] += (alpha * a(r, p)) * b[p][j], for R rows of C and 16 columns at a time.
template <int R>
void rows_times_b(int k, int n, float alpha, const float *a, int lda, bool ta, const float *b, int ldb,
                  float *c, int ldc) {
    auto av = [&](int r, int p) { return alpha * (ta ? a[size_t(p) * lda + r] : a[size_t(r) * lda + p]); };
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        v128_t acc[R][4];
        for (int r = 0; r < R; ++r)
            for (int q = 0; q < 4; ++q) acc[r][q] = wasm_v128_load(c + size_t(r) * ldc + j + 4 * q);
        for (int p = 0; p < k; ++p) {
            const float *brow = b + size_t(p) * ldb + j;
            const v128_t b0 = wasm_v128_load(brow), b1 = wasm_v128_load(brow + 4);
            const v128_t b2 = wasm_v128_load(brow + 8), b3 = wasm_v128_load(brow + 12);
            for (int r = 0; r < R; ++r) {
                const v128_t s = wasm_f32x4_splat(av(r, p));
                acc[r][0] = wasm_f32x4_add(acc[r][0], wasm_f32x4_mul(s, b0));
                acc[r][1] = wasm_f32x4_add(acc[r][1], wasm_f32x4_mul(s, b1));
                acc[r][2] = wasm_f32x4_add(acc[r][2], wasm_f32x4_mul(s, b2));
                acc[r][3] = wasm_f32x4_add(acc[r][3], wasm_f32x4_mul(s, b3));
            }
        }
        for (int r = 0; r < R; ++r)
            for (int q = 0; q < 4; ++q) wasm_v128_store(c + size_t(r) * ldc + j + 4 * q, acc[r][q]);
    }
    for (; j < n; ++j)
        for (int r = 0; r < R; ++r) {
            float sum = c[size_t(r) * ldc + j];
            for (int p = 0; p < k; ++p) sum += av(r, p) * b[size_t(p) * ldb + j];
            c[size_t(r) * ldc + j] = sum;
        }
}

// Four strided partial sums of a[p] * b[p] (lane = p mod 4), combined as (s0 + s1) + (s2 + s3).
inline float finish(v128_t s, const float *a, const float *b, int p, int k) {
    float s0 = wasm_f32x4_extract_lane(s, 0);
    for (; p < k; ++p) s0 += a[p] * b[p];
    return (s0 + wasm_f32x4_extract_lane(s, 1)) + (wasm_f32x4_extract_lane(s, 2) + wasm_f32x4_extract_lane(s, 3));
}

// c[r][q] += alpha * dot(a row r, b row q) for R rows of A and J rows of B at a time.
template <int R, int J>
void dot_block(int k, float alpha, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    v128_t acc[R][J];
    for (int r = 0; r < R; ++r)
        for (int q = 0; q < J; ++q) acc[r][q] = wasm_f32x4_splat(0.f);
    int p = 0;
    for (; p + 4 <= k; p += 4) {
        v128_t bv[J];
        for (int q = 0; q < J; ++q) bv[q] = wasm_v128_load(b + size_t(q) * ldb + p);
        for (int r = 0; r < R; ++r) {
            const v128_t x = wasm_v128_load(a + size_t(r) * lda + p);
            for (int q = 0; q < J; ++q) acc[r][q] = wasm_f32x4_add(acc[r][q], wasm_f32x4_mul(x, bv[q]));
        }
    }
    for (int r = 0; r < R; ++r)
        for (int q = 0; q < J; ++q)
            c[size_t(r) * ldc + q] += alpha * finish(acc[r][q], a + size_t(r) * lda, b + size_t(q) * ldb, p, k);
}
template <int R>
void dot_rows(int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    int j = 0;
    for (; j + 4 <= n; j += 4) dot_block<R, 4>(k, alpha, a, lda, b + size_t(j) * ldb, ldb, c + j, ldc);
    for (; j < n; ++j) dot_block<R, 1>(k, alpha, a, lda, b + size_t(j) * ldb, ldb, c + j, ldc);
}
} // namespace

extern "C" {
// C[m,n] = alpha * op(A)[m,k] * op(B)[k,n] + beta * C. Row-major only.
void cblas_sgemm(CBLAS_ORDER, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, float alpha,
                 const float *a, int lda, const float *b, int ldb, float beta, float *c, int ldc) {
    for (int i = 0; i < m; ++i) {
        float *row = c + size_t(i) * ldc;
        if (beta == 0.f) std::memset(row, 0, size_t(n) * sizeof(float));
        else if (beta != 1.f) for (int j = 0; j < n; ++j) row[j] *= beta;
    }
    const bool at = ta != CblasNoTrans;
    if (tb == CblasNoTrans) {
        // Row i of op(A) starts at a + i * lda, or at column i of A when A is transposed.
        auto a_row = [&](int i) { return at ? a + i : a + size_t(i) * lda; };
        int i = 0;
        for (; i + 4 <= m; i += 4) rows_times_b<4>(k, n, alpha, a_row(i), lda, at, b, ldb, c + size_t(i) * ldc, ldc);
        for (; i < m; ++i) rows_times_b<1>(k, n, alpha, a_row(i), lda, at, b, ldb, c + size_t(i) * ldc, ldc);
    } else if (!at) {
        int i = 0;
        for (; i + 4 <= m; i += 4) dot_rows<4>(n, k, alpha, a + size_t(i) * lda, lda, b, ldb, c + size_t(i) * ldc, ldc);
        for (; i < m; ++i) dot_rows<1>(n, k, alpha, a + size_t(i) * lda, lda, b, ldb, c + size_t(i) * ldc, ldc);
    } else {
        for (int i = 0; i < m; ++i) // both transposed: not used by the engine, kept simple
            for (int j = 0; j < n; ++j) {
                float s0 = 0.f;
                for (int p = 0; p < k; ++p) s0 += a[size_t(p) * lda + i] * b[size_t(j) * ldb + p];
                c[size_t(i) * ldc + j] += alpha * s0;
            }
    }
}

void cblas_sgemv(CBLAS_ORDER, CBLAS_TRANSPOSE ta, int m, int n, float alpha, const float *a, int lda,
                 const float *x, int incx, float beta, float *y, int incy) {
    // Only the unit-stride case is used by the engine.
    (void)incx; (void)incy;
    const int rows = ta == CblasNoTrans ? m : n, cols = ta == CblasNoTrans ? n : m;
    for (int i = 0; i < rows; ++i) y[i] = beta == 0.f ? 0.f : y[i] * beta;
    if (ta == CblasNoTrans) {
        int i = 0;
        for (; i + 4 <= rows; i += 4) dot_block<1, 4>(cols, alpha, x, 0, a + size_t(i) * lda, lda, y + i, 0);
        for (; i < rows; ++i) dot_block<1, 1>(cols, alpha, x, 0, a + size_t(i) * lda, lda, y + i, 0);
    } else {
        for (int p = 0; p < cols; ++p) {
            const float av = alpha * x[p];
            const float *arow = a + size_t(p) * lda;
            for (int i = 0; i < rows; ++i) y[i] += av * arow[i];
        }
    }
}
}
