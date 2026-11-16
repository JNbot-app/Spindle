// Minimal CBLAS for the WebAssembly build: only the two routines the engine calls.
// Row-major, single precision, plain loops that the compiler turns into WASM SIMD.
#ifndef SPINDLE_WASM_CBLAS_H
#define SPINDLE_WASM_CBLAS_H
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;
void cblas_sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k,
                 float alpha, const float *a, int lda, const float *b, int ldb, float beta, float *c,
                 int ldc);
void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE ta, int m, int n, float alpha, const float *a,
                 int lda, const float *x, int incx, float beta, float *y, int incy);
#ifdef __cplusplus
}
#endif
#endif
