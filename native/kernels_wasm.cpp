// WebAssembly SIMD kernels. They keep the same reduction order as the other kernels, so the
// browser gives the same results as the SSE2 build; the gain is the byte shuffle (swizzle),
// which looks up codebook entries in a register instead of through memory.
#include "kernel_template.h"
#include <wasm_simd128.h>
namespace spindle {
struct WasmOps {
    struct Vec { v128_t lo, hi; };
    const float *book;
    explicit WasmOps(const float *p):book(p){}
    Vec lookup(const uint8_t *p) const {
        return {wasm_f32x4_make(book[p[0]&15],book[p[1]&15],book[p[2]&15],book[p[3]&15]),
                wasm_f32x4_make(book[p[0]>>4],book[p[1]>>4],book[p[2]>>4],book[p[3]>>4])};
    }
    static Vec load(const float *a,const float *b) { return {wasm_v128_load(a),wasm_v128_load(b)}; }
    static void store(float *a,float *b,Vec v) { wasm_v128_store(a,v.lo);wasm_v128_store(b,v.hi); }
    static Vec madd(Vec a,Vec w,Vec x) { return {wasm_f32x4_add(a.lo,wasm_f32x4_mul(w.lo,x.lo)),wasm_f32x4_add(a.hi,wasm_f32x4_mul(w.hi,x.hi))}; }
    static Vec scale(Vec v,float s) { auto z=wasm_f32x4_splat(s);return {wasm_f32x4_mul(v.lo,z),wasm_f32x4_mul(v.hi,z)}; }
    static float sum(Vec v) {
        auto s=wasm_f32x4_add(v.lo,v.hi);
        return (wasm_f32x4_extract_lane(s,0)+wasm_f32x4_extract_lane(s,1))+(wasm_f32x4_extract_lane(s,2)+wasm_f32x4_extract_lane(s,3));
    }
    using Table=v128_t; using Bytes=v128_t; using Acc=v128_t;
    static Table table_i8(const int8_t *book) { return wasm_v128_load(book); }
    static Bytes load_i8(const int8_t *p) { return wasm_v128_load(p); }
    static void lookup_i8(Table table,const uint8_t *p,Bytes &lo,Bytes &hi) {
        const auto codes=wasm_v128_load(p);
        lo=wasm_i8x16_swizzle(table,wasm_v128_and(codes,wasm_i8x16_splat(15)));
        hi=wasm_i8x16_swizzle(table,wasm_u8x16_shr(codes,4));
    }
    static Acc dot_acc(Acc acc,Bytes x,Bytes y) {
        acc=wasm_i32x4_add(acc,wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_low_i8x16(x,y)));
        return wasm_i32x4_add(acc,wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extmul_high_i8x16(x,y)));
    }
    static int32_t hsum(Acc acc) {
        return wasm_i32x4_extract_lane(acc,0)+wasm_i32x4_extract_lane(acc,1)+wasm_i32x4_extract_lane(acc,2)+wasm_i32x4_extract_lane(acc,3);
    }
    static void butterfly(float *a,float *b) { auto x=wasm_v128_load(a),y=wasm_v128_load(b);wasm_v128_store(a,wasm_f32x4_add(x,y));wasm_v128_store(b,wasm_f32x4_sub(x,y)); }
};
// A8 quantization: round to nearest even, keeping the sign of zero (as the SSE2 kernel does).
static void wasm_a8(float *x,size_t n) {
    v128_t m=wasm_f32x4_splat(0.f);size_t i=0;
    for(;i+4<=n;i+=4)m=wasm_f32x4_max(m,wasm_f32x4_abs(wasm_v128_load(x+i)));
    float maximum=std::max(std::max(wasm_f32x4_extract_lane(m,0),wasm_f32x4_extract_lane(m,1)),
                           std::max(wasm_f32x4_extract_lane(m,2),wasm_f32x4_extract_lane(m,3)));
    for(;i<n;++i)maximum=std::max(maximum,std::abs(x[i]));
    const float scale=maximum>0?maximum*(1.f/127.f):1.f;
    const auto inv=wasm_f32x4_splat(1.f/scale),s=wasm_f32x4_splat(scale),sign=wasm_i32x4_splat(int(0x80000000u));
    const auto low=wasm_f32x4_splat(-128.f),high=wasm_f32x4_splat(127.f);
    for(i=0;i+4<=n;i+=4){
        auto v=wasm_f32x4_mul(wasm_v128_load(x+i),inv);
        v=wasm_f32x4_min(high,wasm_f32x4_max(low,v));
        auto rounded=wasm_v128_or(wasm_f32x4_nearest(v),wasm_v128_and(v,sign));
        wasm_v128_store(x+i,wasm_f32x4_mul(rounded,s));
    }
    for(;i<n;++i)x[i]=std::max(-128.f,std::min(127.f,std::nearbyint(x[i]/scale)))*scale;
}
const CpuKernels &wasm_kernels(){
    static const CpuKernels k{"wasm",kernel_project<WasmOps>,kernel_integer<WasmOps>,kernel_decode<WasmOps>,kernel_fwht<WasmOps>,wasm_a8};
    return k;
}
} // namespace spindle
