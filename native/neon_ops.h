#pragma once
#include "kernel_template.h"
#include <arm_neon.h>
#include <cstring>
namespace spindle {
struct NeonOps {
    struct Vec { float32x4_t lo{}, hi{}; };
    uint8x16x4_t table;
    explicit NeonOps(const float *book) {
        const auto *p=reinterpret_cast<const uint8_t *>(book);
        for(size_t i=0;i<4;++i) table.val[i]=vld1q_u8(p+i*16);
    }
    Vec lookup(const uint8_t *p) const {
        uint32_t packed; std::memcpy(&packed,p,4);
        const uint8x16_t repeat={0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
        const uint8x16_t offsets={0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
        auto codes=vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(packed)),repeat);
        auto lo=vorrq_u8(vshlq_n_u8(vandq_u8(codes,vdupq_n_u8(15)),2),offsets);
        auto hi=vorrq_u8(vshlq_n_u8(vshrq_n_u8(codes,4),2),offsets);
        return {vreinterpretq_f32_u8(vqtbl4q_u8(table,lo)),vreinterpretq_f32_u8(vqtbl4q_u8(table,hi))};
    }
    static Vec load(const float *a,const float *b) { return {vld1q_f32(a),vld1q_f32(b)}; }
    static void store(float *a,float *b,Vec v) { vst1q_f32(a,v.lo);vst1q_f32(b,v.hi); }
    static Vec madd(Vec a,Vec w,Vec x) { return {vaddq_f32(a.lo,vmulq_f32(w.lo,x.lo)),vaddq_f32(a.hi,vmulq_f32(w.hi,x.hi))}; }
    static Vec scale(Vec v,float s) { return {vmulq_n_f32(v.lo,s),vmulq_n_f32(v.hi,s)}; }
    static float sum(Vec v) { return vaddvq_f32(vaddq_f32(v.lo,v.hi)); }
    using Table=int8x16_t; using Bytes=int8x16_t; using Acc=int32x4_t;
    static Table table_i8(const int8_t *book) { return vld1q_s8(book); }
    static Bytes load_i8(const int8_t *p) { return vld1q_s8(p); }
    static void lookup_i8(Table table,const uint8_t *p,Bytes &lo,Bytes &hi) {
        const auto codes=vld1q_u8(p);
        lo=vqtbl1q_s8(table,vandq_u8(codes,vdupq_n_u8(15)));
        hi=vqtbl1q_s8(table,vshrq_n_u8(codes,4));
    }
    static Acc dot_acc(Acc acc,Bytes a,Bytes b) {
        acc=vpadalq_s16(acc,vmull_s8(vget_low_s8(a),vget_low_s8(b)));
        return vpadalq_s16(acc,vmull_high_s8(a,b));
    }
    static int32_t hsum(Acc acc) { return vaddvq_s32(acc); }
    static void butterfly(float *a,float *b) { auto x=vld1q_f32(a),y=vld1q_f32(b);vst1q_f32(a,vaddq_f32(x,y));vst1q_f32(b,vsubq_f32(x,y)); }
};

// The integer projection of kernel_integer, with the per-group reduction of four rows done in
// one vector. Each lane runs the scalar formula, so the output bits are the same.
template<class Ops>
void neon_integer(const Tensor &t, const IntegerInput &input, size_t begin, size_t rows, float *out) {
    const auto table=Ops::table_i8(input.book);
    const size_t groups=t.padded/128, half=t.padded/2;
    size_t r=0;
    for(;r+4<=rows;r+=4) {
        const uint8_t *codes=t.data+(begin+r)*half;
        const float *norms=t.values.data()+(begin+r)*groups;
        float32x4_t sum=vdupq_n_f32(0.f);
        for(size_t g=0;g<groups;++g) {
            int32x4_t even[4],odd[4];
            for(size_t ri=0;ri<4;++ri) even[ri]=odd[ri]=vdupq_n_s32(0);
            for(size_t k=0;k<64;k+=16) {
                const int8x16_t ev=vld1q_s8(input.activations+g*64+k), od=vld1q_s8(input.activations+half+g*64+k);
                for(size_t ri=0;ri<4;++ri) {
                    int8x16_t low,high;
                    Ops::lookup_i8(table,codes+ri*half+g*64+k,low,high);
                    even[ri]=Ops::dot_acc(even[ri],low,ev);
                    odd[ri]=Ops::dot_acc(odd[ri],high,od);
                }
            }
            const int32x4_t e=vpaddq_s32(vpaddq_s32(even[0],even[1]),vpaddq_s32(even[2],even[3]));
            const int32x4_t o=vpaddq_s32(vpaddq_s32(odd[0],odd[1]),vpaddq_s32(odd[2],odd[3]));
            const float32x4_t value=vaddq_f32(vmulq_n_f32(vcvtq_f32_s32(e),input.scales[g*2]),
                                              vmulq_n_f32(vcvtq_f32_s32(o),input.scales[g*2+1]));
            const float norm[4]={norms[g],norms[groups+g],norms[2*groups+g],norms[3*groups+g]};
            sum=vaddq_f32(sum,vmulq_f32(value,vld1q_f32(norm)));
        }
        vst1q_f32(out+r,sum);
    }
    for(;r<rows;++r) integer_rows<Ops,1>(t,input,table,begin+r,out+r);
}

// Decode sixteen packed bytes at a time with byte-wise codebook tables. A
// structure store interleaves the four FP32 bytes without scalar gathers.
inline void neon_decode(const Tensor &t, size_t begin, size_t rows, float *weights, bool scaled) {
    const size_t half=t.padded/2, groups=t.padded/128;
    const auto unscaled=vld4q_u8(reinterpret_cast<const uint8_t *>(t.book.data()));
    for(size_t r=0;r<rows;++r)
        for(size_t g=0;g<groups;++g) {
            auto table=unscaled;
            if(scaled) {
                float book[16];
                const float scale=t.values[(begin+r)*groups+g];
                for(size_t i=0;i<16;i+=4)
                    vst1q_f32(book+i,vmulq_n_f32(vld1q_f32(t.book.data()+i),scale));
                table=vld4q_u8(reinterpret_cast<const uint8_t *>(book));
            }
            for(size_t k=0;k<64;k+=16) {
                const auto codes=vld1q_u8(t.data+(begin+r)*half+g*64+k);
                const auto low=vandq_u8(codes,vdupq_n_u8(15)), high=vshrq_n_u8(codes,4);
                uint8x16x4_t even,odd;
                for(size_t i=0;i<4;++i) {
                    even.val[i]=vqtbl1q_u8(table.val[i],low);
                    odd.val[i]=vqtbl1q_u8(table.val[i],high);
                }
                float *out=weights+r*t.padded+g*64+k;
                vst4q_u8(reinterpret_cast<uint8_t *>(out),even);
                vst4q_u8(reinterpret_cast<uint8_t *>(out+half),odd);
            }
        }
}
}
