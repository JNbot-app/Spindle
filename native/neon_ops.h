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

// Int8 weights of group g read straight from each layout: the 64 even weights (4 vectors) and
// the 64 odd ones. They match what lookup_i8 gives for the same weights as nibbles.
struct NibbleWeights {
    int8x16_t table; size_t stride;
    void operator()(const uint8_t *row,size_t g,int8x16_t even[4],int8x16_t odd[4]) const {
        for(size_t j=0;j<4;++j) {
            const auto codes=vld1q_u8(row+g*64+j*16);
            even[j]=vqtbl1q_s8(table,vandq_u8(codes,vdupq_n_u8(15)));
            odd[j]=vqtbl1q_s8(table,vshrq_n_u8(codes,4));
        }
    }
};
// A nibble of an archive 2-bit byte holds an even and an odd weight: low[n] is the even one's
// value, high[n] the odd one's. So 16 bytes give 64 weights with only two bit operations, in the
// order of crumb_activations: even weights 0,2,..,30 then 1,3,..,31, then the same from 32.
struct CrumbWeights {
    int8x16_t low, high; size_t stride;
    CrumbWeights(const int8_t *book,size_t stride_) : stride(stride_) {
        int8_t l[16],h[16];
        for(size_t n=0;n<16;++n) { l[n]=book[n&3]; h[n]=book[n>>2]; }
        low=vld1q_s8(l); high=vld1q_s8(h);
    }
    void operator()(const uint8_t *row,size_t g,int8x16_t even[4],int8x16_t odd[4]) const {
        for(size_t part=0;part<2;++part) {
            const auto codes=vld1q_u8(row+g*32+part*16), a=vandq_u8(codes,vdupq_n_u8(15)), b=vshrq_n_u8(codes,4);
            even[2*part]=vqtbl1q_s8(low,a); even[2*part+1]=vqtbl1q_s8(low,b);
            odd[2*part]=vqtbl1q_s8(high,a); odd[2*part+1]=vqtbl1q_s8(high,b);
        }
    }
};
// Int8 activations reordered to match CrumbWeights. Integer sums do not depend on the order,
// so the output bits are the same.
inline IntegerInput crumb_activations(const Tensor &t,const IntegerInput &input) {
    thread_local std::vector<int8_t> buffer;
    buffer.resize(t.padded);
    for(size_t k=0;k<t.padded;k+=32) {
        const int8x16_t a=vld1q_s8(input.activations+k), b=vld1q_s8(input.activations+k+16);
        vst1q_s8(buffer.data()+k,vuzp1q_s8(a,b));
        vst1q_s8(buffer.data()+k+16,vuzp2q_s8(a,b));
    }
    IntegerInput out=input;
    out.activations=buffer.data();
    return out;
}
// The integer projection of kernel_integer for NR rows at once, the per-group reduction of four
// rows done in one vector. Each lane runs the scalar formula, so the output bits are the same.
template<class Ops,size_t NR,class Weights>
inline void neon_integer_rows(const Tensor &t,const IntegerInput &input,const Weights &weights,
                              size_t row,float *out) {
    const size_t groups=t.padded/128, half=t.padded/2;
    const uint8_t *codes=t.data+row*weights.stride;
    const float *norms=t.values.data()+row*groups;
    float32x4_t sum=vdupq_n_f32(0.f);
    for(size_t g=0;g<groups;++g) {
        int32x4_t even[4],odd[4];
        for(size_t ri=0;ri<4;++ri) even[ri]=odd[ri]=vdupq_n_s32(0);
        int8x16_t ev[4],od[4];
        for(size_t j=0;j<4;++j) {
            ev[j]=vld1q_s8(input.activations+g*64+j*16);
            od[j]=vld1q_s8(input.activations+half+g*64+j*16);
        }
        for(size_t ri=0;ri<NR;++ri) {
            int8x16_t we[4],wo[4];
            weights(codes+ri*weights.stride,g,we,wo);
            for(size_t j=0;j<4;++j) {
                even[ri]=Ops::dot_acc(even[ri],we[j],ev[j]);
                odd[ri]=Ops::dot_acc(odd[ri],wo[j],od[j]);
            }
        }
        const int32x4_t e=vpaddq_s32(vpaddq_s32(even[0],even[1]),vpaddq_s32(even[2],even[3]));
        const int32x4_t o=vpaddq_s32(vpaddq_s32(odd[0],odd[1]),vpaddq_s32(odd[2],odd[3]));
        const float32x4_t value=vaddq_f32(vmulq_n_f32(vcvtq_f32_s32(e),input.scales[g*2]),
                                          vmulq_n_f32(vcvtq_f32_s32(o),input.scales[g*2+1]));
        float norm[4]={};
        for(size_t ri=0;ri<NR;++ri) norm[ri]=norms[ri*groups+g];
        sum=vaddq_f32(sum,vmulq_f32(value,vld1q_f32(norm)));
    }
    float result[4];
    vst1q_f32(result,sum);
    std::copy_n(result,NR,out);
}
template<class Ops,class Weights>
inline void neon_integer_layout(const Tensor &t,const IntegerInput &input,const Weights &weights,
                                size_t begin,size_t rows,float *out) {
    size_t r=0;
    for(;r+4<=rows;r+=4) neon_integer_rows<Ops,4>(t,input,weights,begin+r,out+r);
    for(;r<rows;++r) neon_integer_rows<Ops,1>(t,input,weights,begin+r,out+r);
}
template<class Ops>
void neon_integer(const Tensor &t, const IntegerInput &input, size_t begin, size_t rows, float *out) {
    const auto table=vld1q_s8(input.book);
    switch(t.layout) {
    case Layout::nibbles: return neon_integer_layout<Ops>(t,input,NibbleWeights{table,t.row_bytes()},begin,rows,out);
    case Layout::crumbs:
        return neon_integer_layout<Ops>(t,crumb_activations(t,input),CrumbWeights(input.book,t.row_bytes()),begin,rows,out);
    }
}

// Decode sixteen packed bytes at a time with byte-wise codebook tables. A
// structure store interleaves the four FP32 bytes without scalar gathers.
inline uint8x16x4_t float_table(const float *book,float scale,bool scaled) {
    float values[16];
    for(size_t i=0;i<16;i+=4)
        vst1q_f32(values+i,scaled ? vmulq_n_f32(vld1q_f32(book+i),scale) : vld1q_f32(book+i));
    return vld4q_u8(reinterpret_cast<const uint8_t *>(values));
}
inline void store_lookup(float *out,const uint8x16x4_t &table,uint8x16_t index) {
    uint8x16x4_t bytes;
    for(size_t i=0;i<4;++i) bytes.val[i]=vqtbl1q_u8(table.val[i],index);
    vst4q_u8(reinterpret_cast<uint8_t *>(out),bytes);
}
inline void neon_decode(const Tensor &t, size_t begin, size_t rows, float *weights, bool scaled) {
    const size_t half=t.padded/2, groups=t.padded/128;
    if(t.layout==Layout::crumbs) {
        for(size_t r=0;r<rows;++r)
            for(size_t g=0;g<groups;++g) {
                const auto table=float_table(t.book.data(),t.values[(begin+r)*groups+g],scaled);
                for(size_t part=0;part<2;++part) { // bytes 0-15 hold weights 0-63, bytes 16-31 the rest
                    const auto codes=vld1q_u8(t.data+(begin+r)*t.row_bytes()+g*32+part*16), three=vdupq_n_u8(3);
                    const auto e0=vandq_u8(codes,three), e1=vandq_u8(vshrq_n_u8(codes,4),three);
                    const auto o0=vandq_u8(vshrq_n_u8(codes,2),three), o1=vshrq_n_u8(codes,6);
                    float *out=weights+r*t.padded+g*64+part*32;
                    store_lookup(out,table,vzip1q_u8(e0,e1)); store_lookup(out+16,table,vzip2q_u8(e0,e1));
                    store_lookup(out+half,table,vzip1q_u8(o0,o1)); store_lookup(out+half+16,table,vzip2q_u8(o0,o1));
                }
            }
        return;
    }
    nibble_rows(t,begin,rows,[&](const uint8_t *data,size_t first,size_t count) {
        for(size_t r=0;r<count;++r)
            for(size_t g=0;g<groups;++g) {
                const auto table=float_table(t.book.data(),t.values[(first+r)*groups+g],scaled);
                for(size_t k=0;k<64;k+=16) {
                    const auto codes=vld1q_u8(data+r*half+g*64+k);
                    float *out=weights+(first-begin+r)*t.padded+g*64+k;
                    store_lookup(out,table,vandq_u8(codes,vdupq_n_u8(15)));
                    store_lookup(out+half,table,vshrq_n_u8(codes,4));
                }
            }
    });
}
}
