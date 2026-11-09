#include "internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

using namespace spindle;

static bool batched_attention() {
    std::mt19937 rng(41);
    auto random = [&] { return (int(rng() % 255) - 127) * .007f; };
    for (size_t window : {1, 4, 256})
        for (size_t count : {2, 31, 32})
            for (size_t previous : {size_t(0), window - 1})
                for (size_t qdim : {4, 48, 64}) {
                    const size_t heads = 4, kv_heads = 2, vdim = 64, entries = previous + count;
                    std::vector<float> q(count * heads * qdim), k(kv_heads * entries * qdim),
                                       v(kv_heads * entries * vdim), scores(count * entries),
                                       actual(count * heads * vdim + 2, 1234.f);
                    for (auto &x : q) x = random();
                    for (auto &x : k) x = random();
                    for (auto &x : v) x = random();
                    attention_prefill(q.data(), k.data(), v.data(), scores.data(), actual.data() + 1,
                                      count, previous, window, heads, kv_heads, qdim, vdim);
                    if (actual.front() != 1234.f || actual.back() != 1234.f) return false;
                    for (size_t t = 0; t < count; ++t)
                        for (size_t h = 0; h < heads; ++h) {
                            const size_t kh = h / (heads / kv_heads), end = previous + t + 1;
                            const size_t begin = end > window ? end - window : 0;
                            std::vector<double> probabilities(end - begin);
                            for (size_t j = begin; j < end; ++j) {
                                double sum = 0.;
                                for (size_t i = 0; i < qdim; ++i)
                                    sum += double(q[(t * heads + h) * qdim + i]) * k[(kh * entries + j) * qdim + i];
                                probabilities[j - begin] = sum / std::sqrt(double(qdim));
                            }
                            double maximum = *std::max_element(probabilities.begin(), probabilities.end()), total = 0.;
                            for (auto &x : probabilities) { x = std::exp(x - maximum); total += x; }
                            for (size_t i = 0; i < vdim; ++i) {
                                double expected = 0.;
                                for (size_t j = begin; j < end; ++j)
                                    expected += probabilities[j - begin] / total * v[(kh * entries + j) * vdim + i];
                                if (std::abs(actual[1 + (t * heads + h) * vdim + i] - expected) > 2e-6)
                                    return false;
                            }
                        }
                }
    return true;
}

static bool decode_ring() {
    // Grouped decode attention over a wrapped ring, against the dequantized cache.
    std::mt19937 rng(43);
    for (size_t window : {4, 256})
        for (size_t qdim : {4, 48, 64}) {
            const size_t heads = 2, group = 6, vdim = 64;
            KV kv(window, heads, qdim, vdim);
            std::vector<float> key(heads * qdim), value(heads * vdim), q(group * qdim),
                               scores(group * window), out(group * vdim);
            for (size_t pos = 0; pos < window + 33; ++pos) {
                for (auto &x : key) x = (int(rng() % 255) - 127) * .01f;
                for (auto &x : value) x = (int(rng() % 255) - 127) * .02f;
                kv.put(pos, key.data(), value.data());
            }
            for (size_t g = 0; g < group; ++g) {
                for (size_t i = 0; i < qdim; ++i) q[g * qdim + i] = (int(rng() % 255) - 127) * .03f;
                a8(q.data() + g * qdim, qdim);
            }
            const float scale = 1.f / std::sqrt(float(qdim));
            for (size_t first : {size_t(0), window - 3})
                for (size_t h = 0; h < heads; ++h) {
                    kv.attend(h, q.data(), group, first, window, scores.data(), out.data(), scale);
                    for (size_t g = 0; g < group; ++g) {
                        std::vector<double> p(window);
                        for (size_t j = 0; j < window; ++j) {
                            const size_t row = (first + j) % window * heads + h;
                            double dot = 0.;
                            for (size_t i = 0; i < qdim; ++i)
                                dot += double(q[g * qdim + i]) * kv.k[row * qdim + i] * kv.ks[row];
                            p[j] = dot * scale;
                        }
                        const double maximum = *std::max_element(p.begin(), p.end());
                        double total = 0.;
                        for (auto &x : p) { x = std::exp(x - maximum); total += x; }
                        for (size_t i = 0; i < vdim; ++i) {
                            double expected = 0.;
                            for (size_t j = 0; j < window; ++j) {
                                const size_t row = (first + j) % window * heads + h;
                                expected += p[j] / total * kv.v[row * vdim + i] * kv.vs[row];
                            }
                            if (std::abs(out[g * vdim + i] - expected) > 2e-5) return false;
                        }
                    }
                }
        }
    return true;
}

static bool convolution() {
    for (size_t width : {1, 16, 768})
        for (size_t taps_count : {3, 4})
            for (size_t dilation : {1, 3})
                for (size_t window : {1, 2, 256}) {
                    const size_t rows = (taps_count - 1) * dilation + 1;
                    History history(rows, width);
                    Tensor taps;
                    taps.kind = 1; taps.shape[0] = taps_count; taps.elements = taps_count * width;
                    taps.values.resize(taps.elements);
                    for (size_t i = 0; i < taps.elements; ++i) taps.values[i] = std::cos(float(i)) * .13f;
                    std::vector<float> raw(width), expected(width), actual(width), ring(rows * width);
                    for (size_t pos = 0; pos < 40; ++pos) {
                        for (size_t i = 0; i < width; ++i) raw[i] = std::sin(float(pos * width + i));
                        std::copy(raw.begin(), raw.end(), ring.begin() + (pos % rows) * width);
                        const size_t count = std::min(taps_count, pos / dilation + 1);
                        for (size_t i = 0; i < width; ++i) {
                            float sum = 0.f;
                            if (dilation == 1 && taps_count == 3 && window >= 3 && count >= 2) {
                                float previous = taps.values[width + i] * ring[((pos - 1) % rows) * width + i];
                                sum = std::fma(taps.values[i], raw[i], previous);
                                if (count == 3)
                                    sum = std::fma(taps.values[2 * width + i], ring[((pos - 2) % rows) * width + i], sum);
                            } else
                                for (size_t j = 0; j < count; ++j)
                                    if (j * dilation < window)
                                        sum += taps.values[j * width + i] * ring[((pos - j * dilation) % rows) * width + i];
                            expected[i] = sum;
                        }
                        history.convolve(raw.data(), actual.data(), pos, taps, dilation, window);
                        if (std::memcmp(expected.data(), actual.data(), width * sizeof(float))) return false;
                    }
                }
    return true;
}

static bool normalization() {
    std::mt19937 rng(91);
    for(size_t n:{1,3,4,31,32,48,64,127,768,3072,32768}) {
        std::vector<float> x(n),scale(n),out(n);
        double square=0.;
        for(size_t i=0;i<n;++i) {
            x[i]=(int(rng()%2001)-1000)*.031f;
            scale[i]=(int(rng()%101)-50)*.01f;
            square+=double(x[i])*x[i];
        }
        const double rms=std::sqrt(square/double(n)+1e-6);
        norm(x.data(),out.data(),n,scale.data());
        for(size_t i=0;i<n;++i)
            if(std::abs(out[i]-(1.+scale[i])*x[i]/rms)>2e-5) return false;
        auto inplace=x;
        norm(inplace.data(),inplace.data(),n,scale.data());
        if(inplace!=out) return false;
    }
    for(size_t n:{1,4,8}) for(float span:{0.f,1.f,20.f,79.f}) {
        std::vector<float> actual(n*n),expected(n*n);
        for(size_t i=0;i<n*n;++i) actual[i]=expected[i]=float(rng()%1001)*span/1000.f;
        for(int iteration=0;iteration<20;++iteration)
            for(int axis=0;axis<2;++axis)
                for(size_t row=0;row<n;++row) {
                    float maximum=-INFINITY,sum=0.f;
                    for(size_t col=0;col<n;++col) maximum=std::max(maximum,expected[axis?col*n+row:row*n+col]);
                    for(size_t col=0;col<n;++col) sum+=std::exp(expected[axis?col*n+row:row*n+col]-maximum);
                    const float total=maximum+std::log(sum);
                    for(size_t col=0;col<n;++col) expected[axis?col*n+row:row*n+col]-=total;
                }
        if(!sinkhorn_probabilities(actual.data(),n)) return false;
        for(size_t i=0;i<n*n;++i)
            if(std::abs(actual[i]-std::exp(expected[i]))>3e-6f) return false;
    }
    float extreme[]={-1000.f,0.f,0.f,1000.f};
    const std::vector<float> copy(extreme,extreme+4);
    return !sinkhorn_probabilities(extreme,2) && std::equal(copy.begin(),copy.end(),extreme);
}

int main() {
    try { (void)cpu_kernels(); } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << "\n"; return 77;
    }
    if (!batched_attention()) { std::cerr << "Causal/GQA/window attention mismatch\n"; return 1; }
    if (!decode_ring()) { std::cerr << "Decode attention ring mismatch\n"; return 1; }
    if (!convolution()) { std::cerr << "Convolution reduction changed\n"; return 1; }
    if (!normalization()) { std::cerr << "Normalization accuracy mismatch\n"; return 1; }
    std::cout << "Attention, value ring and convolution checks passed\n";
}
