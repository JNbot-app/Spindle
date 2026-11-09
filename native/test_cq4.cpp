#include "internal.h"
#include "kernels.h"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
using namespace spindle;

// Frozen pre-optimization packed projection. Keep this independent of cq4.cpp.
static void reference(const Tensor &t, const float *x, size_t batch, float *out, size_t begin,
                      size_t rows) {
    std::vector<float> rotated_input;
    const size_t cols = t.shape[1];
    size_t padded = t.padded;
    size_t num_groups = padded / 128;
    if (rotated_input.size() < 2 * padded)
        rotated_input.resize(std::max(size_t(8192), 2 * padded));

    float *x_rot = rotated_input.data();
    float *x_even = rotated_input.data() + padded;
    float *x_odd = x_even + padded / 2;

    const uint8_t *codes_base = t.data;
    const float *scales_base = t.values.data();
    const float *book = t.book.data();

    for (size_t b = 0; b < batch; ++b) {
        const float *xb = x + b * cols;
        float *outb = out + b * rows;

        std::copy_n(xb, cols, x_rot);
        if (padded > cols)
            std::fill(x_rot + cols, x_rot + padded, 0.f);

        {
            ProfileTimer timer_fwht(PHASE_MATH_FWHT);
            for (size_t g = 0; g < num_groups; ++g) {
                float *rg = x_rot + g * 128;
                fwht(rg, 128);
                float *ev = x_even + g * 64;
                float *od = x_odd + g * 64;
                for (size_t i = 0; i < 64; ++i) {
                    ev[i] = rg[2 * i];
                    od[i] = rg[2 * i + 1];
                }
            }
        }

        {
            ProfileTimer timer_dot(PHASE_MATH_PACKED_DOT);
            for (size_t i = 0; i < rows; ++i) {
                size_t r = begin + i;
                float sum = 0.f;
                const uint8_t *r_codes = codes_base + r * (padded / 2);
                const float *r_scales = scales_base + r * num_groups;

                for (size_t g = 0; g < num_groups; ++g) {
                    float scale = r_scales[g];
                    const uint8_t *p = r_codes + g * 64;
                    const float *ev = x_even + g * 64;
                    const float *od = x_odd + g * 64;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    float32x4_t v_acc0 = vdupq_n_f32(0.0f);
                    float32x4_t v_acc1 = vdupq_n_f32(0.0f);
                    for (size_t k = 0; k < 64; k += 4) {
                        uint8_t b0 = p[k + 0], b1 = p[k + 1], b2 = p[k + 2], b3 = p[k + 3];
                        float32x4_t w_low = {book[b0 & 15], book[b1 & 15], book[b2 & 15],
                                             book[b3 & 15]};
                        float32x4_t w_high = {book[b0 >> 4], book[b1 >> 4], book[b2 >> 4],
                                              book[b3 >> 4]};
                        v_acc0 = vmlaq_f32(v_acc0, w_low, vld1q_f32(ev + k));
                        v_acc1 = vmlaq_f32(v_acc1, w_high, vld1q_f32(od + k));
                    }
                    float g_sum = vaddvq_f32(vaddq_f32(v_acc0, v_acc1));
                    sum += g_sum * scale;
#else
                    float low[4]{}, high[4]{};
                    for (size_t k = 0; k < 64; ++k) {
                        low[k % 4] += book[p[k] & 15] * ev[k];
                        high[k % 4] += book[p[k] >> 4] * od[k];
                    }
                    float a[4];
                    for (size_t k=0;k<4;++k) a[k]=low[k]+high[k];
                    sum += ((a[0]+a[1])+(a[2]+a[3])) * scale;
#endif
                }
                outb[i] = sum;
            }
        }
    }
}

int main() {
    try { (void)cpu_kernels(); } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << "\n"; return 77;
    }
    std::mt19937 random(42);
    for (size_t threads : {1, 2, 4}) {
        auto workers = std::make_shared<ProjectionWorkers>(threads);
        const int original_rounding = std::fegetround();
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            std::fesetround(rounding);
            for (size_t rows : {0, 1, 3, 4, 9, 64, 65, 1403}) {
                std::vector<int> marks(rows);
                for (int repeat = 0; repeat < 20; ++repeat)
                    workers->run(rows, &marks, [](void *context, size_t begin, size_t end) {
                        auto &values = *static_cast<std::vector<int> *>(context);
                        for (size_t i = begin; i < end; ++i)
                            ++values[i];
                    });
                for (int mark : marks)
                    if (mark != 20)
                        return 1;
            }
            for (size_t cols : {1, 127, 128, 129, 384, 768, 3072, 32768}) {
                size_t total_rows = cols == 768 ? 1403 : 9;
                Tensor t;
                t.kind = 3;
                t.ndim = 2;
                t.bits = 4;
                t.group = 128;
                t.shape = {uint32_t(total_rows), uint32_t(cols), 0, 0};
                t.padded = uint32_t((cols + 127) / 128 * 128);
                size_t groups = t.padded / 128;
                std::vector<uint8_t> codes(total_rows * t.padded / 2);
                for (size_t i = 0; i < codes.size(); ++i)
                    codes[i] = uint8_t(i * 73 + 17);
                t.data = codes.data();
                for (size_t i = 0; i < 16; ++i)
                    t.book[i] = .0317f * (float(i) - 7.f);
                t.book[0] = -0.f;
                t.book[1] = 0.f;
                t.book[2] = 1e-38f;
                t.book[3] = -1e-38f;
                t.values.resize(total_rows * groups);
                for (size_t i = 0; i < t.values.size(); ++i)
                    t.values[i] = i % 7 ? .013f * float(i % 19) : 0.f;
                for (size_t batch : {1, 2, 3, 31, 32}) {
                    std::vector<float> x(batch * cols);
                    for (auto &v : x)
                        v = (int(random() % 2001) - 1000) * .007f;
                    for (size_t i = 0; i < x.size(); i += 13)
                        x[i] = i % 2 ? -0.f : 0.f;
                    for (size_t begin : {size_t(0), size_t(1)}) {
                        size_t rows = total_rows - begin;
                        std::vector<float> expected(batch * rows), actual(batch * rows + 2, 1234.f);
                        reference(t, x.data(), batch, expected.data(), begin, rows);
                        Math math(workers);
                        math.project(t, x.data(), batch, actual.data() + 1, begin, rows);
                        if (actual.front() != 1234.f || actual.back() != 1234.f ||
                            std::memcmp(expected.data(), actual.data() + 1,
                                        expected.size() * sizeof(float))) {
                            std::cerr << "Projection bits differ: cols=" << cols
                                      << " batch=" << batch << " begin=" << begin << '\n';
                            return 1;
                        }
                    }
                }
            }
        }
        std::fesetround(original_rounding);
    }
    std::cout << "Exact CQ4 projections passed\n";
}
