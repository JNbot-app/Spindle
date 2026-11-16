#include "neon_ops.h"
namespace spindle {
const CpuKernels &neon_kernels() {
    // math.cpp retains the register-resident NEON FWHT and quantization kernels.
    static const CpuKernels k{"neon",kernel_project<NeonOps>,neon_integer<NeonOps>,neon_decode,nullptr,nullptr};
    return k;
}
} // namespace spindle
