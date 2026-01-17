#ifndef COTTUS_FUSED_OPS_CUDA_H
#define COTTUS_FUSED_OPS_CUDA_H

#include <cstdint>

namespace cottus {

void fusedAddRMSNormCUDA(
    float* output,
    float* residual_out,
    const float* input,
    const float* residual,
    const float* weight,
    int32_t N,
    float epsilon
);

void fusedSiLUMulCUDA(
    float* output,
    const float* input1,
    const float* input2,
    int32_t N
);

} // namespace cottus

#endif // COTTUS_FUSED_OPS_CUDA_H
