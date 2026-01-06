#pragma once

#include <cstdint>

namespace cottus {
void gemmCUDA(
    float* d_C,
    const float* d_A,
    const float* d_B,
    int32_t M,
    int32_t N,
    int32_t K
);
void rmsnormCUDA(
    float* d_output,
    const float* d_input,
    const float* d_weight,
    int32_t N,
    float epsilon = 1e-5f
);
//rope kernel
void ropeCUDA(
    float* d_output,
    const float* d_input,
    int32_t pos,
    int32_t numHeads,
    int32_t headDim,
    float theta = 10000.0f
);

//residual add kernel
void residualAddCUDA(
    float* d_output,
    const float* d_input1,
    const float* d_input2,
    int32_t N
);
void siluCUDA(
    float* d_output,
    const float* d_input,
    int32_t N
);
void elementwiseMultiplyCUDA(
    float* d_output,
    const float* d_input1,
    const float* d_input2,
    int32_t N
);

} // namespace cottus
