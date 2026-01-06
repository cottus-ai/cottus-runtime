#pragma once

#include <vector>
#include <cstdint>

namespace cottus {
void gemmCPU(
    float* C,
    const float* A,
    const float* B,
    int32_t M,
    int32_t N,
    int32_t K
);

void rmsnormCPU(
    float* output,
    const float* input,
    const float* weight,
    int32_t N,
    float epsilon = 1e-5f
);
void ropeCPU(
    float* output,
    const float* input,
    int32_t pos,
    int32_t numHeads,
    int32_t headDim,
    float theta = 10000.0f
);
void residualAddCPU(
    float* output,
    const float* input1,
    const float* input2,
    int32_t N
);

//silu activation
void siluCPU(
    float* output,
    const float* input,
    int32_t N
);
void elementwiseMultiplyCPU(
    float* output,
    const float* input1,
    const float* input2,
    int32_t N
);

} // namespace cottus
