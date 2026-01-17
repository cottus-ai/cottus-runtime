#pragma once

#include <cstdint>

namespace cottus {

void gemmFP16CUDA(
    void* d_C,
    const void* d_A,
    const void* d_B,
    int32_t M,
    int32_t N,
    int32_t K
);

void rmsnormFP16CUDA(
    void* d_output,
    const void* d_input,
    const void* d_weight,
    int32_t N,
    float epsilon
);

void ropeFP16CUDA(
    void* d_output,
    const void* d_input,
    int32_t pos,
    int32_t numHeads,
    int32_t headDim,
    float theta
);

void fusedAddRMSNormFP16CUDA(
    void* d_output,
    void* d_residual_out,
    const void* d_input,
    const void* d_residual,
    const void* d_weight,
    int32_t N,
    float epsilon
);

void siluFP16CUDA(
    void* d_output,
    const void* d_input,
    int32_t N
);

void fusedSiLUMulFP16CUDA(
    void* d_output,
    const void* d_gate,
    const void* d_up,
    int32_t N
);

void elementwiseMultiplyFP16CUDA(
    void* d_output,
    const void* d_input1,
    const void* d_input2,
    int32_t N
);

void residualAddFP16CUDA(
    void* d_output,
    const void* d_input1,
    const void* d_input2,
    int32_t N
);

}
