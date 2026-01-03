#pragma once

#include <cstdint>

namespace cottus {

// CUDA implementations of compute primitives
// Uses cuBLAS for GEMM, custom kernels for others

// Matrix multiplication using cuBLAS: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
// All pointers are device pointers
void gemmCUDA(
    float* d_C,         // Device output [M, N]
    const float* d_A,   // Device input [M, K]
    const float* d_B,   // Device input [K, N]
    int32_t M,
    int32_t N,
    int32_t K
);

// RMSNorm CUDA kernel
void rmsnormCUDA(
    float* d_output,    // Device output [N]
    const float* d_input,// Device input [N]
    const float* d_weight,// Device weight [N]
    int32_t N,
    float epsilon = 1e-5f
);

// RoPE CUDA kernel
// Applies rotary embeddings at a specific position
void ropeCUDA(
    float* d_output,    // Device output [numHeads, headDim]
    const float* d_input,// Device input [numHeads, headDim]
    int32_t pos,        // Position index
    int32_t numHeads,
    int32_t headDim,
    float theta = 10000.0f
);

// Residual add CUDA kernel
void residualAddCUDA(
    float* d_output,    // Device output [N]
    const float* d_input1,// Device input1 [N]
    const float* d_input2,// Device input2 [N]
    int32_t N
);

// SiLU activation CUDA kernel
void siluCUDA(
    float* d_output,    // Device output [N]
    const float* d_input,// Device input [N]
    int32_t N
);

// Element-wise multiply CUDA kernel
void elementwiseMultiplyCUDA(
    float* d_output,    // Device output [N]
    const float* d_input1,// Device input1 [N]
    const float* d_input2,// Device input2 [N]
    int32_t N
);

} // namespace cottus
