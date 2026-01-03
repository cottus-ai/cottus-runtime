#pragma once

#include <vector>
#include <cstdint>

namespace cottus {

// CPU reference implementations of compute primitives
// These are simple, correct, unoptimized versions for validation

// Matrix multiplication: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
// Row-major layout
void gemmCPU(
    float* C,           // Output [M, N]
    const float* A,     // Input [M, K]
    const float* B,     // Input [K, N]
    int32_t M,
    int32_t N,
    int32_t K
);

// RMSNorm (LLaMA-style)
// output = input * weight / rms(input)
// where rms(input) = sqrt(mean(input^2) + epsilon)
void rmsnormCPU(
    float* output,      // Output [N]
    const float* input, // Input [N]
    const float* weight,// Weight [N]
    int32_t N,
    float epsilon = 1e-5f
);

// Rotary Position Embeddings (RoPE)
// Applies rotary embeddings to query or key tensor at a specific position
// input/output: [numHeads, headDim]
// pos: position index (0-based)
void ropeCPU(
    float* output,      // Output [numHeads, headDim]
    const float* input, // Input [numHeads, headDim]
    int32_t pos,        // Position index
    int32_t numHeads,
    int32_t headDim,
    float theta = 10000.0f
);

// Residual add: output = input1 + input2
void residualAddCPU(
    float* output,      // Output [N]
    const float* input1,// Input1 [N]
    const float* input2,// Input2 [N]
    int32_t N
);

// SiLU activation: output = input * sigmoid(input)
void siluCPU(
    float* output,      // Output [N]
    const float* input, // Input [N]
    int32_t N
);

// Element-wise multiply: output = input1 * input2
void elementwiseMultiplyCPU(
    float* output,      // Output [N]
    const float* input1,// Input1 [N]
    const float* input2,// Input2 [N]
    int32_t N
);

} // namespace cottus
