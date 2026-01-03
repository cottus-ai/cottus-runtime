#include "compute_primitives_cpu.h"
#include <cmath>
#include <algorithm>

namespace cottus {

// Naive GEMM: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
void gemmCPU(
    float* C,
    const float* A,
    const float* B,
    int32_t M,
    int32_t N,
    int32_t K
) {
    // Simple triple loop (not optimized)
    for (int32_t m = 0; m < M; ++m) {
        for (int32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// RMSNorm (LLaMA-style)
void rmsnormCPU(
    float* output,
    const float* input,
    const float* weight,
    int32_t N,
    float epsilon
) {
    // Compute RMS
    float sumSquares = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        sumSquares += input[i] * input[i];
    }
    float rms = std::sqrt(sumSquares / N + epsilon);
    
    // Normalize and scale
    for (int32_t i = 0; i < N; ++i) {
        output[i] = (input[i] / rms) * weight[i];
    }
}

// RoPE (Rotary Position Embeddings)
// Applies rotation to pairs of dimensions at a specific position
void ropeCPU(
    float* output,
    const float* input,
    int32_t pos,
    int32_t numHeads,
    int32_t headDim,
    float theta
) {
    // RoPE rotates pairs of dimensions
    // For each head and each pair (2i, 2i+1):
    // q'[2i] = q[2i] * cos(pos*theta_i) - q[2i+1] * sin(pos*theta_i)
    // q'[2i+1] = q[2i] * sin(pos*theta_i) + q[2i+1] * cos(pos*theta_i)
    
    for (int32_t head = 0; head < numHeads; ++head) {
        for (int32_t d = 0; d < headDim / 2; ++d) {
            // Compute frequency for this dimension pair
            float freq = 1.0f / std::pow(theta, (2.0f * d) / headDim);
            float angle = pos * freq;
            float cosVal = std::cos(angle);
            float sinVal = std::sin(angle);
            
            // Get input indices
            int32_t baseIdx = head * headDim;
            int32_t idx0 = baseIdx + 2 * d;
            int32_t idx1 = baseIdx + 2 * d + 1;
            
            // Apply rotation
            float x0 = input[idx0];
            float x1 = input[idx1];
            output[idx0] = x0 * cosVal - x1 * sinVal;
            output[idx1] = x0 * sinVal + x1 * cosVal;
        }
    }
}

// Residual add
void residualAddCPU(
    float* output,
    const float* input1,
    const float* input2,
    int32_t N
) {
    for (int32_t i = 0; i < N; ++i) {
        output[i] = input1[i] + input2[i];
    }
}

// SiLU activation
void siluCPU(
    float* output,
    const float* input,
    int32_t N
) {
    for (int32_t i = 0; i < N; ++i) {
        float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        output[i] = input[i] * sigmoid;
    }
}

// Element-wise multiply
void elementwiseMultiplyCPU(
    float* output,
    const float* input1,
    const float* input2,
    int32_t N
) {
    for (int32_t i = 0; i < N; ++i) {
        output[i] = input1[i] * input2[i];
    }
}

} // namespace cottus
