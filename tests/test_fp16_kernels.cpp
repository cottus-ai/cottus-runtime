#include "../cottus/csrc/fp16_kernels.h"
#include "../cottus/csrc/paged_attention_fp16.h"
#include "../cottus/csrc/compute_primitives_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace cottus;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

void testGemmFP16() {
    std::cout << "Test: GEMM FP16" << std::endl;
    
    const int M = 32, N = 64, K = 48;
    
    std::vector<float> h_A(M * K), h_B(K * N), h_C_ref(M * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) h_A[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) h_B[i] = dist(rng);
    
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[m * K + k] * h_B[k * N + n];
            }
            h_C_ref[m * N + n] = sum;
        }
    }
    
    std::vector<__half> h_A_fp16(M * K), h_B_fp16(K * N);
    for (int i = 0; i < M * K; ++i) h_A_fp16[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_fp16[i] = __float2half(h_B[i]);
    
    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A_fp16.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_fp16.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));
    
    gemmFP16CUDA(d_C, d_A, d_B, M, N, K);
    
    std::vector<__half> h_C_fp16(M * N);
    CUDA_CHECK(cudaMemcpy(h_C_fp16.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
    
    float maxErr = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(__half2float(h_C_fp16[i]) - h_C_ref[i]);
        maxErr = std::max(maxErr, diff);
    }
    
    std::cout << "  Max error: " << maxErr << std::endl;
    assert(maxErr < 1.0f);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "  PASS" << std::endl;
}

void testRMSNormFP16() {
    std::cout << "Test: RMSNorm FP16" << std::endl;
    
    const int N = 256;
    const float epsilon = 1e-5f;
    
    std::vector<float> h_input(N), h_weight(N), h_output_ref(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(rng);
        h_weight[i] = dist(rng) * 0.1f + 1.0f;
    }
    
    float sumSq = 0.0f;
    for (int i = 0; i < N; ++i) sumSq += h_input[i] * h_input[i];
    float rms = std::sqrt(sumSq / N + epsilon);
    for (int i = 0; i < N; ++i) {
        h_output_ref[i] = (h_input[i] / rms) * h_weight[i];
    }
    
    std::vector<__half> h_input_fp16(N), h_weight_fp16(N);
    for (int i = 0; i < N; ++i) {
        h_input_fp16[i] = __float2half(h_input[i]);
        h_weight_fp16[i] = __float2half(h_weight[i]);
    }
    
    __half *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_weight, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(__half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_fp16.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_fp16.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    
    rmsnormFP16CUDA(d_output, d_input, d_weight, N, epsilon);
    
    std::vector<__half> h_output_fp16(N);
    CUDA_CHECK(cudaMemcpy(h_output_fp16.data(), d_output, N * sizeof(__half), cudaMemcpyDeviceToHost));
    
    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(__half2float(h_output_fp16[i]) - h_output_ref[i]);
        maxErr = std::max(maxErr, diff);
    }
    
    std::cout << "  Max error: " << maxErr << std::endl;
    assert(maxErr < 0.1f);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "  PASS" << std::endl;
}

void testRoPEFP16() {
    std::cout << "Test: RoPE FP16" << std::endl;
    
    const int numHeads = 4;
    const int headDim = 32;
    const int pos = 5;
    const float theta = 10000.0f;
    const int N = numHeads * headDim;
    
    std::vector<float> h_input(N), h_output_ref(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < N; ++i) h_input[i] = dist(rng);
    
    for (int head = 0; head < numHeads; ++head) {
        for (int d = 0; d < headDim / 2; ++d) {
            float freq = 1.0f / std::pow(theta, (2.0f * d) / headDim);
            float angle = pos * freq;
            float cosVal = std::cos(angle);
            float sinVal = std::sin(angle);
            
            int idx0 = head * headDim + d;
            int idx1 = head * headDim + d + headDim / 2;
            
            float x0 = h_input[idx0];
            float x1 = h_input[idx1];
            h_output_ref[idx0] = x0 * cosVal - x1 * sinVal;
            h_output_ref[idx1] = x0 * sinVal + x1 * cosVal;
        }
    }
    
    std::vector<__half> h_input_fp16(N);
    for (int i = 0; i < N; ++i) h_input_fp16[i] = __float2half(h_input[i]);
    
    __half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(__half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_fp16.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    
    ropeFP16CUDA(d_output, d_input, pos, numHeads, headDim, theta);
    
    std::vector<__half> h_output_fp16(N);
    CUDA_CHECK(cudaMemcpy(h_output_fp16.data(), d_output, N * sizeof(__half), cudaMemcpyDeviceToHost));
    
    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(__half2float(h_output_fp16[i]) - h_output_ref[i]);
        maxErr = std::max(maxErr, diff);
    }
    
    std::cout << "  Max error: " << maxErr << std::endl;
    assert(maxErr < 0.01f);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "  PASS" << std::endl;
}

void testFusedSiLUMulFP16() {
    std::cout << "Test: Fused SiLU Mul FP16" << std::endl;
    
    const int N = 512;
    
    std::vector<float> h_gate(N), h_up(N), h_output_ref(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    
    for (int i = 0; i < N; ++i) {
        h_gate[i] = dist(rng);
        h_up[i] = dist(rng);
        float sigmoid = 1.0f / (1.0f + std::exp(-h_gate[i]));
        float silu = h_gate[i] * sigmoid;
        h_output_ref[i] = silu * h_up[i];
    }
    
    std::vector<__half> h_gate_fp16(N), h_up_fp16(N);
    for (int i = 0; i < N; ++i) {
        h_gate_fp16[i] = __float2half(h_gate[i]);
        h_up_fp16[i] = __float2half(h_up[i]);
    }
    
    __half *d_gate, *d_up, *d_output;
    CUDA_CHECK(cudaMalloc(&d_gate, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_up, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(__half)));
    
    CUDA_CHECK(cudaMemcpy(d_gate, h_gate_fp16.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up_fp16.data(), N * sizeof(__half), cudaMemcpyHostToDevice));
    
    fusedSiLUMulFP16CUDA(d_output, d_gate, d_up, N);
    
    std::vector<__half> h_output_fp16(N);
    CUDA_CHECK(cudaMemcpy(h_output_fp16.data(), d_output, N * sizeof(__half), cudaMemcpyDeviceToHost));
    
    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(__half2float(h_output_fp16[i]) - h_output_ref[i]);
        maxErr = std::max(maxErr, diff);
    }
    
    std::cout << "  Max error: " << maxErr << std::endl;
    assert(maxErr < 0.1f);
    
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "  PASS" << std::endl;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA device detected, skipping FP16 kernel tests" << std::endl;
        return 0;
    }
    
    testGemmFP16();
    testRMSNormFP16();
    testRoPEFP16();
    testFusedSiLUMulFP16();
    std::cout << "All FP16 kernel tests passed!" << std::endl;
    return 0;
}
