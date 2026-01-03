#include "../cottus/csrc/compute_primitives_cpu.h"
#include "../cottus/csrc/compute_primitives_cuda.h"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cottus;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Test 1: GEMM parity
void testGEMMParity() {
    std::cout << "Parity Test: GEMM" << std::endl;
    
    int32_t M = 4, N = 3, K = 5;
    
    // Create test matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_cpu(M * N);
    std::vector<float> C_cuda(M * N);
    
    // Initialize with simple values
    for (int i = 0; i < M * K; ++i) A[i] = (i % 7) * 0.1f;
    for (int i = 0; i < K * N; ++i) B[i] = (i % 5) * 0.2f;
    
    // CPU
    gemmCPU(C_cpu.data(), A.data(), B.data(), M, N, K);
    
    // CUDA
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    gemmCUDA(d_C, d_A, d_B, M, N, K);
    
    CUDA_CHECK(cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare
    float maxDiff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(C_cpu[i] - C_cuda[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  Max absolute difference: " << maxDiff << std::endl;
    assert(maxDiff < 1e-4f);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "  PASS" << std::endl;
}

// Test 2: RMSNorm parity
void testRMSNormParity() {
    std::cout << "Parity Test: RMSNorm" << std::endl;
    
    int32_t N = 128;
    
    std::vector<float> input(N);
    std::vector<float> weight(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_cuda(N);
    
    for (int i = 0; i < N; ++i) {
        input[i] = (i % 10) * 0.1f;
        weight[i] = 1.0f + (i % 5) * 0.05f;
    }
    
    // CPU
    rmsnormCPU(output_cpu.data(), input.data(), weight.data(), N);
    
    // CUDA
    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    rmsnormCUDA(d_output, d_input, d_weight, N);
    
    CUDA_CHECK(cudaMemcpy(output_cuda.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxDiff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(output_cpu[i] - output_cuda[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  Max absolute difference: " << maxDiff << std::endl;
    assert(maxDiff < 1e-4f);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "  PASS" << std::endl;
}

// Test 3: RoPE parity
void testRoPEParity() {
    std::cout << "Parity Test: RoPE" << std::endl;
    
    int32_t numHeads = 2;
    int32_t headDim = 8;
    int32_t N = numHeads * headDim;
    int32_t pos = 5;  // Test at position 5
    
    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_cuda(N);
    
    for (int i = 0; i < N; ++i) {
        input[i] = (i % 10) * 0.1f;
    }
    
    // CPU
    ropeCPU(output_cpu.data(), input.data(), pos, numHeads, headDim);
    
    // CUDA
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    ropeCUDA(d_output, d_input, pos, numHeads, headDim);
    
    CUDA_CHECK(cudaMemcpy(output_cuda.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxDiff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(output_cpu[i] - output_cuda[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  Max absolute difference: " << maxDiff << std::endl;
    assert(maxDiff < 1e-4f);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "  PASS" << std::endl;
}

// Test 4: Elementwise ops parity
void testElementwiseParity() {
    std::cout << "Parity Test: Elementwise Ops" << std::endl;
    
    int32_t N = 256;
    
    std::vector<float> input1(N);
    std::vector<float> input2(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_cuda(N);
    
    for (int i = 0; i < N; ++i) {
        input1[i] = (i % 10) * 0.1f;
        input2[i] = (i % 7) * 0.15f;
    }
    
    // Test residual add
    residualAddCPU(output_cpu.data(), input1.data(), input2.data(), N);
    
    float *d_input1, *d_input2, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input1, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input2, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input1, input1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input2, input2.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    residualAddCUDA(d_output, d_input1, d_input2, N);
    
    CUDA_CHECK(cudaMemcpy(output_cuda.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxDiff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(output_cpu[i] - output_cuda[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  Residual Add - Max diff: " << maxDiff << std::endl;
    assert(maxDiff < 1e-6f);
    
    // Test SiLU
    siluCPU(output_cpu.data(), input1.data(), N);
    siluCUDA(d_output, d_input1, N);
    CUDA_CHECK(cudaMemcpy(output_cuda.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    maxDiff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(output_cpu[i] - output_cuda[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  SiLU - Max diff: " << maxDiff << std::endl;
    assert(maxDiff < 1e-4f);
    
    CUDA_CHECK(cudaFree(d_input1));
    CUDA_CHECK(cudaFree(d_input2));
    CUDA_CHECK(cudaFree(d_output));
    
    std::cout << "  PASS" << std::endl;
}

int main() {
    testGEMMParity();
    testRMSNormParity();
    testRoPEParity();
    testElementwiseParity();
    std::cout << "All compute primitives parity tests passed!" << std::endl;
    return 0;
}
