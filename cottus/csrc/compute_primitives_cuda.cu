#include "compute_primitives_cuda.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

namespace cottus {
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)
static cublasHandle_t g_cublasHandle = nullptr;

static void ensureCublasHandle() {
    if (g_cublasHandle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublasHandle));
    }
}
void gemmCUDA(
    float* d_C,
    const float* d_A,
    const float* d_B,
    int32_t M,
    int32_t N,
    int32_t K
) {
    ensureCublasHandle();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        g_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    ));
}

//RMSNorm kernel
__global__ void rmsnormKernel(
    float* output,
    const float* input,
    const float* weight,
    int32_t N,
    float epsilon
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float sumSquares = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i];
        sumSquares += val * val;
    }
    float rms = sqrtf(sumSquares / N + epsilon);
    output[idx] = (input[idx] / rms) * weight[idx];
}
void rmsnormCUDA(
    float* d_output,
    const float* d_input,
    const float* d_weight,
    int32_t N,
    float epsilon
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    rmsnormKernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_input, d_weight, N, epsilon
    );
    
    CUDA_CHECK(cudaGetLastError());
}
__global__ void ropeKernel(
    float* output,
    const float* input,
    int32_t pos,
    int32_t numHeads,
    int32_t headDim,
    float theta
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t totalPairs = numHeads * (headDim / 2);
    if (idx >= totalPairs) return;
    int32_t d = idx % (headDim / 2);
    int32_t head = idx / (headDim / 2);
    float freq = 1.0f / powf(theta, (2.0f * d) / headDim);
    float angle = pos * freq;
    float cosVal = cosf(angle);
    float sinVal = sinf(angle);
    int32_t baseIdx = head * headDim;
    int32_t idx0 = baseIdx + 2 * d;
    int32_t idx1 = baseIdx + 2 * d + 1;
    
    //apply rotation
    float x0 = input[idx0];
    float x1 = input[idx1];
    output[idx0] = x0 * cosVal - x1 * sinVal;
    output[idx1] = x0 * sinVal + x1 * cosVal;
}

void ropeCUDA(
    float* d_output,
    const float* d_input,
    int32_t pos,
    int32_t numHeads,
    int32_t headDim,
    float theta
) {
    int32_t totalPairs = numHeads * (headDim / 2);
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;
    
    ropeKernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_input, pos, numHeads, headDim, theta
    );
    
    CUDA_CHECK(cudaGetLastError());
}

//residual add kernel
__global__ void residualAddKernel(
    float* output,
    const float* input1,
    const float* input2,
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = input1[idx] + input2[idx];
    }
}

void residualAddCUDA(
    float* d_output,
    const float* d_input1,
    const float* d_input2,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    residualAddKernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_input1, d_input2, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

//silu kernel
__global__ void siluKernel(
    float* output,
    const float* input,
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

void siluCUDA(
    float* d_output,
    const float* d_input,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    siluKernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_input, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}
__global__ void elementwiseMultiplyKernel(
    float* output,
    const float* input1,
    const float* input2,
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = input1[idx] * input2[idx];
    }
}

void elementwiseMultiplyCUDA(
    float* d_output,
    const float* d_input1,
    const float* d_input2,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    elementwiseMultiplyKernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_input1, d_input2, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cottus
