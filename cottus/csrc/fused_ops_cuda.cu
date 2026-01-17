#include "fused_ops_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace cottus {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

__global__ void fusedAddRMSNormKernel(
    float* output,
    float* residual_out,
    const float* input,
    const float* residual,
    const float* weight,
    int32_t N,
    float epsilon,
    float invN
) {
    extern __shared__ float sdata[];
    
    int32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float myVal = 0.0f;
    float mySumSq = 0.0f;
    
    if (i < N) {
        myVal = input[i] + residual[i];
        residual_out[i] = myVal;
        mySumSq = myVal * myVal;
    }
    
    sdata[tid] = mySumSq;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    __shared__ float totalSumSq;
    if (tid == 0) {
        totalSumSq = sdata[0];
    }
    __syncthreads();
    
    if (i < N) {
        float rms = sqrtf(totalSumSq * invN + epsilon);
        output[i] = (myVal / rms) * weight[i];
    }
}

__global__ void fusedAddRMSNormKernelLarge(
    float* output,
    float* residual_out,
    const float* input,
    const float* residual,
    const float* weight,
    int32_t N,
    float epsilon
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float myVal = 0.0f;
    if (idx < N) {
        myVal = input[idx] + residual[idx];
        residual_out[idx] = myVal;
    }
    
    __shared__ float partialSum[256];
    float localSum = 0.0f;
    
    for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
        float v = input[i] + residual[i];
        localSum += v * v;
    }
    
    partialSum[threadIdx.x] = localSum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    __shared__ float totalSumSq;
    if (threadIdx.x == 0) {
        totalSumSq = partialSum[0];
    }
    __syncthreads();
    
    if (idx < N) {
        float rms = sqrtf(totalSumSq / N + epsilon);
        output[idx] = (myVal / rms) * weight[idx];
    }
}

void fusedAddRMSNormCUDA(
    float* d_output,
    float* d_residual_out,
    const float* d_input,
    const float* d_residual,
    const float* d_weight,
    int32_t N,
    float epsilon
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    if (numBlocks == 1) {
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        float invN = 1.0f / N;
        fusedAddRMSNormKernel<<<1, threadsPerBlock, sharedMemSize>>>(
            d_output, d_residual_out, d_input, d_residual, d_weight, N, epsilon, invN
        );
    } else {
        fusedAddRMSNormKernelLarge<<<numBlocks, threadsPerBlock>>>(
            d_output, d_residual_out, d_input, d_residual, d_weight, N, epsilon
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// Fused SiLU + Multiply (SwiGLU)
// output = (input1 * sigmoid(input1)) * input2
// Operates in-place if output == input1
__global__ void fusedSiLUMulKernel(
    float* output,
    const float* input1, // Gate
    const float* input2, // Up
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input1[idx];
        float y = input2[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float silu = x * sigmoid;
        output[idx] = silu * y;
    }
}

void fusedSiLUMulCUDA(
    float* d_output,
    const float* d_input1,
    const float* d_input2,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    fusedSiLUMulKernel<<<numBlocks, threadsPerBlock>>>(
        d_output, d_input1, d_input2, N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cottus
