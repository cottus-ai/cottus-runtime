#include "fp16_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
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

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

static cublasHandle_t g_cublasHandleFP16 = nullptr;

static void ensureCublasHandleFP16() {
    if (g_cublasHandleFP16 == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublasHandleFP16));
        CUBLAS_CHECK(cublasSetMathMode(g_cublasHandleFP16, CUBLAS_TENSOR_OP_MATH));
    }
}

void gemmFP16CUDA(
    void* d_C,
    const void* d_A,
    const void* d_B,
    int32_t M,
    int32_t N,
    int32_t K
) {
    ensureCublasHandleFP16();

    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);

    CUBLAS_CHECK(cublasHgemm(
        g_cublasHandleFP16,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha_h,
        static_cast<const __half*>(d_B), N,
        static_cast<const __half*>(d_A), K,
        &beta_h,
        static_cast<__half*>(d_C), N
    ));
}

__device__ inline float warpReduceSumFP16(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void rmsnormFP16Kernel(
    __half* output,
    const __half* input,
    const __half* weight,
    int32_t N,
    float epsilon
) {
    extern __shared__ float sdata[];
    
    int32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float myVal = 0.0f;
    float mySumSq = 0.0f;
    
    if (i < N) {
        myVal = __half2float(input[i]);
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
        float rms = sqrtf(totalSumSq / N + epsilon);
        float w = __half2float(weight[i]);
        float result = (myVal / rms) * w;
        output[i] = __float2half(result);
    }
}

void rmsnormFP16CUDA(
    void* d_output,
    const void* d_input,
    const void* d_weight,
    int32_t N,
    float epsilon
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    
    rmsnormFP16Kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        static_cast<__half*>(d_output),
        static_cast<const __half*>(d_input),
        static_cast<const __half*>(d_weight),
        N,
        epsilon
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void ropeFP16Kernel(
    __half* output,
    const __half* input,
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
    int32_t idx0 = baseIdx + d;
    int32_t idx1 = baseIdx + d + headDim / 2;
    
    float x0 = __half2float(input[idx0]);
    float x1 = __half2float(input[idx1]);
    
    output[idx0] = __float2half(x0 * cosVal - x1 * sinVal);
    output[idx1] = __float2half(x0 * sinVal + x1 * cosVal);
}

void ropeFP16CUDA(
    void* d_output,
    const void* d_input,
    int32_t pos,
    int32_t numHeads,
    int32_t headDim,
    float theta
) {
    int32_t totalPairs = numHeads * (headDim / 2);
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;
    
    ropeFP16Kernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<__half*>(d_output),
        static_cast<const __half*>(d_input),
        pos,
        numHeads,
        headDim,
        theta
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fusedAddRMSNormFP16Kernel(
    __half* output,
    __half* residual_out,
    const __half* input,
    const __half* residual,
    const __half* weight,
    int32_t N,
    float epsilon
) {
    extern __shared__ float sdata[];
    
    int32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float myVal = 0.0f;
    float mySumSq = 0.0f;
    
    if (i < N) {
        myVal = __half2float(input[i]) + __half2float(residual[i]);
        residual_out[i] = __float2half(myVal);
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
        float rms = sqrtf(totalSumSq / N + epsilon);
        float w = __half2float(weight[i]);
        output[i] = __float2half((myVal / rms) * w);
    }
}

void fusedAddRMSNormFP16CUDA(
    void* d_output,
    void* d_residual_out,
    const void* d_input,
    const void* d_residual,
    const void* d_weight,
    int32_t N,
    float epsilon
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    
    fusedAddRMSNormFP16Kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        static_cast<__half*>(d_output),
        static_cast<__half*>(d_residual_out),
        static_cast<const __half*>(d_input),
        static_cast<const __half*>(d_residual),
        static_cast<const __half*>(d_weight),
        N,
        epsilon
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void siluFP16Kernel(
    __half* output,
    const __half* input,
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = __half2float(input[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2half(x * sigmoid);
    }
}

void siluFP16CUDA(
    void* d_output,
    const void* d_input,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    siluFP16Kernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<__half*>(d_output),
        static_cast<const __half*>(d_input),
        N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fusedSiLUMulFP16Kernel(
    __half* output,
    const __half* gate,
    const __half* up,
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-g));
        float silu = g * sigmoid;
        output[idx] = __float2half(silu * u);
    }
}

void fusedSiLUMulFP16CUDA(
    void* d_output,
    const void* d_gate,
    const void* d_up,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    fusedSiLUMulFP16Kernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<__half*>(d_output),
        static_cast<const __half*>(d_gate),
        static_cast<const __half*>(d_up),
        N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void elementwiseMultiplyFP16Kernel(
    __half* output,
    const __half* input1,
    const __half* input2,
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = __half2float(input1[idx]);
        float b = __half2float(input2[idx]);
        output[idx] = __float2half(a * b);
    }
}

void elementwiseMultiplyFP16CUDA(
    void* d_output,
    const void* d_input1,
    const void* d_input2,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    elementwiseMultiplyFP16Kernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<__half*>(d_output),
        static_cast<const __half*>(d_input1),
        static_cast<const __half*>(d_input2),
        N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void residualAddFP16Kernel(
    __half* output,
    const __half* input1,
    const __half* input2,
    int32_t N
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = __half2float(input1[idx]);
        float b = __half2float(input2[idx]);
        output[idx] = __float2half(a + b);
    }
}

void residualAddFP16CUDA(
    void* d_output,
    const void* d_input1,
    const void* d_input2,
    int32_t N
) {
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    residualAddFP16Kernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<__half*>(d_output),
        static_cast<const __half*>(d_input1),
        static_cast<const __half*>(d_input2),
        N
    );
    
    CUDA_CHECK(cudaGetLastError());
}

}
