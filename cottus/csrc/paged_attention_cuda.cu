#include "paged_attention_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace cottus {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

__device__ inline float fp16_to_fp32_device(uint16_t h) {
    __half half_val = *reinterpret_cast<__half*>(&h);
    return __half2float(half_val);
}
__device__ inline float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ inline float blockReduceSum(float val)
{
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if(lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void pagedAttentionKernel(
    float* output,
    const float* query,
    const uint16_t* kvCacheBase,
    const int32_t* blockTable,
    int32_t seqLen,
    int32_t layerIdx,
    int32_t numHeads,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t blockSize,
    int32_t numLayers
)
{
    int32_t qHead = blockIdx.x;
    int32_t tid = threadIdx.x;
    if (qHead >= numHeads) return;
    int32_t kvHead = qHead / (numHeads / numKvHeads);
    float q_val = (tid < headDim) ? query[qHead * headDim + tid] : 0.0f;
    float m_i = -1e20f; 
    float l_i = 0.0f;   
    float acc_i = 0.0f; 
    float scale = 1.0f / sqrtf((float)headDim);
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * numLayers;
    int32_t layerOffset = layerIdx * 2 * elementsPerLayerKV;
    for (int32_t tokenPos = 0; tokenPos < seqLen; ++tokenPos)
    {
        int32_t logicalBlockIdx = tokenPos / blockSize;
        int32_t tokenInBlock = tokenPos % blockSize;
        int32_t physicalBlockId = blockTable[logicalBlockIdx];
        int32_t blockBase = physicalBlockId * elementsPerBlock;
        int32_t keyOffset = blockBase + layerOffset + 
                           tokenInBlock * (numKvHeads * headDim) + kvHead * headDim;

        float k_val = 0.0f;
        if (tid < headDim)
        {
             k_val = fp16_to_fp32_device(kvCacheBase[keyOffset + tid]);
        }
        float thread_prod = q_val * k_val;
        float qk = blockReduceSum(thread_prod);
        __shared__ float s_qk;
        if (tid == 0) s_qk = qk;
        __syncthreads();
        qk = s_qk;
        qk *= scale;
        float m_i_new = fmaxf(m_i, qk);
        float alpha = expf(m_i - m_i_new);
        float beta = expf(qk - m_i_new);
        l_i = l_i * alpha + beta;
        int32_t valOffset = blockBase + layerOffset + elementsPerLayerKV +
                            tokenInBlock * (numKvHeads * headDim) + kvHead * headDim;
        float v_val = 0.0f;
        if (tid < headDim) {
             v_val = fp16_to_fp32_device(kvCacheBase[valOffset + tid]);
        }
        acc_i = acc_i * alpha + beta * v_val;
        m_i = m_i_new;
    }
    if (tid < headDim) {
        output[qHead * headDim + tid] = acc_i / l_i;
    }
}

void pagedAttentionCUDA(
    float* output,
    const float* query,
    const void* kvCacheBase,
    const PageTable& pageTable,
    int32_t seqLen,
    int32_t layerIdx,
    int32_t numHeads,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t blockSize,
    int32_t numLayers
) {
    if (seqLen <= 0) throw std::invalid_argument("seqLen positive");
    
    // Copy block table
    int32_t* d_blockTable;
    size_t blockTableSize = pageTable.numBlocks() * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(&d_blockTable, blockTableSize));
    
    std::vector<int32_t> blockTableHost(pageTable.numBlocks());
    for (int i = 0; i < pageTable.numBlocks(); ++i) {
        blockTableHost[i] = pageTable[i];
    }
    CUDA_CHECK(cudaMemcpy(d_blockTable, blockTableHost.data(), blockTableSize, cudaMemcpyHostToDevice));
    dim3 grid(numHeads, 1, 1);
    dim3 block(128, 1, 1);
    
    pagedAttentionKernel<<<grid, block>>>(
        output, query, static_cast<const uint16_t*>(kvCacheBase), d_blockTable,
        seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, numLayers
    );
    
    CUDA_CHECK(cudaFree(d_blockTable));
}

__global__ void quantizeAndCacheKernel(
    uint16_t* cache,
    const float* k,
    const float* v,
    int32_t totalElements,
    int32_t kOffset,
    int32_t vOffset
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        float k_val = k[idx];
        __half k_half = __float2half(k_val);
        cache[kOffset + idx] = *reinterpret_cast<uint16_t*>(&k_half);
        
        float v_val = v[idx];
        __half v_half = __float2half(v_val);
        cache[vOffset + idx] = *reinterpret_cast<uint16_t*>(&v_half);
    }
}

void quantizeAndCacheCUDA(
    void* d_cache,
    const float* d_k,
    const float* d_v,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t kStartOffset,
    int32_t vStartOffset
) {
    int32_t totalElements = numKvHeads * headDim;
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    
    quantizeAndCacheKernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<uint16_t*>(d_cache),
        d_k,
        d_v,
        totalElements,
        kStartOffset,
        vStartOffset
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cottus
