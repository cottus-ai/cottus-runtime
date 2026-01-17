#include "paged_attention_fp16.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cottus {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

__device__ inline float warpReduceSumPA(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ inline float warpReduceMaxPA(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ inline float blockReduceSumPA(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSumPA(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSumPA(val);
    return val;
}

__global__ void pagedAttentionFP16Kernel(
    __half* output,
    const __half* query,
    const __half* kvCacheBase,
    const int32_t* blockTable,
    int32_t seqLen,
    int32_t layerIdx,
    int32_t numHeads,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t blockSize,
    int32_t numLayers
) {
    int32_t qHead = blockIdx.x;
    int32_t tid = threadIdx.x;
    if (qHead >= numHeads) return;
    
    int32_t kvHead = qHead / (numHeads / numKvHeads);
    
    float q_val = 0.0f;
    if (tid < headDim) {
        q_val = __half2float(query[qHead * headDim + tid]);
    }
    
    float scale = 1.0f / sqrtf((float)headDim);
    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_i = 0.0f;
    
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * numLayers;
    int32_t layerOffset = layerIdx * 2 * elementsPerLayerKV;
    
    for (int32_t tokenPos = 0; tokenPos < seqLen; ++tokenPos) {
        int32_t logicalBlockIdx = tokenPos / blockSize;
        int32_t tokenInBlock = tokenPos % blockSize;
        int32_t physicalBlockId = blockTable[logicalBlockIdx];
        int32_t blockBase = physicalBlockId * elementsPerBlock;
        
        int32_t keyOffset = blockBase + layerOffset + 
                           tokenInBlock * (numKvHeads * headDim) + kvHead * headDim;
        
        float k_val = 0.0f;
        if (tid < headDim) {
            k_val = __half2float(kvCacheBase[keyOffset + tid]);
        }
        
        float thread_prod = q_val * k_val;
        float qk = blockReduceSumPA(thread_prod);
        
        __shared__ float s_qk;
        if (tid == 0) s_qk = qk;
        __syncthreads();
        qk = s_qk * scale;
        
        float m_i_new = fmaxf(m_i, qk);
        float alpha = expf(m_i - m_i_new);
        float beta = expf(qk - m_i_new);
        l_i = l_i * alpha + beta;
        
        int32_t valOffset = blockBase + layerOffset + elementsPerLayerKV +
                           tokenInBlock * (numKvHeads * headDim) + kvHead * headDim;
        
        float v_val = 0.0f;
        if (tid < headDim) {
            v_val = __half2float(kvCacheBase[valOffset + tid]);
        }
        
        acc_i = acc_i * alpha + beta * v_val;
        m_i = m_i_new;
    }
    
    if (tid < headDim) {
        output[qHead * headDim + tid] = __float2half(acc_i / l_i);
    }
}

void pagedAttentionFP16CUDA(
    void* output,
    const void* query,
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
    if (seqLen <= 0) return;
    
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
    
    pagedAttentionFP16Kernel<<<grid, block>>>(
        static_cast<__half*>(output),
        static_cast<const __half*>(query),
        static_cast<const __half*>(kvCacheBase),
        d_blockTable,
        seqLen,
        layerIdx,
        numHeads,
        numKvHeads,
        headDim,
        blockSize,
        numLayers
    );
    
    CUDA_CHECK(cudaFree(d_blockTable));
}

__global__ void pagedAttentionBatchedFP16Kernel(
    __half* output,
    const __half* queries,
    const __half* kvCacheBase,
    const int32_t* seqLens,
    const int32_t* blockTableOffsets,
    const int32_t* blockTables,
    int32_t maxSeqLen,
    int32_t layerIdx,
    int32_t numHeads,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t blockSize,
    int32_t numLayers
) {
    int32_t batchIdx = blockIdx.y;
    int32_t qHead = blockIdx.x;
    int32_t tid = threadIdx.x;
    
    if (qHead >= numHeads) return;
    
    int32_t seqLen = seqLens[batchIdx];
    if (seqLen <= 0) return;
    
    int32_t queryOffset = batchIdx * numHeads * headDim;
    int32_t kvHead = qHead / (numHeads / numKvHeads);
    
    float q_val = 0.0f;
    if (tid < headDim) {
        q_val = __half2float(queries[queryOffset + qHead * headDim + tid]);
    }
    
    float scale = 1.0f / sqrtf((float)headDim);
    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_i = 0.0f;
    
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * numLayers;
    int32_t layerOffset = layerIdx * 2 * elementsPerLayerKV;
    
    int32_t blockTableOffset = blockTableOffsets[batchIdx];
    
    for (int32_t tokenPos = 0; tokenPos < seqLen; ++tokenPos) {
        int32_t logicalBlockIdx = tokenPos / blockSize;
        int32_t tokenInBlock = tokenPos % blockSize;
        int32_t physicalBlockId = blockTables[blockTableOffset + logicalBlockIdx];
        int32_t blockBase = physicalBlockId * elementsPerBlock;
        
        int32_t keyOffset = blockBase + layerOffset + 
                           tokenInBlock * (numKvHeads * headDim) + kvHead * headDim;
        
        float k_val = 0.0f;
        if (tid < headDim) {
            k_val = __half2float(kvCacheBase[keyOffset + tid]);
        }
        
        float thread_prod = q_val * k_val;
        float qk = blockReduceSumPA(thread_prod);
        
        __shared__ float s_qk;
        if (tid == 0) s_qk = qk;
        __syncthreads();
        qk = s_qk * scale;
        
        float m_i_new = fmaxf(m_i, qk);
        float alpha = expf(m_i - m_i_new);
        float beta = expf(qk - m_i_new);
        l_i = l_i * alpha + beta;
        
        int32_t valOffset = blockBase + layerOffset + elementsPerLayerKV +
                           tokenInBlock * (numKvHeads * headDim) + kvHead * headDim;
        
        float v_val = 0.0f;
        if (tid < headDim) {
            v_val = __half2float(kvCacheBase[valOffset + tid]);
        }
        
        acc_i = acc_i * alpha + beta * v_val;
        m_i = m_i_new;
    }
    
    int32_t outputOffset = batchIdx * numHeads * headDim;
    if (tid < headDim) {
        output[outputOffset + qHead * headDim + tid] = __float2half(acc_i / l_i);
    }
}

void pagedAttentionBatchedFP16CUDA(
    void* output,
    const void* queries,
    const void* kvCacheBase,
    const int32_t* seqLens,
    const int32_t* blockTableOffsets,
    const int32_t* blockTables,
    int32_t batchSize,
    int32_t maxSeqLen,
    int32_t layerIdx,
    int32_t numHeads,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t blockSize,
    int32_t numLayers
) {
    if (batchSize <= 0) return;
    
    dim3 grid(numHeads, batchSize, 1);
    dim3 block(128, 1, 1);
    
    pagedAttentionBatchedFP16Kernel<<<grid, block>>>(
        static_cast<__half*>(output),
        static_cast<const __half*>(queries),
        static_cast<const __half*>(kvCacheBase),
        seqLens,
        blockTableOffsets,
        blockTables,
        maxSeqLen,
        layerIdx,
        numHeads,
        numKvHeads,
        headDim,
        blockSize,
        numLayers
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void quantizeAndCacheFP16Kernel(
    __half* cache,
    const __half* k,
    const __half* v,
    int32_t totalElements,
    int32_t kOffset,
    int32_t vOffset
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        cache[kOffset + idx] = k[idx];
        cache[vOffset + idx] = v[idx];
    }
}

void quantizeAndCacheFP16CUDA(
    void* d_cache,
    const void* d_k,
    const void* d_v,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t kStartOffset,
    int32_t vStartOffset
) {
    int32_t totalElements = numKvHeads * headDim;
    int32_t threadsPerBlock = 256;
    int32_t numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    
    quantizeAndCacheFP16Kernel<<<numBlocks, threadsPerBlock>>>(
        static_cast<__half*>(d_cache),
        static_cast<const __half*>(d_k),
        static_cast<const __half*>(d_v),
        totalElements,
        kStartOffset,
        vStartOffset
    );
    
    CUDA_CHECK(cudaGetLastError());
}

}
