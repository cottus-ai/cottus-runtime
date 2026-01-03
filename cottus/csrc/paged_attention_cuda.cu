#include "paged_attention_cpu.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>

namespace cottus {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Device function: FP16 to FP32 conversion
__device__ inline float fp16_to_fp32_device(uint16_t h) {
    __half half_val = *reinterpret_cast<__half*>(&h);
    return __half2float(half_val);
}

// CUDA Kernel: PagedAttention v1
// Option A: One thread per query head (simple, correct, not optimized)
__global__ void pagedAttentionKernel(
    float* output,              // [numHeads, headDim]
    const float* query,         // [numHeads, headDim]
    const uint16_t* kvCacheBase,// KV cache (fp16)
    const int32_t* blockTable,  // Physical block IDs [numLogicalBlocks]
    int32_t seqLen,
    int32_t layerIdx,
    int32_t numHeads,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t blockSize,
    int32_t numLayers
) {
    // One thread per query head
    int32_t qHead = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (qHead >= numHeads) return;
    
    // GQA: Map query head to KV head
    int32_t kvHead = (qHead * numKvHeads) / numHeads;
    
    // KV layout constants
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * numLayers;
    
    // Initialize headOutput accumulator
    float headOutput[256]; // Max headDim = 256
    for (int32_t d = 0; d < headDim; ++d) {
        headOutput[d] = 0.0f;
    }
    
    // --- PASS 1: Find Max QK for numerical stability ---
    float maxQk = -1e20f; // Negative infinity
    
    for (int32_t tokenPos = 0; tokenPos < seqLen; ++tokenPos) {
        // Use separate scope or helper to avoid register spills if possible?
        // Replicating indexing logic for correctness
        
        int32_t logicalBlockIdx = tokenPos / blockSize;
        int32_t tokenInBlock = tokenPos % blockSize;
        
        int32_t physicalBlockId = blockTable[logicalBlockIdx];
        int32_t blockBase = physicalBlockId * elementsPerBlock;
        int32_t layerOffset = layerIdx * 2 * elementsPerLayerKV;
        int32_t keyOffset = blockBase + layerOffset + 
                           tokenInBlock * (numKvHeads * headDim) + 
                           kvHead * headDim;
                           
        float qk = 0.0f;
        for (int32_t d = 0; d < headDim; ++d) {
            float q = query[qHead * headDim + d];
            float k = fp16_to_fp32_device(kvCacheBase[keyOffset + d]);
            qk += q * k;
        }
        
        // Scale
        float scale = 1.0f / sqrtf(static_cast<float>(headDim));
        qk *= scale;
        
        if (qk > maxQk) {
            maxQk = qk;
        }
    }
    
    // --- PASS 2: Compute Softmax and Accumulate V ---
    float sumExp = 0.0f;
    
    for (int32_t tokenPos = 0; tokenPos < seqLen; ++tokenPos) {
        // Re-compute QK
        int32_t logicalBlockIdx = tokenPos / blockSize;
        int32_t tokenInBlock = tokenPos % blockSize;
        
        int32_t physicalBlockId = blockTable[logicalBlockIdx];
        int32_t blockBase = physicalBlockId * elementsPerBlock;
        int32_t layerOffset = layerIdx * 2 * elementsPerLayerKV;
        int32_t keyOffset = blockBase + layerOffset + 
                           tokenInBlock * (numKvHeads * headDim) + 
                           kvHead * headDim;
                           
        float qk = 0.0f;
        for (int32_t d = 0; d < headDim; ++d) {
            float q = query[qHead * headDim + d];
            float k = fp16_to_fp32_device(kvCacheBase[keyOffset + d]);
            qk += q * k;
        }
        
        float scale = 1.0f / sqrtf(static_cast<float>(headDim));
        qk *= scale;
        
        // Subtract max for stability
        float expQk = expf(qk - maxQk);
        sumExp += expQk;
        
        // Accumulate V
        int32_t valueOffset = blockBase + layerOffset + elementsPerLayerKV +
                             tokenInBlock * (numKvHeads * headDim) + 
                             kvHead * headDim;
                             
        for (int32_t d = 0; d < headDim; ++d) {
            float v = fp16_to_fp32_device(kvCacheBase[valueOffset + d]);
            headOutput[d] += expQk * v;
        }
    }
    
    // Normalize
    for (int32_t d = 0; d < headDim; ++d) {
        output[qHead * headDim + d] = headOutput[d] / sumExp;
    }
}

// Host-side wrapper
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
    // Validate inputs
    if (seqLen <= 0) throw std::invalid_argument("seqLen must be positive");
    if (layerIdx < 0) throw std::invalid_argument("layerIdx must be non-negative");
    if (numHeads <= 0 || numKvHeads <= 0) throw std::invalid_argument("numHeads must be positive");
    if (headDim <= 0 || headDim > 256) throw std::invalid_argument("headDim must be in (0, 256]");
    if (blockSize <= 0) throw std::invalid_argument("blockSize must be positive");
    
    // Allocate device memory
    float* d_output;
    float* d_query;
    uint16_t* d_kvCache;
    int32_t* d_blockTable;
    
    size_t outputSize = numHeads * headDim * sizeof(float);
    size_t querySize = numHeads * headDim * sizeof(float);
    
    // Calculate KV cache size
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * numLayers;
    size_t kvCacheSize = pageTable.numBlocks() * elementsPerBlock * sizeof(uint16_t);
    
    size_t blockTableSize = pageTable.numBlocks() * sizeof(int32_t);
    
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
    CUDA_CHECK(cudaMalloc(&d_query, querySize));
    CUDA_CHECK(cudaMalloc(&d_kvCache, kvCacheSize));
    CUDA_CHECK(cudaMalloc(&d_blockTable, blockTableSize));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_query, query, querySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvCache, kvCacheBase, kvCacheSize, cudaMemcpyHostToDevice));
    
    // Copy block table to device
    std::vector<int32_t> blockTableHost(pageTable.numBlocks());
    for (int i = 0; i < pageTable.numBlocks(); ++i) {
        blockTableHost[i] = pageTable[i];
    }
    CUDA_CHECK(cudaMemcpy(d_blockTable, blockTableHost.data(), blockTableSize, cudaMemcpyHostToDevice));
    
    // Launch kernel: One thread per head
    dim3 grid(numHeads, 1, 1);
    dim3 block(1, 1, 1);
    
    pagedAttentionKernel<<<grid, block>>>(
        d_output, d_query, d_kvCache, d_blockTable,
        seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, numLayers
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_kvCache));
    CUDA_CHECK(cudaFree(d_blockTable));
}

} // namespace cottus
