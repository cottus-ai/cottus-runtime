#pragma once

#include <vector>
#include <cstdint>
#include "page_table.h"
namespace cottus {

//CUDA wrapper for paged attention
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
);

//quantize FP32 K/V to FP16 and write to GPU KV cache
void quantizeAndCacheCUDA(
    void* d_kvCache,
    const float* d_k,
    const float* d_v,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t kStartOffset,
    int32_t vStartOffset
);

} // namespace cottus
