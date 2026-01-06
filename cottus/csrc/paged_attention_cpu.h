#pragma once

#include <vector>
#include <cstdint>
#include "page_table.h"

namespace cottus {
void pagedAttentionCPU(
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

} // namespace cottus
