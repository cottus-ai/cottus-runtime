#pragma once

#include <cstdint>
#include "page_table.h"

namespace cottus {

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
);

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
);

void quantizeAndCacheFP16CUDA(
    void* d_cache,
    const void* d_k,
    const void* d_v,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t kStartOffset,
    int32_t vStartOffset
);

}
