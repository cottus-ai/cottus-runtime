#pragma once

#include <vector>
#include <cstdint>
#include "page_table.h"

namespace cottus {

// CPU reference implementation of paged attention.
// Uses simple nested loops for maximum clarity and correctness.
//
// This is NOT optimized. It exists to:
// 1. Validate the CUDA kernel
// 2. Serve as ground truth for testing
// 3. Document the exact attention algorithm
//
// Invariants:
// - FP32 accumulation for numerical stability
// - FP16 storage in KV cache
// - No fusion, no tricks
// - Bounds-checked access
void pagedAttentionCPU(
    float* output,              // [numHeads, headDim] - output tensor
    const float* query,         // [numHeads, headDim] - query tensor  
    const void* kvCacheBase,    // Base pointer to KV cache (fp16)
    const PageTable& pageTable, // Maps logical positions to physical blocks
    int32_t seqLen,             // Current sequence length
    int32_t layerIdx,           // Which transformer layer
    int32_t numHeads,           // Number of query heads
    int32_t numKvHeads,         // Number of KV heads (for GQA)
    int32_t headDim,            // Dimension per head
    int32_t blockSize,          // Tokens per block
    int32_t numLayers           // Total number of transformer layers
);

} // namespace cottus
