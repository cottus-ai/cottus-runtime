#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include "block_allocator.h"
#include "page_table.h"

namespace cottus {

// Configuration struct for the Engine.
// Properties match Phase 3 contract.
struct EngineConfig {
    int32_t vocabSize;
    int32_t hiddenDim;
    int32_t numLayers;
    int32_t numHeads;
    int32_t numKvHeads;
    int32_t headDim;
    int32_t intermediateDim;  // FFN intermediate dimension (typically hiddenDim * 4)
    int32_t maxSeqLen;
    int32_t blockSize;
    float ropeTheta;
    float normEpsilon;
    std::string device;
    std::string dtype;
};

// Main inference engine.
// Owns the BlockAllocator and manages the lifecycle of PageTables.
//
// Invariants:
// - Single-threaded usage.
// - BlockAllocator initialized at construction.
// - PageTable created per forward() call (ephemeral).
// - No memory leaks (all blocks freed after forward).
class Engine {
public:
    // Initialize engine, allocate KV cache memory.
    // Throws std::invalid_argument if config is invalid.
    Engine(const EngineConfig& config, const std::unordered_map<std::string, uintptr_t>& weightPtrs);
    
    ~Engine();

    // Perform a blocking inference pass (prefill + decode).
    // Creates a fresh PageTable, processes input, generates tokens,
    // and strictly cleans up all resources before returning.
    //
    // Returns: Generated token IDs.
    std::vector<int32_t> forward(const std::vector<int32_t>& inputIds);

    // Generate new tokens using greedy decoding.
    //
    // inputIds: Prompt token IDs
    // maxNewTokens: Maximum number of tokens to generate
    //
    // Semantics:
    // 1. Process all prompt tokens (prefill)
    // 2. Generate up to maxNewTokens using greedy argmax
    // 3. Free all KV blocks before returning
    //
    // Returns: Generated token IDs (excluding prompt)
    std::vector<int32_t> generate(
        const std::vector<int32_t>& inputIds,
        int32_t maxNewTokens
    );

    // Reset internal state (no-op for stateless v0.1, but part of contract).
    void reset();

    // Get current free block count (for testing memory leaks)
    int32_t getFreeBlockCount() const;

private:
    EngineConfig config_;
    std::unique_ptr<BlockAllocator> blockAllocator_;
    std::unique_ptr<class GenericTransformer> transformer_;
    std::vector<uint16_t> kvCache_;
    
    // Future: PagedAttention kernel, Model instance
};

} // namespace cottus
