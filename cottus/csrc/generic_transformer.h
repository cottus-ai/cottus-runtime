#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include "engine.h"  // For EngineConfig
#include "page_table.h"

namespace cottus {

struct LayerWeights {
    // Attention
    uintptr_t wq; // Shape: [hiddenDim, numHeads * headDim]
    uintptr_t wk; // Shape: [hiddenDim, numKvHeads * headDim]
    uintptr_t wv; // Shape: [hiddenDim, numKvHeads * headDim]
    uintptr_t wo; // Shape: [numHeads * headDim, hiddenDim]
    
    // FFN (MLP)
    uintptr_t w1; // Gate: [hiddenDim, intermediateDim]
    uintptr_t w2; // Down: [intermediateDim, hiddenDim]
    uintptr_t w3; // Up:   [hiddenDim, intermediateDim]
    
    // Norms (RMSNorm weights)
    uintptr_t attention_norm; // [hiddenDim]
    uintptr_t ffn_norm;       // [hiddenDim]
};

// Generic Transformer Model Skeleton.
// Holds pointers to all model weights and defines the forward pass usage.
//
// Invariants:
// - All weight pointers are validated at construction.
// - Weights are owned by Python (we borrowed them).
// - No dynamic allocation during forward().
// - Stateless (except for constant weights).
class GenericTransformer {
public:
    // Initialize model, validate all required weights exist.
    // Throws std::invalid_argument if weights missing or shapes invalid (in future).
    GenericTransformer(const EngineConfig& config, const std::unordered_map<std::string, uintptr_t>& weightPtrs);

    // Perform forward pass for a single token.
    //
    // token: input token ID
    // pos: position index (for RoPE)
    // pageTable: KV cache mapping for this request
    // kvCacheBase: Base pointer to physical KV cache memory
    // device: "cpu" or "cuda"
    //
    // Returns: Logits [vocabSize]
    std::vector<float> forwardToken(
        int32_t token, 
        int32_t pos, 
        const PageTable& pageTable, 
        uintptr_t kvCacheBase,
        const std::string& device = "cuda"
    );

    // Destructor to free GPU weights if needed
    ~GenericTransformer();

private:
    EngineConfig config_;
    std::vector<void*> allocated_gpu_weights_; // Track GPU allocations for cleanup
    
    // Model Weights
    uintptr_t token_embedding_table_; // [vocabSize, hiddenDim]
    uintptr_t output_norm_;           // [hiddenDim]
    uintptr_t output_head_;           // [hiddenDim, vocabSize]
    
    std::vector<LayerWeights> layers_;
};

} // namespace cottus
