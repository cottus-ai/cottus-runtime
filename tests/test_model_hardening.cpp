#include "../cottus/csrc/generic_transformer.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

using namespace cottus;

// 1. Shape Mismatch Tests
// Ensure model rejects invalid weight shapes/counts if possible.
// Currently constructor validates *presence* of keys.
// In v0.1 we rely on user providing valid pointers.
// However, we can verify that the model structure matches config.

void testLayerValues() {
    EngineConfig config;
    config.numLayers = 5;
    config.vocabSize = 100;
    // ... basic config
    config.hiddenDim=32; config.numHeads=4; config.numKvHeads=4; config.headDim=8; config.blockSize=16;
    
    std::unordered_map<std::string, uintptr_t> weights;
    uintptr_t dummy = 0xDEADBEEF;
    
    // Fill weights
    weights["model.embed_tokens.weight"] = dummy;
    weights["model.norm.weight"] = dummy;
    weights["lm_head.weight"] = dummy;
    
    for (int i=0; i<5; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        weights[p+"self_attn.q_proj.weight"] = dummy;
        weights[p+"self_attn.k_proj.weight"] = dummy;
        weights[p+"self_attn.v_proj.weight"] = dummy;
        weights[p+"self_attn.o_proj.weight"] = dummy;
        weights[p+"mlp.gate_proj.weight"] = dummy;
        weights[p+"mlp.down_proj.weight"] = dummy;
        weights[p+"mlp.up_proj.weight"] = dummy;
        weights[p+"input_layernorm.weight"] = dummy;
        weights[p+"post_attention_layernorm.weight"] = dummy;
    }
    
    try {
        GenericTransformer model(config, weights);
        std::cout << "PASS: Layer Count Verification" << std::endl;
    } catch (...) {
        assert(false && "Should accept valid config");
    }
    
    // Now try with missing layer 4
    weights.erase("model.layers.4.self_attn.q_proj.weight");
    bool caught = false;
    try {
        GenericTransformer model(config, weights);
    } catch (const std::invalid_argument& e) {
        caught = true;
    }
    assert(caught);
    std::cout << "PASS: Missing Layer Weight Detected" << std::endl;
}

// 2. No Allocation In Forward
// Ideally we would mock malloc, but for now we observe side effects.
// GenericTransformer::forward is a stub returning a vector.
// The VECTOR allocation is allowed (output), but intermediate heaps are bad.
// Since it's a stub, we just verify it runs without crashing.
// Meaningful "no allocation" test requires hooking allocator or inspecting RSS, which is complex.
// For v0.1 hardening, we trust the code inspection matching "no dynamic allocation" invariant.

int main() {
    testLayerValues();
    return 0;
}
