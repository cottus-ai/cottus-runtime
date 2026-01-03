#include "../cottus/csrc/generic_transformer.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

using namespace cottus;

// Helper: Create dummy weights
std::unordered_map<std::string, uintptr_t> createDummyWeights(const EngineConfig& config) {
    std::unordered_map<std::string, uintptr_t> weights;
    
    // Allocate dummy weight buffers (never freed in this test, but that's OK for unit test)
    auto allocWeight = [](size_t size) -> uintptr_t {
        float* ptr = new float[size]();
        // Initialize to small values
        for (size_t i = 0; i < size; ++i) {
            ptr[i] = 0.01f * (i % 10);
        }
        return reinterpret_cast<uintptr_t>(ptr);
    };
    
    // Global weights
    weights["model.embed_tokens.weight"] = allocWeight(config.vocabSize * config.hiddenDim);
    weights["model.norm.weight"] = allocWeight(config.hiddenDim);
    weights["lm_head.weight"] = allocWeight(config.hiddenDim * config.vocabSize);
    
    // Per-layer weights
    for (int32_t i = 0; i < config.numLayers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        
        weights[prefix + "self_attn.q_proj.weight"] = allocWeight(config.hiddenDim * config.numHeads * config.headDim);
        weights[prefix + "self_attn.k_proj.weight"] = allocWeight(config.hiddenDim * config.numKvHeads * config.headDim);
        weights[prefix + "self_attn.v_proj.weight"] = allocWeight(config.hiddenDim * config.numKvHeads * config.headDim);
        weights[prefix + "self_attn.o_proj.weight"] = allocWeight(config.numHeads * config.headDim * config.hiddenDim);
        
        weights[prefix + "mlp.gate_proj.weight"] = allocWeight(config.hiddenDim * config.intermediateDim);
        weights[prefix + "mlp.down_proj.weight"] = allocWeight(config.intermediateDim * config.hiddenDim);
        weights[prefix + "mlp.up_proj.weight"] = allocWeight(config.hiddenDim * config.intermediateDim);
        
        weights[prefix + "input_layernorm.weight"] = allocWeight(config.hiddenDim);
        weights[prefix + "post_attention_layernorm.weight"] = allocWeight(config.hiddenDim);
    }
    
    return weights;
}

// Test 1: Single layer, single token forward pass
void testSingleLayerForward() {
    std::cout << "Test: Single Layer Forward Pass" << std::endl;
    
    // Minimal config
    EngineConfig config;
    config.vocabSize = 100;
    config.hiddenDim = 64;
    config.numLayers = 1;
    config.numHeads = 2;
    config.numKvHeads = 2;
    config.headDim = 32;
    config.intermediateDim = 256;  // 4x hiddenDim
    config.maxSeqLen = 16;
    config.blockSize = 4;
    config.ropeTheta = 10000.0f;
    config.normEpsilon = 1e-5f;
    config.device = "cpu";
    config.dtype = "float16";
    
    // Create dummy weights
    auto weights = createDummyWeights(config);
    
    // Create transformer
    GenericTransformer model(config, weights);
    
    // Create dummy KV cache
    int32_t elementsPerLayerKV = config.blockSize * config.numKvHeads * config.headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * config.numLayers;
    std::vector<uint16_t> kvCache(elementsPerBlock, 0);
    
    // Create PageTable
    PageTable pageTable(config.blockSize);
    pageTable.appendBlock(0);  // Single block
    
    // Run forward pass
    try {
        std::vector<float> logits = model.forwardToken(
            50,  // token
            0,   // pos
            pageTable,
            reinterpret_cast<uintptr_t>(kvCache.data()),
            "cpu"
        );
        
        // Verify output shape
        assert(logits.size() == static_cast<size_t>(config.vocabSize));
        
        // Verify outputs are finite
        bool allFinite = true;
        for (float val : logits) {
            if (!std::isfinite(val)) {
                allFinite = false;
                break;
            }
        }
        assert(allFinite);
        
        std::cout << "  Output shape: " << logits.size() << std::endl;
        std::cout << "  Sample logits: " << logits[0] << ", " << logits[1] << ", " << logits[2] << std::endl;
        std::cout << "  PASS" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "  FAIL: " << e.what() << std::endl;
        assert(false);
    }
}

// Test 2: Determinism check
void testDeterminism() {
    std::cout << "Test: Determinism" << std::endl;
    
    EngineConfig config;
    config.vocabSize = 50;
    config.hiddenDim = 32;
    config.numLayers = 1;
    config.numHeads = 1;
    config.numKvHeads = 1;
    config.headDim = 32;
    config.intermediateDim = 128;
    config.maxSeqLen = 8;
    config.blockSize = 4;
    config.ropeTheta = 10000.0f;
    config.normEpsilon = 1e-5f;
    config.device = "cpu";
    config.dtype = "float16";
    
    auto weights = createDummyWeights(config);
    GenericTransformer model(config, weights);
    
    int32_t elementsPerLayerKV = config.blockSize * config.numKvHeads * config.headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * config.numLayers;
    std::vector<uint16_t> kvCache(elementsPerBlock, 0);
    
    PageTable pageTable(config.blockSize);
    pageTable.appendBlock(0);
    
    // Run twice
    std::vector<float> logits1 = model.forwardToken(10, 0, pageTable, reinterpret_cast<uintptr_t>(kvCache.data()), "cpu");
    std::vector<float> logits2 = model.forwardToken(10, 0, pageTable, reinterpret_cast<uintptr_t>(kvCache.data()), "cpu");
    
    // Compare
    assert(logits1.size() == logits2.size());
    for (size_t i = 0; i < logits1.size(); ++i) {
        assert(logits1[i] == logits2[i]);  // Bitwise identical
    }
    
    std::cout << "  PASS: Outputs are bitwise identical" << std::endl;
}

int main() {
    testSingleLayerForward();
    testDeterminism();
    std::cout << "All single-layer forward tests passed!" << std::endl;
    return 0;
}
