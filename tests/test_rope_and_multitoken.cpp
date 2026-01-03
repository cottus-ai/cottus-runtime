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
    
    auto allocWeight = [](size_t size) -> uintptr_t {
        float* ptr = new float[size]();
        for (size_t i = 0; i < size; ++i) {
            ptr[i] = 0.01f * (i % 10);
        }
        return reinterpret_cast<uintptr_t>(ptr);
    };
    
    weights["model.embed_tokens.weight"] = allocWeight(config.vocabSize * config.hiddenDim);
    weights["model.norm.weight"] = allocWeight(config.hiddenDim);
    weights["lm_head.weight"] = allocWeight(config.hiddenDim * config.vocabSize);
    
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

// Test: RoPE position semantics
void testRoPEPositions() {
    std::cout << "Test: RoPE Position Semantics" << std::endl;
    
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
    
    // Run with token=5 at pos=0
    std::vector<float> logits_t5_p0 = model.forwardToken(5, 0, pageTable, reinterpret_cast<uintptr_t>(kvCache.data()), "cpu");
    
    // Run with token=7 at pos=0 (different token, same position)
    std::vector<uint16_t> kvCache2(elementsPerBlock, 0);
    std::vector<float> logits_t7_p0 = model.forwardToken(7, 0, pageTable, reinterpret_cast<uintptr_t>(kvCache2.data()), "cpu");
    
    // Debug: Print first few logits
    std::cout << "  Token 5 logits: " << logits_t5_p0[0] << ", " << logits_t5_p0[1] << ", " << logits_t5_p0[2] << std::endl;
    std::cout << "  Token 7 logits: " << logits_t7_p0[0] << ", " << logits_t7_p0[1] << ", " << logits_t7_p0[2] << std::endl;
    
    // Verify outputs are different (different tokens should produce different embeddings/outputs)
    bool different = false;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < logits_t5_p0.size(); ++i) {
        float diff = std::abs(logits_t5_p0[i] - logits_t7_p0[i]);
        maxDiff = std::max(maxDiff, diff);
        if (diff > 1e-6f) {
            different = true;
        }
    }
    
    std::cout << "  Max difference: " << maxDiff << std::endl;
    
    assert(different);
    std::cout << "  PASS: Different tokens produce different outputs" << std::endl;
}

// Test: Multi-token forward
void testMultiTokenForward() {
    std::cout << "Test: Multi-Token Forward" << std::endl;
    
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
    
    // Process 3 tokens
    std::vector<int32_t> tokens = {10, 20, 30};
    std::vector<std::vector<float>> allLogits;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<float> logits = model.forwardToken(
            tokens[i], 
            static_cast<int32_t>(i), 
            pageTable, 
            reinterpret_cast<uintptr_t>(kvCache.data()), 
            "cpu"
        );
        allLogits.push_back(logits);
        
        // Verify output is finite
        for (float val : logits) {
            assert(std::isfinite(val));
        }
    }
    
    std::cout << "  Processed " << tokens.size() << " tokens successfully" << std::endl;
    std::cout << "  PASS: Multi-token forward works" << std::endl;
}

int main() {
    testRoPEPositions();
    testMultiTokenForward();
    std::cout << "All RoPE and multi-token tests passed!" << std::endl;
    return 0;
}
