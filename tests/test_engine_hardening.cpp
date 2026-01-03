#include "../cottus/csrc/engine.h"
#include <cassert>
#include <iostream>
#include <vector>

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

EngineConfig createTestConfig() {
    EngineConfig config;
    config.vocabSize = 50;
    config.hiddenDim = 32;
    config.numLayers = 1;
    config.numHeads = 1;
    config.numKvHeads = 1;
    config.headDim = 32;
    config.intermediateDim = 128;
    config.maxSeqLen = 16;
    config.blockSize = 4;
    config.ropeTheta = 10000.0f;
    config.normEpsilon = 1e-5f;
    config.device = "cpu";
    config.dtype = "float16";
    return config;
}

// Test: Max Pressure - exact capacity match
void testMaxPressure() {
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    Engine engine(config, weights);

    // Request that fits exactly in capacity
    std::vector<int32_t> input = {1, 2, 3, 4, 5};  // 5 tokens
    
    try {
        std::vector<int32_t> output = engine.generate(input, 3);  // Request 3 new tokens
        std::cout << "PASS: Exact Capacity Match - Generated " << output.size() << " tokens" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: Failed at exact capacity: " << e.what() << std::endl;
        assert(false);
    }
}

// Test: Block Recycling
void testBlockRecycling() {
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    Engine engine(config, weights);
    
    int32_t initialBlocks = engine.getFreeBlockCount();
    
    // Run multiple iterations
    for (int i = 0; i < 3; ++i) {
        std::vector<int32_t> input = {1, 2, 3};
        engine.generate(input, 2);
        
        int32_t currentBlocks = engine.getFreeBlockCount();
        assert(currentBlocks == initialBlocks);
    }
    
    std::cout << "PASS: Block Recycling" << std::endl;
}

// Test: Empty input rejection
void testEmptyInputRejection() {
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    Engine engine(config, weights);
    
    bool caught = false;
    try {
        std::vector<int32_t> empty;
        engine.generate(empty, 5);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::cout << "PASS: Empty Input Rejected - " << e.what() << std::endl;
    }
    assert(caught);
}

int main() {
    testMaxPressure();
    testBlockRecycling();
    testEmptyInputRejection();
    std::cout << "PASS: All engine hardening tests" << std::endl;
    return 0;
}
