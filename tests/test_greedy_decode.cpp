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

// Test 1: Basic greedy decode
void testGreedyDecodeBasic() {
    std::cout << "Test: Greedy Decode Basic" << std::endl;
    
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    
    Engine engine(config, weights);
    
    std::vector<int32_t> prompt = {5, 3, 7};
    std::vector<int32_t> generated = engine.generate(prompt, 3);
    
    // Verify output
    assert(generated.size() == 3);
    
    // Verify all tokens are in valid range
    for (int32_t token : generated) {
        assert(token >= 0 && token < config.vocabSize);
    }
    
    std::cout << "  Generated: ";
    for (int32_t t : generated) std::cout << t << " ";
    std::cout << std::endl;
    
    std::cout << "  PASS" << std::endl;
}

// Test 2: No memory leak
void testGreedyDecodeNoLeak() {
    std::cout << "Test: Greedy Decode No Leak" << std::endl;
    
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    
    Engine engine(config, weights);
    
    int32_t initialFreeBlocks = engine.getFreeBlockCount();
    
    // Run generate twice
    std::vector<int32_t> prompt = {1, 2, 3};
    engine.generate(prompt, 3);
    int32_t afterFirst = engine.getFreeBlockCount();
    
    engine.generate(prompt, 5);
    int32_t afterSecond = engine.getFreeBlockCount();
    
    // Verify all blocks freed
    assert(afterFirst == initialFreeBlocks);
    assert(afterSecond == initialFreeBlocks);
    
    std::cout << "  Initial blocks: " << initialFreeBlocks << std::endl;
    std::cout << "  After first: " << afterFirst << std::endl;
    std::cout << "  After second: " << afterSecond << std::endl;
    std::cout << "  PASS: No memory leak" << std::endl;
}

// Test 3: Zero new tokens
void testGreedyDecodeZeroTokens() {
    std::cout << "Test: Greedy Decode Zero Tokens" << std::endl;
    
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    
    Engine engine(config, weights);
    
    std::vector<int32_t> prompt = {5, 3};
    std::vector<int32_t> generated = engine.generate(prompt, 0);
    
    assert(generated.empty());
    
    std::cout << "  PASS: Returns empty vector" << std::endl;
}

// Test 4: Prompt only (longer prompt)
void testGreedyDecodePromptOnly() {
    std::cout << "Test: Greedy Decode Prompt Only" << std::endl;
    
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    
    Engine engine(config, weights);
    
    // Longer prompt that spans multiple blocks
    std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6};
    std::vector<int32_t> generated = engine.generate(prompt, 2);
    
    assert(generated.size() == 2);
    
    std::cout << "  Prompt length: " << prompt.size() << std::endl;
    std::cout << "  Generated: " << generated.size() << " tokens" << std::endl;
    std::cout << "  PASS" << std::endl;
}

// Test 5: Determinism
void testGreedyDecodeDeterminism() {
    std::cout << "Test: Greedy Decode Determinism" << std::endl;
    
    EngineConfig config = createTestConfig();
    auto weights = createDummyWeights(config);
    
    Engine engine(config, weights);
    
    std::vector<int32_t> prompt = {5, 3, 7};
    
    // Run multiple times
    std::vector<int32_t> run1 = engine.generate(prompt, 3);
    std::vector<int32_t> run2 = engine.generate(prompt, 3);
    std::vector<int32_t> run3 = engine.generate(prompt, 3);
    
    // Verify identical outputs
    assert(run1 == run2);
    assert(run2 == run3);
    
    std::cout << "  PASS: All runs identical" << std::endl;
}

int main() {
    testGreedyDecodeBasic();
    testGreedyDecodeNoLeak();
    testGreedyDecodeZeroTokens();
    testGreedyDecodePromptOnly();
    testGreedyDecodeDeterminism();
    std::cout << "All greedy decode tests passed!" << std::endl;
    return 0;
}
