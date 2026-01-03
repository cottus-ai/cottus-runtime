#include "../cottus/csrc/generic_transformer.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

using namespace cottus;

// Helper: Create dummy weights with valid memory
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

// Test: Weight Loading and Validation
void testWeightLoading() {
    EngineConfig config;
    config.vocabSize = 50;
    config.numLayers = 1;
    config.hiddenDim = 32;
    config.numHeads = 1;
    config.numKvHeads = 1;
    config.headDim = 32;
    config.intermediateDim = 128;
    config.blockSize = 4;
    config.maxSeqLen = 16;
    config.ropeTheta = 10000.0f;
    config.normEpsilon = 1e-5f;
    config.device = "cpu";
    config.dtype = "float16";
    
    auto weights = createDummyWeights(config);
    
    // 1. Success case
    {
        GenericTransformer model(config, weights);
        std::cout << "PASS: Weight loading success" << std::endl;
        
        // Test forward with valid memory
        int32_t elementsPerLayerKV = config.blockSize * config.numKvHeads * config.headDim;
        int32_t elementsPerBlock = 2 * elementsPerLayerKV * config.numLayers;
        std::vector<uint16_t> kvCache(elementsPerBlock, 0);
        
        PageTable pt(config.blockSize);
        pt.appendBlock(0);
        
        std::vector<float> logits = model.forwardToken(5, 0, pt, reinterpret_cast<uintptr_t>(kvCache.data()), "cpu");
        assert(logits.size() == static_cast<size_t>(config.vocabSize));
        std::cout << "PASS: Forward token success" << std::endl;
    }

    // 2. Missing weight case
    {
        auto badWeights = weights;
        badWeights.erase("lm_head.weight");
        bool caught = false;
        try {
            GenericTransformer model(config, badWeights);
        } catch (const std::invalid_argument& e) {
            caught = true;
            std::string msg = e.what();
            if (msg.find("Missing weight") != std::string::npos) {
                std::cout << "PASS: Caught missing weight" << std::endl;
            }
        }
        assert(caught);
    }
}

int main() {
    testWeightLoading();
    return 0;
}
