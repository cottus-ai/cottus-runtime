#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include "engine.h"  //EngineConfig
#include "page_table.h"

namespace cottus {

struct LayerWeights {
    uintptr_t wq; 
    uintptr_t wk; 
    uintptr_t wv; 
    uintptr_t wqkv;
    uintptr_t wo;
    uintptr_t w1; 
    uintptr_t w2; 
    uintptr_t w3;
    uintptr_t attention_norm;
    uintptr_t ffn_norm;
};
class GenericTransformer {
public:
    GenericTransformer(const EngineConfig& config, const std::unordered_map<std::string, uintptr_t>& weightPtrs);
    std::vector<float> forwardToken(
        int32_t token, 
        int32_t pos, 
        const PageTable& pageTable, 
        uintptr_t kvCacheBase,
        void* d_kvCacheVal,
        const std::string& device = "cuda"
    );
    ~GenericTransformer();
private:
    EngineConfig config_;
    std::vector<void*> allocated_gpu_weights_;
    uintptr_t token_embedding_table_;
    uintptr_t output_norm_;
    uintptr_t output_head_;
    std::vector<LayerWeights> layers_;
    float* d_x_ = nullptr;
    float* d_xb_ = nullptr;
    float* d_q_ = nullptr;
    float* d_k_ = nullptr;
    float* d_v_ = nullptr;
    float* d_att_ = nullptr;
    float* d_hb_ = nullptr;
    float* d_hb2_ = nullptr;
    
    uintptr_t d_output_head_ = 0;
    float* d_logits_ = nullptr;
};

} // namespace cottus
