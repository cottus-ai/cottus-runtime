#include "generic_transformer.h"
#include "compute_primitives_cpu.h"
#include "compute_primitives_cuda.h"
#include "paged_attention_cpu.h"
#include "paged_attention_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace cottus {

#define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } \
    while(0)

GenericTransformer::GenericTransformer(const EngineConfig& config, const std::unordered_map<std::string, uintptr_t>& weightPtrs)
    : config_(config) {
    
    auto getWeight = [&](const std::string& name) -> uintptr_t {
        auto it = weightPtrs.find(name);
        if (it == weightPtrs.end()) {
            throw std::invalid_argument("Missing weight: " + name);
        }
        return it->second;
    };
    token_embedding_table_ = getWeight("model.embed_tokens.weight");
    output_norm_ = getWeight("model.norm.weight");
    output_head_ = getWeight("lm_head.weight");
    layers_.resize(config.numLayers);
    for (int32_t i = 0; i < config.numLayers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        layers_[i].wq = getWeight(prefix + "self_attn.q_proj.weight");
        layers_[i].wk = getWeight(prefix + "self_attn.k_proj.weight");
        layers_[i].wv = getWeight(prefix + "self_attn.v_proj.weight");
        layers_[i].wo = getWeight(prefix + "self_attn.o_proj.weight");
        layers_[i].w1 = getWeight(prefix + "mlp.gate_proj.weight");
        layers_[i].w2 = getWeight(prefix + "mlp.down_proj.weight");
        layers_[i].w3 = getWeight(prefix + "mlp.up_proj.weight");
        layers_[i].attention_norm = getWeight(prefix + "input_layernorm.weight");
        layers_[i].ffn_norm = getWeight(prefix + "post_attention_layernorm.weight");
    }
    if(config_.device == "cuda")
    {
        auto upload = [&](uintptr_t hostPtr, size_t sizeBytes) -> uintptr_t
        {
            void* devPtr = nullptr;
            CUDA_CHECK(cudaMalloc(&devPtr, sizeBytes));
            CUDA_CHECK(cudaMemcpy(devPtr, reinterpret_cast<void*>(hostPtr), sizeBytes, cudaMemcpyHostToDevice));
            allocated_gpu_weights_.push_back(devPtr);
            return reinterpret_cast<uintptr_t>(devPtr);
        };
        int32_t vocabSize = config.vocabSize;
        int32_t hidden = config.hiddenDim;
        int32_t headDim = config.headDim;
        int32_t numHeads = config.numHeads;
        int32_t numKvHeads = config.numKvHeads;
        int32_t intermediate = config.intermediateDim;
        output_norm_ = upload(output_norm_, hidden * sizeof(float)); 
        for (int32_t i = 0; i < config.numLayers; ++i) {
            layers_[i].wq = upload(layers_[i].wq, hidden * numHeads * headDim * sizeof(float));
            layers_[i].wk = upload(layers_[i].wk, hidden * numKvHeads * headDim * sizeof(float));
            layers_[i].wv = upload(layers_[i].wv, hidden * numKvHeads * headDim * sizeof(float));
            layers_[i].wo = upload(layers_[i].wo, numHeads * headDim * hidden * sizeof(float));
            
            layers_[i].w1 = upload(layers_[i].w1, hidden * intermediate * sizeof(float));
            layers_[i].w2 = upload(layers_[i].w2, intermediate * hidden * sizeof(float));
            layers_[i].w3 = upload(layers_[i].w3, hidden * intermediate * sizeof(float));
            
            layers_[i].attention_norm = upload(layers_[i].attention_norm, hidden * sizeof(float));
            layers_[i].ffn_norm = upload(layers_[i].ffn_norm, hidden * sizeof(float));
        }
    }
}

GenericTransformer::~GenericTransformer() {
    for (void* ptr : allocated_gpu_weights_) {
        cudaFree(ptr);
    }
    allocated_gpu_weights_.clear();
}

std::vector<float> GenericTransformer::forwardToken(
    int32_t token,
    int32_t pos,
    const PageTable& pageTable,
    uintptr_t kvCacheBase,
    const std::string& device
) {
    if (token < 0 || token >= config_.vocabSize) {
        throw std::out_of_range("Token ID out of vocab range");
    }
    bool useCuda = (device == "cuda");
    int32_t hiddenDim = config_.hiddenDim;
    int32_t numHeads = config_.numHeads;
    int32_t numKvHeads = config_.numKvHeads;
    int32_t headDim = config_.headDim;
    int32_t intermediateDim = config_.intermediateDim;

#ifdef COTTUS_DEBUG_PARITY
    if (pos == 0) {     
        std::cout << "\n[DEBUG] Config Values (C++):" << std::endl;
        std::cout << "  hiddenDim: " << hiddenDim << std::endl;
        std::cout << "  numHeads: " << numHeads << std::endl;
        std::cout << "  numKvHeads: " << numKvHeads << std::endl;
        std::cout << "  headDim: " << headDim << std::endl;
        std::cout << "  intermediateDim: " << intermediateDim << std::endl;
    }
#endif
    
    std::vector<float> x(hiddenDim);           
    std::vector<float> xb(hiddenDim);          
    std::vector<float> q(numHeads * headDim);  
    std::vector<float> k(numKvHeads * headDim);
    std::vector<float> v(numKvHeads * headDim);
    std::vector<float> att(numHeads * headDim);
    std::vector<float> hb(intermediateDim);    
    std::vector<float> hb2(intermediateDim);   
    float *d_x = nullptr, *d_xb = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_att = nullptr, *d_hb = nullptr, *d_hb2 = nullptr;
    
    if (useCuda) {
        CUDA_CHECK(cudaMalloc(&d_x, hiddenDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_xb, hiddenDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q, numHeads * headDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k, numKvHeads * headDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, numKvHeads * headDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_att, numHeads * headDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hb, intermediateDim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hb2, intermediateDim * sizeof(float)));
    }
    
    try {
        const float* embedTable = reinterpret_cast<const float*>(token_embedding_table_);
        std::memcpy(x.data(), embedTable + token * hiddenDim, hiddenDim * sizeof(float));
        
        if (useCuda) {
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), hiddenDim * sizeof(float), cudaMemcpyHostToDevice));
        }
        for (int32_t layer = 0; layer < config_.numLayers; ++layer) {
            const LayerWeights& weights = layers_[layer];
            if (useCuda) {
                CUDA_CHECK(cudaMemcpy(d_xb, d_x, hiddenDim * sizeof(float), cudaMemcpyDeviceToDevice));
            } else {
                std::memcpy(xb.data(), x.data(), hiddenDim * sizeof(float));
            }
            if (useCuda) {
                rmsnormCUDA(d_x, d_x, reinterpret_cast<const float*>(weights.attention_norm), hiddenDim, config_.normEpsilon);
            } else {
                rmsnormCPU(x.data(), x.data(), reinterpret_cast<const float*>(weights.attention_norm), hiddenDim, config_.normEpsilon);
            }
            
#ifdef COTTUS_DEBUG_PARITY
            if (!useCuda && layer < 2) {
                std::cout << "\n[DEBUG] After pre-attn RMSNorm (layer " << layer << "):" << std::endl;
                std::cout << "  First 8 values: ";
                for (int i = 0; i < std::min(8, hiddenDim); ++i) {
                    std::cout << x[i] << " ";
                }
                std::cout << std::endl;
            }
#endif
            if (useCuda) {
                gemmCUDA(d_q, d_x, reinterpret_cast<const float*>(weights.wq), 1, numHeads * headDim, hiddenDim);
                gemmCUDA(d_k, d_x, reinterpret_cast<const float*>(weights.wk), 1, numKvHeads * headDim, hiddenDim);
                gemmCUDA(d_v, d_x, reinterpret_cast<const float*>(weights.wv), 1, numKvHeads * headDim, hiddenDim);
            } else {
                gemmCPU(q.data(), x.data(), reinterpret_cast<const float*>(weights.wq), 1, numHeads * headDim, hiddenDim);
                gemmCPU(k.data(), x.data(), reinterpret_cast<const float*>(weights.wk), 1, numKvHeads * headDim, hiddenDim);
                gemmCPU(v.data(), x.data(), reinterpret_cast<const float*>(weights.wv), 1, numKvHeads * headDim, hiddenDim);
            }
            
#ifdef COTTUS_DEBUG_PARITY
            if (!useCuda && layer < 2) {
                std::cout << "\n[DEBUG] After Q projection (layer " << layer << "):" << std::endl;
                std::cout << "  First 8 values: ";
                for (int i = 0; i < std::min(8, (int)(numHeads * headDim)); ++i) {
                    std::cout << q[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "\n[DEBUG] After K projection (layer " << layer << "):" << std::endl;
                std::cout << "  First 8 values: ";
                for (int i = 0; i < std::min(8, (int)(numKvHeads * headDim)); ++i) {
                    std::cout << k[i] << " ";
                }
                std::cout << std::endl;
            }
#endif
            if (useCuda) {
                ropeCUDA(d_q, d_q, pos, numHeads, headDim, config_.ropeTheta);
                ropeCUDA(d_k, d_k, pos, numKvHeads, headDim, config_.ropeTheta);
            } else {
                ropeCPU(q.data(), q.data(), pos, numHeads, headDim, config_.ropeTheta);
                ropeCPU(k.data(), k.data(), pos, numKvHeads, headDim, config_.ropeTheta);
            }
            
#ifdef COTTUS_DEBUG_PARITY
            if (!useCuda && layer < 2) {
                std::cout << "\n[DEBUG] After RoPE on Q (layer " << layer << ", pos=" << pos << "):" << std::endl;
                std::cout << "  First 8 values: ";
                for (int i = 0; i < std::min(8, (int)(numHeads * headDim)); ++i) {
                    std::cout << q[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "\n[DEBUG] After RoPE on K (layer " << layer << ", pos=" << pos << "):" << std::endl;
                std::cout << "  First 8 values: ";
                for (int i = 0; i < std::min(8, (int)(numKvHeads * headDim)); ++i) {
                    std::cout << k[i] << " ";
                }
                std::cout << std::endl;
            }
#endif
            int32_t logicalBlockIdx = pos / config_.blockSize;
            int32_t tokenInBlock = pos % config_.blockSize;
            
            if (logicalBlockIdx >= pageTable.numBlocks()) {
                throw std::out_of_range("Position exceeds page table");
            }
            
            int32_t physicalBlockId = pageTable[logicalBlockIdx];
            int32_t elementsPerLayerKV = config_.blockSize * numKvHeads * headDim;
            int32_t elementsPerBlock = 2 * elementsPerLayerKV * config_.numLayers;
            
            int32_t blockBase = physicalBlockId * elementsPerBlock;
            int32_t layerOffset = layer * 2 * elementsPerLayerKV;
            
#ifdef COTTUS_DEBUG_PARITY
            if (!useCuda && layer < 2) {
                std::cout << "\n[DEBUG] KV Cache Write (layer " << layer << ", pos=" << pos << "):" << std::endl;
                std::cout << "  physicalBlockId: " << physicalBlockId << std::endl;
                std::cout << "  blockBase: " << blockBase << std::endl;
                std::cout << "  layerOffset: " << layerOffset << std::endl;
                std::cout << "  K write base: " << (blockBase + layerOffset) << std::endl;
            }
#endif
            if (useCuda) {
                CUDA_CHECK(cudaMemcpy(k.data(), d_k, numKvHeads * headDim * sizeof(float), cudaMemcpyDeviceToHost));
            }
            
            uint16_t* kvCachePtr = reinterpret_cast<uint16_t*>(kvCacheBase);
            for (int32_t h = 0; h < numKvHeads; ++h) {
                for (int32_t d = 0; d < headDim; ++d) {
                    int32_t keyOffset = blockBase + layerOffset + 
                                       tokenInBlock * (numKvHeads * headDim) + 
                                       h * headDim + d;
                    float val = k[h * headDim + d];
                    uint32_t bits;
                    std::memcpy(&bits, &val, sizeof(float));
                    uint32_t sign = (bits & 0x80000000) >> 16;
                    int32_t exp = ((bits & 0x7F800000) >> 23) - 127 + 15;
                    uint32_t mant = (bits & 0x007FFFFF) >> 13;
                    if (exp <= 0) kvCachePtr[keyOffset] = static_cast<uint16_t>(sign);
                    else if (exp >= 31) kvCachePtr[keyOffset] = static_cast<uint16_t>(sign | 0x7C00);
                    else kvCachePtr[keyOffset] = static_cast<uint16_t>(sign | (exp << 10) | mant);
                }
            }
            if (useCuda) {
                CUDA_CHECK(cudaMemcpy(v.data(), d_v, numKvHeads * headDim * sizeof(float), cudaMemcpyDeviceToHost));
            }
            
            for (int32_t h = 0; h < numKvHeads; ++h) {
                for (int32_t d = 0; d < headDim; ++d) {
                    int32_t valueOffset = blockBase + layerOffset + elementsPerLayerKV +
                                         tokenInBlock * (numKvHeads * headDim) + 
                                         h * headDim + d;
                    float val = v[h * headDim + d];
                    uint32_t bits;
                    std::memcpy(&bits, &val, sizeof(float));
                    uint32_t sign = (bits & 0x80000000) >> 16;
                    int32_t exp = ((bits & 0x7F800000) >> 23) - 127 + 15;
                    uint32_t mant = (bits & 0x007FFFFF) >> 13;
                    if (exp <= 0) kvCachePtr[valueOffset] = static_cast<uint16_t>(sign);
                    else if (exp >= 31) kvCachePtr[valueOffset] = static_cast<uint16_t>(sign | 0x7C00);
                    else kvCachePtr[valueOffset] = static_cast<uint16_t>(sign | (exp << 10) | mant);
                }
            }
            if (useCuda) {
                CUDA_CHECK(cudaMemcpy(q.data(), d_q, numHeads * headDim * sizeof(float), cudaMemcpyDeviceToHost));
                pagedAttentionCUDA(d_att, d_q, reinterpret_cast<const void*>(kvCacheBase), pageTable, pos + 1, layer, numHeads, numKvHeads, headDim, config_.blockSize, config_.numLayers);
                CUDA_CHECK(cudaMemcpy(att.data(), d_att, numHeads * headDim * sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                pagedAttentionCPU(att.data(), q.data(), reinterpret_cast<const void*>(kvCacheBase), 
                                 pageTable, pos + 1, layer, numHeads, numKvHeads, headDim, config_.blockSize, config_.numLayers);
            }
            
#ifdef COTTUS_DEBUG_PARITY
            if (!useCuda && layer == 1) {
                std::cout << "\n[DEBUG] After PagedAttention (layer 1, pos=" << pos << "):" << std::endl;
                std::cout << "  First 8 values: ";
                for (int i = 0; i < std::min(8, (int)(numHeads * headDim)); ++i) {
                    std::cout << att[i] << " ";
                }
                std::cout << std::endl;
            }
#endif
            
            if (useCuda) {
                CUDA_CHECK(cudaMemcpy(d_att, att.data(), numHeads * headDim * sizeof(float), cudaMemcpyHostToDevice));
                gemmCUDA(d_x, d_att, reinterpret_cast<const float*>(weights.wo), 1, hiddenDim, numHeads * headDim);
            } else {
                gemmCPU(x.data(), att.data(), reinterpret_cast<const float*>(weights.wo), 1, hiddenDim, numHeads * headDim);
            }

#ifdef COTTUS_DEBUG_PARITY
            if (!useCuda && layer <= 1) {
                std::cout << "\n[DEBUG] After O_Proj (Layer " << layer << ", pos=" << pos << "):" << std::endl;
                std::cout << "  First 8 values: ";
                for (int i = 0; i < std::min(8, hiddenDim); ++i) std::cout << x[i] << " ";
                std::cout << std::endl;
            }
#endif
            if (useCuda)
            {
                residualAddCUDA(d_x, d_x, d_xb, hiddenDim);
            }
            else
            {
                residualAddCPU(x.data(), x.data(), xb.data(), hiddenDim);
            }
            if
            (useCuda)
            {
                CUDA_CHECK(cudaMemcpy(d_xb, d_x, hiddenDim * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            else
            {
                std::memcpy(xb.data(), x.data(), hiddenDim * sizeof(float));
            }
            if(useCuda)
            {
                rmsnormCUDA(d_x, d_x, reinterpret_cast<const float*>(weights.ffn_norm), hiddenDim, config_.normEpsilon);
            }
            
            else
            {
                rmsnormCPU(x.data(), x.data(), reinterpret_cast<const float*>(weights.ffn_norm), hiddenDim, config_.normEpsilon);
            }
            if (useCuda) {
                gemmCUDA(d_hb, d_x, reinterpret_cast<const float*>(weights.w1), 1, intermediateDim, hiddenDim);  
                gemmCUDA(d_hb2, d_x, reinterpret_cast<const float*>(weights.w3), 1, intermediateDim, hiddenDim); 
                siluCUDA(d_hb, d_hb, intermediateDim);
                elementwiseMultiplyCUDA(d_hb, d_hb, d_hb2, intermediateDim);
                gemmCUDA(d_x, d_hb, reinterpret_cast<const float*>(weights.w2), 1, hiddenDim, intermediateDim);
            } else {
                gemmCPU(hb.data(), x.data(), reinterpret_cast<const float*>(weights.w1), 1, intermediateDim, hiddenDim);
                gemmCPU(hb2.data(), x.data(),  reinterpret_cast<const float*>(weights.w3), 1, intermediateDim, hiddenDim);
                
                siluCPU(hb.data(), hb.data(), intermediateDim);
                
                elementwiseMultiplyCPU(hb.data(), hb.data(), hb2.data(), intermediateDim);
                
                gemmCPU(x.data(), hb.data(), reinterpret_cast<const float*>(weights.w2), 1, hiddenDim, intermediateDim);
            }
            if (useCuda) {
                residualAddCUDA(d_x, d_x, d_xb, hiddenDim);
            } else {
                residualAddCPU(x.data(), x.data(), xb.data(), hiddenDim);
            }
        }
        if (useCuda) {
            rmsnormCUDA(d_x, d_x, reinterpret_cast<const float*>(output_norm_), hiddenDim, config_.normEpsilon);
            CUDA_CHECK(cudaMemcpy(x.data(), d_x, hiddenDim * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            rmsnormCPU(x.data(), x.data(), reinterpret_cast<const float*>(output_norm_), hiddenDim, config_.normEpsilon);
        }
        std::vector<float> logits(config_.vocabSize);
        gemmCPU(logits.data(), x.data(), reinterpret_cast<const float*>(output_head_), 1, config_.vocabSize, hiddenDim);
        if (useCuda) {
            CUDA_CHECK(cudaFree(d_x));
            CUDA_CHECK(cudaFree(d_xb));
            CUDA_CHECK(cudaFree(d_q));
            CUDA_CHECK(cudaFree(d_k));
            CUDA_CHECK(cudaFree(d_v));
            CUDA_CHECK(cudaFree(d_att));
            CUDA_CHECK(cudaFree(d_hb));
            CUDA_CHECK(cudaFree(d_hb2));
        }
        
        return logits;
        
    } catch (...) {
        if (useCuda) {
            cudaFree(d_x);
            cudaFree(d_xb);
            cudaFree(d_q);
            cudaFree(d_k);
            cudaFree(d_v);
            cudaFree(d_att);
            cudaFree(d_hb);
            cudaFree(d_hb2);
        }
        throw;
    }
}

} // namespace cottus
