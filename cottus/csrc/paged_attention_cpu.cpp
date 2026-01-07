#include "paged_attention_cpu.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <limits>

namespace cottus {
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    
    if (exp == 0) return 0.0f;
    exp = (exp - 15 + 127) << 23;
    mant = mant << 13;
    uint32_t bits = sign | exp | mant;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

void pagedAttentionCPU(
    float* output,
    const float* query,
    const void* kvCacheBase,
    const PageTable& pageTable,
    int32_t seqLen,
    int32_t layerIdx,
    int32_t numHeads,
    int32_t numKvHeads,
    int32_t headDim,
    int32_t blockSize,
    int32_t numLayers
) {
    if (seqLen <= 0) throw std::invalid_argument("seqLen must be positive");
    if (layerIdx < 0) throw std::invalid_argument("layerIdx must be non-negative");
    if (numHeads <= 0 || numKvHeads <= 0) throw std::invalid_argument("numHeads must be positive");
    if (headDim <= 0) throw std::invalid_argument("headDim must be positive");
    if (blockSize <= 0) throw std::invalid_argument("blockSize must be positive");
    
    const uint16_t* kvCache = static_cast<const uint16_t*>(kvCacheBase);
    
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * numLayers;
    for (int32_t qHead = 0; qHead < numHeads; ++qHead) {
        int32_t kvHead = (qHead * numKvHeads) / numHeads;
        std::vector<float> headOutput(headDim, 0.0f);
        float sumExp = 0.0f;
        std::vector<float> scores(seqLen);
        float maxScore = -std::numeric_limits<float>::infinity();
        for (int32_t tokenPos = 0; tokenPos < seqLen; ++tokenPos) {
            float qk = 0.0f;
            int32_t logicalBlockIdx = tokenPos / blockSize;
            int32_t tokenInBlock = tokenPos % blockSize;
            
            if (logicalBlockIdx >= pageTable.numBlocks()) {
                throw std::out_of_range("Token position exceeds page table");
            }
            int32_t physicalBlockId = pageTable[logicalBlockIdx];

            int32_t blockBase = physicalBlockId * elementsPerBlock;
            int32_t layerOffset = layerIdx * 2 * elementsPerLayerKV;
            int32_t keyOffset = blockBase + layerOffset + 
                               tokenInBlock * (numKvHeads * headDim) + 
                               kvHead * headDim;
            
            for (int32_t d = 0; d < headDim; ++d) {
                float qVal = query[qHead * headDim + d];
                float kVal = fp16_to_fp32(kvCache[keyOffset + d]);
                qk += qVal * kVal;
            }
            
            float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
            qk *= scale;
            
            scores[tokenPos] = qk;
            if (qk > maxScore) maxScore = qk;
        }
        for (int32_t tokenPos = 0; tokenPos < seqLen; ++tokenPos) {
            float qk = scores[tokenPos];
            float expQk = std::exp(qk - maxScore); //subtracting max
            sumExp += expQk;
            int32_t logicalBlockIdx = tokenPos / blockSize;
            int32_t tokenInBlock = tokenPos % blockSize;
            int32_t physicalBlockId = pageTable[logicalBlockIdx];
            int32_t blockBase = physicalBlockId * elementsPerBlock;
            int32_t layerOffset = layerIdx * 2 * elementsPerLayerKV;
            int32_t valueOffset = blockBase + layerOffset + elementsPerLayerKV +
                                 tokenInBlock * (numKvHeads * headDim) + 
                                 kvHead * headDim;
            
            for (int32_t d = 0; d < headDim; ++d) {
                float v = fp16_to_fp32(kvCache[valueOffset + d]);
                headOutput[d] += expQk * v;
            }
        }
        for (int32_t d = 0; d < headDim; ++d) {
            output[qHead * headDim + d] = headOutput[d] / sumExp;
        }
    }
}

} // namespace cottus
