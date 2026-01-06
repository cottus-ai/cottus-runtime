#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include "block_allocator.h"
#include "page_table.h"

namespace cottus {
struct EngineConfig
{
    int32_t vocabSize;
    int32_t hiddenDim;
    int32_t numLayers;
    int32_t numHeads;
    int32_t numKvHeads;
    int32_t headDim;
    int32_t intermediateDim;
    int32_t maxSeqLen;
    int32_t blockSize;
    float ropeTheta;
    float normEpsilon;
    std::string device;
    std::string dtype;
};
class Engine
{
public:
    Engine(const EngineConfig& config, const std::unordered_map<std::string, uintptr_t>& weightPtrs);
    ~Engine();
    std::vector<int32_t> forward(const std::vector<int32_t>& inputIds);
    std::vector<int32_t> generate(
        const std::vector<int32_t>& inputIds,
        int32_t maxNewTokens
    );
    void reset();
    int32_t getFreeBlockCount() const;

private:
    EngineConfig config_;
    std::unique_ptr<BlockAllocator> blockAllocator_;
    std::unique_ptr<class GenericTransformer> transformer_;
    std::vector<uint16_t> kvCache_;
};

} // namespace cottus
