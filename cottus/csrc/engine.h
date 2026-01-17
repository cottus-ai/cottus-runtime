#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include "block_allocator.h"
#include "logits_processor.h"
#include "request.h"

namespace cottus {
class Scheduler; // Forward declaration

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
    int32_t eosTokenId = 2;
    std::vector<int32_t> eosTokenIds;
    float temperature = 0.0f;
    int32_t topK = 50;
    float repetitionPenalty = 1.1f;
    float frequencyPenalty = 0.0f;
    int32_t noRepeatNGramSize = 0;
    std::vector<std::vector<int32_t>> stopTokenSequences;
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

    void addRequest(std::unique_ptr<Request> req);
    void step();
    // Helper to get finished requests from Scheduler
    std::vector<std::unique_ptr<Request>> pullFinishedRequests();
    bool hasActiveRequests() const;
    
    void reset();
    int32_t getFreeBlockCount() const;

private:
    EngineConfig config_;
    std::unique_ptr<class Scheduler> scheduler_;
    std::unique_ptr<class GenericTransformer> transformer_;
    
    // Scheduler owns request state now. 
    // Engine only keeps scratch buffers for execution.
    std::vector<uint16_t> kvCache_;
    void* d_kvCache_ = nullptr;
    int nextRequestId_ = 1;

    std::vector<int32_t> scratchInputIds_;
    std::vector<int32_t> scratchPositions_;
    std::vector<float> scratchLogits_;
    std::vector<int32_t> scratchSlotIndices_;
};

} // namespace cottus
