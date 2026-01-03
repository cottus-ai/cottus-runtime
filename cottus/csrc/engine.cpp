#include "engine.h"
#include "generic_transformer.h"
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace cottus {

Engine::Engine(const EngineConfig& config, const std::unordered_map<std::string, uintptr_t>& weightPtrs)
    : config_(config) {
    // Basic validation
    if (config.blockSize <= 0) throw std::invalid_argument("blockSize must be positive");
    if (config.maxSeqLen <= 0) throw std::invalid_argument("maxSeqLen must be positive");
    
    // Calculate required blocks
    int32_t totalBlocks = (config.maxSeqLen + config.blockSize - 1) / config.blockSize;

    // Initialize BlockAllocator
    blockAllocator_ = std::make_unique<BlockAllocator>(totalBlocks, config.blockSize);
    
    // Initialize Transformer
    transformer_ = std::make_unique<GenericTransformer>(config, weightPtrs);
    
    // Allocate KV cache
    // Layout: [numBlocks * elementsPerBlock] where elementsPerBlock = 2 * numLayers * blockSize * numKvHeads * headDim
    int32_t elementsPerLayerKV = config.blockSize * config.numKvHeads * config.headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * config.numLayers;
    kvCache_.resize(totalBlocks * elementsPerBlock, 0);
}

Engine::~Engine() = default;

std::vector<int32_t> Engine::forward(const std::vector<int32_t>& inputIds) {
    // Delegate to generate with 0 new tokens
    return generate(inputIds, 0);
}

std::vector<int32_t> Engine::generate(
    const std::vector<int32_t>& inputIds,
    int32_t maxNewTokens
) {
    if (inputIds.empty()) {
        throw std::invalid_argument("inputIds cannot be empty");
    }
    if (maxNewTokens < 0) {
        throw std::invalid_argument("maxNewTokens cannot be negative");
    }
    
    // Check context window
    if (inputIds.size() + maxNewTokens > static_cast<size_t>(config_.maxSeqLen)) {
        throw std::length_error("Total sequence length (prompt + gen) exceeds maxSeqLen");
    }
    
    // Create ephemeral PageTable for this request
    PageTable pageTable(config_.blockSize);
    std::vector<int32_t> generatedTokens;
    int32_t currentPos = 0;
    
    // Track allocated blocks for cleanup
    std::vector<int32_t> allocatedBlocks;
    
    try {
        // 1. Prefill: Process all input tokens
        for (size_t i = 0; i < inputIds.size(); ++i) {
            // Allocate new block at start of every chunk
            if (currentPos % config_.blockSize == 0) {
                int32_t blockId = blockAllocator_->allocateBlock();
                pageTable.appendBlock(blockId);
                allocatedBlocks.push_back(blockId);
            }
            
            // Run forward pass for this token
            std::vector<float> logits = transformer_->forwardToken(
                inputIds[i],
                currentPos,
                pageTable,
                reinterpret_cast<uintptr_t>(kvCache_.data()),
                config_.device
            );
            
            currentPos++;
        }
        
        // 2. Decode: Generate new tokens using greedy argmax
        int32_t lastToken = inputIds.back();
        
        for (int32_t step = 0; step < maxNewTokens; ++step) {
            // Allocate block if needed
            if (currentPos % config_.blockSize == 0) {
                int32_t blockId = blockAllocator_->allocateBlock();
                pageTable.appendBlock(blockId);
                allocatedBlocks.push_back(blockId);
            }
            
            // Run forward pass
            std::vector<float> logits = transformer_->forwardToken(
                lastToken,
                currentPos,
                pageTable,
                reinterpret_cast<uintptr_t>(kvCache_.data()),
                config_.device
            );
            
            // Greedy argmax
            int32_t nextToken = static_cast<int32_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin()
            );
            
            generatedTokens.push_back(nextToken);
            lastToken = nextToken;
            currentPos++;
            
            // Check for max sequence length
            if (currentPos >= config_.maxSeqLen) {
                break;
            }
        }
        
    } catch (...) {
        // Cleanup on exception
        for (int32_t blockId : allocatedBlocks) {
            try {
                blockAllocator_->freeBlock(blockId);
            } catch (...) {}
        }
        throw;
    }
    
    // 3. Cleanup: Free all allocated blocks
    for (int32_t blockId : allocatedBlocks) {
        blockAllocator_->freeBlock(blockId);
    }
    
    // Clear KV cache for next request
    std::fill(kvCache_.begin(), kvCache_.end(), static_cast<uint16_t>(0));
    
    return generatedTokens;
}

void Engine::reset() {
    // Clear KV cache
    std::fill(kvCache_.begin(), kvCache_.end(), static_cast<uint16_t>(0));
}

int32_t Engine::getFreeBlockCount() const {
    return blockAllocator_->numFreeBlocks();
}

} // namespace cottus
