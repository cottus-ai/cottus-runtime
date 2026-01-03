#include "block_allocator.h"
#include <stdexcept>
#include <string>

namespace cottus {

BlockAllocator::BlockAllocator(int32_t totalBlocks, int32_t blockSize)
    : totalBlocks_(totalBlocks), blockSize_(blockSize) {
    if (totalBlocks <= 0) {
        throw std::invalid_argument("totalBlocks must be positive");
    }
    if (blockSize <= 0) {
        throw std::invalid_argument("blockSize must be positive");
    }

    // Initialize free list with all block IDs in reverse order
    // (so block 0 is allocated first when popping from back)
    freeList_.reserve(totalBlocks);
    for (int32_t i = totalBlocks - 1; i >= 0; --i) {
        freeList_.push_back(i);
    }

    // Initialize allocation tracker (all blocks start free)
    allocated_.resize(totalBlocks, false);
}

int32_t BlockAllocator::allocateBlock() {
    if (freeList_.empty()) {
        throw std::runtime_error("Out of memory: no free blocks");
    }

    // Pop from free list (stack-based LIFO)
    int32_t blockId = freeList_.back();
    freeList_.pop_back();

    // Mark as allocated
    allocated_[blockId] = true;

    return blockId;
}

void BlockAllocator::freeBlock(int32_t blockId) {
    // Validate block ID is in range
    if (blockId < 0 || blockId >= totalBlocks_) {
        throw std::invalid_argument("Invalid block ID: " + std::to_string(blockId));
    }

    // Detect double-free
    if (!allocated_[blockId]) {
        throw std::runtime_error("Double free detected for block " + std::to_string(blockId));
    }

    // Mark as free and return to free list
    allocated_[blockId] = false;
    freeList_.push_back(blockId);
}

int32_t BlockAllocator::numFreeBlocks() const {
    return static_cast<int32_t>(freeList_.size());
}

int32_t BlockAllocator::totalBlocks() const {
    return totalBlocks_;
}

int32_t BlockAllocator::blockSize() const {
    return blockSize_;
}

}  // namespace cottus
