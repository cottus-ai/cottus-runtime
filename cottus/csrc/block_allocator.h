#pragma once

#include <cstdint>
#include <vector>

namespace cottus {

// BlockAllocator manages a fixed pool of KV cache blocks.
// Blocks are identified by integer IDs in range [0, totalBlocks).
// Uses a simple stack-based free list for allocation.
//
// Invariants:
// - Total capacity is fixed at construction
// - No dynamic resizing
// - No silent failures (all errors throw)
// - No double-free allowed
class BlockAllocator {
public:
    // Construct allocator with fixed capacity.
    // totalBlocks: number of blocks in the pool
    // blockSize: number of tokens per block
    // Throws std::invalid_argument if either parameter is non-positive.
    BlockAllocator(int32_t totalBlocks, int32_t blockSize);

    // Allocate a single block from the free list.
    // Returns the block ID.
    // Throws std::runtime_error if no free blocks available.
    int32_t allocateBlock();

    // Free a previously allocated block.
    // blockId: the block to return to free list
    // Throws std::invalid_argument if blockId is out of range.
    // Throws std::runtime_error if block is already free (double-free).
    void freeBlock(int32_t blockId);

    // Query number of currently free blocks.
    int32_t numFreeBlocks() const;

    // Query total capacity.
    int32_t totalBlocks() const;

    // Query tokens per block.
    int32_t blockSize() const;

private:
    int32_t totalBlocks_;
    int32_t blockSize_;
    std::vector<int32_t> freeList_;  // Stack of free block IDs
    std::vector<bool> allocated_;    // Track allocation status for double-free detection
};

}  // namespace cottus
