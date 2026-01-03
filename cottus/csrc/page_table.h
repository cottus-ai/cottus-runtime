#pragma once

#include <cstdint>
#include <vector>

namespace cottus {

// PageTable maps logical token positions to physical KV cache block IDs.
// It is ephemeral and scoped to a single generate() call.
//
// Invariants:
// - Append-only growth (no removal or modification)
// - Bounds-checked access (out-of-range throws)
// - Does NOT allocate physical blocks (only stores IDs)
// - Lifetime must not outlive the blocks it references
class PageTable {
public:
    // Construct empty page table.
    // blockSize: tokens per block (for validation/metadata)
    // Throws std::invalid_argument if blockSize is non-positive.
    explicit PageTable(int32_t blockSize);

    // Append a block ID to the end of the table.
    // blockId: physical block ID from BlockAllocator
    // This extends the logical address space by one block.
    void appendBlock(int32_t blockId);

    // Get the physical block ID for a given logical index.
    // logicalIndex: index into the table [0, numBlocks())
    // Returns the physical block ID.
    // Throws std::out_of_range if index is invalid.
    int32_t getBlock(int32_t logicalIndex) const;

    // Operator overload for array-style access.
    // Equivalent to getBlock(logicalIndex).
    int32_t operator[](int32_t logicalIndex) const;

    // Query number of blocks currently in the table.
    int32_t numBlocks() const;

    // Query tokens per block.
    int32_t blockSize() const;

private:
    std::vector<int32_t> logicalToPhysical_;  // Maps logical index to block ID
    int32_t blockSize_;                        // Tokens per block
};

}  // namespace cottus
