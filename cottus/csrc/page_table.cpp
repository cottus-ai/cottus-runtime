#include "page_table.h"
#include <stdexcept>
#include <string>

namespace cottus {

PageTable::PageTable(int32_t blockSize) : blockSize_(blockSize) {
    if (blockSize <= 0) {
        throw std::invalid_argument("blockSize must be positive");
    }
    // Start with empty table (no blocks allocated yet)
}

void PageTable::appendBlock(int32_t blockId) {
    // Append block ID to the end of the mapping
    // No validation of blockId itself (trust BlockAllocator)
    logicalToPhysical_.push_back(blockId);
}

int32_t PageTable::getBlock(int32_t logicalIndex) const {
    // Validate index is in range
    if (logicalIndex < 0 || logicalIndex >= static_cast<int32_t>(logicalToPhysical_.size())) {
        throw std::out_of_range("Logical index " + std::to_string(logicalIndex) + 
                                " out of range [0, " + std::to_string(logicalToPhysical_.size()) + ")");
    }
    return logicalToPhysical_[logicalIndex];
}

int32_t PageTable::operator[](int32_t logicalIndex) const {
    return getBlock(logicalIndex);
}

int32_t PageTable::numBlocks() const {
    return static_cast<int32_t>(logicalToPhysical_.size());
}

int32_t PageTable::blockSize() const {
    return blockSize_;
}

}  // namespace cottus
