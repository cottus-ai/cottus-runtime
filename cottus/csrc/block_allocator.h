#pragma once

#include <cstdint>
#include <vector>

namespace cottus {
class BlockAllocator {
public:
    BlockAllocator(int32_t totalBlocks, int32_t blockSize);
    int32_t allocateBlock();
    void freeBlock(int32_t blockId);
    int32_t numFreeBlocks() const;
    int32_t totalBlocks() const;
    int32_t blockSize() const;

private:
    int32_t totalBlocks_;
    int32_t blockSize_;
    std::vector<int32_t> freeList_;     
    std::vector<bool> allocated_;
};

}  // namespace cottus
