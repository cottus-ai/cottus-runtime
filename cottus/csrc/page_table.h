#pragma once

#include <cstdint>
#include <vector>

namespace cottus {
class PageTable {
public:
    explicit PageTable(int32_t blockSize);
    void appendBlock(int32_t blockId);
    int32_t getBlock(int32_t logicalIndex) const;
    int32_t operator[](int32_t logicalIndex) const;
    int32_t numBlocks() const;
    int32_t blockSize() const;

private:
    std::vector<int32_t> logicalToPhysical_;
    int32_t blockSize_;                        
};

}  // namespace cottus
