#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

namespace cottus {

class SwapSpace {
public:
    SwapSpace(int32_t maxBlocks, int32_t blockSizeBytes);
    ~SwapSpace();

    int32_t allocateCpuBlock();
    void freeCpuBlock(int32_t cpuBlockId);
    int32_t numFreeCpuBlocks() const;

    void swapOut(int32_t gpuBlockId, int32_t cpuBlockId, 
                 void* gpuKvCache, int32_t blockSizeBytes, cudaStream_t stream = 0);
    void swapIn(int32_t cpuBlockId, int32_t gpuBlockId,
                void* gpuKvCache, int32_t blockSizeBytes, cudaStream_t stream = 0);

    void* getCpuBlockPtr(int32_t cpuBlockId) const;

private:
    int32_t maxBlocks_;
    int32_t blockSizeBytes_;
    std::vector<uint8_t> cpuMemory_;
    std::vector<int32_t> freeList_;
    std::vector<bool> allocated_;
};

}
