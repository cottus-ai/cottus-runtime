#include "swap_space.h"
#include <stdexcept>
#include <cstring>

namespace cottus {

SwapSpace::SwapSpace(int32_t maxBlocks, int32_t blockSizeBytes)
    : maxBlocks_(maxBlocks), blockSizeBytes_(blockSizeBytes) {
    if (maxBlocks <= 0) throw std::invalid_argument("maxBlocks must be positive");
    if (blockSizeBytes <= 0) throw std::invalid_argument("blockSizeBytes must be positive");

    cpuMemory_.resize(static_cast<size_t>(maxBlocks) * blockSizeBytes);
    allocated_.resize(maxBlocks, false);
    freeList_.reserve(maxBlocks);
    for (int32_t i = maxBlocks - 1; i >= 0; --i) {
        freeList_.push_back(i);
    }
}

SwapSpace::~SwapSpace() {}

int32_t SwapSpace::allocateCpuBlock() {
    if (freeList_.empty()) {
        throw std::runtime_error("SwapSpace: no free CPU blocks");
    }
    int32_t blockId = freeList_.back();
    freeList_.pop_back();
    allocated_[blockId] = true;
    return blockId;
}

void SwapSpace::freeCpuBlock(int32_t cpuBlockId) {
    if (cpuBlockId < 0 || cpuBlockId >= maxBlocks_) {
        throw std::invalid_argument("Invalid CPU block ID");
    }
    if (!allocated_[cpuBlockId]) {
        throw std::runtime_error("Double free on CPU block");
    }
    allocated_[cpuBlockId] = false;
    freeList_.push_back(cpuBlockId);
}

int32_t SwapSpace::numFreeCpuBlocks() const {
    return static_cast<int32_t>(freeList_.size());
}

void* SwapSpace::getCpuBlockPtr(int32_t cpuBlockId) const {
    if (cpuBlockId < 0 || cpuBlockId >= maxBlocks_) {
        throw std::invalid_argument("Invalid CPU block ID");
    }
    return const_cast<uint8_t*>(cpuMemory_.data() + static_cast<size_t>(cpuBlockId) * blockSizeBytes_);
}

void SwapSpace::swapOut(int32_t gpuBlockId, int32_t cpuBlockId,
                        void* gpuKvCache, int32_t blockSizeBytes, cudaStream_t stream) {
    if (!gpuKvCache) {
        throw std::invalid_argument("gpuKvCache is null");
    }
    void* cpuPtr = getCpuBlockPtr(cpuBlockId);
    uint8_t* gpuPtr = static_cast<uint8_t*>(gpuKvCache) + static_cast<size_t>(gpuBlockId) * blockSizeBytes;
    
    cudaError_t err = cudaMemcpyAsync(cpuPtr, gpuPtr, blockSizeBytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync DtoH failed");
    }
}

void SwapSpace::swapIn(int32_t cpuBlockId, int32_t gpuBlockId,
                       void* gpuKvCache, int32_t blockSizeBytes, cudaStream_t stream) {
    if (!gpuKvCache) {
        throw std::invalid_argument("gpuKvCache is null");
    }
    void* cpuPtr = getCpuBlockPtr(cpuBlockId);
    uint8_t* gpuPtr = static_cast<uint8_t*>(gpuKvCache) + static_cast<size_t>(gpuBlockId) * blockSizeBytes;
    
    cudaError_t err = cudaMemcpyAsync(gpuPtr, cpuPtr, blockSizeBytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync HtoD failed");
    }
}

}
