#pragma once

#include <deque>
#include <vector>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include "sequence.h"
#include "block_allocator.h"
#include "swap_space.h"

namespace cottus {

struct SchedulerConfig {
    int32_t maxBatchSize = 64;
    int32_t maxPrefillTokens = 4096;
    int32_t blockSize = 16;
    int32_t gpuBlocks = 4096;
    int32_t cpuSwapBlocks = 1024;
    int32_t blockSizeBytes = 0;
};

struct SchedulerOutputs {
    std::vector<SequenceGroup*> scheduled_seq_groups;
    std::vector<SequenceGroup*> preempted;
    std::vector<SequenceGroup*> swapped_in;
    std::vector<SequenceGroup*> swapped_out;
    int32_t num_prefill_groups = 0;
    int32_t num_decode_groups = 0;
};

struct SwapMapping {
    std::unordered_map<int32_t, int32_t> gpu_to_cpu;
};

class Scheduler {
public:
    Scheduler(const SchedulerConfig& config);

    void addSequenceGroup(std::unique_ptr<SequenceGroup> seq_group);
    SchedulerOutputs schedule();
    std::vector<std::unique_ptr<SequenceGroup>> getFinishedGroups();
    bool hasPendingWork() const;
    int32_t getNumUnfinishedSeqs() const;

    BlockAllocator& getBlockAllocator() { return *gpu_allocator_; }
    SwapSpace& getSwapSpace() { return *swap_space_; }

    void executeSwapOut(SequenceGroup& sg, void* gpuKvCache);
    void executeSwapIn(SequenceGroup& sg, void* gpuKvCache);

private:
    SchedulerConfig config_;
    std::unique_ptr<BlockAllocator> gpu_allocator_;
    std::unique_ptr<SwapSpace> swap_space_;

    std::deque<std::unique_ptr<SequenceGroup>> waiting_;
    std::deque<std::unique_ptr<SequenceGroup>> running_;
    std::deque<std::unique_ptr<SequenceGroup>> swapped_;
    std::vector<std::unique_ptr<SequenceGroup>> finished_;

    std::unordered_map<int32_t, SwapMapping> swap_mappings_;

    bool canAllocate(const SequenceGroup& seq_group) const;
    void allocateBlocks(SequenceGroup& seq_group);
    void freeBlocks(SequenceGroup& seq_group);
    void freeSwapBlocks(int32_t request_id);
};

}
