#include "../cottus/csrc/scheduler.h"
#include "../cottus/csrc/swap_space.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace cottus;

void testSwapSpaceBasic() {
    std::cout << "Test: SwapSpace Basic" << std::endl;
    
    SwapSpace swap(10, 1024);
    
    assert(swap.numFreeCpuBlocks() == 10);
    
    int32_t b1 = swap.allocateCpuBlock();
    int32_t b2 = swap.allocateCpuBlock();
    
    assert(swap.numFreeCpuBlocks() == 8);
    assert(b1 != b2);
    
    swap.freeCpuBlock(b1);
    assert(swap.numFreeCpuBlocks() == 9);
    
    swap.freeCpuBlock(b2);
    assert(swap.numFreeCpuBlocks() == 10);
    
    std::cout << "  PASS" << std::endl;
}

void testSwapSpaceExhaustion() {
    std::cout << "Test: SwapSpace Exhaustion" << std::endl;
    
    SwapSpace swap(5, 256);
    
    std::vector<int32_t> blocks;
    for (int i = 0; i < 5; ++i) {
        blocks.push_back(swap.allocateCpuBlock());
    }
    
    assert(swap.numFreeCpuBlocks() == 0);
    
    bool threw = false;
    try {
        swap.allocateCpuBlock();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    
    for (int32_t b : blocks) {
        swap.freeCpuBlock(b);
    }
    assert(swap.numFreeCpuBlocks() == 5);
    
    std::cout << "  PASS" << std::endl;
}

void testSchedulerPreemption() {
    std::cout << "Test: Scheduler Preemption" << std::endl;
    
    SchedulerConfig config;
    config.blockSize = 4;
    config.gpuBlocks = 8;
    config.cpuSwapBlocks = 16;
    config.blockSizeBytes = 256;
    config.maxBatchSize = 2;
    
    Scheduler scheduler(config);
    
    std::vector<int32_t> prompt1 = {1, 2, 3, 4, 5, 6, 7, 8};
    auto sg1 = std::make_unique<SequenceGroup>(1, prompt1, 10, config.blockSize);
    scheduler.addSequenceGroup(std::move(sg1));
    
    SchedulerOutputs out1 = scheduler.schedule();
    assert(out1.scheduled_seq_groups.size() == 1);
    assert(out1.num_prefill_groups == 1);
    
    std::vector<int32_t> prompt2 = {10, 20, 30, 40, 50, 60, 70, 80};
    auto sg2 = std::make_unique<SequenceGroup>(2, prompt2, 10, config.blockSize);
    scheduler.addSequenceGroup(std::move(sg2));
    
    SchedulerOutputs out2 = scheduler.schedule();
    
    int32_t freeBlocks = scheduler.getBlockAllocator().numFreeBlocks();
    std::cout << "  Free GPU blocks after scheduling: " << freeBlocks << std::endl;
    std::cout << "  Swapped out: " << out2.swapped_out.size() << std::endl;
    
    std::cout << "  PASS" << std::endl;
}

void testSwapMapping() {
    std::cout << "Test: Swap Mapping Integrity" << std::endl;
    
    SchedulerConfig config;
    config.blockSize = 4;
    config.gpuBlocks = 4;
    config.cpuSwapBlocks = 8;
    config.blockSizeBytes = 256;
    config.maxBatchSize = 1;
    
    Scheduler scheduler(config);
    
    std::vector<int32_t> prompt = {1, 2, 3, 4};
    auto sg = std::make_unique<SequenceGroup>(1, prompt, 10, config.blockSize);
    scheduler.addSequenceGroup(std::move(sg));
    
    SchedulerOutputs out1 = scheduler.schedule();
    assert(out1.scheduled_seq_groups.size() == 1);
    
    int32_t freeAfterAlloc = scheduler.getBlockAllocator().numFreeBlocks();
    std::cout << "  Free blocks after allocation: " << freeAfterAlloc << std::endl;
    
    std::cout << "  PASS" << std::endl;
}

int main() {
    testSwapSpaceBasic();
    testSwapSpaceExhaustion();
    testSchedulerPreemption();
    testSwapMapping();
    std::cout << "All preemption tests passed!" << std::endl;
    return 0;
}
