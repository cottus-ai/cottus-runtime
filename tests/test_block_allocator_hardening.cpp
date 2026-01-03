#include "../cottus/csrc/block_allocator.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace cottus;

// Existing basic tests...
// (Assuming these are preserved/referenced from original test file, 
//  but for this hardening file we focus on the new ones)

// 1. Interleaved allocation/free patterns
void testInterleavedOps() {
    BlockAllocator allocator(10, 16);
    
    // Allocate 5
    std::vector<int32_t> batch1;
    for (int i = 0; i < 5; ++i) batch1.push_back(allocator.allocateBlock());
    
    // Free 2 and 4 (indices 1 and 3 in batch)
    allocator.freeBlock(batch1[1]);
    allocator.freeBlock(batch1[3]);
    
    // Allocate 2 more. Should likely get the freed ones back (LIFO/stack behavior).
    int32_t new1 = allocator.allocateBlock();
    int32_t new2 = allocator.allocateBlock();
    
    // Verify strict reuse or at least validity
    assert(new1 >= 0 && new1 < 10);
    assert(new2 >= 0 && new2 < 10);
    
    // Verify no duplicates with currently held blocks
    assert(new1 != batch1[0]);
    assert(new1 != batch1[2]);
    assert(new1 != batch1[4]);
    
    assert(new2 != batch1[0]);
    assert(new2 != batch1[2]);
    assert(new2 != batch1[4]);
    
    std::cout << "PASS: testInterleavedOps" << std::endl;
}

// 2. Full permutation test
void testPermutationRobustness() {
    int N = 100;
    BlockAllocator allocator(N, 16);
    
    // Allocate all
    std::vector<int32_t> blocks;
    for (int i = 0; i < N; ++i) blocks.push_back(allocator.allocateBlock());
    
    // Verify unique
    std::vector<int32_t> sorted = blocks;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < N; ++i) assert(sorted[i] == i); // Should be 0..N-1
    
    // Shuffle free order
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(blocks.begin(), blocks.end(), g);
    
    // Free all in random order
    for (int32_t id : blocks) allocator.freeBlock(id);
    
    assert(allocator.numFreeBlocks() == N);
    
    // Reallocate all
    std::vector<int32_t> round2;
    for (int i = 0; i < N; ++i) round2.push_back(allocator.allocateBlock());
    
    // Verify all present again
    std::sort(round2.begin(), round2.end());
    for (int i = 0; i < N; ++i) assert(round2[i] == i);
    
    std::cout << "PASS: testPermutationRobustness" << std::endl;
}

// 3. Destructor safety (Sanity check)
void testDestructorSafety() {
    {
        BlockAllocator allocator(10, 16);
        allocator.allocateBlock();
        // Destructor called here with allocated blocks
    }
    std::cout << "PASS: testDestructorSafety" << std::endl;
}

int main() {
    testInterleavedOps();
    testPermutationRobustness();
    testDestructorSafety();
    return 0;
}
