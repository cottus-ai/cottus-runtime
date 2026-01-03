#include "../cottus/csrc/block_allocator.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace cottus;

// Test: Normal allocation and free
void testNormalAllocationAndFree() {
    BlockAllocator allocator(10, 16);
    
    assert(allocator.totalBlocks() == 10);
    assert(allocator.blockSize() == 16);
    assert(allocator.numFreeBlocks() == 10);
    
    // Allocate a block
    int32_t block1 = allocator.allocateBlock();
    assert(block1 >= 0 && block1 < 10);
    assert(allocator.numFreeBlocks() == 9);
    
    // Allocate another
    int32_t block2 = allocator.allocateBlock();
    assert(block2 >= 0 && block2 < 10);
    assert(block2 != block1);
    assert(allocator.numFreeBlocks() == 8);
    
    // Free first block
    allocator.freeBlock(block1);
    assert(allocator.numFreeBlocks() == 9);
    
    // Free second block
    allocator.freeBlock(block2);
    assert(allocator.numFreeBlocks() == 10);
    
    std::cout << "PASS: testNormalAllocationAndFree" << std::endl;
}

// Test: Allocator exhaustion
void testAllocatorExhaustion() {
    BlockAllocator allocator(3, 16);
    
    // Allocate all blocks
    int32_t b1 = allocator.allocateBlock();
    int32_t b2 = allocator.allocateBlock();
    int32_t b3 = allocator.allocateBlock();
    
    assert(allocator.numFreeBlocks() == 0);
    
    // Attempt to allocate when exhausted
    bool caughtException = false;
    try {
        allocator.allocateBlock();
    } catch (const std::runtime_error& e) {
        caughtException = true;
        std::string msg(e.what());
        assert(msg.find("Out of memory") != std::string::npos);
    }
    assert(caughtException);
    
    std::cout << "PASS: testAllocatorExhaustion" << std::endl;
}

// Test: Double free detection
void testDoubleFree() {
    BlockAllocator allocator(5, 16);
    
    int32_t block = allocator.allocateBlock();
    allocator.freeBlock(block);
    
    // Attempt double free
    bool caughtException = false;
    try {
        allocator.freeBlock(block);
    } catch (const std::runtime_error& e) {
        caughtException = true;
        std::string msg(e.what());
        assert(msg.find("Double free") != std::string::npos);
    }
    assert(caughtException);
    
    std::cout << "PASS: testDoubleFree" << std::endl;
}

// Test: Invalid block ID
void testInvalidBlockId() {
    BlockAllocator allocator(5, 16);
    
    // Test negative ID
    bool caughtNegative = false;
    try {
        allocator.freeBlock(-1);
    } catch (const std::invalid_argument& e) {
        caughtNegative = true;
        std::string msg(e.what());
        assert(msg.find("Invalid block ID") != std::string::npos);
    }
    assert(caughtNegative);
    
    // Test out-of-range ID
    bool caughtOutOfRange = false;
    try {
        allocator.freeBlock(100);
    } catch (const std::invalid_argument& e) {
        caughtOutOfRange = true;
        std::string msg(e.what());
        assert(msg.find("Invalid block ID") != std::string::npos);
    }
    assert(caughtOutOfRange);
    
    std::cout << "PASS: testInvalidBlockId" << std::endl;
}

// Test: Invalid constructor arguments
void testInvalidConstructorArgs() {
    // Test zero totalBlocks
    bool caughtZeroBlocks = false;
    try {
        BlockAllocator allocator(0, 16);
    } catch (const std::invalid_argument& e) {
        caughtZeroBlocks = true;
        std::string msg(e.what());
        assert(msg.find("totalBlocks must be positive") != std::string::npos);
    }
    assert(caughtZeroBlocks);
    
    // Test negative blockSize
    bool caughtNegativeSize = false;
    try {
        BlockAllocator allocator(10, -1);
    } catch (const std::invalid_argument& e) {
        caughtNegativeSize = true;
        std::string msg(e.what());
        assert(msg.find("blockSize must be positive") != std::string::npos);
    }
    assert(caughtNegativeSize);
    
    std::cout << "PASS: testInvalidConstructorArgs" << std::endl;
}

// Test: Free list integrity (no silent reuse)
void testFreeListIntegrity() {
    BlockAllocator allocator(5, 16);
    
    // Allocate all blocks
    std::vector<int32_t> blocks;
    for (int i = 0; i < 5; ++i) {
        blocks.push_back(allocator.allocateBlock());
    }
    
    // Verify all IDs are unique
    for (size_t i = 0; i < blocks.size(); ++i) {
        for (size_t j = i + 1; j < blocks.size(); ++j) {
            assert(blocks[i] != blocks[j]);
        }
    }
    
    // Free all blocks
    for (int32_t block : blocks) {
        allocator.freeBlock(block);
    }
    
    // Allocate again and verify no duplicates
    std::vector<int32_t> blocks2;
    for (int i = 0; i < 5; ++i) {
        blocks2.push_back(allocator.allocateBlock());
    }
    
    for (size_t i = 0; i < blocks2.size(); ++i) {
        for (size_t j = i + 1; j < blocks2.size(); ++j) {
            assert(blocks2[i] != blocks2[j]);
        }
    }
    
    std::cout << "PASS: testFreeListIntegrity" << std::endl;
}

int main() {
    testNormalAllocationAndFree();
    testAllocatorExhaustion();
    testDoubleFree();
    testInvalidBlockId();
    testInvalidConstructorArgs();
    testFreeListIntegrity();
    
    std::cout << "\nAll BlockAllocator tests passed!" << std::endl;
    return 0;
}
