#include "../cottus/csrc/page_table.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace cottus;

// Test: Empty PageTable behavior
void testEmptyPageTable() {
    PageTable table(16);
    
    assert(table.blockSize() == 16);
    assert(table.numBlocks() == 0);
    
    // Accessing empty table should throw
    bool caughtException = false;
    try {
        table.getBlock(0);
    } catch (const std::out_of_range& e) {
        caughtException = true;
        std::string msg(e.what());
        assert(msg.find("out of range") != std::string::npos);
    }
    assert(caughtException);
    
    std::cout << "PASS: testEmptyPageTable" << std::endl;
}

// Test: Append and access behavior
void testAppendAndAccess() {
    PageTable table(16);
    
    // Append some blocks
    table.appendBlock(5);
    table.appendBlock(3);
    table.appendBlock(7);
    
    assert(table.numBlocks() == 3);
    
    // Access via getBlock
    assert(table.getBlock(0) == 5);
    assert(table.getBlock(1) == 3);
    assert(table.getBlock(2) == 7);
    
    // Access via operator[]
    assert(table[0] == 5);
    assert(table[1] == 3);
    assert(table[2] == 7);
    
    std::cout << "PASS: testAppendAndAccess" << std::endl;
}

// Test: Bounds checking
void testBoundsChecking() {
    PageTable table(16);
    table.appendBlock(10);
    table.appendBlock(20);
    
    // Test negative index
    bool caughtNegative = false;
    try {
        table.getBlock(-1);
    } catch (const std::out_of_range& e) {
        caughtNegative = true;
        std::string msg(e.what());
        assert(msg.find("out of range") != std::string::npos);
    }
    assert(caughtNegative);
    
    // Test index at boundary (valid)
    assert(table.getBlock(1) == 20);
    
    // Test index beyond boundary
    bool caughtBeyond = false;
    try {
        table.getBlock(2);
    } catch (const std::out_of_range& e) {
        caughtBeyond = true;
        std::string msg(e.what());
        assert(msg.find("out of range") != std::string::npos);
    }
    assert(caughtBeyond);
    
    // Test far out of range
    bool caughtFar = false;
    try {
        table.getBlock(100);
    } catch (const std::out_of_range& e) {
        caughtFar = true;
    }
    assert(caughtFar);
    
    std::cout << "PASS: testBoundsChecking" << std::endl;
}

// Test: Sequential growth correctness
void testSequentialGrowth() {
    PageTable table(16);
    
    // Append blocks in sequence
    for (int32_t i = 0; i < 10; ++i) {
        table.appendBlock(i * 100);
        assert(table.numBlocks() == i + 1);
    }
    
    // Verify all blocks are accessible and correct
    for (int32_t i = 0; i < 10; ++i) {
        assert(table[i] == i * 100);
    }
    
    std::cout << "PASS: testSequentialGrowth" << std::endl;
}

// Test: No reuse of indices (negative confirmation)
void testNoIndexReuse() {
    PageTable table(16);
    
    // Append blocks
    table.appendBlock(1);
    table.appendBlock(2);
    table.appendBlock(3);
    
    // Verify indices are stable
    int32_t block0 = table[0];
    int32_t block1 = table[1];
    int32_t block2 = table[2];
    
    // Append more blocks
    table.appendBlock(4);
    table.appendBlock(5);
    
    // Original indices must still return same values
    assert(table[0] == block0);
    assert(table[1] == block1);
    assert(table[2] == block2);
    
    // New indices are accessible
    assert(table[3] == 4);
    assert(table[4] == 5);
    
    std::cout << "PASS: testNoIndexReuse" << std::endl;
}

// Test: Invalid constructor arguments
void testInvalidConstructorArgs() {
    // Test zero blockSize
    bool caughtZero = false;
    try {
        PageTable table(0);
    } catch (const std::invalid_argument& e) {
        caughtZero = true;
        std::string msg(e.what());
        assert(msg.find("blockSize must be positive") != std::string::npos);
    }
    assert(caughtZero);
    
    // Test negative blockSize
    bool caughtNegative = false;
    try {
        PageTable table(-5);
    } catch (const std::invalid_argument& e) {
        caughtNegative = true;
        std::string msg(e.what());
        assert(msg.find("blockSize must be positive") != std::string::npos);
    }
    assert(caughtNegative);
    
    std::cout << "PASS: testInvalidConstructorArgs" << std::endl;
}

// Test: No mutation after append (negative confirmation)
void testNoMutationAfterAppend() {
    PageTable table(16);
    
    table.appendBlock(42);
    
    // Access multiple times, verify value is stable
    assert(table[0] == 42);
    assert(table[0] == 42);
    assert(table.getBlock(0) == 42);
    
    // Append another block, original should be unchanged
    table.appendBlock(99);
    assert(table[0] == 42);
    assert(table[1] == 99);
    
    std::cout << "PASS: testNoMutationAfterAppend" << std::endl;
}

int main() {
    testEmptyPageTable();
    testAppendAndAccess();
    testBoundsChecking();
    testSequentialGrowth();
    testNoIndexReuse();
    testInvalidConstructorArgs();
    testNoMutationAfterAppend();
    
    std::cout << "\nAll PageTable tests passed!" << std::endl;
    return 0;
}
