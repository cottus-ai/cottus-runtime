#include "../cottus/csrc/paged_attention_cpu.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

using namespace cottus;

// Helper: Convert fp32 to fp16 (simple version)
static uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits & 0x80000000) >> 16;
    int32_t exp = ((bits & 0x7F800000) >> 23) - 127 + 15;
    uint32_t mant = (bits & 0x007FFFFF) >> 13;
    
    if (exp <= 0) return static_cast<uint16_t>(sign); // Underflow to zero
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00); // Overflow to inf
    
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

// Test 1: Single head, single block, hand-verifiable values
void testSingleHeadSingleBlock() {
    std::cout << "Test: Single Head, Single Block" << std::endl;
    
    // Config
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 4;
    int32_t blockSize = 2;
    int32_t seqLen = 2;
    int32_t layerIdx = 0;
    
    // Create query: [1, 0, 0, 0]
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    
    // Create KV cache with known values
    // Block 0, Layer 0:
    //   Keys: token0=[1,0,0,0], token1=[0,1,0,0]
    //   Values: token0=[1,1,1,1], token1=[2,2,2,2]
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim; // 2*1*4 = 8
    int32_t elementsPerBlock = 2 * elementsPerLayerKV; // 16
    
    std::vector<uint16_t> kvCache(elementsPerBlock, fp32_to_fp16(0.0f));
    
    // Fill Keys
    kvCache[0] = fp32_to_fp16(1.0f); // token0, head0, dim0
    kvCache[4] = fp32_to_fp16(0.0f); // token1, head0, dim0
    kvCache[5] = fp32_to_fp16(1.0f); // token1, head0, dim1
    
    // Fill Values (offset by elementsPerLayerKV)
    for (int i = 0; i < 4; ++i) kvCache[8 + i] = fp32_to_fp16(1.0f); // token0
    for (int i = 0; i < 4; ++i) kvCache[12 + i] = fp32_to_fp16(2.0f); // token1
    
    // Create PageTable
    PageTable pageTable(blockSize);
    pageTable.appendBlock(0); // Physical block 0
    
    // Run attention
    std::vector<float> output(numHeads * headDim);
    pagedAttentionCPU(output.data(), query.data(), kvCache.data(), 
                     pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    // Expected:
    // Q·K[0] = 1*1 = 1, scaled = 1/2 = 0.5, exp = 1.649
    // Q·K[1] = 1*0 = 0, scaled = 0, exp = 1.0
    // softmax = [1.649/(1.649+1.0), 1.0/(1.649+1.0)] = [0.622, 0.378]
    // output = 0.622*[1,1,1,1] + 0.378*[2,2,2,2] = [1.378, 1.378, 1.378, 1.378]
    
    float expected = 0.622f * 1.0f + 0.378f * 2.0f; // ~1.378
    for (int i = 0; i < headDim; ++i) {
        float diff = std::abs(output[i] - expected);
        assert(diff < 0.01f); // Tolerance for fp16 conversion
    }
    
    std::cout << "  PASS" << std::endl;
}

// Test 2: Multi-head attention
void testMultiHead() {
    std::cout << "Test: Multi-Head Attention" << std::endl;
    
    int32_t numHeads = 2;
    int32_t numKvHeads = 2;
    int32_t headDim = 2;
    int32_t blockSize = 1;
    int32_t seqLen = 1;
    int32_t layerIdx = 0;
    
    // Query: head0=[1,0], head1=[0,1]
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 1.0f};
    
    // KV cache: 1 block, 1 token
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim; // 1*2*2 = 4
    int32_t elementsPerBlock = 2 * elementsPerLayerKV; // 8
    
    std::vector<uint16_t> kvCache(elementsPerBlock);
    // Keys: head0=[1,0], head1=[0,1]
    kvCache[0] = fp32_to_fp16(1.0f);
    kvCache[1] = fp32_to_fp16(0.0f);
    kvCache[2] = fp32_to_fp16(0.0f);
    kvCache[3] = fp32_to_fp16(1.0f);
    
    // Values: head0=[2,0], head1=[0,3]
    kvCache[4] = fp32_to_fp16(2.0f);
    kvCache[5] = fp32_to_fp16(0.0f);
    kvCache[6] = fp32_to_fp16(0.0f);
    kvCache[7] = fp32_to_fp16(3.0f);
    
    PageTable pageTable(blockSize);
    pageTable.appendBlock(0);
    
    std::vector<float> output(numHeads * headDim);
    pagedAttentionCPU(output.data(), query.data(), kvCache.data(),
                     pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    // Expected: Each head attends to itself (single token, softmax=1)
    // head0: Q·K = 1, output = V = [2,0]
    // head1: Q·K = 1, output = V = [0,3]
    assert(std::abs(output[0] - 2.0f) < 0.01f);
    assert(std::abs(output[1] - 0.0f) < 0.01f);
    assert(std::abs(output[2] - 0.0f) < 0.01f);
    assert(std::abs(output[3] - 3.0f) < 0.01f);
    
    std::cout << "  PASS" << std::endl;
}

// Test 3: Multiple blocks (paging)
void testMultipleBlocks() {
    std::cout << "Test: Multiple Blocks (Paging)" << std::endl;
    
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 2;
    int32_t blockSize = 2;
    int32_t seqLen = 3; // Spans 2 blocks
    int32_t layerIdx = 0;
    
    std::vector<float> query = {1.0f, 0.0f};
    
    // 2 physical blocks
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim; // 4
    int32_t elementsPerBlock = 2 * elementsPerLayerKV; // 8
    std::vector<uint16_t> kvCache(2 * elementsPerBlock);
    
    // Block 0: tokens 0,1
    kvCache[0] = fp32_to_fp16(1.0f); // token0 K
    kvCache[2] = fp32_to_fp16(1.0f); // token1 K
    kvCache[4] = fp32_to_fp16(1.0f); // token0 V
    kvCache[6] = fp32_to_fp16(2.0f); // token1 V
    
    // Block 1: token 2
    kvCache[8] = fp32_to_fp16(1.0f); // token2 K
    kvCache[12] = fp32_to_fp16(3.0f); // token2 V
    
    PageTable pageTable(blockSize);
    pageTable.appendBlock(0);
    pageTable.appendBlock(1);
    
    std::vector<float> output(numHeads * headDim);
    pagedAttentionCPU(output.data(), query.data(), kvCache.data(),
                     pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    // All Q·K = 1 (after scaling), equal attention
    // output ≈ (1+2+3)/3 = 2.0 for first dim
    assert(std::abs(output[0] - 2.0f) < 0.1f);
    
    std::cout << "  PASS" << std::endl;
}

int main() {
    testSingleHeadSingleBlock();
    testMultiHead();
    testMultipleBlocks();
    std::cout << "All PagedAttention CPU tests passed!" << std::endl;
    return 0;
}
