#include "../cottus/csrc/paged_attention_cpu.h"
#include "../cottus/csrc/paged_attention_cuda.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace cottus;

// Helper: Convert fp32 to fp16
static uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits & 0x80000000) >> 16;
    int32_t exp = ((bits & 0x7F800000) >> 23) - 127 + 15;
    uint32_t mant = (bits & 0x007FFFFF) >> 13;
    
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

// Test 1: Single head, single block - CPU vs CUDA parity
void testParitySingleHead() {
    std::cout << "Parity Test: Single Head, Single Block" << std::endl;
    
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 4;
    int32_t blockSize = 2;
    int32_t seqLen = 2;
    int32_t layerIdx = 0;
    
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV;
    
    std::vector<uint16_t> kvCache(elementsPerBlock, fp32_to_fp16(0.0f));
    
    // Keys
    kvCache[0] = fp32_to_fp16(1.0f);
    kvCache[4] = fp32_to_fp16(0.0f);
    kvCache[5] = fp32_to_fp16(1.0f);
    
    // Values
    for (int i = 0; i < 4; ++i) kvCache[8 + i] = fp32_to_fp16(1.0f);
    for (int i = 0; i < 4; ++i) kvCache[12 + i] = fp32_to_fp16(2.0f);
    
    PageTable pageTable(blockSize);
    pageTable.appendBlock(0);
    
    // Run CPU
    std::vector<float> outputCPU(numHeads * headDim);
    pagedAttentionCPU(outputCPU.data(), query.data(), kvCache.data(),
                     pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    // Run CUDA
    std::vector<float> outputCUDA(numHeads * headDim);
    pagedAttentionCUDA(outputCUDA.data(), query.data(), kvCache.data(),
                      pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    // Compare
    float maxDiff = 0.0f;
    for (int i = 0; i < numHeads * headDim; ++i) {
        float diff = std::abs(outputCPU[i] - outputCUDA[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  Max absolute difference: " << maxDiff << std::endl;
    assert(maxDiff < 1e-3f);
    std::cout << "  PASS" << std::endl;
}

// Test 2: Multi-head with GQA
void testParityMultiHead() {
    std::cout << "Parity Test: Multi-Head with GQA" << std::endl;
    
    int32_t numHeads = 2;
    int32_t numKvHeads = 2;
    int32_t headDim = 2;
    int32_t blockSize = 1;
    int32_t seqLen = 1;
    int32_t layerIdx = 0;
    
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 1.0f};
    
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV;
    
    std::vector<uint16_t> kvCache(elementsPerBlock);
    kvCache[0] = fp32_to_fp16(1.0f);
    kvCache[1] = fp32_to_fp16(0.0f);
    kvCache[2] = fp32_to_fp16(0.0f);
    kvCache[3] = fp32_to_fp16(1.0f);
    kvCache[4] = fp32_to_fp16(2.0f);
    kvCache[5] = fp32_to_fp16(0.0f);
    kvCache[6] = fp32_to_fp16(0.0f);
    kvCache[7] = fp32_to_fp16(3.0f);
    
    PageTable pageTable(blockSize);
    pageTable.appendBlock(0);
    
    std::vector<float> outputCPU(numHeads * headDim);
    std::vector<float> outputCUDA(numHeads * headDim);
    
    pagedAttentionCPU(outputCPU.data(), query.data(), kvCache.data(),
                     pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    pagedAttentionCUDA(outputCUDA.data(), query.data(), kvCache.data(),
                      pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    float maxDiff = 0.0f;
    for (int i = 0; i < numHeads * headDim; ++i) {
        float diff = std::abs(outputCPU[i] - outputCUDA[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  Max absolute difference: " << maxDiff << std::endl;
    assert(maxDiff < 1e-3f);
    std::cout << "  PASS" << std::endl;
}

// Test 3: Multi-block paging
void testParityMultiBlock() {
    std::cout << "Parity Test: Multi-Block Paging" << std::endl;
    
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 2;
    int32_t blockSize = 2;
    int32_t seqLen = 3;
    int32_t layerIdx = 0;
    
    std::vector<float> query = {1.0f, 0.0f};
    
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV;
    std::vector<uint16_t> kvCache(2 * elementsPerBlock);
    
    // Block 0
    kvCache[0] = fp32_to_fp16(1.0f);
    kvCache[2] = fp32_to_fp16(1.0f);
    kvCache[4] = fp32_to_fp16(1.0f);
    kvCache[6] = fp32_to_fp16(2.0f);
    
    // Block 1
    kvCache[8] = fp32_to_fp16(1.0f);
    kvCache[12] = fp32_to_fp16(3.0f);
    
    PageTable pageTable(blockSize);
    pageTable.appendBlock(0);
    pageTable.appendBlock(1);
    
    std::vector<float> outputCPU(numHeads * headDim);
    std::vector<float> outputCUDA(numHeads * headDim);
    
    pagedAttentionCPU(outputCPU.data(), query.data(), kvCache.data(),
                     pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    pagedAttentionCUDA(outputCUDA.data(), query.data(), kvCache.data(),
                      pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    float maxDiff = 0.0f;
    for (int i = 0; i < numHeads * headDim; ++i) {
        float diff = std::abs(outputCPU[i] - outputCUDA[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    
    std::cout << "  Max absolute difference: " << maxDiff << std::endl;
    assert(maxDiff < 1e-3f);
    std::cout << "  PASS" << std::endl;
}

// Test 4: Determinism (run CUDA 10 times)
void testDeterminism() {
    std::cout << "Determinism Test: 10 CUDA runs" << std::endl;
    
    int32_t numHeads = 2;
    int32_t numKvHeads = 2;
    int32_t headDim = 4;
    int32_t blockSize = 2;
    int32_t seqLen = 3;
    int32_t layerIdx = 0;
    
    std::vector<float> query(numHeads * headDim, 1.0f);
    
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV;
    std::vector<uint16_t> kvCache(2 * elementsPerBlock, fp32_to_fp16(1.0f));
    
    PageTable pageTable(blockSize);
    pageTable.appendBlock(0);
    pageTable.appendBlock(1);
    
    std::vector<float> referenceOutput(numHeads * headDim);
    pagedAttentionCUDA(referenceOutput.data(), query.data(), kvCache.data(),
                      pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
    
    // Run 9 more times and compare
    for (int run = 0; run < 9; ++run) {
        std::vector<float> output(numHeads * headDim);
        pagedAttentionCUDA(output.data(), query.data(), kvCache.data(),
                          pageTable, seqLen, layerIdx, numHeads, numKvHeads, headDim, blockSize, 1);
        
        for (int i = 0; i < numHeads * headDim; ++i) {
            assert(output[i] == referenceOutput[i]); // Bitwise identical
        }
    }
    
    std::cout << "  PASS: All 10 runs bitwise identical" << std::endl;
}

int main() {
    testParitySingleHead();
    testParityMultiHead();
    testParityMultiBlock();
    testDeterminism();
    std::cout << "All parity tests passed!" << std::endl;
    return 0;
}
