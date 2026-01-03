#include <cassert>
#include <iostream>
#include <vector>
#include <cstdint>

// Phase 6 Specification for KV Cache Layout
// Re-implementing logic here for independent verification
// 
// Shapes: [blockSize, numKvHeads, headDim]
// Strides: 
//   stride_token = numKvHeads * headDim
//   stride_head = headDim
//   stride_dim = 1
//
// Offsets:
//   keyLayerOffset = layerIdx * 2 * elementsPerLayerKV
//   keyOffset = keyLayerOffset + tokenIdx * stride_token + headIdx * stride_head + dimIdx
//   valueLayerOffset = keyLayerOffset + elementsPerLayerKV

// Helper to calculate expected offset
int32_t getKeyOffset(int32_t layerIdx, int32_t tokenIdx, int32_t headIdx, int32_t dimIdx,
                     int32_t blockSize, int32_t numKvHeads, int32_t headDim, int32_t numLayers) {
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t layerBase = layerIdx * 2 * elementsPerLayerKV;
    int32_t inLayerOffset = tokenIdx * (numKvHeads * headDim) + headIdx * headDim + dimIdx;
    return layerBase + inLayerOffset;
}

int32_t getValueOffset(int32_t layerIdx, int32_t tokenIdx, int32_t headIdx, int32_t dimIdx,
                       int32_t blockSize, int32_t numKvHeads, int32_t headDim, int32_t numLayers) {
    int32_t elementsPerLayerKV = blockSize * numKvHeads * headDim;
    int32_t layerBase = layerIdx * 2 * elementsPerLayerKV;
    int32_t inLayerOffset = tokenIdx * (numKvHeads * headDim) + headIdx * headDim + dimIdx;
    return layerBase + elementsPerLayerKV + inLayerOffset; // Skip Keys
}

// 1. Index formula correctness
void testIndexFormulas() {
    int32_t blockSize = 16;
    int32_t numKvHeads = 4;
    int32_t headDim = 32;
    int32_t numLayers = 2;
    
    // First element
    assert(getKeyOffset(0, 0, 0, 0, blockSize, numKvHeads, headDim, numLayers) == 0);
    
    // Boundary of Layer 0 Keys -> Values
    int32_t elementsPerKV = blockSize * numKvHeads * headDim;
    // Last element of KEYS layer 0
    int32_t lastKey = getKeyOffset(0, blockSize - 1, numKvHeads - 1, headDim - 1, blockSize, numKvHeads, headDim, numLayers);
    assert(lastKey == elementsPerKV - 1);
    
    // First element of VALUES layer 0
    assert(getValueOffset(0, 0, 0, 0, blockSize, numKvHeads, headDim, numLayers) == elementsPerKV);
    
    // Boundary Layer 0 Values -> Layer 1 Keys
    int32_t lastVal = getValueOffset(0, blockSize - 1, numKvHeads - 1, headDim - 1, blockSize, numKvHeads, headDim, numLayers);
    assert(lastVal == 2 * elementsPerKV - 1);
    
    // First element of Layer 1 Keys
    assert(getKeyOffset(1, 0, 0, 0, blockSize, numKvHeads, headDim, numLayers) == 2 * elementsPerKV);
    
    std::cout << "PASS: testIndexFormulas" << std::endl;
}

// 2. Bounds safety logic (Independent Check)
// Simulating bounds checking logic that kernels must implement
void testBoundsLogic() {
    int32_t blockSize = 16;
    
    // Logic check: token index inside [0, blockSize)
    auto isValidToken = [&](int32_t idx) { return idx >= 0 && idx < blockSize; };
    
    assert(isValidToken(0));
    assert(isValidToken(15));
    assert(!isValidToken(16));
    assert(!isValidToken(-1));
    
    std::cout << "PASS: testBoundsLogic" << std::endl;
}

// 3. Stride Contiguity
// Verify strides map to contiguous indices
void testContiguity() {
    int32_t blockSize = 16;
    int32_t numKvHeads = 4;
    int32_t headDim = 32;
    int32_t numLayers = 1;

    for (int t = 0; t < blockSize; ++t) {
        for (int h = 0; h < numKvHeads; ++h) {
            for (int d = 0; d < headDim; ++d) {
                int32_t current = getKeyOffset(0, t, h, d, blockSize, numKvHeads, headDim, numLayers);
                
                // If not last element, next dim element should be +1
                if (d < headDim - 1) {
                    int32_t nextDim = getKeyOffset(0, t, h, d+1, blockSize, numKvHeads, headDim, numLayers);
                    assert(nextDim == current + 1);
                }
            }
        }
    }
    std::cout << "PASS: testContiguity" << std::endl;
}


int main() {
    testIndexFormulas();
    testBoundsLogic();
    testContiguity();
    return 0;
}
