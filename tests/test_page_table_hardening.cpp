#include "../cottus/csrc/page_table.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace cottus;

// 1. Immutability check (Logical)
// PageTable doesn't have "set" methods, but we verify that
// accessing via operator[] multiple times returns same values.
void testImmutability() {
    PageTable pt(16);
    pt.appendBlock(10);
    pt.appendBlock(20);
    
    int32_t v1 = pt[0];
    int32_t v2 = pt[1];
    
    // Append more
    pt.appendBlock(30);
    
    // Old values MUST be stable
    assert(pt[0] == v1);
    assert(pt[1] == v2);
    
    std::cout << "PASS: testImmutability" << std::endl;
}

// 2. Stress Append
void testStressAppend() {
    PageTable pt(16);
    int N = 10000;
    for (int i = 0; i < N; ++i) {
        pt.appendBlock(i);
    }
    
    assert(pt.numBlocks() == N);
    for (int i = 0; i < N; ++i) {
        assert(pt[i] == i);
    }
    
    std::cout << "PASS: testStressAppend" << std::endl;
}

// 3. Copy behavior
// If PageTable is copyable, copies must be independent.
void testCopyIndependence() {
    PageTable pt1(16);
    pt1.appendBlock(100);
    
    PageTable pt2 = pt1; // Copy
    
    pt2.appendBlock(200); // Modifying copy
    
    assert(pt1.numBlocks() == 1);
    assert(pt2.numBlocks() == 2);
    assert(pt1[0] == 100);
    assert(pt2[0] == 100);
    assert(pt2[1] == 200);
    
    std::cout << "PASS: testCopyIndependence" << std::endl;
}

int main() {
    testImmutability();
    testStressAppend();
    testCopyIndependence();
    return 0;
}
