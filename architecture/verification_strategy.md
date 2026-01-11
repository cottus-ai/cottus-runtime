# Verification Strategy

Cottus employs a multi-layered verification strategy to ensure correctness, stability, and performance. The core principle is **"Golden Master Parity"**, treating HuggingFace Transformers as the ground truth.

## 1. Verification Layers

### Layer 1: Unit & Component Tests (C++)
**Goal**: Verify memory safety and logic of individual primitives.
- **Scope**: `BlockAllocator`, `PageTable`, `Rope`, `RMSNorm`.
- **Tool**: CTest / GoogleTest pattern (custom implementation).
- **Location**: `tests/*.cpp`
- **Key Check**: Memory leak detection (tracking allocator stats), boundary checks.

### Layer 2: Numerical Parity Tests (C++/CUDA)
**Goal**: Ensure custom kernels match reference implementations bit-for-bit (or within FP32 tolerance).
- **Strategy**: 
    1. Run reference CPU implementation.
    2. Run Optimized CUDA kernel.
    3. Assert `norm(diff) < epsilon`.
- **Location**: `tests/test_*_parity.cpp`.

### Layer 3: End-to-End Parity (Python)
**Goal**: Verify the entire engine produces the exact same token sequence as HuggingFace.
- **Strategy**:
    1. Load model in HF (`transformers`).
    2. Load same model in `cottus`.
    3. Run greedy generation on same prompt.
    4. Assert `hf_tokens == cottus_tokens`.
- **Location**: `tests/test_hf_parity.py`.

## 2. Hardening (Negative Testing)

We explicitly test failure modes to ensure the engine fails gracefully (see `05_error_handling_strategy.md`).

- **OOM Hardening**: `tests/test_engine_memory.cpp` fills memory to breaking point.
- **Invalid Inputs**: `tests/test_engine_hardening.cpp` sends bad token IDs and config.
- **Concurrency**: Basic checks for single-thread invariant enforcement.

## 3. Continuous Integration Gates

Every commit must pass the following pipeline:
1.  **Build**: `cmake --build .` (Strict compiler flags).
2.  **Unit Tests**: `ctest --output-on-failure`.
3.  **Python Install**: `pip install .`.
4.  **Parity Check**: `python tests/test_hf_parity.py`.

## 4. Release Criteria

A release (v0.x) is only valid if:
- All CI gates pass.
- `auditwheel` repair succeeds (Linux compatibility).
- Package size is <100MB (CUDA dependencies excluded).
- Parity tests pass on both CPU (fallback) and CUDA (primary) backends.
