# Error Handling Strategy

Cottus prioritizes system stability and predictable failure over partial degradation. The error handling strategy relies on C++ exceptions, RAII resource management, and strict boundary translation.

## 1. Core Philosophy

1.  **Fail Fast**: Detect invalid state (config, inputs, memory) immediately.
2.  **No Zombie State**: If an operation fails, resources are cleaned up. The system never remains in a "half-allocated" or undefined state.
3.  **Explicit Propagation**: Errors bubble up to the User Space (Python) as standard exceptions.

## 2. Exception Hierarchy

We map C++ standard exceptions to Python exceptions via PyBind11.

| C++ Exception | Python Equivalent | Trigger Scenario | Recoverable? |
|---|---|---|---|
| `std::invalid_argument` | `ValueError` | Invalid config, mismatching weight shapes. | Yes (Fix args) |
| `std::out_of_range` | `IndexError` | Token ID > vocab_size, Context length exceeded. | Yes (Fix input) |
| `std::runtime_error` | `RuntimeError` | CUDA error (700), OOM, Internal invariant violation. | Varies (OOM: Yes, CUDA: No) |
| `std::bad_alloc` | `MemoryError` | Host OOM. | No |

## 3. Boundary Safety (PyBind11)

The Interface Layer acts as a firewall.
- **Translation**: All C++ exceptions escaping `Engine::forward` or `generate` are caught by PyBind11 and translated to Python exceptions.
- **No Segfaults**: Use of `at()` instead of `[]` for vector access in critical paths (where perf allows) or explicit bounds checking.

## 4. Resource Management (RAII)

We rely on **Resource Acquisition Is Initialization (RAII)** to ensure cleanup during stack unwinding.

### 4.1. Automatic Cleanup
- **BlockAllocator**: GPU memory block (pre-allocated) is owned by a `unique_ptr`. If `Engine` construction fails, the allocator is destroyed and memory freed.
- **PageTable**: Created on the stack (or `unique_ptr`) for each request. If `forward()` throws, `PageTable` destructor runs, returning blocks to the `BlockAllocator` free list.

### 4.2. Manual Cleanup (CUDA Kernel Launch)
In raw CUDA paths where RAII wrappers (like `thrust`) are avoided for control:
```cpp
try {
    CUDA_CHECK(cudaMalloc(...));
    kernel<<<...>>>();
} catch(...) {
    cudaFree(...); // Explicit rollback in catch block
    throw;
}
```
*Note: Future refactoring targets moving raw pointers to smart pointers/views to remove manual `try-catch` blocks.*

## 5. Failure Scenarios

### Scenario A: Out of Memory (OOM)
- **Trigger**: `BlockAllocator` cannot satisfy a `allocate()` request during decode.
- **Action**: Throws `std::runtime_error("OOM")`.
- **Cleanup**: `PageTable` destructor runs. All blocks used by the failed request are returned to the free pool.
- **System State**: The `Engine` returns to a valid state. Other requests (if batching existed) would be unaffected.

### Scenario B: CUDA Error
- **Trigger**: `cudaMalloc` fails or kernel execution fails.
- **Action**: `CUDA_CHECK` macro throws `std::runtime_error`.
- **System State**: GPU context might be tainted depending on the error. We treat CUDA errors as fatal for the process generally, though the exception is propagated.

### Scenario C: Invalid Input
- **Trigger**: User passes `token_id = -1`.
- **Action**: `Engine` methods validate inputs *before* touching GPU memory. Throws `std::invalid_argument`.
- **Cost**: Zero (failed cheap).
