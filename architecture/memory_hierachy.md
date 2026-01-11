# Memory Hierarchy & PagedAttention

Cottus uses a **Paged Memory Management** system inspired by OS virtual memory to eliminate fragmentation and handle dynamic sequence lengths efficiently.

## Block-Based Allocation

Instead of allocating contiguous memory for each request (which causes fragmentation), we break the Key-Value (KV) Cache into fixed-size **Blocks**.

- **Block Size**: Configurable (default 16 tokens).
- **Physical Block**: A contiguous slice of GPU memory holding KV data for `blockSize` tokens.
- **Logical Block**: A virtual view of a block seen by the sequence.

## Address Translation

The `PageTable` maps Logical Block IDs to Physical Block IDs.

```mermaid
graph LR
    classDef logical fill:#f3e5f5,stroke:#4a148c;
    classDef table fill:#e3f2fd,stroke:#0d47a1;
    classDef physical fill:#e8f5e9,stroke:#1b5e20;

    subgraph Sequence ["Sequence (Logical View)"]
        S1["Seq A: Tokens 0-15"]:::logical
        S2["Seq A: Tokens 16-31"]:::logical
        S3["Seq A: Tokens 32-47"]:::logical
    end

    subgraph Mapping ["Page Table"]
        T1["Logical 0 -> Physical 42"]:::table
        T2["Logical 1 -> Physical 07"]:::table
        T3["Logical 2 -> Physical 99"]:::table
    end

    subgraph VRAM ["GPU Memory (Physical Heap)"]
        P0["Block 00 (Free)"]:::physical
        P7["Block 07 (Allocated)"]:::physical
        P42["Block 42 (Allocated)"]:::physical
        P99["Block 99 (Allocated)"]:::physical
        PX["..."]:::physical
    end

    S1 --> T1 --> P42
    S2 --> T2 --> P7
    S3 --> T3 --> P99
```

This allows non-contiguous physical memory to appear contiguous to the attention mechanism via the `PagedAttention` kernel.

## GPU-Resident KV Cache (v0.2+)

The KV cache is **persistent on GPU** to eliminate PCIe bottlenecks:

- **`d_kvCache_`**: Pre-allocated at `Engine` construction, freed at destruction
- **Quantization on GPU**: FP32→FP16 via `quantizeAndCacheCUDA()` kernel
- **Zero CPU Round-Trips**: K/V computed on GPU → quantized on GPU → stored in GPU cache

```
Old Flow (per token):
  GPU: Compute K,V → cudaMemcpy D2H → CPU: Quantize → cudaMemcpy H2D → GPU Cache

New Flow (per token):
  GPU: Compute K,V → quantizeAndCacheKernel → GPU Cache (no memcpy!)
```

See `notes/architecture_sync/20260112_gpu_memory_subsystem.md` for details.
