# System Overview

This document outlines the high-level architecture of the Cottus Runtime.

## Component Interaction

The system is divided into three main layers:
1.  **User Space (Python)**: Handles configuration, weight loading, and tokenizer interaction.
2.  **Interface Layer (C++/PyBind11)**: Manages state, memory, and orchestrates the pipeline.
3.  **Compute Layer (C++/CUDA)**: Executes the actual math operations on the hardware.

```mermaid
graph TD
    classDef python fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef cpp fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef cuda fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    subgraph UserSpace ["User Space (Python)"]
        UserScript["User Script"]:::python -->|1. Setup| Config["EngineConfig"]:::python
        UserScript -->|2. Weights| HFModel["HuggingFace Weights"]:::python
        UserScript -->|3. Generate| EngineAPI["cottus.Engine"]:::python
    end

    subgraph Interface ["Interface Layer (C++ / PyBind11)"]
        EngineAPI <-->|Binding| PyEngine["PyEngine Wrapper"]:::cpp
        PyEngine -->|Manage| EngineCore["Engine (Core)"]:::cpp
    end

    subgraph Core ["Compute Layer (C++20 / CUDA)"]
        EngineCore -->|Schedule| Scheduler["Scheduler"]:::cpp
        EngineCore -->|allocate| BlockAlloc["BlockAllocator"]:::cpp
        
        Scheduler -->|Execute| Pipeline["GenericTransformer"]:::cpp
        
        Pipeline -->|Attention| PagedAttn["PagedAttention Kernel"]:::cuda
        Pipeline -->|Layers| DenseKernels["cuBLAS / GEMM"]:::cuda
        Pipeline -->|Ops| ElementWise["RoPE / SiLU / Norm"]:::cuda
    end

    BlockAlloc -.->|Virtual Mapping| PagedAttn
```
