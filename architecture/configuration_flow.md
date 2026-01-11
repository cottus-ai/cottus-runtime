# Configuration Architecture

The configuration system bridges the flexible Python ecosystem (HuggingFace Transformers) with the strict, performance-critical C++ runtime.

## 1. Data Flow

Configuration flows unidirectionally from User Space to the Compute Layer during initialization.

```mermaid
graph LR
    HF["HuggingFace config.json"] -->|AutoConfig| PyObj["Python Object (HF Config)"]
    PyObj -->|Adapter Logic (model.py)| CStruct["C++ EngineConfig (Struct)"]
    CStruct -->|PyBind11| Engine["C++ Engine"]
    Engine -->|Copy| Transformer["GenericTransformer"]
```

## 2. Configuration Schema (`cottus::EngineConfig`)

The canonical definition resides in C++ (`engine.h`). It is a Plain Old Data (POD) struct designed for zero-overhead access.

### Core Dimensions
| Field | Description | Source (HF) |
|---|---|---|
| `vocabSize` | Token vocabulary size | `vocab_size` |
| `hiddenDim` | Embedding dimension | `hidden_size` |
| `numLayers` | Number of transformer blocks | `num_hidden_layers` |
| `numHeads` | Attention heads | `num_attention_heads` |
| `numKvHeads` | KV heads (GQA support) | `num_key_value_heads` |
| `headDim` | Dimesion per head | `hidden_size / num_attention_heads` |

### Generation Parameters
| Field | Description | Constraints |
|---|---|---|
| `maxSeqLen` | Max context window | Fixed at init (Memory Allocation) |
| `blockSize` | PagedAttention block size | Fixed (Currently 16) |

### Numerical Stability
| Field | Description | Default |
|---|---|---|
| `ropeTheta` | RoPE base frequency | 10000.0 |
| `normEpsilon` | RMSNorm epsilon | 1e-5 |

## 3. Immutability & Validation

### Immutability
Once the `Engine` is constructed, the configuration is **immutable**.
- **Reason**: Memory buffers (KV cache, lookup tables) are pre-allocated based on these dimensions. Resizing would violate the zero-allocation invariant during inference.

### Validation
The `Engine` constructor performs strict validation before resource allocation:
- Checks for negative dimensions.
- Verifies `headDim * numHeads == hiddenDim`.
- Ensures `maxSeqLen` is within hardware limits (if applicable).
- Throws `std::invalid_argument` if any check fails, preventing zombie states.

## 4. Extension Pattern

To support new model architectures (different from Llama), the schema must be extended:
1.  Add field to C++ struct in `engine.h`.
2.  Update Python adapter in `model.py` to populate it.
3.  Update `Engine` validation.
4.  Update `GenericTransformer` to use it.
