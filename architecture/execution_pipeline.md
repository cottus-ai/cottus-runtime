# Execution Pipeline

The inference loop is the heart of the engine. It processes input tokens and generates new ones in an autoregressive manner.

## Step-by-Step Flow

1.  **Prefill**: The prompt is processed in parallel (all tokens at once).
2.  **Decode**: New tokens are generated one by one.

```mermaid
sequenceDiagram
    participant User as User (Python)
    participant Engine as Engine (C++)
    participant Sch as Scheduler
    participant Mod as Transformer
    participant Mem as Memory (GPU)

    User->>Engine: generate(prompt_ids)
    Engine->>Sch: New Request
    Sch->>Mem: Allocate Blocks for Prompt
    
    rect rgb(240, 248, 255)
        note right of Engine: Prefill Phase
        Sch->>Mod: forward(prompt_tokens)
        Mod->>Mem: Write KV Cache (Parallel)
        Mod->>Mod: Compute Logits
        Mod->>Engine: new_token
    end

    loop Generation
        rect rgb(255, 248, 240)
            Engine->>Sch: Schedule Next Step
            Sch->>Mem: Allocate Block (if needed)
            Sch->>Mod: forward(new_token)
            
            note right of Mod: Decode Phase
            Mod->>Mem: Read Paged KV Cache
            Mod->>Mod: PagedAttention
            Mod->>Mod: FFN / Norm / Residual
            
            Mod->>Engine: next_token
        end
    end

    Engine->>Mem: Free Blocks
    Engine->>User: Return Output
```
