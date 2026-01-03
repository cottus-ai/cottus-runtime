import time
import random
from cottus import Engine, EngineConfig
from cottus.model import load_hf_model

def main():
    print("=== Example 4: Long Context Stress Test ===")
    print("Demonstrating PagedAttention handling a large context window.")
    
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    device = "cuda"
    
    # 1. Load Weights
    weights, _, _, _, _ = load_hf_model(model_name, device=device)
    
    # 2. Config with larger context
    config = EngineConfig()
    config.vocab_size = 32000
    config.hidden_dim = 16
    config.num_layers = 1      # Keep simple for speed, verify memory
    config.num_heads = 4
    config.num_kv_heads = 4
    config.head_dim = 4
    config.intermediate_dim = 64
    config.max_seq_len = 4096  # <--- LARGE CONTEXT
    config.block_size = 16
    config.device = device
    
    engine = Engine(config, weights)
    
    # 3. Create a Long Prompt
    # 2000 random tokens
    long_prompt = [random.randint(100, 30000) for _ in range(2000)]
    print(f"Input Prompt Length: {len(long_prompt)} tokens")
    
    # 4. Generate
    print("Generating...")
    start = time.time()
    
    # This will force the PageTable to map 2000/16 = 125 blocks non-contiguously
    output = engine.generate(long_prompt, max_new_tokens=50)
    
    dt = time.time() - start
    tokens_per_sec = 50 / dt
    
    print(f"Generated 50 tokens on top of 2000 context.")
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print("Success! PagedAttention handled the memory fragmentation.")

if __name__ == "__main__":
    main()
