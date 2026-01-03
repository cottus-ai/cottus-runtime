import time
from cottus import Engine, EngineConfig
from cottus.model import load_hf_model

def main():
    print("=== Example 2: CPU Inference ===")
    print("Demonstrating fallback to CPU backend for devices without NVIDIA GPUs.")
    
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    device = "cpu"
    
    # 1. Load Weights
    print(f"Loading {model_name} on {device}...")
    weights, config_dummy, _, _, _ = load_hf_model(model_name, device=device)
    
    # 2. Configure Engine explicitly for CPU
    config = EngineConfig()
    config.vocab_size = 32000
    config.hidden_dim = 16
    config.num_layers = 2
    config.num_heads = 4
    config.num_kv_heads = 4
    config.head_dim = 4
    config.intermediate_dim = 64
    config.max_seq_len = 512
    config.block_size = 16
    config.rope_theta = 10000.0
    config.norm_epsilon = 1e-5
    config.dtype = "float32"
    config.device = "cpu"  # <--- FORCE CPU DEVICE
    
    # 3. Initialize
    print("Initializing Engine (C++ Backend)...")
    engine = Engine(config, weights)
    
    # 4. Generate
    prompt = [1, 502, 33, 992] # "Mock prompt"
    print(f"Prompt IDs: {prompt}")
    
    start = time.time()
    output = engine.generate(prompt, max_new_tokens=10)
    dt = time.time() - start
    
    print(f"Output IDs: {output}")
    print(f"Time: {dt:.4f}s (CPU is slower than CUDA, but works everywhere)")

if __name__ == "__main__":
    main()
