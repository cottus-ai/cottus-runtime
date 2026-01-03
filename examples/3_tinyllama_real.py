import argparse
from cottus import Engine, EngineConfig
from cottus.model import load_hf_model

def main():
    print("=== Example 3: Real Model (TinyLlama-1.1B) ===")
    print("NOTE: This requires downloading a 1.1B parameter model (~2.2GB).")
    print("      Ensure you have enough RAM/VRAM.")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda"
    
    try:
        # 1. Load Weights
        print(f"Loading {model_name}... (may take a while)")
        weights, _, _, tokenizer, _ = load_hf_model(model_name, device=device)
    except OSError:
        print(f"ERROR: Model {model_name} not found locally or access denied.")
        print("Please run `huggingface-cli login` or check your internet connection.")
        return

    # 2. Configure Engine for TinyLlama Architecture
    # Values taken from config.json of TinyLlama-1.1B
    config = EngineConfig()
    config.vocab_size = 32000
    config.hidden_dim = 2048      # 2048
    config.num_layers = 22        # 22 layers
    config.num_heads = 32         # 32 heads
    config.num_kv_heads = 4       # Grouped Query Attention (32 q-heads / 4 kv-heads = 8x grouping)
    config.head_dim = 64          # 2048 / 32 = 64
    config.intermediate_dim = 5632
    config.max_seq_len = 2048
    config.block_size = 16
    config.rope_theta = 10000.0
    config.norm_epsilon = 1e-5
    config.device = device
    
    print("Initializing Engine...")
    engine = Engine(config, weights)
    
    # 3. Interactive Chat Loop
    print("\n--- TinyLlama Chat (Ctrl+C to exit) ---")
    print("Type something to chat with the model!\n")
    
    while True:
        try:
            user_input = input("User: ")
            
            # Format as chat template (simplified)
            # <|user|>\n...</s>\n<|assistant|>\n
            prompt_text = f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
            input_ids = tokenizer.encode(prompt_text)
            
            # Generate
            # We assume the engine runs synchronously for now
            output_ids = engine.generate(input_ids, max_new_tokens=64)
            
            # Decode only the NEW tokens (Cottus returns full sequence including prompt? 
            # Check implementation. Engine::generate appends. So we slice.)
            new_tokens = output_ids[len(input_ids):]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            print(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
