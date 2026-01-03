import argparse
import time
from cottus import Engine, EngineConfig
from cottus.model import load_hf_model

def main():
    parser = argparse.ArgumentParser(description="Run Cottus Inference")
    parser.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-LlamaForCausalLM", help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max new tokens to generate")
    args = parser.parse_args()

    print(f"Loading model: {args.model} on {args.device}...")
    
    # 1. Load Weights
    # weight_ptrs, config, model, tokenizer, tensors
    weights, _, _, _, _ = load_hf_model(args.model, device=args.device)
    
    # 2. Configure Engine
    # (In v0.2, config will be auto-loaded from model, but for v0.1 we set it manually or rely on defaults matching the model)
    config = EngineConfig()
    config.vocab_size = 32000
    config.hidden_dim = 16  # Tiny Random Llama specific
    config.num_layers = 2
    config.num_heads = 4
    config.num_kv_heads = 4
    config.head_dim = 4
    config.intermediate_dim = 64
    config.max_seq_len = 2048
    config.block_size = 16
    config.rope_theta = 10000.0
    config.norm_epsilon = 1e-5
    config.device = args.device
    config.dtype = "float32"

    print("Initializing Engine...")
    engine = Engine(config, weights)

    # 3. Tokenize (Mock for v0.1 - using raw IDs for simplicity or a real HF tokenizer)
    # Ideally, we would use 'transformers' here just for tokenization
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        input_ids = tokenizer.encode(args.prompt)
        print(f"Prompt IDs: {input_ids}")
    except ImportError:
        print("Transformers library not found, using dummy input IDs")
        input_ids = [1, 15043, 3186] # "Hello world"

    # 4. Generate
    start_time = time.time()
    output_ids = engine.generate(input_ids, max_new_tokens=args.max_tokens)
    end_time = time.time()

    print(f"Generated IDs: {output_ids}")
    
    # 5. Decode
    try:
        decoded_text = tokenizer.decode(output_ids)
        print(f"Output Text: {decoded_text}")
    except NameError:
        pass

    print(f"Generation took: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    main()
