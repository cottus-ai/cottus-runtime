#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/ruskaruma/Desktop/GITHUB/cottus-runtime')
sys.path.insert(0, '/home/ruskaruma/Desktop/GITHUB/cottus-runtime/build/cottus/csrc')
import torch
from cottus.model import load_hf_model, create_cottus_engine
from cottus.deterministic_bypass import DeterministicBypass
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS = 256
print(f"Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_ptrs, config, hf_model, tokenizer, tensors = load_hf_model(MODEL_NAME, device=device, release_hf_model=True)
config.temperature = 0.7
config.top_k = 40
config.frequency_penalty = 0.3
config.repetition_penalty = 1.1
config.no_repeat_ngram_size = 3
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = tokenizer.eos_token or "</s>"
print(f"Debug: EOS '{EOS_TOKEN}' ID={tokenizer.eos_token_id}")
config.eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(EOS_TOKEN)]
if tokenizer.convert_tokens_to_ids(USER_TOKEN) is not None:
    config.eos_token_ids.append(tokenizer.convert_tokens_to_ids(USER_TOKEN))
stop_strs = [USER_TOKEN, "\nUser:", "\n" + USER_TOKEN, ASSISTANT_TOKEN, "\n" + ASSISTANT_TOKEN]
config.stop_token_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]
print(f"Debug: Stop sequences: {config.stop_token_sequences}")
print(f"Debug: EOS IDs: {config.eos_token_ids}")
engine = create_cottus_engine(weight_ptrs, config)
print("Ready! Type your message (or 'quit' to exit)")
print("=" * 60)
bypass = DeterministicBypass()

history = []
while True:
    try:
        user_input = input("\nYou: ").strip()
    except EOFError:
        break
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    history.append({"role": "user", "content": user_input})
    is_resolved, result = bypass.resolve(user_input)
    if is_resolved:
        print(f"\nAssistant: {result}")
        history.append({"role": "assistant", "content": result})
        continue

    input_ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")[0].tolist()
    output_ids = engine.generate(input_ids, MAX_TOKENS)
    
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if not response_text:
        print("\nAssistant: [Error: Empty response]") 
        continue
    if "<|user|>" in response_text or "<|assistant|>" in response_text or "[system]" in response_text:
         print(f"Debug: Role Leak Detected in output: {response_text[:50]}...")
         # Sanitize
         response_text = response_text.split("<|user|>")[0].split("<|assistant|>")[0].strip()
    
    print(f"\nAssistant: {response_text}")
    
    history.append({"role": "assistant", "content": response_text})
    
    engine.reset()
