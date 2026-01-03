"""
HuggingFace Layer 1 Intermediate Extraction using Forward Hooks

This script uses PyTorch forward hooks to capture intermediate tensors
from HF Layer 1 and compares them 1:1 with Cottus debug output.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_path = os.path.join(_project_root, 'build', 'cottus', 'csrc')
sys.path.insert(0, _build_path)

from load_hf_weights import load_hf_model, create_cottus_engine


# Storage for hook outputs
hook_outputs = {}


def create_hook(name):
    """Create a forward hook that stores output."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            hook_outputs[name] = output[0].detach().clone()
        else:
            hook_outputs[name] = output.detach().clone()
    return hook


def main():
    print("="*70)
    print("HF LAYER 1 PARITY DEBUG - FORWARD HOOKS")
    print("="*70)
    
    # Load model
    weight_ptrs, config, hf_model, tokenizer, tensors = load_hf_model()
    print(f"DEBUG: Config head_dim = {config.head_dim}")
    print(f"DEBUG: Config hidden_dim = {config.hidden_dim}")
    print(f"DEBUG: Config num_heads = {config.num_heads}")
    engine = create_cottus_engine(weight_ptrs, config)
    
    # Long prompt from parity test
    text = "The quick brown fox jumps over the lazy dog and runs away into the forest"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"\nInput prompt: '{text}'")
    print(f"Token count: {input_ids.shape[1]}")
    print(f"Testing single-token forward pass (prefill)")
    
    # Register hooks on Layer 0 and 1
    layer0 = hf_model.model.layers[0]
    layer1 = hf_model.model.layers[1]
    
    hooks = []
    
    # Embeddings
    hooks.append(hf_model.model.embed_tokens.register_forward_hook(create_hook('embeddings')))
    
    # Layer 0 Output (Input to Layer 1)
    hooks.append(layer0.register_forward_hook(create_hook('layer0_output')))
    
    # Layer 0 FFN Output (MLP)
    hooks.append(layer0.mlp.register_forward_hook(create_hook('layer0_mlp')))
    
    # Layer 0 Attention Output
    hooks.append(layer0.self_attn.register_forward_hook(create_hook('layer0_attn')))
    
    # Layer 1 Inputs
    hooks.append(layer1.input_layernorm.register_forward_hook(create_hook('layer1_ln')))
    
    # Layer 1 V-Proj
    hooks.append(layer1.self_attn.v_proj.register_forward_hook(create_hook('layer1_v_proj')))
    
    # Layer 1 Attention Output (before o_proj)
    hooks.append(layer1.self_attn.register_forward_hook(create_hook('self_attn')))
    
    # Run HF forward
    print("\n" + "-"*70)
    print("Running HF forward pass...")
    print("-"*70)
    
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Extract and display captured tensors
    print("\n" + "-"*70)
    print("HF LAYER 1 INTERMEDIATE TENSORS")
    print("-"*70)
    
    # Extract and display captured tensors
    print("\n" + "-"*70)
    print("HF INTERMEDIATE TENSORS (LAST TOKEN - Pos 17)")
    print("-"*70)
    
    if 'embeddings' in hook_outputs:
        emb = hook_outputs['embeddings']
        print(f"\n[HF] Embeddings (Token 0) first 8: {emb[0, 0, :8]}")

    if 'layer0_mlp' in hook_outputs:
        l0_mlp = hook_outputs['layer0_mlp']
        print(f"[HF] Layer 0 MLP Output (Pos 17) first 8: {l0_mlp[0, -1, :8]}")

    if 'layer0_output' in hook_outputs:
        l0_out = hook_outputs['layer0_output']
        # Layer output is a tuple (hidden_states,)
        if isinstance(l0_out, tuple): l0_out = l0_out[0]
        print(f"\n[HF] Layer 0 Output (Input to Layer 1) first 8: {l0_out[0, -1, :8]}")
        
    if 'layer0_attn' in hook_outputs:
        l0_attn = hook_outputs['layer0_attn']
        print(f"[HF] Layer 0 Self-Attention Output first 8: {l0_attn[0, -1, :8]}")
        
    if 'layer1_ln' in hook_outputs:
        ln = hook_outputs['layer1_ln']
        print(f"[HF] Layer 1 Pre-Attn Norm first 8: {ln[0, -1, :8]}")
        
    if 'layer1_v_proj' in hook_outputs:
        v = hook_outputs['layer1_v_proj']
        print(f"[HF] Layer 1 V-Proj first 8: {v[0, -1, :8]}")

    if 'self_attn' in hook_outputs:
        attn_out = hook_outputs['self_attn']
        print(f"\n[HF] Self-Attention output (first 8): {attn_out[0, -1, :8]}")
        
    if 'gate_proj' in hook_outputs:
        gate = hook_outputs['gate_proj']
        gated_silu = torch.nn.functional.silu(gate)
        print(f"\n[HF] FFN Gate (after SiLU) first 8: {gated_silu[0, -1, :8]}")
        
    if 'up_proj' in hook_outputs:
        up = hook_outputs['up_proj'] 
        print(f"[HF] FFN Up first 8: {up[0, -1, :8]}")
        
        # Gated product
        if 'gate_proj' in hook_outputs:
            gated = gated_silu * up
            print(f"[HF] FFN Gated Product first 8: {gated[0, -1, :8]}")
    
    # Run Cottus
    print("\n" + "-"*70)
    print("COTTUS LAYER 1 INTERMEDIATE TENSORS")
    print("-"*70)
    print("(From debug output below)")
    print()
    
    # Convert tensor to list of ints
    token_list = input_ids[0].tolist()
    result = engine.generate(token_list, 1)
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print("Compare the values above to identify first divergence > 1e-5")
    print("="*70)


if __name__ == "__main__":
    main()
