"""
Cottus Runtime - HuggingFace Weight Loader

Loads weights from a HuggingFace model and converts them to Cottus format.
Python owns all weight tensors for the lifetime of the Engine.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, Tuple
import sys
import os
import gc

#add build directory to path for _cottus_C module
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_path = os.path.join(_project_root, 'build', 'cottus', 'csrc')
sys.path.insert(0, _build_path)

try:
    import _cottus_C
except ImportError as e:
    print(f"Warning: Could not import _cottus_C: {e}")
    print(f"Searched in: {_build_path}")
    print("Make sure to build the project first with `cmake --build build`")
    _cottus_C = None


def load_hf_model(model_name: str = "hf-internal-testing/tiny-random-LlamaForCausalLM", device: str = "cpu", release_hf_model: bool = False):
    """
    Load a HuggingFace model and extract weights for Cottus.
    
    Args:
        model_name: HF model identifier
        device: "cpu" or "cuda"
        release_hf_model: If True, release the HF model memory after extraction to save RAM
        
    Returns:
        Tuple of (weight_ptrs, config, model, tokenizer, weight_tensors)
        - weight_ptrs: Dict[str, int] mapping weight names to data_ptr()
        - config: Cottus EngineConfig
        - model: HF model (None if release_hf_model=True)
        - tokenizer: HF tokenizer
        - weight_tensors: Dict keeping weights alive
    """
    from transformers import AutoTokenizer
    
    print(f"Loading HuggingFace model: {model_name} (device={device})")
    
    #load model and tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    hf_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    hf_config = hf_model.config
    
    #extract model config
    cottus_config = None
    if _cottus_C is not None:
        cottus_config = _cottus_C.EngineConfig()
        cottus_config.vocab_size = hf_config.vocab_size
        cottus_config.hidden_dim = hf_config.hidden_size
        cottus_config.num_layers = hf_config.num_hidden_layers
        cottus_config.num_heads = hf_config.num_attention_heads
        cottus_config.num_kv_heads = getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)
        cottus_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        cottus_config.intermediate_dim = hf_config.intermediate_size
        cottus_config.max_seq_len = getattr(hf_config, 'max_position_embeddings', 2048)
        cottus_config.block_size = 16
        cottus_config.rope_theta = getattr(hf_config, 'rope_theta', 10000.0)
        cottus_config.norm_epsilon = getattr(hf_config, 'rms_norm_eps', 1e-5)
        cottus_config.device = device
        cottus_config.dtype = "float32"
        cottus_config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        cottus_config.temperature = 0.7
        cottus_config.top_k = 40
    weight_ptrs = {}
    weight_tensors = {}  
    
    state_dict = hf_model.state_dict()
    if release_hf_model:
        del hf_model
        gc.collect()
    
    layers_to_skip = ['embed_tokens.weight']  
    transposed_count = 0
    keys = list(state_dict.keys())
    
    for name in keys:
        tensor = state_dict[name]
        tensor = tensor.contiguous().float()
        should_transpose = (
            len(tensor.shape) == 2 and
            not any(skip in name for skip in layers_to_skip)
        )
        
        if should_transpose:
            tensor = tensor.T.contiguous()
            transposed_count += 1
        
        weight_tensors[name] = tensor
        weight_ptrs[name] = tensor.data_ptr()
        if release_hf_model:
            del state_dict[name]
    
    num_layers = cottus_config.num_layers if cottus_config else 0
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}.self_attn"
        q_name = f"{prefix}.q_proj.weight"
        k_name = f"{prefix}.k_proj.weight"
        v_name = f"{prefix}.v_proj.weight"
        
        if q_name in weight_tensors and k_name in weight_tensors and v_name in weight_tensors:
            q_weight = weight_tensors[q_name]
            k_weight = weight_tensors[k_name]
            v_weight = weight_tensors[v_name]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1).contiguous()
            fused_name = f"{prefix}.qkv_proj.weight"
            weight_tensors[fused_name] = qkv_weight
            weight_ptrs[fused_name] = qkv_weight.data_ptr()
            
    if release_hf_model:
        gc.collect()
        
    print(f"Loaded {len(weight_ptrs)} weight tensors ({transposed_count} transposed)")
    
    return weight_ptrs, cottus_config, (None if release_hf_model else hf_model), tokenizer, weight_tensors


def create_cottus_engine(weight_ptrs: Dict[str, int], config) -> "_cottus_C.Engine":
    """
    Create a Cottus Engine with the given weights and config.
    
    Args:
        weight_ptrs: Dict mapping weight names to data_ptr() values
        config: Cottus EngineConfig
        
    Returns:
        Cottus Engine instance
    """
    if _cottus_C is None:
        raise RuntimeError("_cottus_C module not loaded")
    
    return _cottus_C.Engine(config, weight_ptrs)


if __name__ == "__main__":
    #load model
    weight_ptrs, config, model, tokenizer, tensors, hf_config = load_hf_model()
    
    if hasattr(hf_config, "no_repeat_ngram_size") and hf_config.no_repeat_ngram_size:
        config.no_repeat_ngram_size = hf_config.no_repeat_ngram_size
    else:
        config.no_repeat_ngram_size = 3 # Default safety
        
    print(f"\nModel config:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_kv_heads: {config.num_kv_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  intermediate_dim: {config.intermediate_dim}")
    
    print(f"\nWeight names (first 10):")
    for i, name in enumerate(list(weight_ptrs.keys())[:10]):
        print(f"  {name}")
