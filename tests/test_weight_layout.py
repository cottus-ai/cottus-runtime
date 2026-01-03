"""
Diagnostic test to verify HF Linear vs Cottus GEMM weight layout.

This test will:
1. Load a single HF linear layer
2. Compare output with our GEMM using the same input
3. Identify if transpose is needed
"""

import torch
from transformers import AutoModelForCausalLM
import sys
import os

# Add build path for _cottus_C
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_path = os.path.join(_project_root, 'build', 'cottus', 'csrc')
sys.path.insert(0, _build_path)

try:
    import _cottus_C
    from load_hf_weights import load_hf_model
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)


def test_linear_transpose():
    """
    Test if we need to transpose weights for GEMM.
    """
    print("="*60)
    print("Diagnostic: HF Linear vs Cottus GEMM Weight Layout")
    print("="*60)
    
    # Load HF model
    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype=torch.float32
    )
    model.eval()
    
    # Get a linear layer (e.g., q_proj from layer 0)
    linear_layer = model.model.layers[0].self_attn.q_proj
    
    print(f"\nHF Linear Layer: q_proj")
    print(f"  Weight shape: {linear_layer.weight.shape}")
    print(f"  Has bias: {linear_layer.bias is not None}")
    
    # Create test input: [1, in_features]
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    
    test_input = torch.randn(1, in_features)
    
    # HF Linear computation
    with torch.no_grad():
        hf_output = linear_layer(test_input)
    
    print(f"\nTest Input shape: {test_input.shape}")
    print(f"HF Output shape: {hf_output.shape}")
    print(f"Expected: [1, {out_features}]")
    
    # Manual computation with PyTorch semantics: y = x @ W^T
    manual_output = test_input @ linear_layer.weight.T
    if linear_layer.bias is not None:
        manual_output += linear_layer.bias
    
    print(f"\nManual (x @ W^T) output shape: {manual_output.shape}")
    print(f"Max diff (HF vs Manual): {(hf_output - manual_output).abs().max().item()}")
    
    # Now test what our GEMM would compute
    # Our GEMM: C = A * B where A=[M,K], B=[K,N], C=[M,N]
    # Without transpose: C = input @ weight (expecting weight as [in, out])
    weight_no_transpose = linear_layer.weight.detach().numpy()  # [out, in]
    output_no_transpose = test_input.detach().numpy() @ weight_no_transpose.T  # Matches HF
    
    # With transpose: weight becomes [in, out]
    weight_transposed = linear_layer.weight.T.detach().numpy()  # [in, out]
    output_with_transpose = test_input.detach().numpy() @ weight_transposed  # Should match HF
    
    print(f"\n--- Testing GEMM Semantics ---")
    print(f"Weight (HF): {linear_layer.weight.shape} [out, in]")
    print(f"Weight (transposed): {weight_transposed.shape} [in, out]")
    
    import numpy as np
    diff_no_transpose = np.abs(hf_output.numpy() - output_no_transpose).max()
    diff_with_transpose = np.abs(hf_output.numpy() - output_with_transpose).max()
    
    print(f"\nMax diff (HF vs input@W^T): {diff_no_transpose:.2e}")
    print(f"Max diff (HF vs input@W_transposed): {diff_with_transpose:.2e}")
    
    print(f"\n{'='*60}")
    if diff_with_transpose < 1e-5:
        print("CONCLUSION: Need to transpose weights from [out, in] → [in, out]")
        print("This matches GEMM semantics: C = A * B")
        return True
    else:
        print("UNEXPECTED: Transpose did not fix the issue")
        return False


def enumerate_all_layers():
    """
    Enumerate all linear layers in the model and their shapes.
    """
    print("\n" + "="*60)
    print("Enumerating All Linear Layers")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype=torch.float32
    )
    
    print("\nLayers requiring transpose:")
    layers_to_transpose = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            layers_to_transpose.append(name)
            print(f"  {name}: {param.shape} → {param.T.shape}")
    
    print(f"\nTotal linear layers: {len(layers_to_transpose)}")
    
    # Special cases
    print("\nLayers NOT requiring transpose:")
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 1:
            print(f"  {name}: {param.shape} (1D - normalization)")
    
    return layers_to_transpose


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Weight Layout Diagnostic Suite")
    print("#"*60)
    
    # Test 1: Verify transpose requirement
    needs_transpose = test_linear_transpose()
    
    # Test 2: Enumerate all layers
    layers = enumerate_all_layers()
    
    print("\n" + "="*60)
    if needs_transpose:
        print("✓ Diagnosis COMPLETE: Transpose required for all 2D weight tensors")
    else:
        print("✗ Diagnosis UNCLEAR: Further investigation needed")
    print("="*60)
