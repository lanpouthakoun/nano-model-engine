"""
Test to verify custom Qwen implementation matches HuggingFace hidden states
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from InferenceModel.models.qwen import QwenForCausalLM

hf_result_map = {}
custom_result_map = {}

def setup_hooks(model, result_map, model_name="model"):
    """Attach hooks to a model and store results in result_map"""
    hooks = []  # Store hook handles so you can remove them later
    
    for name, module in model.named_modules():
        def make_hook(name, model_name):
            def hook(module, inp, output):
                full_name = f"{model_name}.{name}" if name else model_name
                
                # Handle input safely
                if isinstance(inp, tuple):
                    if len(inp) > 0:
                        input_data = inp[0]
                    else:
                        input_data = None  # Empty tuple
                else:
                    input_data = inp
                
                result_map[full_name] = {
                    "input": input_data,
                    "output": output
                }
                
                if hasattr(module, 'weight') and module.weight is not None:
                    result_map[full_name]["weight"] = module.weight
                if hasattr(module, 'bias') and module.bias is not None:
                    result_map[full_name]["bias"] = module.bias
            return hook
        
        handle = module.register_forward_hook(make_hook(name, model_name))
        hooks.append(handle)
    
    return hooks
def load_models(model_name: str = "Qwen/Qwen3-0.6B", device: str = "cpu"):
    """Load both HuggingFace and custom models"""
    print(f"Loading models on {device}...")
    
    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU testing
        device_map=device,
        attn_implementation='eager'
    )
    hf_model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load custom model with HF weights
    config = hf_model.config
    custom_model = QwenForCausalLM(config)
    custom_model.load_state_dict(hf_model.state_dict(), strict=False)
    custom_model = custom_model.to(dtype=torch.float32, device=device)
    custom_model.eval()
    
    return hf_model, custom_model, tokenizer, config

# Create separate result maps for each model


# In your test function:
def test_single_input_hidden_states(device: str = "cpu"):
    """Test that hidden states match for a single input"""
    hf_model, custom_model, tokenizer, _ = load_models(device=device)
    
    # Clear result maps
    hf_result_map.clear()
    custom_result_map.clear()
    
    # Attach hooks to both models
    print("Setting up hooks...")
    hf_hooks = setup_hooks(hf_model, hf_result_map, "hf")
    custom_hooks = setup_hooks(custom_model, custom_result_map, "custom")
    print(f"Attached {len(hf_hooks)} hooks to HF model")
    print(f"Attached {len(custom_hooks)} hooks to custom model")
    
    # Simple test input
    test_text = "The quick brown fox jumps over the lazy dog"
    print(f"\nTest text: '{test_text}'")
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Sequence length: {seq_len}")
    
    # Forward pass
    with torch.no_grad():
        # HuggingFace forward
        hf_outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Custom model forward
        custom_logits, custom_layer_outputs = custom_model(
            input_ids=input_ids,
            position_ids=positions,
            attention_mask=attention_mask
        )
    
    # Now compare module outputs between the two models
    print("\n" + "="*60)
    print("COMPARING MODULE OUTPUTS (via hooks)")
    print("="*60)
    
    # Find matching module names
    hf_names = set(hf_result_map.keys())
    custom_names = set(custom_result_map.keys())
    
    # You might need to map names if they differ
    # For example: "hf.model.layers.0.self_attn" -> "custom.layers.0.attention"
    
    print(f"\nHF model has {len(hf_names)} modules")
    print(f"Custom model has {len(custom_names)} modules")
    
    # Compare matching modules (adjust name mapping as needed)
    for hf_name in sorted(hf_names):
        # Simple name mapping - adjust based on your architecture
        custom_name = hf_name.replace("hf.", "custom.")
        
        if custom_name in custom_names:
            hf_data = hf_result_map[hf_name]
            custom_data = custom_result_map[custom_name]
            
            print(f"\n{'-'*60}")
            print(f"Comparing: {hf_name.replace('hf.', '')}")
            
            # Compare outputs
            if "output" in hf_data and "output" in custom_data:
                hf_out = hf_data["output"]
                custom_out = custom_data["output"]
                
                # Handle tuple outputs (some modules return tuples)
                if isinstance(hf_out, tuple):
                    hf_out = hf_out[0]
                if isinstance(custom_out, tuple):
                    custom_out = custom_out[0]
                
                if hasattr(hf_out, 'shape') and hasattr(custom_out, 'shape'):
                    print(f"  Output shapes - HF: {hf_out.shape}, Custom: {custom_out.shape}")
                    
                    if hf_out.shape == custom_out.shape:
                        diff = (hf_out - custom_out).abs()
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        
                        print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
                        
                        try:
                            torch.testing.assert_close(
                                custom_out,
                                hf_out,
                                rtol=1e-4,
                                atol=1e-4
                            )
                            print(f"  ✅ MATCH!")
                        except AssertionError:
                            print(f"  ❌ MISMATCH!")
                            print(f"    Custom: min={custom_out.min():.6f}, max={custom_out.max():.6f}, mean={custom_out.mean():.6f}")
                            print(f"    HF: min={hf_out.min():.6f}, max={hf_out.max():.6f}, mean={hf_out.mean():.6f}")
                    else:
                        print(f"  ⚠️ Shape mismatch!")
    
    # Compare layer outputs (your existing code)
    print("\n" + "="*60)
    print("COMPARING LAYER OUTPUTS")
    print("="*60)
    
    num_layers = min(len(custom_layer_outputs), len(hf_outputs.hidden_states) - 1)
    
    for i in range(num_layers):
        custom_hidden = custom_layer_outputs[i]
        hf_hidden = hf_outputs.hidden_states[i + 1]
        
        print(f"\nLayer {i}:")
        print(f"  Custom shape: {custom_hidden.shape}")
        print(f"  HF shape: {hf_hidden.shape}")
        
        diff = (custom_hidden - hf_hidden).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        
        try:
            torch.testing.assert_close(
                custom_hidden,
                hf_hidden,
                rtol=1e-4,
                atol=1e-4
            )
            print(f"  ✅ Layer {i} MATCHES!")
        except AssertionError:
            print(f"  ❌ Layer {i} MISMATCH!")
    
    # Compare final logits
    print("\n" + "="*60)
    print("COMPARING FINAL LOGITS")
    print("="*60)
    hf_logits = hf_outputs.logits
    
    print(f"Custom logits shape: {custom_logits.shape}")
    print(f"HF logits shape: {hf_logits.shape}")
    
    diff = (custom_logits - hf_logits).abs()
    print(f"Max difference: {diff.max().item():.6e}")
    print(f"Mean difference: {diff.mean().item():.6e}")
    # In test function, after forward pass:
    print(f"\nHook captured for model.layers.27:")
    print(f"\nComparing layer 27:")
    print(f"Custom layer_outputs[27]:")
    print(f"  {custom_layer_outputs[27].min():.6f}, {custom_layer_outputs[27].mean():.6f}, {custom_layer_outputs[27].max():.6f}")
    print(f"HF hidden_states[28]:")
    hf_27 = hf_outputs.hidden_states[27]
    print(f"  {hf_27.min():.6f}, {hf_27.mean():.6f}, {hf_27.max():.6f}")
    try:
        torch.testing.assert_close(
            custom_logits,
            hf_logits,
            rtol=1e-4,
            atol=1e-4
        )
        print("✅ Final logits MATCH!")
    except AssertionError:
        print("❌ Final logits MISMATCH!")
    
    # Clean up hooks
    for hook in hf_hooks + custom_hooks:
        hook.remove()
    
    return hf_result_map, custom_result_map


def test_multiple_inputs(device: str = "cpu"):
    """Test with multiple different inputs"""
    hf_model, custom_model, tokenizer, _ = load_models(device=device)
    
    test_cases = [
        "Hello, world!",
        "The capital of France is",
        "1 + 1 =",
    ]
    
    print("\n" + "="*60)
    print("TESTING MULTIPLE INPUTS")
    print("="*60)
    
    for test_text in test_cases:
        print(f"\nTest: '{test_text}'")
        
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        with torch.no_grad():
            hf_outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            custom_logits, custom_layer_outputs = custom_model(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=attention_mask
            )
        
        # Just check final layer
        final_custom = custom_layer_outputs[-1]
        final_hf = hf_outputs.hidden_states[-1]
        
        diff = (final_custom - final_hf).abs()
        print(f"  Final layer - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
        
        try:
            torch.testing.assert_close(final_custom, final_hf, rtol=1e-4, atol=1e-4)
            print(f"  ✅ PASS")
        except AssertionError:
            print(f"  ❌ FAIL")


if __name__ == "__main__":
    print("Testing custom Qwen implementation - Hidden States Only\n")
    
    # Use CPU for easier debugging (or change to "mps" if you prefer)
    device = "cpu"
    
    try:
        # Main test
        test_single_input_hidden_states(device=device)
        
        # Additional tests
        print("\n" + "="*60)
        test_multiple_inputs(device=device)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()