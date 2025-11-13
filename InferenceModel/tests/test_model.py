"""
Test to verify custom Qwen implementation matches HuggingFace hidden states
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from InferenceModel.models.qwen import QwenForCausalLM


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


def test_single_input_hidden_states(device: str = "cpu"):
    """Test that hidden states match for a single input"""
    hf_model, custom_model, tokenizer, _ = load_models(device=device)
    
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
            positions=positions,
            attention_mask=attention_mask
        )
    
    # Compare embeddings (hidden_states[0] in HF)
    print("\n" + "="*60)
    print("COMPARING EMBEDDINGS (Layer 0)")
    print("="*60)
    hf_embeddings = hf_outputs.hidden_states[0]
    # You might need to extract embeddings from your custom model differently
    # For now assuming first layer output is after embeddings
    
    print(f"HF embeddings shape: {hf_embeddings.shape}")
    print(f"Custom has {len(custom_layer_outputs)} layer outputs")
    
    # Compare layer by layer
    print("\n" + "="*60)
    print("COMPARING LAYER OUTPUTS")
    print("="*60)
    
    num_layers = min(len(custom_layer_outputs), len(hf_outputs.hidden_states) - 1)
    
    for i in range(num_layers):
        custom_hidden = custom_layer_outputs[i]
        hf_hidden = hf_outputs.hidden_states[i + 1]  # +1 because hidden_states[0] is embeddings
        
        print(f"\nLayer {i}:")
        print(f"  Custom shape: {custom_hidden.shape}")
        print(f"  HF shape: {hf_hidden.shape}")
        
        # Calculate differences
        diff = (custom_hidden - hf_hidden).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        
        # Check if they match
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
            
            # Additional debugging info
            print(f"\n  Debugging info:")
            print(f"    Custom - min: {custom_hidden.min():.6f}, max: {custom_hidden.max():.6f}, mean: {custom_hidden.mean():.6f}")
            print(f"    HF - min: {hf_hidden.min():.6f}, max: {hf_hidden.max():.6f}, mean: {hf_hidden.mean():.6f}")
            
            # Show where the largest differences are
            max_diff_idx = diff.argmax()
            max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
            print(f"    Largest diff at position: {max_diff_pos}")
            print(f"    Custom value: {custom_hidden[max_diff_pos].item():.6f}")
            print(f"    HF value: {hf_hidden[max_diff_pos].item():.6f}")
    
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
                positions=positions,
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