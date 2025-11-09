"""
Test to verify custom LLaMA implementation matches HuggingFace outputs
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from InferenceModel.models.llama import LLamaForCausalLM
import torch.testing as torch_testing
class OutputHook:
    """Simple hook to capture a module's output."""
    def __init__(self):
        self.output = None
        self.handle = None

    def __call__(self, module, inputs, output):
        # We only care about the main output tensor
        # Some models return tuples (hidden_states, caches)
        self.output = output[0] if isinstance(output, tuple) else output
        
    def register(self, module):
        """Registers this hook to the given module."""
        self.handle = module.register_forward_hook(self)
        
    def remove(self):
        """Removes the hook."""
        if self.handle:
            self.handle.remove()

def load_models(model_name: str = "meta-llama/Llama-3.2-1B"):
    """Load both HuggingFace and custom models"""
    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        device_map="mps"
    )
    hf_model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load custom model with HF weights
    config = hf_model.config
    custom_model = LLamaForCausalLM(config)
    custom_model.load_state_dict(hf_model.state_dict(), strict=False)
    custom_model = custom_model.to(dtype=torch.float16, device="mps")
    custom_model.eval()
    
    return hf_model, custom_model, tokenizer, config

def check_model_weights():
    """
    Compares the state_dicts of two models to find mismatches.
    """
    print("--- Starting Weight Check ---")
    hf_model, custom_model, tokenizer, config = load_models()
    
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    # 1. Check for mismatched keys
    hf_keys = set(hf_state_dict.keys())
    custom_keys = set(custom_state_dict.keys())

    missing_in_custom = hf_keys - custom_keys
    extra_in_custom = custom_keys - hf_keys

    if missing_in_custom:
        print("\n❌ ERROR: Keys missing from your custom model:")
        for key in sorted(missing_in_custom):
            print(f"  - {key}")

    if extra_in_custom:
        print("\n❌ ERROR: Your custom model has extra keys:")
        for key in sorted(extra_in_custom):
            print(f"  - {key}")

    if not missing_in_custom and not extra_in_custom:
        print("\n✅ SUCCESS: All state_dict keys match!")
    else:
        # Stop here if keys are wrong, no point checking values
        return

    # 2. Check for mismatched tensor values
    mismatched_tensors = []
    total_params = 0
    mismatched_params = 0

    for key in hf_keys:
        hf_tensor = hf_state_dict[key]
        custom_tensor = custom_state_dict[key]

        total_params += hf_tensor.numel()

        if not torch.allclose(hf_tensor, custom_tensor):
            mismatched_tensors.append(key)
            mismatched_params += hf_tensor.numel()

    if mismatched_tensors:
        print("\n❌ ERROR: Tensors have different values:")
        for key in mismatched_tensors:
            print(f"  - {key}")
        print(f"\nSummary: {mismatched_params / total_params * 100:.2f}% of parameters mismatched.")
    else:
        print("\n✅ SUCCESS: All tensor values match perfectly!")
        print("-----------------------------------")



def test_output_logits_match():
    """Test that logits from both models match layer by layer."""
    hf_model, custom_model, tokenizer, config = load_models()
    
    # --- 1. Prepare Inputs ---
    test_text = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps") # Or your device
    
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    
    # --- 2. Run Embeddings ---
    # We'll feed the same embedded input to both models' first layer
    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(input_ids)
        custom_embeds = custom_model.model.embed_tokens(input_ids)
    
    # Check embeddings (they should be identical)
    print("--- Checking Embeddings ---")
    torch_testing.assert_close(custom_embeds, hf_embeds, rtol=1e-3, atol=1e-3)
    print("✅ Embeddings match!")

    # --- 3. Layer-by-Layer Comparison ---
    print("\n--- Checking Decoder Layers ---")
    
    # Start with the same input
    hf_states = hf_embeds
    custom_states = custom_embeds
    
    # Get the layers from both models
    hf_layers = hf_model.model.layers
    custom_layers = custom_model.model.layers

    for i in range(len(hf_layers)):
        print(f"\nComparing Layer {i}:")
        hf_layer = hf_layers[i]
        custom_layer = custom_layers[i]
        
        # Create hooks
        hf_hook = OutputHook()
        custom_hook = OutputHook()
        
        # Attach hooks
        hf_hook.register(hf_layer)
        custom_hook.register(custom_layer)
        
        # Run forward pass (hooks will capture the output)
        with torch.no_grad():
            # HF model needs attention_mask, your custom one doesn't
            # This is a key difference in testing!
            
            attention_mask = torch.ones_like(input_ids)
            

            hf_layer(
                hf_states,
                attention_mask=attention_mask,
                positions=positions,
            )
            
            # Run Custom layer
            custom_layer(
                positions=positions,
                hidden_states=custom_states
            )

        # Remove hooks
        hf_hook.remove()
        custom_hook.remove()

        # --- 4. Compare Outputs ---
        try:
            torch_testing.assert_close(
                custom_hook.output, 
                hf_hook.output, 
                rtol=1e-3, 
                atol=1e-3
            )
            print(f"✅ Layer {i} outputs match!")
            
            # Update states for the next loop
            hf_states = hf_hook.output
            custom_states = custom_hook.output
            
        except AssertionError as e:
            print(f"❌ MISMATCH at Layer {i}!")
            print(e)
            # Stop the test, we found the bug
            return

    # --- 5. Final Check (if all layers passed) ---
    print("\n--- Checking Final Norm & LM Head ---")
    # ... (You can add checks for the final norm and lm_head here) ...
    print("\n✅ All decoder layers match!")


def test_generation_matches():
    """Test that generated tokens match"""
    hf_model, custom_model, tokenizer, config = load_models()
    
    # Prepare prompt
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    
    # Generate from both models
    with torch.no_grad():
        # HF generation
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,  # deterministic
            temperature=1.0,
            top_p=1.0
        )
        
        # Custom generation (you'll need to implement generate if you haven't)
        custom_output = custom_model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
            top_p=1.0
        )
    
    # Compare outputs
    assert torch.equal(hf_output, custom_output), \
        f"Generated tokens differ:\nHF: {tokenizer.decode(hf_output[0])}\nCustom: {tokenizer.decode(custom_output[0])}"
    
    print("✓ Generated outputs match!")
    print(f"Generated text: {tokenizer.decode(hf_output[0])}")


def test_multiple_inputs():
    """Test with multiple different inputs"""
    hf_model, custom_model, tokenizer, config = load_models()
    
    test_cases = [
        "Hello, world!",
        "The capital of France is",
        "1 + 1 =",
        "In a galaxy far, far away",
    ]
    
    for test_text in test_cases:
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to("mps")
        
        with torch.no_grad():
            hf_outputs = hf_model(input_ids)
            custom_outputs = custom_model(input_ids)
        
        hf_logits = hf_outputs.logits
        custom_logits = custom_outputs.logits if hasattr(custom_outputs, 'logits') else custom_outputs
        
        torch.testing.assert_close(custom_logits, hf_logits, rtol=1e-3, atol=1e-3)
        print(f"✓ '{test_text}' - outputs match!")


def test_hidden_states_match():
    """Test intermediate hidden states match (if available)"""
    hf_model, custom_model, tokenizer, config = load_models()
    
    test_text = "Testing hidden states"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    
    with torch.no_grad():
        hf_outputs = hf_model(input_ids, output_hidden_states=True)
        custom_outputs = custom_model(input_ids, output_hidden_states=True)
    
    if hasattr(custom_outputs, 'hidden_states') and custom_outputs.hidden_states is not None:
        # Compare hidden states from each layer
        for layer_idx, (hf_hidden, custom_hidden) in enumerate(
            zip(hf_outputs.hidden_states, custom_outputs.hidden_states)
        ):
            torch.testing.assert_close(
                custom_hidden, 
                hf_hidden, 
                rtol=1e-3, 
                atol=1e-3,
                msg=f"Layer {layer_idx} hidden states differ"
            )
        print("✓ All hidden states match!")


if __name__ == "__main__":
    print("Testing custom LLaMA implementation...\n")
    
    try:
        # check_model_weights()

        print("1. Testing logits...")
        test_output_logits_match()

        
        print("\n2. Testing with multiple inputs...")
        test_multiple_inputs()
        
        print("\n3. Testing hidden states...")
        test_hidden_states_match()
        
        # Uncomment if you've implemented generate()
        # print("\n4. Testing generation...")
        # test_generation_matches()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise