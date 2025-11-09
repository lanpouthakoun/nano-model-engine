"""
This file is meant to check if my implementation of LLama appropriately loads model weights

No KVCache
"""
import torch
from transformers import AutoModelForCausalLM, LlamaConfig
from InferenceModel.models.llama import LLamaForCausalLM

def load_pretrained(model_name: str):
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16, 
        device_map="mps"
    )
    config = hf_model.config
    custom_model = LLamaForCausalLM(config)
    custom_state_dict = {}

    
    state_dict = hf_model.state_dict()
    for key, value in state_dict.items():
        custom_state_dict[key] = value
    
    custom_model.load_state_dict(custom_state_dict, strict=False)
    
    return custom_model, config

model, config = load_pretrained("meta-llama/Llama-3.2-1B")

