import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
from InferenceModel.models.layers.layernorm import RMSNorm
from InferenceModel.models.layers.simple_attention import SimpleAttention
from InferenceModel.models.layers.mlp import MLP
from InferenceModel.models.layers.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb
from typing import Optional, Union


class QwenAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.self_attn = SimpleAttention(
            config.num_attention_heads,
            self.head_dim,
            self.scaling,
            config.num_key_value_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output, attn_weights = self.self_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
class QwenDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.self_attn = QwenAttention(config, layer_idx)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        r_1 = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = r_1 + hidden_states
        r_2 = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return hidden_states + r_2
    

class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        
    def forward(self, input_ids, position_ids, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        layer_outputs = []  # ✅ Add this
        
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            if idx == 27:
                    print(f"Layer 27 output being stored:")
                    print(f"  Shape: {hidden_states.shape}")
                    print(f"  Min: {hidden_states.min():.6f}, Max: {hidden_states.max():.6f}")
                    print(f"  Mean: {hidden_states.mean():.6f}")
                
            layer_outputs.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        layer_outputs[-1] = hidden_states
        return hidden_states, layer_outputs  # ✅ Return both
    
class QwenForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = QwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        # Initialize weights and apply final processing


    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states, layer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)
        return logits, layer_outputs 