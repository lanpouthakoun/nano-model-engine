"""
Implementation of LLama with Custom Attnetion Mechanism

"""
import torch
from torch import nn
from transformers import LlamaConfig
from layers.layernorm import LlamaRMSNorm
from layers.attention import Attention
from layers.rotary_embedding import get_rope

class LLamaAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            max_position: int = 4096 * 32,
            attention_bias: bool= True,
            head_dim: int | None = None,
            layer_idx: int | None = None,
            rope_theta: float = 10000,
            rope_scaling: tuple | None = None,):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.max_position = max_position
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=attention_bias
        )
    
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )

        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=attention_bias
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,

        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
    def forward(self, positions, hidden_states):
        """
        Need to take care of the cache manager first before i can do this
        """
        k,v,q = self.k_proj(hidden_states), self.v_proj(hidden_states), self.q_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q,k = self.rotary_emb(positions, q, k)
        
        o = self.attn(q,k,v)

        output = self.o_proj(o.flatten(1,-1))
        return output

class LLamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=bias
        )
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )

class LLamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLamaAttention(config = config, layer_idx = layer_idx)

        self.mlp = LLamaMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        

        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class LlamaModel(nn.Module):
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList([LLamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LLamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
    
    def load_model(self, model):
        pass
