import torch
from torch import nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, query, key, value, attention_mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key: (batch_size, num_kv_heads, seq_len, head_dim)
            value: (batch_size, num_kv_heads, seq_len, head_dim)
            attention_mask: Optional (batch_size, 1, seq_len, seq_len)
        
        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
            attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # --- Handle Grouped Query Attention (GQA) ---
        _,_,seq_len, _ = query.shape
        key_states = self.repeat_kv(key, self.num_key_value_groups)
        value_states = self.repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * self.scale
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


def create_causal_mask(
    batch_size: int,
    seq_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    attention_mask=None,
) -> torch.Tensor:
    """
    Create a 4D causal mask for eager attention.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        device: Device to create mask on
        dtype: Data type for the mask
        attention_mask: Optional 2D padding mask (batch_size, seq_length) with 1s for valid tokens, 0s for padding
    
    Returns:
        4D mask of shape (batch_size, 1, seq_length, seq_length) with 0s where attention is allowed, -inf where masked
    """
    causal_mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.where(
        causal_mask, 
        torch.tensor(min_dtype, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device)
    )
    
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    if attention_mask is not None:
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=dtype, device=device)
        padding_mask = (1.0 - padding_mask) * min_dtype
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length) + padding_mask
    else:
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
    
    return causal_mask