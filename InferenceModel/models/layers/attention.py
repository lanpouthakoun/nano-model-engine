import torch
from torch import nn
import torch.nn.functional as F
# import triton
# import triton.language as tl

# @triton.jit
# def store_kvcache_kernel(
#     key_ptr,
#     key_stride,
#     value_ptr,
#     value_stride,
#     k_cache_ptr,
#     v_cache_ptr,
#     slot_mapping_ptr,
#     D: tl.constexpr,
# ):
#     idx = tl.program_id(0)
#     slot = tl.load(slot_mapping_ptr + idx)
#     if slot == -1: 
#         return
    
#     key_offsets = idx * key_stride + tl.arange(0, D)
#     value_offsets = idx * value_stride + tl.arange(0, D)
#     key = tl.load(key_ptr + key_offsets)
#     value = tl.load(value_ptr + value_offsets)
    
#     cache_offsets = slot * D + tl.arange(0, D)
#     tl.store(k_cache_ptr + cache_offsets, key)
#     tl.store(v_cache_ptr + cache_offsets, value)

# def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
#     N, num_heads, head_dim = key.shape
#     D = num_heads * head_dim
#     assert key.stride(-1) == 1 and value.stride(-1) == 1
#     assert key.stride(1) == head_dim and value.stride(1) == head_dim
#     assert k_cache.stride(1) == D and v_cache.stride(1) == D
#     assert slot_mapping.numel() == N
#     store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):
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
        self.k_cache = self.v_cache = torch.tensor([])
    
    def forward(self, query, key, value):
        """
        Compute scaled dot-product attention with KV caching.
        
        Args:
            query: (total_tokens, num_heads, head_dim)
            key: (total_tokens, num_kv_heads, head_dim)
            value: (total_tokens, num_kv_heads, head_dim)
        
        Returns:
            output: (total_tokens, num_heads, head_dim)
        """
        context = get_context() # make a context tool
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store new KV pairs in cache
        # if k_cache.numel() and v_cache.numel():
        #     # only store the projected kv_values
        #     store_kvcache(key, value, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            output = self._prefill_attention(query, key, value, context)
        else:
            output = self._decode_attention(query, k_cache, v_cache, context)
        
        return output
    
    def _prefill_attention(self, query, key, value, context):
        """
        Prefill attention: compute attention over prompt tokens.
        Handles variable-length sequences using cu_seqlens.
        """
        # query: (total_tokens, num_heads, head_dim) -> need batch dimension
        
        # Get cumulative sequence lengths to separate sequences
        cu_seqlens_q = context.cu_seqlens_q.cpu().tolist()
        cu_seqlens_k = context.cu_seqlens_k.cpu().tolist()
        
        num_seqs = len(cu_seqlens_q) - 1
        
        # If prefix cache is used, read from cache instead
        if context.block_tables is not None:
            key = self._read_from_cache(self.k_cache, context.block_tables, cu_seqlens_k)
            value = self._read_from_cache(self.v_cache, context.block_tables, cu_seqlens_k)
        
        if self.num_kv_heads < self.num_heads:
            key = self._repeat_kv(key, self.num_heads // self.num_kv_heads)
            value = self._repeat_kv(value, self.num_heads // self.num_kv_heads)
        
        outputs = []
        for i in range(num_seqs):
            q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
            
            q = query[q_start:q_end]  
            k = key[k_start:k_end]  
            v = value[k_start:k_end] 
            
            q = q.permute(1, 0, 2).unsqueeze(0) 
            k = k.permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0) 
            
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            seq_len_q, seq_len_k = q.size(2), k.size(2)
            causal_mask = self._get_causal_mask(seq_len_q, seq_len_k, q.device)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            output = torch.matmul(attn_weights, v)
            
            output = output.squeeze(0).permute(1, 0, 2)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)  
    
    def _decode_attention(self, query, k_cache, v_cache, context):
        """
        Decode attention: attend new token to cached KV.
        Each sequence generates exactly 1 new token.
        """
        batch_size = query.size(0)  # Number of sequences
        
        # query: (batch_size, num_heads, head_dim)
        # Each query attends to its own cached context
        
        outputs = []
        for i in range(batch_size):
            q = query[i:i+1] 
            
            # Get block table and context length for this sequence
            block_table = context.block_tables[i]  # (num_blocks,)
            context_len = context.context_lens[i].item()  # Total sequence length
            
            # Read cached KV for this sequence
            k = self._read_cached_kv(k_cache, block_table, context_len)  # (context_len, num_kv_heads, head_dim)
            v = self._read_cached_kv(v_cache, block_table, context_len)  # (context_len, num_kv_heads, head_dim)
            
            # Expand KV heads if using grouped-query attention
            if self.num_kv_heads < self.num_heads:
                k = self._repeat_kv(k, self.num_heads // self.num_kv_heads)
                v = self._repeat_kv(v, self.num_heads // self.num_kv_heads)
            
            # Reshape for attention
            q = q.unsqueeze(0).unsqueeze(2)  # (1, num_heads, 1, head_dim)
            k = k.permute(1, 0, 2).unsqueeze(0)  # (1, num_heads, context_len, head_dim)
            v = v.permute(1, 0, 2).unsqueeze(0)  # (1, num_heads, context_len, head_dim)
            
            # Compute attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # (1, num_heads, 1, context_len)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            output = torch.matmul(attn_weights, v)
            # (1, num_heads, 1, head_dim)
            
            # Reshape: (1, num_heads, 1, head_dim) -> (1, num_heads, head_dim)
            output = output.squeeze(0).squeeze(1)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)  # (batch_size, num_heads, head_dim)
    
    def _read_cached_kv(self, cache, block_table, context_len):
        """
        Read KV cache for a single sequence using its block table.
        
        Args:
            cache: (num_blocks, block_size, num_kv_heads, head_dim)
            block_table: (num_blocks,) - which blocks belong to this sequence
            context_len: Total length of the sequence
        
        Returns:
            (context_len, num_kv_heads, head_dim)
        """
        block_size = cache.size(1)
        num_blocks = (context_len + block_size - 1) // block_size
        
        kv_list = []
        for i in range(num_blocks):
            block_id = block_table[i].item()
            if block_id == -1:
                break
            
            block_data = cache[block_id]  # (block_size, num_kv_heads, head_dim)
            
            # Last block might not be full
            if i == num_blocks - 1:
                tokens_in_block = context_len - i * block_size
                block_data = block_data[:tokens_in_block]
            
            kv_list.append(block_data)
        
        return torch.cat(kv_list, dim=0)  # (context_len, num_kv_heads, head_dim)
    
    def _read_from_cache(self, cache, block_tables, cu_seqlens):
        """
        Read from cache for prefix caching scenario.
        """
        all_kv = []
        num_seqs = len(cu_seqlens) - 1
        
        for i in range(num_seqs):
            seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
            block_table = block_tables[i]
            kv = self._read_cached_kv(cache, block_table, seq_len)
            all_kv.append(kv)
        
        return torch.cat(all_kv, dim=0)
    
    def _repeat_kv(self, x, n_rep):
        """
        Repeat KV heads for grouped-query attention.
        x: (seq_len, num_kv_heads, head_dim)
        """
        if n_rep == 1:
            return x
        seq_len, num_kv_heads, head_dim = x.shape
        x = x.unsqueeze(2).expand(seq_len, num_kv_heads, n_rep, head_dim)
        return x.reshape(seq_len, num_kv_heads * n_rep, head_dim)
    
    def _get_causal_mask(self, seq_len_q, seq_len_k, device):
        q_indices = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_indices = torch.arange(seq_len_k, device=device).unsqueeze(0)
        
        offset = seq_len_k - seq_len_q
        q_indices = q_indices + offset
        
        mask = q_indices < k_indices  
        return mask.unsqueeze(0).unsqueeze(0)  