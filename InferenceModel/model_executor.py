import torch
from torch import nn
from pagedManager import PagedAttentionManager

class Executor:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager

    def execute_model(self):
        pass

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    
    def sample(self):
        pass
    def run(self, requests, prefill):
        """
        Start prefill and start decode are both functions 
        """
        if prefill:
            input_ids, positions = self.cache_manager.start_prefill(requests)
        else:
            input_ids, positions = self.cache_manager.start_decode(requests)
        
        temperatures = self.prepare_sample(requests) if self.rank == 0 else None

        logits = self.model.compute_logits(self.model(input_ids, positions))
        token_ids = self.sample(logits, temperatures).tolist() if self.rank == 0 else None
        reset_instructions()
        return token_ids
