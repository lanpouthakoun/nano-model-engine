from kv_manager import KVCacheManager
import torch
from torch import nn
from collections import deque
from request import Request
from utils.instructions import set_instructions
from utils.block import Block


class PagedAttentionManager(KVCacheManager):
    """
    I need to somehow organize and see how many blocks i need

    1. Calculate Num Blocks (could this be done in the kvcachemanager?)

        
    """
    def __init__(self, block_size):
        super().__init__(block_size)
        
        self.free_block_queue = deque([i for i in range(self.num_blocks)])
        self.block_size = block_size
        self.blocks = [] # blocks of 
        self.used_blocks = set()
        self.token_ids_to_block_id: dict[int, int] = dict()

    def _allocate(self):
        if not self.can_allocate(1):
            return False
        id = self.free_block_queue.popleft()
        self.used_blocks.add(id)
        return self.blocks[id]

    def allocate(self, req: Request):
        """"
        This is only called during prefill.

        We initially set the hash to be -1
        For the number of initial required blocks,
        we get the tokens associated with each block
        if it is already in the kv cache, we can increase its reference count (this is so we don't free something being used)

        We allocate a block if it doesnt exist in the kv cache
        If it does, we update its reference count
        """
        h = -1
        assert self.can_allocate(req.num_blocks)
        for i in range(req.num_blocks):
            token_ids = req.get_block(i)
            block_id = self.token_ids_to_block_id.get(token_ids, -1)
            if block_id == -1:
                block = self._allocate()
            else:
                req.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate() #freed the block but its still in the hash
            if h != -1:
                block.update(h, token_ids)
                self.token_ids_to_block_id[h] = block_id
            req.blocks.append(block_id)

    def free(self, request: Request):
        for block_id in reversed(request.blocks):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.deallocate_block(block_id)
            request.num_cached_tokens = 0
            request.block_table.clear()

    def deallocate_block(self, block_id):
        self.used_blocks.remove(block_id)
        self.free_block_queue.append(block_id)

    def can_allocate(self, n):
        # suppose each request has a number of blocks that we can 
        return n <= len(self.free_block_queue)
    

    def can_run(self, request: Request):
        pass


    def start_prefill(self, seqs: list[Request]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_instructions(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def start_decode(self, seqs: list[Request]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_instructions(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions
    def start_decode(self):
        pass

    def prepare_block_tables(self, seqs: list[Request]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables