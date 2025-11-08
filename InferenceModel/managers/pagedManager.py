from kv_manager import KVCacheManager
import torch
from torch import nn
from collections import deque
from request import Request


class Block:

    def __init__(self, block_id):
        self.block_id = block_id

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