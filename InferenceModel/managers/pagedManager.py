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

        
    I could in theory comp
    """
    def __init__(self, block_size):
        super().__init__(block_size)
        
        self.free_block_queue = deque([Block(i) for i in range(self.num_blocks)])
        self.block_size = block_size
        self.used_blocks = set()

    def allocate(self, request: Request):
        """
        This funciton allocates the appropriate amount of blocks for a given request.
        (Includes teh necessary blocks for an upcoming token generation)
        """
        necessary_blocks = request.num_blocks_needed
        
        for i in range(necessary_blocks):
            block = self.free_block_queue.popleft()
            request.add_block(block)
            self.used_blocks.add(block)

        
    def free(self, request: Request):
        blocks = request.get_blocks_for_free
        for block in blocks:
            self.used_blocks.remove(block)
            self.free_block_queue.append(block)

    def can_allocate(self, request: Request):
        necessary_blocks = request.num_blocks_needed
        if len(self.free_block_queue) < necessary_blocks:
            return False
        else:
            return True

    def can_run(self):
        pass

    def organize(self):
        return self.free_block_queue