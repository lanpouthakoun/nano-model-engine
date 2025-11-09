import torch
from torch import nn
from typing import Optional, Dict, List, Tuple
from collections import deque
from utils.block import Block

class RadixNode:
    def __init__(self):
        self.tokens: List[int] = []
        # self.children: Dict[int, RadixNode] = {}
        self.ref_count = 0
        self.block_ids: List[int] = []
        self.ref_count: int = 0
        self.children: Dict[int, Tuple[List[int], 'RadixNode']] = {}
        self.is_terminal: bool = False
    def __repr__(self):
        return (f"RadixNode(terminal={self.is_terminal}, "
                f"ref={self.ref_count}, "
                f"blocks={self.block_ids}, "
                f"children={len(self.children)})")




class RadixTree:
    def __init__(self, block_size):
        self.root = RadixNode()
        self.block_size = block_size
        self.free_block_queue = deque([i for i in range(self.num_blocks)])

    def insert(self, tokens: List[int]) -> Tuple[RadixNode, int, List[int]]:
        if not tokens:
            return self.root, 0, []
        matched_blocks, allocated_blocks = self._recurse(
            self.root,
            tokens,
            token_offset=0,
            path_tokens=[]
        )
        
        num_cached_tokens = matched_blocks * self.block_size
        
        return self.root, num_cached_tokens, allocated_blocks
    def recurse(self):
        pass