"""
Implementing a trie based structure for holding blocks. Never get rid of blocks. 
"""
from kv_manager import KVCacheManager
import torch
from torch import nn
from typing import Optional, Dict, List
from collections import deque


class Block:

    def __init__(self, block_id):
        self.block_id = block_id



class RadixAttentionManager(KVCacheManager):
    def __init__(self):
        self.root = RadixNode()
        self.free_block_queue = deque(self.num_blocks)


    def allocate(self, request):
        pass

    def free(self, requests):
        pass

    def match_prefix(self, requests):
        """
        Take in many requests, find the prefix nodes that match
        """
        pass
