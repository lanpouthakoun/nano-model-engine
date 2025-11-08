from collections import deque
from abc import ABC, abstractmethod

class KVCacheManager(ABC):
    def __init__(self, ):
        self.free_block_queue = deque()        
        pass

    @abstractmethod
    def allocate(self,):
        """
        Synonomous to Allocate Slots
        """
        pass

    @abstractmethod
    def free(self):
        pass
    
    @abstractmethod
    def can_allocate(self):
        pass


