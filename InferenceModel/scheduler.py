from collections import deque
from request import Request
from managers.pagedManager import PagedAttentionManager


class Scheduler:
    def __init__(self, cache_manager, max_batch_size, eos = None, block_size = 32):
        self.KVCacheManager = PagedAttentionManager(block_size)
        self.waiting = deque()
        self.running = deque()
        self.max_batch_size = max_batch_size
        self.eos = eos # End of String Token

    def add_request(self, request):
        """
        This function adds a request to the queue
        """
        self.waiting.append(request)

    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0

    def schedule(self):
        """
        This function decides what to run at this specific iteration, it decides when to allocate requests (move from waiting to running)
        """
        num_requests = 0
        current_step = []
        # we want to prioritize getting some outputs at least, so prioritize maximizing batch_size always
        while self.waiting and num_requests < self.max_batch_size:
            # We want to add all requests from the waiting to the runnign queue
            if not self.KVCacheManager.can_allocate(self.waiting[0]):
                break
            request = self.waiting.popleft()
            blocks = self.KVCacheManager.allocate(request)
            num_requests += 1
            request.prefill_blocks(blocks)
            self.running.append(request)
            current_step.append(request)
        if current_step: # we want to only run prefill requests together
            return current_step, True
        
        # Here, we decode

        while self.running and num_requests < self.max_batch_size:
            request = self.running.popleft()
            while not self.KVCacheManager.can_allocate(request): # can we even decode this request?
                # if not, preempt other requests
                if self.running:
                    self.preempt(self.running.popleft())
                else:
                    self.preempt(request)
                    break
            else:
                self.KVCacheManager.allocate_decode(request)
                num_requests += 1
                current_step.append(request)
        for i in current_step:
            self.running.append(i)
        return current_step, False

    
    def preempt(self, req: Request):
        """
        We were generating this request, but we no longer have space,
        make this a waiting request again and come back to it next up
        """
        req.set_status() = "WAITING"
        self.KVCacheManager.free(req)
        self.waiting.appendleft(req)

    def postproc(self, requests, new_tokens):
        """
        This function takes in a list of requests that have been processed and a list of new_tokens to add to those requests

        It also decides when to deallocate a request
        
        """
        for request, token in zip(requests, new_tokens):
            request.add_token(token)
            if token == self.eos or request.num_tokens == self.max_tokens:
                request.status = "COMPLETE"
                self.KVCacheManager.free(request)
                self.running.remove(request)