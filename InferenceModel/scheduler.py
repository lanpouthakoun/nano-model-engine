from collections import deque
from request import Request
from managers.pagedManager import PagedAttentionManager


class Scheduler:
    def __init__(self, cache_manager, max_batch_size, eos = None, block_size = 32):
        self.KVCacheManager = PagedAttentionManager(block_size)
        self.waiting = deque()
        self.running = []
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
        resulting_requests = []

        while self.waiting and num_requests < self.max_batch_size:
            request = self.waiting[0]
            if not self.KVCacheManager.can_allocate(request):
                break
            num_requests += 1
            self.KVCacheManager.allocate(request)
            # check here if we can even process this request

            request.set_status("RUNNING")
            self.waiting.popleft()
            self.running.append(request)
            resulting_requests.append(request)
        
        if resulting_requests:
            # this means we have things to prefill
            return resulting_requests, True
    
        while self.running and num_requests < self.max_batch_size:
            req = self.running.popleft()
            while not self.KVCacheManager.can_run(req): #were in a queue, so we want the front one to run the most
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(req)
                    break
            else:
                resulting_requests.append(req)
                num_requests += 1

        return resulting_requests, False
    
    def preempt(self, req: Request):
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
                request.status = "FINISHED"
                self.KVCacheManager.deallocate(request)
                self.running.remove(request)