from collections import deque


class Scheduler:
    def __init__(self, cache_manager, max_batch_size, eos = None):
        self.KVCacheManager = cache_manager
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
        pass

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


    def preempt(self):
        pass
