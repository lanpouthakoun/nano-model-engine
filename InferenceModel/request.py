import torch
from torch import nn

class Request:
    request_id = 1
    def __init__(self, prompt, tokenizer):
        self.prompt = prompt
        self.tokens_ids = tokenizer(prompt)
        self.num_tokens = len(self.tokens)
        self.num_tokens_so_far = 0
        self.status = "WAITING"
        self.id = Request.request_id
        Request.request_id += 1
        self.num_blocks = 0
        self.block_table = []
        self.eos_token = tokenizer.eos_token


    def __len__(self):
        return self.num_tokens
    def __getitem__(self, key):
        return self.token_ids[key]
    
    
    def add_token(self, token):
        """
        This function adds a token to an already created request
        adding to the block and possibly allocating extra blocks
        This should be called at the end of a step.
        """
        self.token_ids.append(token)
        self.num_tokens_so_far += 1
        if token == self.eos_token:
            self.status = 'COMPLETE'
        
    def add_block(self, block_id):
        self.num_blocks += 1
        self.block_table.append(block_id)

    def prefill_blocks(self, blocks):
        """
        This is called when we are doing prefill on a request,
        it gives the request the appropriate set of blocks
        also expects to change its status to running
        """
        self.status = 'RUNNING'
        self.block_table = blocks
        self.num_blocks = len(blocks)
    


    def request_length(self):
        return len(self.tokens)

    def get_block(self, i):
        assert 0 <= i < self.num_blocks
        return self.tokens[i*self.block_size: (i+1)*self.block_size]

