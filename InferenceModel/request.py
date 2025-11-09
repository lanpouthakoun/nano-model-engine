class Request:
    request_id = 1
    def __init__(self, prompt, tokenizer):
        self.prompt = prompt
        self.tokens = tokenizer(prompt)
        self.num_tokens = len(self.tokens)
        self.status = "WAITING"
        self.id = Request.request_id
        Request.request_id += 1
        self.num_blocks = 0
        self.blocks = []
        self.completion_token_ids = []

    def __len__(self):
        return self.num_tokens
    def __getitem__(self, key):
        return self.token_ids[key]
    
    
    def add_token(self):
        """
        This function adds a token to an already created request
        adding to the block and possibly allocating extra blocks
        """
        pass

    def request_length(self):
        return len(self.tokens)

    def get_block(self, i):
        assert 0 <= i < self.num_blocks
        return self.tokens[i*self.block_size: (i+1)*self.block_size]

