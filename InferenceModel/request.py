class Request:
    request_id = 1
    def __init__(self, prompt, tokenizer):
        self.prompt = prompt
        self.tokens = tokenizer(prompt)
        # Create a static variable to keep track of request ids
        self.status = "WAITING"
        self.id = Request.request_id
        Request.request_id += 1


    def add_token(self):
        """
        This function adds a token to an already created request
        adding to the block and possibly allocating extra blocks
        """
        pass

