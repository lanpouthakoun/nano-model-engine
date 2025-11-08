import torch
from InferenceModel.request import Request
from InferenceModel.scheduler import Scheduler


class Engine:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.scheduler = Scheduler()


    def step(self):
        """
        self.scheduler.get_requests - This line is to get which requests to run in this step

        - Then, we run a forward pass
        then we post process everything
        """
        requests = self.scheduler.schedule()
        new_tokens = self.model.run(requests)
        outputs = self.scheduler.postproc(requests, new_tokens) 



    def is_finished(self):
        return self.scheduler.is_finished()
    
    def add_request(self, prompt):
        self.scheduler.add_request(Request(prompt))
    def generate(self,prompts):
        """
        This function takes in a list of formatted prompts

        It tokenizes these prompts and turns them all into Request Objects

        We then use the scheduler to generate all possible things

        Step
        
        """

        for prompt in prompts:
            self.add_request(prompt)
        
        # While queue is not empty:
            # we call step
        results = {}
        while not self.is_finished():
            output = self.step()
            for seq_id, token_ids in output:
                results[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        return outputs