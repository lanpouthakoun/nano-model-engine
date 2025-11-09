import torch
from InferenceModel.request import Request
from InferenceModel.scheduler import Scheduler
from InferenceModel.model_executor import Executor
from dataclasses import fields
from transformers import AutoTokenizer
from InferenceModel.utils.config import Config


class Engine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.model_runner = Executor(config)


    def step(self):
        """
        self.scheduler.get_requests - This line is to get which requests to run in this step

        - Then, we run a forward pass
        then we post process everything
        """
        requests, _ = self.scheduler.schedule()
        new_tokens = self.model.run(requests)
        self.scheduler.postprocess(requests, new_tokens)
        outputs = [(request.request_id, request.completion_token_ids) for request in requests if request.is_finished]
        return outputs

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
        
        results = {}
        while not self.is_finished():
            output = self.step()
            for seq_id, token_ids in output:
                results[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        return outputs