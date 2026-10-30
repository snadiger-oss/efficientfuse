"""Ordinary execution of independently compiled groups."""

from time import perf_counter

from .buffers import check_stream, current_stream_id, synchronize


class LaunchExecutor:
    mode = "launch"

    def __init__(self, compiled):
        self.compiled = compiled
        self.contract = compiled.contract
        self.device = self.contract.device
        self.stream_id = current_stream_id(self.device)
        self.closed = False
        self.setup_seconds = {}

    def run(self, inputs: dict):
        import torch

        if self.closed:
            raise RuntimeError("Executor is closed")
        check_stream(self.device, self.stream_id)
        ordered = self.contract.validate(inputs)
        with torch.inference_mode():
            output, _ = self.compiled.evaluate(ordered)
        return output

    def warmup(self, inputs: dict, iterations: int = 3):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("Warmup iterations must be a positive integer")
        start = perf_counter()
        for _ in range(iterations):
            self.run(inputs)
        synchronize(self.device)
        self.setup_seconds["warmup"] = perf_counter() - start

    def buffer_contract(self):
        return {**self.contract.to_dict(), "input_update": "use caller's resident inputs directly", "static_input_bytes": 0}

    def close(self):
        if not self.closed:
            synchronize(self.device)
            self.compiled.close()
            self.closed = True
