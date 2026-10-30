"""Explicit whole-partition CUDA Graph capture and replay; no fallback mode."""

from time import perf_counter

from .buffers import StaticBuffers, UnsupportedExecutionError, check_stream, current_stream_id, synchronize


class GraphExecutor:
    mode = "graph"

    @classmethod
    def prepare(cls, compiled, inputs: dict, *, warmup_iterations: int = 3):
        import torch

        device = compiled.contract.device
        if device.type != "cuda" or not torch.cuda.is_available():
            raise UnsupportedExecutionError("Explicit CUDA Graph execution requires an available NVIDIA CUDA device")
        if type(warmup_iterations) is not int or warmup_iterations < 1:
            raise ValueError("Warmup iterations must be a positive integer")
        self = cls()
        self.compiled = compiled
        self.contract = compiled.contract
        self.device = device
        self.stream_id = current_stream_id(device)
        self.closed = False
        self.setup_seconds = {}
        self.buffers = StaticBuffers(self.contract, inputs)
        self.graph = None
        self.output = None
        self.intermediates = ()
        try:
            with torch.cuda.device(device), torch.inference_mode():
                side = torch.cuda.Stream(device=device)
                side.wait_stream(torch.cuda.current_stream(device))
                start = perf_counter()
                with torch.cuda.stream(side):
                    for _ in range(warmup_iterations):
                        output, intermediates = compiled.evaluate(self.buffers.values)
                torch.cuda.current_stream(device).wait_stream(side)
                synchronize(device)
                del output, intermediates
                self.setup_seconds["warmup"] = perf_counter() - start
                start = perf_counter()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=side):
                    self.output, self.intermediates = compiled.evaluate(self.buffers.values)
                synchronize(device)
                self.setup_seconds["capture"] = perf_counter() - start
            return self
        except Exception:
            # Keep the original failure visible even if CUDA cleanup also fails.
            try:
                self.close()
            except Exception:
                pass
            raise

    def run(self, inputs: dict):
        import torch

        if self.closed:
            raise RuntimeError("Executor is closed")
        check_stream(self.device, self.stream_id)
        with torch.cuda.device(self.device), torch.inference_mode():
            # Copies are intentionally part of every logical run, even when the
            # caller repeats the same input values. Measurement must include them.
            self.buffers.update(inputs)
            self.graph.replay()
        return self.output

    def buffer_contract(self):
        return {
            **self.contract.to_dict(),
            "input_update": "copy all caller inputs into owned buffers on every run",
            "static_input_bytes": self.buffers.nbytes,
            "retained_output_and_intermediate_bytes": sum(value.numel() * value.element_size() for value in self.intermediates),
            "pool_note": "Graph-private allocator pool may reserve additional memory",
        }

    def close(self):
        import torch

        if not self.closed:
            with torch.cuda.device(self.device):
                synchronize(self.device)
                if self.graph is not None:
                    self.graph.reset()
                self.graph = None
                self.output = None
                self.intermediates = ()
                self.buffers.close()
                self.compiled.close()
                self.closed = True
