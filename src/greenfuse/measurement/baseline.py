"""Original eager workload as an external baseline, excluded from selection."""

from greenfuse.execution.buffers import InputContract, check_stream, current_stream_id, synchronize
from greenfuse.execution.prepare import ExecutionRecord, PreparedExecution


class EagerExecutor:
    mode = "launch"

    def __init__(self, workload, inputs, *, allow_cpu=False):
        self.workload = workload
        self.contract = InputContract.from_inputs(inputs, tuple(inputs), allow_cpu=allow_cpu)
        self.device = self.contract.device
        self.stream_id = current_stream_id(self.device)
        self.closed = False

    def run(self, inputs):
        import torch
        if self.closed:
            raise RuntimeError("Executor is closed")
        check_stream(self.device, self.stream_id)
        self.contract.validate(inputs)
        with torch.inference_mode():
            return self.workload(**inputs)

    def close(self):
        if not self.closed:
            synchronize(self.device)
            self.closed = True


def prepare_eager(workload, inputs, graph_fingerprint, *, allow_cpu=False):
    executor = EagerExecutor(workload, inputs, allow_cpu=allow_cpu)
    record = ExecutionRecord("eager", graph_fingerprint, "launch", 0,
                             status="prepared", device=str(executor.device),
                             buffers={**executor.contract.to_dict(), "input_update": "use caller GPU-resident inputs directly"})
    return PreparedExecution(record, executor)
