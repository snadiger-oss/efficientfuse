"""Prepare one requested partition/mode. This is not the search/selection planner."""

from dataclasses import asdict, dataclass, field
import math
from time import perf_counter

from greenfuse.compiler.inductor import CompilationError, CompilerConfig, compile_partition
from greenfuse.compiler.inspect import KernelEvidence, profile_executor
from greenfuse.measurement.correctness import check_correctness
from .buffers import InputContract, UnsupportedExecutionError, synchronize
from .graphs import GraphExecutor
from .launch import LaunchExecutor


@dataclass
class ExecutionRecord:
    partition_id: str
    graph_fingerprint: str
    mode: str
    requested_groups: int
    status: str = "pending"
    device: str | None = None
    failure_stage: str | None = None
    reason: str | None = None
    compilation: dict | None = None
    correctness: dict | None = None
    buffers: dict | None = None
    profiling: dict = field(default_factory=lambda: KernelEvidence(reason="Not profiled").to_dict())
    setup_seconds: dict = field(default_factory=dict)
    ready_for_gpu_measurement: bool = False

    @property
    def id(self):
        return f"{self.partition_id}_{self.mode}"

    def to_dict(self):
        return {"id": self.id, **asdict(self)}


@dataclass
class PreparedExecution:
    record: ExecutionRecord
    executor: object | None = None

    def close(self):
        if self.executor is not None:
            self.executor.close()


def prepare_execution(built, inputs: dict, *, mode="launch", reference=None,
                      config: CompilerConfig | None = None, allow_cpu=False,
                      warmup_iterations=3, rtol=1e-5, atol=1e-6, trace_path=None) -> PreparedExecution:
    """Return an executor and evidence, or a visible failure with no executor."""
    if mode not in ("launch", "graph"):
        raise ValueError("mode must be launch or graph")
    record = ExecutionRecord(built.spec.id, built.spec.graph_fingerprint, mode, len(built.groups))
    compiled = None
    executor = None
    stage = "support"
    start = perf_counter()
    try:
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (rtol, atol)):
            raise ValueError("Correctness tolerances must be finite and nonnegative")
        contract = InputContract.from_inputs(inputs, tuple(item.name for item in built.inputs), allow_cpu=allow_cpu)
        record.device = str(contract.device)
        if mode == "graph" and contract.device.type != "cuda":
            raise UnsupportedExecutionError("CUDA Graph mode cannot run on the CPU")
        if type(warmup_iterations) is not int or warmup_iterations < 1:
            raise ValueError("Warmup iterations must be a positive integer")
        stage = "compilation"
        compiled = compile_partition(built, inputs, config=config, allow_cpu=allow_cpu)
        record.compilation = compiled.record.to_dict()
        record.setup_seconds["compilation"] = compiled.record.setup_seconds
        stage = "capture" if mode == "graph" else "warmup"
        if mode == "graph":
            executor = GraphExecutor.prepare(compiled, inputs, warmup_iterations=warmup_iterations)
        else:
            executor = LaunchExecutor(compiled)
            executor.warmup(inputs, warmup_iterations)
        record.setup_seconds.update(executor.setup_seconds)
        record.buffers = executor.buffer_contract()
        stage = "correctness"
        check_start = perf_counter()
        reference = reference or (lambda **values: built.run(values))
        record.correctness = check_correctness(executor, reference, inputs, rtol=rtol, atol=atol)
        record.setup_seconds["correctness"] = perf_counter() - check_start
        stage = "profiling"
        if trace_path is not None:
            profile_start = perf_counter()
            evidence = profile_executor(executor, inputs, trace_path)
            record.setup_seconds["profiling"] = perf_counter() - profile_start
            record.profiling = evidence.to_dict()
            if evidence.replay_check == "failed":
                raise RuntimeError(evidence.reason)
        record.ready_for_gpu_measurement = contract.device.type == "cuda" and record.profiling["replay_check"] == "passed"
        record.status = "ready" if record.ready_for_gpu_measurement else (
            "cpu_validated" if contract.device.type == "cpu" else "correct_profile_unverified"
        )
        record.setup_seconds["total"] = perf_counter() - start
        return PreparedExecution(record, executor)
    except Exception as error:
        record.status = "unsupported" if isinstance(error, UnsupportedExecutionError) else "failed"
        record.failure_stage = stage
        record.reason = f"{type(error).__name__}: {error}"
        if isinstance(error, CompilationError):
            record.compilation = error.record.to_dict()
        if stage == "correctness":
            record.correctness = {"status": "failed", "reason": str(error)}
        try:
            if executor is not None:
                executor.close()
            elif compiled is not None:
                synchronize(compiled.contract.device)
                compiled.close()
        except Exception as cleanup_error:
            record.reason += f"; cleanup also failed: {cleanup_error}"
        record.setup_seconds["total"] = perf_counter() - start
        return PreparedExecution(record)
