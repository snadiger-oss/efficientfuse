"""Compile groups independently. Never compile the outer group orchestrator."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from greenfuse.execution.buffers import InputContract, synchronize


@dataclass(frozen=True)
class CompilerConfig:
    """Fixed controlled policy. Explicit replay belongs only to graphs.py."""

    artifact_dir: str | None = None
    # default (cuBLAS), autotune (cuBLAS vs Triton), triton (Triton only), or fuse_epilogue:
    # a group that fuses a GEMM with other operations uses a Triton GEMM (needed to fuse the
    # epilogue into it); a group holding only the GEMM keeps cuBLAS.
    gemm_backend: str = "default"

    def options(self, available_options) -> dict:
        """Options shared by every group (fuse_epilogue adds per-group options; see group_options)."""
        required = {"triton.cudagraphs": False}
        if self.gemm_backend not in ("default", "autotune", "triton", "fuse_epilogue"):
            raise ValueError("gemm_backend must be default, autotune, triton, or fuse_epilogue")
        if self.gemm_backend in ("autotune", "triton"):
            # GEMM autotuning only: plain max_autotune would also turn on Inductor's own CUDA Graphs.
            # Triton templates can fuse the bias/GELU epilogue into the GEMM kernel.
            required["max_autotune_gemm"] = True
            required["max_autotune_gemm_backends"] = "TRITON" if self.gemm_backend == "triton" else "ATEN,TRITON"
        if self.artifact_dir is not None:
            required.update({"trace.enabled": True, "trace.debug_dir": str(Path(self.artifact_dir).resolve())})
        missing = set(required) - set(available_options)
        if missing:
            raise RuntimeError(f"Installed Inductor lacks required options: {sorted(missing)}")
        return required

    def group_options(self, base: dict, module) -> dict:
        """Per-group options. Only fuse_epilogue differs between groups."""
        if self.gemm_backend != "fuse_epilogue":
            return dict(base)
        import operator
        import torch

        targets = [node.target for node in module.graph.nodes if node.op in ("call_function", "call_method")]
        has_gemm = any(t in (operator.matmul, torch.matmul, "matmul") for t in targets)
        if has_gemm and len(targets) > 1:
            return {**base, "max_autotune_gemm": True, "max_autotune_gemm_backends": "TRITON"}
        return dict(base)


@dataclass
class CompilationRecord:
    backend: str = "inductor"
    fullgraph: bool = True
    dynamic: bool = False
    options: dict = field(default_factory=dict)
    torch_version: str | None = None
    cuda_build: str | None = None
    device: str | None = None
    groups: list[dict] = field(default_factory=list)
    setup_seconds: float = 0.0
    generated_code_files: list[str] = field(default_factory=list)
    generated_code_note: str = "Not requested; source-file counts are not kernel counts"

    def to_dict(self):
        return asdict(self)


class CompilationError(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


@dataclass
class CompiledStep:
    callable: Any
    input_slots: tuple[int, ...]
    output_id: str
    output_count: int = 1  # >1 when a group exports an intermediate for fan-out


@dataclass
class CompiledPartition:
    partition_id: str
    graph_fingerprint: str
    contract: InputContract
    steps: tuple[CompiledStep, ...]
    output_slot: int
    record: CompilationRecord

    def evaluate(self, ordered_inputs: tuple):
        """Return output plus intermediate owners; caller sets inference context."""
        values = list(ordered_inputs)
        for step in self.steps:
            result = step.callable(*(values[index] for index in step.input_slots))
            values.extend(result if step.output_count > 1 else (result,))
        return values[self.output_slot], tuple(values[len(ordered_inputs):])

    def close(self):
        self.steps = ()


def compile_partition(built, inputs: dict, *, config=None, allow_cpu=False) -> CompiledPartition:
    import torch
    import torch._inductor

    config = config or CompilerConfig()
    names = tuple(item.name for item in built.inputs)
    contract = InputContract.from_inputs(inputs, names, allow_cpu=allow_cpu)
    record = CompilationRecord(
        torch_version=str(torch.__version__), cuda_build=torch.version.cuda, device=str(contract.device),
        groups=[{"index": i, "node_ids": list(group.bindings.node_ids), "status": "not_attempted"}
                for i, group in enumerate(built.groups)],
    )
    started = perf_counter()
    existing_artifacts = set()
    try:
        if torch._dynamo.config.suppress_errors:
            raise RuntimeError("Disable torch._dynamo.config.suppress_errors; compilation failures must stay visible")
        record.options = config.options(torch._inductor.list_options())
        if config.artifact_dir:
            Path(config.artifact_dir).mkdir(parents=True, exist_ok=True)
            existing_artifacts = set(Path(config.artifact_dir).rglob("output_code.py"))
        values = list(contract.validate(inputs))
        slots = {item.id: i for i, item in enumerate(built.inputs)}
        steps = []
        with torch.inference_mode():
            for index, group in enumerate(built.groups):
                details = record.groups[index]
                details["status"] = "compiling"
                group_started = perf_counter()
                input_slots = tuple(slots[key] for key in group.bindings.input_ids)
                options = config.group_options(record.options, group.module)
                if options != record.options:
                    details["options"] = options
                fn = torch.compile(group.module, backend="inductor", fullgraph=True, dynamic=False, options=options)
                # Compilation is lazy: invoke the wrapper and synchronize here.
                output = fn(*(values[slot] for slot in input_slots))
                synchronize(contract.device)
                details.update(status="compiled", setup_seconds=perf_counter() - group_started)
                exported = group.bindings.output_ids
                outputs = tuple(output) if len(exported) > 1 else (output,)
                for key, value in zip(exported, outputs):
                    slots[key] = len(values)
                    values.append(value)
                steps.append(CompiledStep(fn, input_slots, group.bindings.output_id, len(exported)))
        record.setup_seconds = perf_counter() - started
        if config.artifact_dir:
            generated = set(Path(config.artifact_dir).rglob("output_code.py")) - existing_artifacts
            record.generated_code_files = [str(path.resolve()) for path in sorted(generated)]
            record.generated_code_note = "Inductor debug artifacts; may be absent on compiler cache hits; never a kernel-count proxy"
        return CompiledPartition(built.spec.id, built.spec.graph_fingerprint, contract, tuple(steps), slots[built.output_id], record)
    except Exception as error:
        record.setup_seconds = perf_counter() - started
        for details in record.groups:
            if details["status"] == "compiling":
                details.update(status="failed", reason=str(error))
        raise CompilationError(f"Inductor preparation failed: {error}", record) from error
