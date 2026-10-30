"""Interpret an isolated, one-invocation profiler trace; never guess kernels."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

INVOCATION_MARKER = "greenfuse.logical_invocation"


@dataclass
class KernelEvidence:
    status: str = "unavailable"
    reason: str | None = None
    kernel_launches_per_invocation: int | None = None
    kernel_names: list[str] = field(default_factory=list)
    graph_launch_api_events: int | None = None
    replay_check: str = "unavailable"
    trace_path: str | None = None
    logical_invocations: int = 1

    def to_dict(self):
        return asdict(self)


def summarize_kernel_trace(trace: dict, *, mode: str, device_index: int) -> KernelEvidence:
    """Consume the Chrome trace from profile_executor, not an arbitrary session."""
    if mode not in ("launch", "graph"):
        raise ValueError("Expected launch or graph mode")
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        return KernelEvidence(reason="Trace has no event list")
    complete = [event for event in events if isinstance(event, dict) and event.get("ph") == "X"]
    # Newer PyTorch also mirrors the marker onto the GPU timeline as gpu_user_annotation; bound by the CPU range.
    markers = [event for event in complete if event.get("name") == INVOCATION_MARKER
               and event.get("cat") != "gpu_user_annotation"]
    if len(markers) != 1:
        return KernelEvidence(reason="Expected exactly one complete logical-invocation marker")

    def interval(event):
        values = event.get("ts"), event.get("dur")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            return None
        start, duration = values
        return (start, start + duration) if duration >= 0 else None

    bounds = interval(markers[0])
    if bounds is None or bounds[0] == bounds[1]:
        return KernelEvidence(reason="Invalid logical-invocation interval")
    for event in events:
        if not isinstance(event, dict) or "kernel" not in str(event.get("cat", "")).split(","):
            continue
        span = interval(event)
        if span is None or event.get("ph") != "X":
            return KernelEvidence(reason="Incomplete GPU kernel events; refusing a partial count")
        if span[0] < bounds[1] and span[1] > bounds[0] and not (
            span[0] >= bounds[0] and span[1] <= bounds[1]
        ):
            return KernelEvidence(reason="GPU work crosses the invocation boundary; scope is ambiguous")
    selected = []
    for event in complete:
        span = interval(event)
        if span is not None and span[0] >= bounds[0] and span[1] <= bounds[1]:
            selected.append(event)
    kernels = [event for event in selected if "kernel" in str(event.get("cat", "")).split(",")]
    runtime = [event for event in selected if event.get("cat") in ("cuda_runtime", "cuda_driver")]
    graph_launches = sum("graphlaunch" in str(event.get("name", "")).lower() for event in runtime)
    if mode == "launch" and graph_launches:
        return KernelEvidence(
            status="invalid", reason="Ordinary execution unexpectedly submitted a CUDA Graph",
            graph_launch_api_events=graph_launches, replay_check="failed",
        )
    if not kernels:
        return KernelEvidence(reason="No GPU kernel events visible; graph submission alone is not a kernel count", graph_launch_api_events=graph_launches if runtime else None)
    for event in kernels:
        args = event.get("args", {})
        if not isinstance(args, dict) or args.get("device") != device_index:
            return KernelEvidence(reason="Kernel device is missing or does not match the requested CUDA device")
    launches = [event for event in runtime if "launchkernel" in str(event.get("name", "")).lower()]
    replay_verified = bool(graph_launches) if mode == "graph" else bool(launches)
    return KernelEvidence(
        status="observed",
        reason=None if replay_verified else "Kernel events are visible, but runtime launch events cannot verify the replay policy",
        kernel_launches_per_invocation=len(kernels),
        kernel_names=[str(event.get("name", "unknown")) for event in kernels],
        graph_launch_api_events=graph_launches if runtime else None,
        replay_check="passed" if replay_verified else "unavailable",
    )


def profile_executor(executor, inputs: dict, trace_path) -> KernelEvidence:
    """Diagnostic pass only. Person 3 owns the later measurement harness."""
    import torch

    if executor.device.type != "cuda":
        return KernelEvidence(reason="GPU kernel counts require CUDA; CPU execution is only a development check", replay_check="not_applicable")
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        return KernelEvidence(reason="CUDA profiler activity is unavailable in this PyTorch environment")
    path = Path(trace_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        executor.run(inputs)
        torch.cuda.synchronize(executor.device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            with torch.profiler.record_function(INVOCATION_MARKER):
                executor.run(inputs)
                torch.cuda.synchronize(executor.device)
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        evidence = summarize_kernel_trace(trace, mode=executor.mode, device_index=executor.device.index)
        evidence.trace_path = str(path.resolve())
        return evidence
    except Exception as error:
        return KernelEvidence(reason=f"Profiler evidence unavailable: {type(error).__name__}: {error}")
