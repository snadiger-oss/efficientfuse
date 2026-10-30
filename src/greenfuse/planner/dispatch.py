"""Process-local executable reuse, distinct from the portable decision cache."""

from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from time import perf_counter

from greenfuse.execution.buffers import current_stream_id
from .cache import DecisionCache


@dataclass
class AcquiredExecution:
    prepared: object
    decision_key: str
    selected_id: str
    decision_cache_status: str
    live_executor_reused: bool
    decision_lookup_seconds: float
    reconstruction_seconds: float

    def run(self, inputs):
        """Steady-state calls use the live executor directly, without disk I/O."""
        return self.prepared.executor.run(inputs)

    def to_dict(self):
        return {"decision_key": self.decision_key, "selected_id": self.selected_id,
                "decision_cache_status": self.decision_cache_status, "live_executor_reused": self.live_executor_reused,
                "decision_lookup_seconds": self.decision_lookup_seconds,
                "reconstruction_seconds": self.reconstruction_seconds,
                "compiler_cache_status": "managed by PyTorch; not inferred from a decision-cache hit",
                "setup_energy": None, "break_even_executions": None}


class ExecutorRegistry:
    """Single-caller registry. CUDA stream identity is part of each live key."""

    def __init__(self):
        self.owner = get_ident()
        self.entries = {}
        self.closed = False

    def acquire(self, key, factory, *, context_token, require_gpu=True):
        if self.closed or get_ident() != self.owner:
            raise RuntimeError("Executor registry is closed or accessed from another thread")
        live_key = key, context_token
        if live_key in self.entries:
            return self.entries[live_key], True, 0.0
        start = perf_counter()
        prepared = factory()
        try:
            if prepared.executor is None:
                raise RuntimeError(prepared.record.reason or "Selected executor could not be prepared")
            if require_gpu and (prepared.executor.device.type != "cuda" or not prepared.record.ready_for_gpu_measurement):
                raise RuntimeError("Selected GPU executor failed correctness or mode verification")
            if prepared.record.correctness is None or prepared.record.correctness.get("status") != "passed":
                raise RuntimeError("Selected executor lacks correctness evidence")
        except Exception:
            prepared.close()
            raise
        self.entries[live_key] = prepared
        return prepared, False, perf_counter() - start

    def close(self):
        errors = []
        for prepared in self.entries.values():
            try:
                prepared.close()
            except Exception as error:
                errors.append(str(error))
        self.entries.clear()
        self.closed = True
        if errors:
            raise RuntimeError("Executor cleanup failed: " + "; ".join(errors))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CachedDispatcher:
    def __init__(self, cache_dir="results/decisions", artifact_dir="results/dispatch"):
        self.cache = DecisionCache(cache_dir)
        self.artifact_dir = Path(artifact_dir)
        self.registry = ExecutorRegistry()

    def acquire(self, context):
        start = perf_counter()
        lookup = self.cache.lookup(context.identity)
        lookup_seconds = perf_counter() - start
        if lookup.status != "hit":
            raise RuntimeError(f"Decision cache {lookup.status}: {lookup.reason}; run calibration for this configuration")
        if context.environment.get("status") != "cuda_available" or context.environment.get("energy", {}).get("reason"):
            raise RuntimeError("Cached GPU execution requires a matching, qualified exclusive GPU environment")
        inputs = context.ensure_inputs()
        device = next(iter(inputs.values())).device
        token = get_ident(), current_stream_id(device)
        selected_id = lookup.decision["selected_id"]
        prepared, reused, elapsed = self.registry.acquire(
            lookup.key, lambda: context.prepare_candidate(selected_id, self.artifact_dir / lookup.key / selected_id, profile=True),
            context_token=token)
        return AcquiredExecution(prepared, lookup.key, selected_id, lookup.status, reused, lookup_seconds, elapsed)

    def close(self):
        self.registry.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
