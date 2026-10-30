"""Measure supplied candidates; selecting/caching a winner belongs to Person 4."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Callable
import uuid

from greenfuse.execution.buffers import synchronize
from greenfuse.results import SCHEMA_VERSION, Trial, fingerprint, portable, save_report, summarize_candidate
from .correctness import check_correctness
from .energy import NvmlEnergyMeter, measure_energy
from .profiling import profile_prepared
from .timing import calibrate_iterations, measure_timing


@dataclass(frozen=True)
class MeasurementConfig:
    rounds: int = 5
    window_seconds: float = 3.0
    seed: int = 0
    warmup_iterations: int = 3
    max_iterations: int = 10_000_000
    allow_cpu: bool = False
    rtol: float = 1e-5
    atol: float = 1e-6

    def __post_init__(self):
        for key in ("rounds", "warmup_iterations", "max_iterations"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or type(self.allow_cpu) is not bool:
            raise ValueError("seed must be an integer and allow_cpu a boolean")
        if type(self.window_seconds) not in (float, int) or not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        if not self.allow_cpu and self.window_seconds < 3:
            raise ValueError("GPU windows must be at least three seconds")
        for value in (self.rtol, self.atol):
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("Tolerances must be finite and nonnegative")


@dataclass(frozen=True)
class MeasurementCandidate:
    id: str
    # Factory(path_for_this_attempt) returns Person 2's PreparedExecution.
    prepare: Callable
    role: str = "candidate"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.id):
            raise ValueError("Candidate IDs must be safe simple names")
        if self.role not in ("candidate", "external_baseline") or not callable(self.prepare):
            raise ValueError("Invalid candidate role or factory")


def trial_schedule(ids, rounds, seed):
    """Separate passes and a reproducible shuffle within every round."""
    rng = random.Random(seed)
    for kind in ("timing", "energy"):
        for round_index in range(rounds):
            order = list(ids)
            rng.shuffle(order)
            for order_index, candidate_id in enumerate(order):
                yield kind, round_index, order_index, candidate_id


def _invalidate(trial, reason):
    trial.status, trial.reason = "invalid", reason
    trial.seconds_per_execution = trial.energy_joules_per_execution = None


def record_unavailable(requested, config, metadata, environment, reason, output):
    """Save requested coverage when GPU/input preparation cannot start."""
    report = {"schema_version": SCHEMA_VERSION, "run_id": uuid.uuid4().hex[:12], "scope": "unavailable",
              "config": asdict(config), "metadata": metadata, "environment": environment,
              "comparison_fingerprint": fingerprint({"metadata": metadata, "config": asdict(config), "environment": environment}),
              "requested_ids": [item[0] for item in requested], "complete": True, "comparison_complete": False,
              "preflight_failure": reason, "candidates": {}}
    for candidate_id, role in requested:
        trials = [Trial(kind, 0, candidate_id, round_index, status="unavailable", reason=reason).to_dict()
                  for kind in ("timing", "energy") for round_index in range(config.rounds)]
        entry = {"id": candidate_id, "role": role, "preparations": [], "pilots": [], "trials": trials, "failures": []}
        entry["summary"] = summarize_candidate(entry, config.rounds, gpu_run=False)
        report["candidates"][candidate_id] = entry
    save_report(report, output)
    return report


def run_measurements(candidates, inputs, reference, *, config=None, metadata=None, environment=None,
                     energy_meter=None, output_path="results/measurement.json", progress=None):
    """Rebuild/close one executor at a time and checkpoint every attempted slot.

All candidates receive the same resident tensors. Setup, correctness, profiling
and pilot windows are recorded separately. Graph input copies remain inside run.
Missing energy does not prevent the independent timing pass.
"""
    config = config or MeasurementConfig()
    candidates = tuple(candidates)
    by_id = {candidate.id: candidate for candidate in candidates}
    if not candidates or len(by_id) != len(candidates):
        raise ValueError("Require nonempty, unique candidate IDs")
    if not inputs:
        raise ValueError("Inputs cannot be empty")
    device = next(iter(inputs.values())).device
    if device.type != "cuda" and not (device.type == "cpu" and config.allow_cpu):
        raise ValueError("Only NVIDIA CUDA or explicitly enabled CPU development is supported")
    if device.type == "cuda" and (config.allow_cpu or config.window_seconds < 3):
        raise ValueError("CPU development options cannot relax GPU measurement policy")
    scope = "gpu" if device.type == "cuda" else "cpu_development"
    run_id = uuid.uuid4().hex[:12]
    output_path = Path(output_path)
    artifact_dir = output_path.parent / f"measurement_artifacts_{run_id}"
    owned_meter = energy_meter is None
    meter = energy_meter or NvmlEnergyMeter.open(device)
    identity = {"metadata": metadata or {}, "config": asdict(config), "environment": environment or {},
                "energy_identity": meter.identity}
    report = {"schema_version": SCHEMA_VERSION, "run_id": run_id,
              "created_utc": datetime.now(timezone.utc).isoformat(), "scope": scope,
              "comparison_fingerprint": fingerprint(identity), **identity,
              "requested_ids": list(by_id), "complete": False, "comparison_complete": False,
              "protocol": {"runtime": "synchronized wall-clock seconds per completed full executor call",
                           "energy": "NVML whole-GPU delta mJ / 1000 / completed calls; no idle subtraction",
                           "inputs": "identical resident input values for every measurement window; initial transfer excluded",
                           "copies": "required graph input updates included; borrowed output, no retained clone",
                           "passes": "separate timing and energy rounds; not paired samples",
                           "pilot": "target 1.25 times minimum window; bounded growth, record pilots separately",
                           "memory": "prepare and close one executor per trial; compiler/allocator caches may persist",
                           "counter_boundaries": "counter reads bracket work after synchronization; read latency recorded",
                           "baseline": "original eager workload is external and excluded from selection"},
              "candidates": {c.id: {"id": c.id, "role": c.role, "metadata": c.metadata,
                                      "preparations": [], "pilots": [], "trials": [], "failures": []}
                             for c in candidates}}
    iterations_by_candidate = {}
    report = portable(report)

    def checkpoint():
        for entry in report["candidates"].values():
            entry["summary"] = summarize_candidate(entry, config.rounds, gpu_run=scope == "gpu")
        selected_scope = [c for c in report["candidates"].values() if c["role"] == "candidate"]
        report["comparison_complete"] = bool(report["complete"] and selected_scope and all(
            c["summary"]["eligible_for_selection"] for c in selected_scope))
        save_report(report, output_path)

    try:
        checkpoint()
        for kind, round_index, order_index, candidate_id in trial_schedule(by_id, config.rounds, config.seed):
            candidate = by_id[candidate_id]
            entry = report["candidates"][candidate_id]
            trial = Trial(kind, 0, candidate_id, round_index, order_index)
            prepared = None
            attempt = artifact_dir / f"{kind}_{round_index}_{candidate_id}"
            stage = "preparation"
            preparation_start = perf_counter()
            try:
                if kind == "energy" and (scope != "gpu" or not meter.available):
                    trial.status = "unavailable"
                    trial.reason = "CPU development has no GPU energy" if scope != "gpu" else meter.reason
                else:
                    prepared = candidate.prepare(attempt)
                    if prepared.executor is None:
                        trial.status, trial.reason = "unavailable", prepared.record.reason or "Executor was not prepared"
                    else:
                        executor = prepared.executor
                        if executor.device != device:
                            raise ValueError("Candidate executor device differs from common input device")
                        stage = "correctness"
                        begin = perf_counter()
                        prepared.record.correctness = check_correctness(executor, reference, inputs, rtol=config.rtol, atol=config.atol)
                        prepared.record.setup_seconds["measurement_correctness"] = perf_counter() - begin
                        stage = "profiling"
                        if scope == "gpu":
                            profile_prepared(prepared, inputs, attempt / "kernels.json")
                            if not prepared.record.ready_for_gpu_measurement:
                                raise RuntimeError(prepared.record.profiling.get("reason") or "Execution mode could not be verified")
                        stage = "warmup"
                        begin = perf_counter()
                        for _ in range(config.warmup_iterations):
                            executor.run(inputs)
                        synchronize(device)
                        prepared.record.setup_seconds["measurement_warmup"] = perf_counter() - begin
                        stage = "pilot"
                        if candidate_id not in iterations_by_candidate:
                            begin = perf_counter()
                            pilot = calibrate_iterations(executor, inputs, config.window_seconds, max_iterations=config.max_iterations)
                            prepared.record.setup_seconds["measurement_pilot"] = perf_counter() - begin
                            entry["pilots"].append(pilot)
                            if pilot["status"] != "ready":
                                raise RuntimeError(pilot["reason"])
                            iterations_by_candidate[candidate_id] = pilot["iterations"]
                        stage = kind
                        prepared.record.setup_seconds["measurement_preparation_total"] = perf_counter() - preparation_start
                        count = iterations_by_candidate[candidate_id]
                        if kind == "timing":
                            before = meter.snapshot()
                            trial = measure_timing(executor, inputs, count)
                            trial.environment_before, trial.environment_after = before, meter.snapshot()
                            # An asserted exclusive allocation with observable
                            # contention invalidates timing as well as energy.
                            if meter.available and meter.exclusive_gpu:
                                reason = meter.qualification_reason(before) or meter.qualification_reason(trial.environment_after)
                                if reason:
                                    _invalidate(trial, reason)
                        else:
                            trial = measure_energy(executor, inputs, count, meter)
                        if trial.status == "ok" and trial.elapsed_seconds < config.window_seconds:
                            _invalidate(trial, "Measured window shorter than configured minimum; retained for diagnosis, rerun with longer windows")
            except Exception as error:
                trial.status, trial.reason = "failed", f"{stage}: {type(error).__name__}: {error}"
                trial.seconds_per_execution = trial.energy_joules_per_execution = None
                entry["failures"].append({"kind": kind, "round_index": round_index, "stage": stage, "reason": trial.reason})
            finally:
                if prepared is not None:
                    preparation = portable(prepared.record.to_dict())
                    preparation.update(measurement_pass=kind, round_index=round_index, order_index=order_index)
                    entry["preparations"].append(preparation)
                    try:
                        prepared.close()
                    except Exception as error:
                        _invalidate(trial, f"Cleanup failed: {error}")
                        entry["failures"].append({"kind": kind, "round_index": round_index, "stage": "cleanup", "reason": str(error)})
            trial.candidate_id, trial.round_index, trial.order_index = candidate_id, round_index, order_index
            entry["trials"].append(trial.to_dict())
            checkpoint()
            if progress:
                progress(f"{kind} round {round_index + 1}/{config.rounds} {candidate_id}: {trial.status}"
                         + (f" — {trial.reason}" if trial.reason else ""))
        report["complete"] = True  # Every requested slot attempted, not every slot successful.
        checkpoint()
        return report
    finally:
        if owned_meter:
            meter.close()
