"""Coordinate capture, compilation, measurement, fresh validation and caching."""

from dataclasses import dataclass, replace
from pathlib import Path
import uuid

from greenfuse.compiler.inductor import CompilerConfig
from greenfuse.execution.prepare import prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.measurement.baseline import prepare_eager
from greenfuse.measurement.energy import NvmlEnergyMeter
from greenfuse.measurement.environment import environment_report
from greenfuse.measurement.runner import MeasurementCandidate, record_unavailable, run_measurements
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.results import fingerprint, portable, write_summary_csv
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.registry import get_workload, get_workload_spec
from .cache import DecisionCache, write_json_atomic
from .candidates import combine_candidates
from .identity import build_identity, stable_environment
from .select import select_report, validate_repeat


@dataclass
class ExperimentContext:
    config: object
    workload: object
    spec: object
    captured: object
    candidates: tuple
    environment: dict
    identity: dict
    inputs: dict | None = None

    @property
    def key(self):
        return fingerprint(self.identity)

    @property
    def metadata(self):
        partitions = {c.partition.id: c.partition.to_dict() for c in self.candidates}
        return {"workload": self.spec.to_dict(), "input_spec_fingerprint": self.spec.fingerprint,
                "graph_fingerprint": self.captured.description.fingerprint, "partitions": list(partitions.values()),
                "compiler_policy": self.identity["execution_contract"]["compiler"],
                "planner_identity": self.identity, "planner_key": self.key}

    def ensure_inputs(self):
        if self.inputs is None:
            self.inputs = make_inputs(self.spec, device=self.config.device)
        return self.inputs

    def prepare_candidate(self, candidate_id, path, *, profile=False):
        candidate = next((c for c in self.candidates if c.id == candidate_id), None)
        if candidate is None:
            raise ValueError("Candidate is absent from this experiment")
        return prepare_execution(build_partition(self.captured, candidate.partition), self.ensure_inputs(),
                                 mode=candidate.mode, reference=self.workload,
                                 config=CompilerConfig(str(Path(path) / "compiler") if self.config.compiler_artifacts else None),
                                 allow_cpu=self.config.device == "cpu", warmup_iterations=self.config.warmup_iterations,
                                 rtol=self.config.rtol, atol=self.config.atol,
                                 trace_path=Path(path) / "kernels.json" if profile else None)

    def measurement_candidates(self):
        candidates = [MeasurementCandidate(c.id, lambda path, name=c.id: self.prepare_candidate(name, path),
                                           metadata=c.to_dict()) for c in self.candidates]
        if self.config.eager_baseline:
            candidates.append(MeasurementCandidate("eager_launch", lambda path: prepare_eager(
                self.workload, self.ensure_inputs(), self.captured.description.fingerprint,
                allow_cpu=self.config.device == "cpu"), role="external_baseline"))
        return tuple(candidates)


def create_context(config, shape, *, exclusive_gpu=False):
    config = config.for_shape(shape)
    workload = get_workload(config.workload)
    spec = replace(get_workload_spec(config.workload, m=shape[0], n=shape[1], seed=config.seed),
                   rtol=config.rtol, atol=config.atol)
    captured = capture_graph(workload)
    partitions = enumerate_partitions(captured.description, max_partitions=config.max_partitions)
    experiment_key = fingerprint({"config": config.to_dict(), "graph": captured.description.fingerprint, "inputs": spec.fingerprint})
    candidates = combine_candidates(partitions, config.modes, experiment_key)
    environment = environment_report(config.device, exclusive_gpu=exclusive_gpu)
    identity = build_identity(config, spec, captured.description, candidates, environment)
    return ExperimentContext(config, workload, spec, captured, candidates, environment, identity)


def benchmark_context(context, output, *, validation=False, exclusive_gpu=False, progress=None):
    output = Path(output)
    config = context.config.measurement(validation=validation)
    requested = [(c.id, "candidate") for c in context.candidates]
    if context.config.eager_baseline:
        requested.append(("eager_launch", "external_baseline"))
    reason = context.environment.get("reason") if context.environment.get("status") == "unavailable" else None
    if reason is None:
        try:
            context.ensure_inputs()
        except Exception as error:
            reason = f"Input preparation failed: {type(error).__name__}: {error}"
    if reason is not None:
        return record_unavailable(requested, config, context.metadata, context.environment, reason, output)
    import torch
    meter = NvmlEnergyMeter.open(torch.device(context.config.device), exclusive_gpu=exclusive_gpu)
    try:
        report = run_measurements(context.measurement_candidates(), context.inputs, context.workload, config=config,
                                  metadata=context.metadata, environment=context.environment, energy_meter=meter,
                                  output_path=output, progress=progress)
    finally:
        meter.close()
    write_summary_csv(report, output.with_suffix(".csv"))
    return report


def calibrate_context(context, *, output_dir="results/calibrations", cache_dir="results/decisions",
                      exclusive_gpu=False, progress=None):
    """No fallback on an incomplete or unstable comparison; preserve evidence."""
    directory = Path(output_dir) / f"{context.key[:12]}_{uuid.uuid4().hex[:8]}"
    initial_path, fresh_path = directory / "initial.json", directory / "validation.json"
    result = {"status": "pending", "planner_key": context.key, "selected_id": None,
              "initial_report": str(initial_path.resolve()), "validation_report": None, "cache_written": False}
    write_json_atomic(directory / "decision.json", result)
    initial = benchmark_context(context, initial_path, exclusive_gpu=exclusive_gpu, progress=progress)
    selection = select_report(initial)
    result["selection"] = selection
    if selection["status"] != "selected":
        result.update(status=selection["status"], reason=selection["reason"])
    else:
        if progress:
            progress(f"Provisional winner: {selection['selected_id']}; starting a fresh full-set validation")
        fresh_environment = environment_report(context.config.device, exclusive_gpu=exclusive_gpu)
        if stable_environment(fresh_environment) != context.identity["environment"]:
            result.update(status="inconclusive", reason="Environment changed after calibration; no decision cached")
        else:
            context.environment = fresh_environment
            fresh = benchmark_context(context, fresh_path, validation=True, exclusive_gpu=exclusive_gpu, progress=progress)
            result["validation_report"] = str(fresh_path.resolve())
            validation = validate_repeat(initial, fresh)
            result["validation"] = validation
            result.update(status=validation["status"], reason=validation["reason"])
            if validation["status"] == "validated":
                cached = DecisionCache(cache_dir).store(context.identity, initial, fresh)
                result.update(selected_id=cached["selected_id"], cache_written=True,
                              cache_path=str((Path(cache_dir) / f"{context.key}.json").resolve()))
    write_json_atomic(directory / "decision.json", result)
    result["decision_path"] = str((directory / "decision.json").resolve())
    return portable(result)
