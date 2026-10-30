"""SYNTHETIC TEST DATA ONLY. These numbers are not GPU observations."""

from dataclasses import asdict

from greenfuse.config import ExperimentConfig
from greenfuse.graph.ir import InputBinding, NormalizedGraph, Operation, ValueRef
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.planner.candidates import combine_candidates
from greenfuse.planner.identity import build_identity
from greenfuse.results import Trial, fingerprint, portable
from greenfuse.workloads.specs import w1_spec


def synthetic_report(*, fresh=False, costs=None, exclusion_policy="none"):
    graph = NormalizedGraph(tuple(InputBinding(f"input_{i}", name) for i, name in enumerate(("x", "bias", "residual"))), (
        Operation("op_0", "add", "operator.add", (ValueRef("input_0"), ValueRef("input_1"))),
        Operation("op_1", "relu", "torch.relu", (ValueRef("op_0"),)),
        Operation("op_2", "add", "operator.add", (ValueRef("op_1"), ValueRef("input_2"))),
    ), ValueRef("op_2"))
    config = ExperimentConfig(rounds=3, exclusion_policy=exclusion_policy)
    spec = w1_spec()
    partitions = enumerate_partitions(graph)
    candidates = combine_candidates(partitions, config.modes, "synthetic-experiment")
    gpu_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    energy_identity = {"cuda_uuid": gpu_uuid, "nvml_uuid": gpu_uuid, "exclusive_physical_gpu_asserted": True,
                       "driver_version": "SYNTHETIC driver", "pci_bus_id": "0000:01:00.0"}
    environment = {"status": "cuda_available", "python": "SYNTHETIC Python", "platform": "SYNTHETIC host",
                   "versions": {"torch": "SYNTHETIC torch", "triton": "SYNTHETIC triton"}, "cuda_build": "SYNTHETIC CUDA",
                   "device": {"logical_index": 0, "uuid": gpu_uuid},
                   "energy": {"identity": energy_identity, "state": {"power_limit_mw": 250000}, "reason": None}}
    identity = build_identity(config, spec, graph, candidates, environment)
    metadata = {"source": "SYNTHETIC TEST ONLY", "workload": spec.to_dict(), "input_spec_fingerprint": spec.fingerprint,
                "graph_fingerprint": graph.fingerprint, "partitions": [p.to_dict() for p in partitions],
                "planner_identity": identity, "planner_key": fingerprint(identity)}
    costs = costs or {"p00_launch": (100, 80), "p00_graph": (101, 90), "p01_launch": (104, 40),
                      "p01_graph": (105, 50), "p10_launch": (107, 10), "p10_graph": (108, 10),
                      "p11_launch": (120, 60), "p11_graph": (118, 50)}
    entries = {}
    clock = 10000.0 if fresh else 1000.0
    for candidate in candidates:
        name, partition, mode = candidate.id, candidate.partition, candidate.mode
        microseconds, microjoules = costs[name]
        entry = {"id": name, "role": "candidate", "preparations": [], "failures": [], "trials": []}
        for kind in ("timing", "energy"):
            for round_index in range(config.rounds):
                count = 100000
                seconds = count * microseconds / 1e6 if kind == "timing" else 4.0
                trial = Trial(kind, count, name, round_index, status="ok", submitted_iterations=count,
                              completed_iterations=count, start_monotonic_seconds=clock, end_monotonic_seconds=clock + seconds,
                              elapsed_seconds=seconds, seconds_per_execution=microseconds / 1e6 if kind == "timing" else None,
                              energy_start_mj=1000 if kind == "energy" else None,
                              energy_end_mj=1000 + int(count * microjoules / 1000) if kind == "energy" else None,
                              energy_joules_per_execution=microjoules / 1e6 if kind == "energy" else None)
                clock += seconds + 1
                entry["trials"].append(trial.to_dict())
                entry["preparations"].append({
                    "id": name, "partition_id": partition.id, "graph_fingerprint": graph.fingerprint,
                    "mode": mode, "requested_groups": len(partition.groups), "device": "cuda:0", "status": "ready",
                    "ready_for_gpu_measurement": True, "measurement_pass": kind, "round_index": round_index,
                    "compilation": {"backend": "inductor", "fullgraph": True, "dynamic": False,
                                    "options": {"triton.cudagraphs": False}, "device": "cuda:0",
                                    "torch_version": environment["versions"]["torch"], "cuda_build": environment["cuda_build"],
                                    "groups": [{"status": "compiled", "node_ids": list(g)} for g in partition.groups]},
                    "correctness": {"status": "passed", "rtol": config.rtol, "atol": config.atol,
                                    "checks": [{"inputs": "original", "max_absolute_error": 0},
                                               {"inputs": "changed", "max_absolute_error": 0}]},
                    "buffers": {"forward_only": True, "autocast": False, "tensors": [
                        {"name": t.name, "shape": list(t.shape), "dtype": "torch.float32", "device": "cuda:0"}
                        for t in spec.inputs]},
                    "profiling": {"replay_check": "passed", "kernel_launches_per_invocation": 2},
                })
        entries[name] = entry
    # Eager is deliberately much faster/cheaper; it must not enter selection.
    entries["eager_launch"] = {"id": "eager_launch", "role": "external_baseline", "preparations": [], "failures": [],
                               "trials": [Trial(k, 0, "eager_launch", r, status="unavailable", reason="Synthetic unused baseline").to_dict()
                                          for k in ("timing", "energy") for r in range(config.rounds)]}
    report = {"schema_version": 1, "run_id": "synthetic-validation" if fresh else "synthetic-initial", "scope": "gpu",
              "complete": True, "metadata": metadata, "environment": environment, "energy_identity": energy_identity,
              "config": asdict(config.measurement(validation=fresh)), "requested_ids": list(entries), "candidates": entries}
    report["comparison_fingerprint"] = fingerprint({k: report[k] for k in ("metadata", "environment", "energy_identity", "config")})
    return portable(report)


def make_failed_candidate(report, candidate_id):
    entry = report["candidates"][candidate_id]
    for preparation in entry["preparations"]:
        preparation.update(status="failed", failure_stage="compilation", reason="SYNTHETIC compiler failure",
                           ready_for_gpu_measurement=False)
    for trial in entry["trials"]:
        trial.update(status="unavailable", reason="SYNTHETIC compiler failure", submitted_iterations=0, completed_iterations=0,
                     seconds_per_execution=None, energy_joules_per_execution=None)
