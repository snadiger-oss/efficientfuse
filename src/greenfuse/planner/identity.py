"""Stable specialization/cache identity, separate from transient GPU telemetry."""

from pathlib import Path

from greenfuse.results import fingerprint, portable


def implementation_fingerprint():
    root = Path(__file__).resolve().parents[1]
    sources = {str(path.relative_to(root)): path.read_text() for path in sorted(root.rglob("*.py"))}
    return fingerprint(sources)


def stable_environment(environment):
    energy = environment.get("energy", {})
    state, identity = energy.get("state", {}), energy.get("identity", {})
    return portable({
        "python": environment.get("python"), "platform": environment.get("platform"),
        "versions": environment.get("versions"), "cuda_build": environment.get("cuda_build"),
        "device": environment.get("device"), "driver": identity.get("driver_version"),
        "physical_gpu_uuid": identity.get("nvml_uuid"), "pci_bus_id": identity.get("pci_bus_id"),
        "power_limit_mw": state.get("power_limit_mw"),
        "application_graphics_clock_mhz": state.get("application_graphics_clock_mhz"),
        "application_memory_clock_mhz": state.get("application_memory_clock_mhz"),
        "torch_threads": environment.get("torch_threads"),
        "numerics": {key: environment.get(key) for key in ("float32_matmul_precision", "deterministic_algorithms",
                                                            "cudnn_benchmark", "cudnn_deterministic")},
        "execution_environment": {key: environment.get("environment", {}).get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "OMP_NUM_THREADS",
            "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")},
    })


def build_identity(config, workload_spec, graph, candidates, environment):
    return portable({
        "schema_version": 1, "scope": "gpu" if environment.get("status") == "cuda_available" else environment.get("status"),
        "workload": workload_spec.to_dict(), "graph": graph.to_dict(), "graph_fingerprint": graph.fingerprint,
        "experiment": config.to_dict(), "candidates": [c.to_dict() for c in candidates],
        "environment": stable_environment(environment), "implementation": implementation_fingerprint(),
        "execution_contract": {"forward_only": True, "autocast": False, "input_update": "every invocation",
                               "output": "borrowed until next invocation", "graph_scope": "whole compiled partition",
                               "compiler": {"backend": "inductor", "fullgraph": True, "dynamic": False, "triton.cudagraphs": False}},
        "policy": {"latency_tolerance": config.latency_tolerance, "exclusion_policy": config.exclusion_policy,
                   "aggregation": "median; all trials retained", "validation": "fresh full-set repeat with same median winner",
                   "tie_break": ["energy", "runtime", "candidate_id"]},
    })
