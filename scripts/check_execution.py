"""Person 2 correctness/setup runner. This does not measure runtime or energy."""

import argparse
import json
from pathlib import Path
import uuid

from greenfuse.compiler.inductor import CompilerConfig
from greenfuse.execution.prepare import ExecutionRecord, prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="cuda:N for GPU validation; cpu for explicit ordinary-only development")
    parser.add_argument("--modes", nargs="+", choices=["launch", "graph"], default=["launch", "graph"])
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compiler-artifacts", action="store_true", help="Ask Inductor to save debug code; cache hits may not produce files")
    parser.add_argument("--output", type=Path, default=Path("results/person2_execution.json"))
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup must be positive")
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        parser.error("Only CUDA and explicit CPU development are supported")
    run_id = uuid.uuid4().hex[:12]
    artifacts = args.output.parent / f"person2_artifacts_{run_id}"
    spec = w1_spec(args.m, args.n, seed=args.seed)
    captured = capture_graph(workload_w1)
    partitions = enumerate_partitions(captured.description)
    modes = tuple(dict.fromkeys(args.modes))
    report = {
        "schema_version": 1, "run_id": run_id,
        "scope": "Person 2 execution correctness, setup costs, and diagnostic kernel profiling; no runtime/energy trials",
        "device": str(device), "torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
        "workload": spec.to_dict(), "input_spec_fingerprint": spec.fingerprint,
        "graph_fingerprint": captured.description.fingerprint, "executions": [],
        "requested_ids": [f"{partition.id}_{mode}" for partition in partitions for mode in modes],
        "complete": False,
    }
    unavailable = None
    preflight_status, preflight_stage = "unsupported", "support"
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.version.cuda is None:
            unavailable = "CUDA is unavailable in this environment; no CPU fallback was performed"
        elif device.index is not None and device.index >= torch.cuda.device_count():
            unavailable = f"CUDA device {device.index} does not exist"
        else:
            report["gpu_name"] = torch.cuda.get_device_name(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    inputs = None
    if unavailable is None:
        try:
            inputs = make_inputs(spec, device=str(device))
        except Exception as error:
            unavailable = f"Input preparation failed: {type(error).__name__}: {error}"
            preflight_status, preflight_stage = "failed", "inputs"
    # Prepare/check/release one candidate at a time; do not keep eight graph pools alive.
    for partition in partitions:
        for mode in modes:
            if unavailable:
                record = ExecutionRecord(partition.id, partition.graph_fingerprint, mode, len(partition.groups),
                                         status=preflight_status, device=str(device), failure_stage=preflight_stage, reason=unavailable)
            else:
                candidate_dir = artifacts / f"{partition.id}_{mode}"
                config = CompilerConfig(str(candidate_dir / "compiler") if args.compiler_artifacts else None)
                prepared = prepare_execution(
                    build_partition(captured, partition), inputs, mode=mode, reference=workload_w1,
                    config=config, allow_cpu=device.type == "cpu", warmup_iterations=args.warmup,
                    rtol=spec.rtol, atol=spec.atol, trace_path=candidate_dir / "execution_trace.json",
                )
                record = prepared.record
                try:
                    prepared.close()
                except Exception as error:
                    record.status = "failed"
                    record.failure_stage = "cleanup"
                    record.reason = str(error)
                    record.ready_for_gpu_measurement = False
            report["executions"].append(record.to_dict())
            count = record.profiling["kernel_launches_per_invocation"]
            print(f"{record.id}: {record.status} | requested groups={record.requested_groups} | GPU kernels={count}", flush=True)
            if record.reason:
                print(f"  {record.failure_stage}: {record.reason}", flush=True)
            # Preserve earlier outcomes even if a later candidate is interrupted.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Execution evidence: {args.output}")
    print("Timing and GPU energy measurement belong to Person 3 and are not implemented by this runner.")
    allowed = {"cpu_validated"} if device.type == "cpu" else {"ready"}
    return 0 if all(item["status"] in allowed for item in report["executions"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
