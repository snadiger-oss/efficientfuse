"""W1 measurement entry point. No selection, caching, or invented GPU results."""

import argparse
from dataclasses import replace
from pathlib import Path
import uuid

from greenfuse.compiler.inductor import CompilerConfig
from greenfuse.execution.prepare import prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.measurement.baseline import prepare_eager
from greenfuse.measurement.energy import NvmlEnergyMeter
from greenfuse.measurement.environment import environment_report
from greenfuse.measurement.runner import MeasurementCandidate, MeasurementConfig, record_unavailable, run_measurements
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions, select_partitions
from greenfuse.results import write_summary_csv
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.registry import WORKLOADS, get_workload, get_workload_spec


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Use cpu explicitly for development timing only")
    parser.add_argument("--modes", nargs="+", choices=["launch", "graph"])
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="w1")
    parser.add_argument("--gemm-backend", choices=["default", "autotune", "triton"], default="default",
                        help="GEMM policy for w2: cuBLAS, autotuned cuBLAS-vs-Triton, or Triton templates only")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--partitions", nargs="+", metavar="NAME",
                        help="Subset to run: separate, first_pair, last_pair, full (or p11, p01, p10, p00). Default: all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--exclusive-gpu", action="store_true", help="Assert a reserved whole physical GPU; required for energy")
    parser.add_argument("--compiler-artifacts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CUDA and explicit CPU development are supported")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        spec = get_workload_spec(args.workload, m=args.m, n=args.n, seed=args.seed)
        spec = replace(spec, rtol=spec.rtol if args.rtol is None else args.rtol, atol=spec.atol if args.atol is None else args.atol)
        config = MeasurementConfig(rounds=args.rounds, window_seconds=args.window_seconds, seed=args.seed,
                                   warmup_iterations=args.warmup, allow_cpu=device.type == "cpu",
                                   rtol=spec.rtol, atol=spec.atol)
        workload = get_workload(args.workload, n=args.n)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    output = args.output or Path("results") / f"person3_{uuid.uuid4().hex[:12]}" / "measurements.json"
    modes = tuple(dict.fromkeys(args.modes or (["launch"] if device.type == "cpu" else ["launch", "graph"])))
    captured = capture_graph(workload)
    try:
        partitions = select_partitions(enumerate_partitions(captured.description), args.partitions)
    except ValueError as error:
        parser.error(str(error))
    metadata = {"workload": spec.to_dict(), "input_spec_fingerprint": spec.fingerprint,
                "graph_fingerprint": captured.description.fingerprint,
                "partitions": [p.to_dict() for p in partitions],
                "compiler_policy": {"backend": "inductor", "fullgraph": True, "dynamic": False, "triton.cudagraphs": False}}
    requested = [(f"{p.id}_{mode}", "candidate") for p in partitions for mode in modes] + [("eager_launch", "external_baseline")]
    environment = environment_report(str(device), exclusive_gpu=args.exclusive_gpu)
    failure = environment["reason"] if environment["status"] == "unavailable" else None
    if failure is None:
        try:
            inputs = make_inputs(spec, device=str(device))
        except Exception as error:
            failure = f"Input creation failed: {type(error).__name__}: {error}"
    if failure:
        report = record_unavailable(requested, config, metadata, environment, failure, output)
        print(f"Unavailable: {failure}\nRequested coverage saved: {output}")
        return 1

    candidates = []
    for partition in partitions:
        for mode in modes:
            def factory(path, partition=partition, mode=mode):
                return prepare_execution(build_partition(captured, partition), inputs, mode=mode, reference=workload,
                                         config=CompilerConfig(str(path / "compiler") if args.compiler_artifacts else None, gemm_backend=args.gemm_backend),
                                         allow_cpu=device.type == "cpu", warmup_iterations=args.warmup,
                                         rtol=spec.rtol, atol=spec.atol)
            candidates.append(MeasurementCandidate(f"{partition.id}_{mode}", factory,
                                                    metadata={"partition_id": partition.id, "mode": mode,
                                                              "requested_groups": len(partition.groups)}))
    candidates.append(MeasurementCandidate("eager_launch", lambda path: prepare_eager(
        workload, inputs, captured.description.fingerprint, allow_cpu=device.type == "cpu"), role="external_baseline"))
    meter = NvmlEnergyMeter.open(device, exclusive_gpu=args.exclusive_gpu)
    try:
        report = run_measurements(candidates, inputs, workload, config=config, metadata=metadata,
                                  environment=environment, energy_meter=meter, output_path=output,
                                  progress=lambda line: print(line, flush=True))
    finally:
        meter.close()
    write_summary_csv(report, output.with_suffix(".csv"))
    print(f"Raw evidence: {output}\nSummary: {output.with_suffix('.csv')}")
    if device.type == "cpu":
        print("CPU development only. GPU energy, CUDA replay, and GPU comparisons remain unverified.")
        return 0 if all(c["summary"]["timing"]["complete"] for c in report["candidates"].values()) else 1
    print(f"All requested candidates eligible for comparison: {report['comparison_complete']}")
    return 0 if report["comparison_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
