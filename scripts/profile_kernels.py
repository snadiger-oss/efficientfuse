"""Per-kernel GPU time for one grouping, measured in a fresh process.

Prepares one partition (ordinary execution; replay runs the same kernels), warms it up, then profiles
`--calls` invocations and reports each kernel's name, launches per call and average GPU duration.
Run one partition per process: PyTorch drops CUPTI kernel records in later profiler sessions.

Example:
  python scripts/profile_kernels.py --workload w2 --m 64 --n 4096 --gemm-backend fuse_epilogue \
    --partition p00 --output results/bhargav/kernel_profiles/w2_64x4096_p00.json
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    import torch
    from torch.profiler import ProfilerActivity, profile

    from greenfuse.compiler.inductor import CompilerConfig
    from greenfuse.execution.prepare import prepare_execution
    from greenfuse.graph.capture import capture_graph
    from greenfuse.naming import grouping_name
    from greenfuse.partition.build import build_partition
    from greenfuse.partition.enumerate import enumerate_partitions, select_partitions
    from greenfuse.workloads.inputs import make_inputs
    from greenfuse.workloads.registry import get_workload, get_workload_spec

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--gemm-backend", default="default")
    parser.add_argument("--partition", required=True, help="ID such as p00 or p0101, or an alias like full")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workload = get_workload(args.workload, n=args.n)
    spec = get_workload_spec(args.workload, m=args.m, n=args.n)
    captured = capture_graph(workload)
    kinds = [op.kind for op in captured.description.operations]
    partition, = select_partitions(enumerate_partitions(captured.description), [args.partition])
    inputs = make_inputs(spec, device="cuda:0")
    prepared = prepare_execution(build_partition(captured, partition), inputs, mode="launch", reference=workload,
                                 config=CompilerConfig(gemm_backend=args.gemm_backend), rtol=1e-4, atol=1e-3,
                                 warmup_iterations=10)
    if prepared.executor is None:
        raise SystemExit(f"Preparation failed: {prepared.record.reason}")
    executor = prepared.executor
    for _ in range(10):
        executor.run(inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.calls):
            executor.run(inputs)
        torch.cuda.synchronize()
    kernels = defaultdict(list)
    for event in prof.events():
        if event.device_type.name == "CUDA" and not event.name.startswith(("Memcpy", "Memset", "## ")):
            kernels[event.name].append(event.device_time if hasattr(event, "device_time") else event.cuda_time)
    rows = [{"kernel": name, "launches_per_call": len(times) / args.calls, "avg_gpu_us": sum(times) / len(times),
             "gpu_us_per_call": sum(times) / args.calls} for name, times in kernels.items()]
    rows.sort(key=lambda row: -row["gpu_us_per_call"])
    result = {"workload": args.workload, "shape": f"{args.m}x{args.n}", "gemm_backend": args.gemm_backend,
              "partition": partition.id, "grouping": grouping_name(partition, kinds),
              "groups": [list(g) for g in partition.groups], "calls_profiled": args.calls,
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
              "kernels": rows, "total_gpu_us_per_call": sum(r["gpu_us_per_call"] for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.workload} {args.m}x{args.n} {result['grouping']}: total GPU {result['total_gpu_us_per_call']:.1f} us/call")
    for row in rows:
        print(f"  {row['gpu_us_per_call']:8.1f} us  x{row['launches_per_call']:.0f}  {row['kernel'][:90]}")
    prepared.close()


if __name__ == "__main__":
    raise SystemExit(main())
