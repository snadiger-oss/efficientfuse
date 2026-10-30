"""Quick single-window runtime + GPU energy per plan (ordinary execution). Not a calibration.

One synchronized window of ~--window seconds per plan, read from the NVML cumulative energy
counter (whole GPU device). Use for indicative numbers only; calibrations use 5 shuffled rounds.

Example:
  python scripts/quick_energy.py --workload w1 --m 4096 --n 4096 --output results/bhargav/quick_energy/w1_4096.json
"""

import argparse
import json
from pathlib import Path
from time import perf_counter


def main():
    import pynvml
    import torch

    from greenfuse.compiler.inductor import CompilerConfig
    from greenfuse.execution.prepare import prepare_execution
    from greenfuse.graph.capture import capture_graph
    from greenfuse.measurement.baseline import prepare_eager
    from greenfuse.naming import grouping_name
    from greenfuse.partition.build import build_partition
    from greenfuse.partition.enumerate import enumerate_partitions
    from greenfuse.workloads.inputs import make_inputs
    from greenfuse.workloads.registry import get_workload, get_workload_spec

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--gemm-backend", default="default")
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    workload = get_workload(args.workload, n=args.n)
    captured = capture_graph(workload)
    kinds = [op.kind for op in captured.description.operations]
    inputs = make_inputs(get_workload_spec(args.workload, m=args.m, n=args.n), device="cuda:0")
    plans = [("eager", None)] + [(grouping_name(p, kinds), p) for p in enumerate_partitions(captured.description)]
    results = []
    for name, partition in plans:
        if partition is None:
            prepared = prepare_eager(workload, inputs, captured.description.fingerprint)
        else:
            prepared = prepare_execution(build_partition(captured, partition), inputs, mode="launch", reference=workload,
                                         config=CompilerConfig(gemm_backend=args.gemm_backend), rtol=1e-4, atol=1e-3)
        executor = prepared.executor
        for _ in range(20):
            executor.run(inputs)
        torch.cuda.synchronize()
        count, elapsed = 10, 0.0
        while elapsed < 0.2:
            count *= 2
            start = perf_counter()
            for _ in range(count):
                executor.run(inputs)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
        count = max(1, int(count * args.window / elapsed))
        torch.cuda.synchronize()
        energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        start = perf_counter()
        for _ in range(count):
            executor.run(inputs)
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy_start
        results.append({"plan": name, "calls": count, "window_s": seconds,
                        "runtime_us": seconds / count * 1e6, "energy_mJ": energy_mj / count})
        print(f"{args.workload} {args.m}x{args.n} {name:12s} {seconds / count * 1e6:9.1f} us  {energy_mj / count:9.3f} mJ", flush=True)
        prepared.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "quick single-window measurement, ordinary execution, not a calibration",
                                       "workload": args.workload, "shape": f"{args.m}x{args.n}",
                                       "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
                                       "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
