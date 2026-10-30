"""Quick runtime screen: where does partial grouping + CUDA Graph replay match full fusion?

For each workload:shape case, prepare all 8 compiled candidates (correctness-checked) plus eager,
time short synchronized batches, and compare the best partial+replay candidate with the best
full-fusion candidate. This is a cheap pre-filter for choosing calibration runs: runtime only,
no energy, no selection, no cache. Replay uses on_change input refresh (inputs resident on GPU).

Example:
  python scripts/screen_candidates.py --cases w1:32x256 w5:64x131072 --output results/bhargav/screening/screen.json
"""

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter

from greenfuse.compiler.inductor import CompilerConfig
from greenfuse.execution.buffers import synchronize
from greenfuse.execution.prepare import prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.measurement.baseline import prepare_eager
from greenfuse.naming import grouping_name
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.registry import get_workload, get_workload_spec



def time_executor(executor, inputs, device, window, repeats):
    """Median synchronized wall-clock seconds per call over `repeats` batches of ~window seconds."""
    for _ in range(3):
        executor.run(inputs)
    synchronize(device)
    count, elapsed = 8, 0.0
    while elapsed < window / 4:  # pilot: grow the batch until it lasts a quarter window
        count *= 2
        start = perf_counter()
        for _ in range(count):
            executor.run(inputs)
        synchronize(device)
        elapsed = perf_counter() - start
    count = max(1, int(count * window / elapsed))
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(count):
            executor.run(inputs)
        synchronize(device)
        samples.append((perf_counter() - start) / count)
    return statistics.median(samples), min(samples), max(samples)


def screen_case(name, m, n, args):
    import torch

    device = torch.device(args.device)
    spec = get_workload_spec(name, m=m, n=n, seed=0)
    workload = get_workload(name, n=n)
    captured = capture_graph(workload)
    inputs = make_inputs(spec, device=str(device))
    rows = {}
    kinds = [op.kind for op in captured.description.operations]
    for partition in enumerate_partitions(captured.description):
        for mode in ("launch", "graph"):
            cid = f"{grouping_name(partition, kinds)}_{'replay' if mode == 'graph' else 'ordinary'}"
            prepared = prepare_execution(build_partition(captured, partition), inputs, mode=mode, reference=workload,
                                         config=CompilerConfig(gemm_backend=args.gemm_backend), rtol=args.tol,
                                         atol=args.tol, input_refresh="on_change")
            if prepared.executor is None:
                rows[cid] = {"status": prepared.record.status, "reason": prepared.record.reason}
            else:
                median, low, high = time_executor(prepared.executor, inputs, device, args.window, args.repeats)
                rows[cid] = {"status": "ok", "us": median * 1e6, "us_min": low * 1e6, "us_max": high * 1e6}
                prepared.close()
            torch.cuda.empty_cache()
    eager = prepare_eager(workload, inputs, captured.description.fingerprint)
    median, low, high = time_executor(eager.executor, inputs, device, args.window, args.repeats)
    rows["eager"] = {"status": "ok", "us": median * 1e6, "us_min": low * 1e6, "us_max": high * 1e6}
    eager.close()
    ok = {k: v["us"] for k, v in rows.items() if v["status"] == "ok"}
    full = min((ok[k], k) for k in ("full_ordinary", "full_replay") if k in ok)
    partial_replay = min((v, k) for k, v in ok.items() if k.endswith("_replay") and k != "full_replay")
    fastest = min((v, k) for k, v in ok.items() if k != "eager")
    return {"case": f"{name}:{m}x{n}", "candidates": rows, "best_full": full[1], "best_partial_replay": partial_replay[1],
            "partial_replay_over_full": partial_replay[0] / full[0], "fastest_compiled": fastest[1],
            "partial_replay_within_5pct_of_fastest": partial_replay[0] <= 1.05 * fastest[0]}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", required=True, help="workload:MxN, e.g. w5:64x131072")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=float, default=0.3, help="Seconds per timed batch")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-5, help="rtol = atol for correctness")
    parser.add_argument("--gemm-backend", default="default", choices=["default", "autotune", "triton", "fuse_epilogue"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {"scope": "runtime screen only (synchronized batch wall-clock); not a calibration",
               "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
               "window_seconds": args.window, "repeats": args.repeats, "replay_input_refresh": "on_change", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        name, shape = case.split(":")
        m, n = (int(v) for v in shape.lower().split("x"))
        try:
            result = screen_case(name, m, n, args)
        except Exception as error:  # keep screening other cases; record the failure
            result = {"case": case, "error": f"{type(error).__name__}: {error}"}
        results["cases"].append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        if "error" in result:
            print(f"{case}: ERROR {result['error']}", flush=True)
            continue
        cells = " ".join(f"{k.replace('_ordinary', '_ord').replace('_replay', '_rep')}={v.get('us', float('nan')):.1f}"
                         for k, v in sorted(result["candidates"].items()))
        flag = "PASS" if result["partial_replay_within_5pct_of_fastest"] else "----"
        print(f"{flag} {case}: best partial+replay {result['best_partial_replay']} / best full {result['best_full']} "
              f"= {result['partial_replay_over_full']:.3f} | {cells}", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
