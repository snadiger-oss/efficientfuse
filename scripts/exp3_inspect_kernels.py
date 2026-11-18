"""Explain Experiment 3 from the generated code: what each grouping does inside its kernels.

For every (shape, k, candidate) this saves the TorchInductor-generated Triton
source and reports, per kernel: kind (pointwise / persistent reduction / looped
reduction), passes over a row, sin/tanh evaluations written in the code, loads,
stores, and the profiled GPU time per call. It also checks whether each kernel
re-evaluates the expensive chain (a sign of recomputation).
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_screen_fanout import make_candidates  # noqa: E402

KERNEL = re.compile(r"async_compile\.triton\('(\w+)',\s*'''(.*?)'''", re.S)
KINDS = {"triton_poi": "pointwise", "triton_per": "persistent_reduction", "triton_red": "looped_reduction"}


def analyze_source(source):
    kernels = []
    for name, body in KERNEL.findall(source):
        kind = next((v for p, v in KINDS.items() if name.startswith(p)), "other")
        loops = len(re.findall(r"for r\w*offset in (?:tl\.)?range\(", body))
        kernels.append({
            "kernel": name, "kind": kind,
            "row_passes": loops if kind == "looped_reduction" else 1,
            "sin_calls": len(re.findall(r"\.sin\(", body)), "tanh_calls": len(re.findall(r"\.tanh\(", body)),
            "loads": len(re.findall(r"tl\.load\(", body)), "stores": len(re.findall(r"tl\.store\(", body)),
        })
    return kernels


def profile_kernels(fn, x, iterations):
    import torch

    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            fn(x)
        torch.cuda.synchronize()
    totals = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            totals[event.name] = totals.get(event.name, 0.0) + event.device_time_total
    return {name: total / iterations for name, total in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["8192x8192", "2048x32768"])
    parser.add_argument("--k", nargs="+", type=int, default=[4, 8, 12, 16, 24, 32])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/haripreeth/exp3_fanout_screen/kernel_analysis"))
    args = parser.parse_args()

    import torch
    import torch._inductor.config as inductor_config
    from torch._inductor.utils import run_and_get_code

    inductor_config.triton.cudagraphs = False
    rows = []
    for shape_text in args.shapes:
        shape = tuple(int(v) for v in shape_text.split("x"))
        for k in args.k:
            torch._dynamo.reset()
            torch.manual_seed(0)
            x = torch.randn(shape, device="cuda")
            for name, fn in make_candidates(k).items():
                if name == "eager_baseline":
                    continue
                _, sources = run_and_get_code(fn, x)
                folder = args.output_dir / f"{shape_text}_k{k}" / name
                folder.mkdir(parents=True, exist_ok=True)
                kernels = []
                for index, source in enumerate(sources):
                    (folder / f"group{index + 1}_generated.py").write_text(source)
                    kernels += [{**kern, "group": index + 1} for kern in analyze_source(source)]
                times = profile_kernels(fn, x, args.iterations)
                for kern in kernels:
                    kern["gpu_us_per_call"] = round(times.get(kern["kernel"], float("nan")), 1)
                    rows.append({"shape": shape_text, "k": k, "candidate": name, **kern})
                (folder / "kernels.json").write_text(json.dumps(kernels, indent=2))
                total = sum(kern["sin_calls"] + kern["tanh_calls"] for kern in kernels)
                print(f"{shape_text} k={k:>2} {name:24} kernels={len(kernels)} "
                      f"expensive-op evaluations per element={total} (k={k}) | "
                      + "; ".join(f"{kern['kind']} passes={kern['row_passes']} sin+tanh={kern['sin_calls'] + kern['tanh_calls']} "
                                  f"loads={kern['loads']} stores={kern['stores']} {kern['gpu_us_per_call']}us" for kern in kernels),
                      flush=True)
            del x
            torch.cuda.empty_cache()
    with open(args.output_dir / "kernels_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
