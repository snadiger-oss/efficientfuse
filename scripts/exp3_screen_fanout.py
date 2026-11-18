"""Experiment 3 screen: can a partial grouping beat full fusion?

Workload (shared expensive intermediate, "fan-out"):

    y   = E_k(x)                     # k chained transcendental ops (sin/tanh alternating)
    out = y * y.sum(-1, keepdim=True)

y feeds both a row reduction and an elementwise product. When the reduction is
not persistent (long rows), a single fused group may recompute E_k(x) in each
pass instead of materializing y. Materializing y costs one extra write and read;
recomputing costs k transcendental ops per element. The hypothesis is that
above some k, splitting at y beats full fusion.

Standalone screen: ordinary execution only (no CUDA Graph replay), synchronized
wall-clock time per call and NVML GPU-device energy per call measured over the
same window, shuffled candidate order per round. Promote any winner into the
planner (with separate timing/energy passes and fresh validation) before
claiming it.
"""

import argparse
import csv
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time


def make_expensive(k):
    import torch

    def expensive(x):
        y = x
        for i in range(k):
            y = torch.sin(y) if i % 2 == 0 else torch.tanh(y)
        return y
    return expensive


def make_candidates(k):
    import torch

    expensive = make_expensive(k)

    def whole(x):
        y = expensive(x)
        return y * y.sum(-1, keepdim=True)

    def reduce_and_scale(y):
        return y * y.sum(-1, keepdim=True)

    def row_sum(y):
        return y.sum(-1, keepdim=True)

    def scale(y, s):
        return y * s

    def compile_(fn):
        return torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)

    full = compile_(whole)
    g_y, g_rs = compile_(expensive), compile_(reduce_and_scale)
    s_y, s_sum, s_scale = compile_(expensive), compile_(row_sum), compile_(scale)
    return {
        "eager_baseline": whole,
        "compiled_full": full,
        "compiled_split_after_y": lambda x: g_rs(g_y(x)),
        "compiled_separate": lambda x: (lambda y: s_scale(y, s_sum(y)))(s_y(x)),
    }


class EnergyMeter:
    def __init__(self, torch_device):
        import pynvml
        import torch

        pynvml.nvmlInit()
        self.nvml = pynvml
        uuid = str(torch.cuda.get_device_properties(torch_device).uuid)
        uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        self.name = pynvml.nvmlDeviceGetName(self.handle)

    def millijoules(self):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


def kernel_launches(fn, x):
    import torch

    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return len(names), sorted(set(names))


def measure_window(fn, x, meter, window_s):
    import torch

    calls = 0
    torch.cuda.synchronize()
    e0, t0 = meter.millijoules(), time.perf_counter()
    while True:
        for _ in range(8):
            fn(x)
        calls += 8
        torch.cuda.synchronize()
        if time.perf_counter() - t0 >= window_s:
            break
    t1, e1 = time.perf_counter(), meter.millijoules()
    if e1 <= e0:
        return None
    return {"calls": calls, "seconds_per_call": (t1 - t0) / calls, "joules_per_call": (e1 - e0) / 1000 / calls}


def screen(shape, k, args, meter, out_dir, rows, log):
    import torch

    torch._dynamo.reset()
    torch.manual_seed(args.seed)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    candidates = make_candidates(k)
    reference = candidates["eager_baseline"](x)
    scale = reference.abs().max().item()
    info = {}
    for name, fn in candidates.items():
        start = time.perf_counter()
        for _ in range(args.warmup):
            result = fn(x)
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - start
        rel_err = ((result - reference).abs().max().item() / scale) if scale else 0.0
        count, names = kernel_launches(fn, x)
        info[name] = {"setup_seconds": setup_s, "max_rel_error": rel_err, "correct": rel_err <= args.rel_tol,
                      "kernels_per_call": count, "kernel_names": names, "trials": []}
        del result
    order = list(candidates)
    rng = random.Random(args.seed)
    for round_index in range(args.rounds):
        rng.shuffle(order)
        for name in order:
            trial = measure_window(candidates[name], x, meter, args.window)
            info[name]["trials"].append({"round": round_index, **(trial or {"invalid": "energy counter did not advance"})})
    label = f"{shape[0]}x{shape[1]}"
    summary = {}
    for name, entry in info.items():
        valid = [t for t in entry["trials"] if "joules_per_call" in t]
        if len(valid) != args.rounds:
            summary[name] = None
            continue
        ts = [t["seconds_per_call"] for t in valid]
        es = [t["joules_per_call"] for t in valid]
        summary[name] = {"us": statistics.median(ts) * 1e6, "us_min": min(ts) * 1e6, "us_max": max(ts) * 1e6,
                         "mJ": statistics.median(es) * 1e3, "mJ_min": min(es) * 1e3, "mJ_max": max(es) * 1e3}
    compiled = {n: s for n, s in summary.items() if n != "eager_baseline" and s and info[n]["correct"]}
    winner, cutoff = None, None
    if len(compiled) == 3:
        fastest = min(s["us"] for s in compiled.values())
        cutoff = fastest * 1.05
        eligible = {n: s for n, s in compiled.items() if s["us"] <= cutoff}
        winner = min(eligible, key=lambda n: (eligible[n]["mJ"], eligible[n]["us"], n))
    for name, s in summary.items():
        rows.append({"shape": label, "k_transcendentals": k, "candidate": name,
                     "runtime_us_median": round(s["us"], 2) if s else "", "runtime_us_min": round(s["us_min"], 2) if s else "",
                     "runtime_us_max": round(s["us_max"], 2) if s else "",
                     "energy_mJ_median": round(s["mJ"], 3) if s else "", "energy_mJ_min": round(s["mJ_min"], 3) if s else "",
                     "energy_mJ_max": round(s["mJ_max"], 3) if s else "",
                     "kernels_per_call": info[name]["kernels_per_call"], "max_rel_error": f"{info[name]['max_rel_error']:.2e}",
                     "correct": info[name]["correct"], "winner": name == winner})
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw" / f"{label}_k{k}.json").write_text(json.dumps(
        {"shape": list(shape), "k": k, "candidates": info, "summary": summary, "winner": winner,
         "runtime_cutoff_us": cutoff}, indent=2))
    full, split = summary.get("compiled_full"), summary.get("compiled_split_after_y")
    note = ""
    if full and split:
        note = f" | split_after_y vs full: time {split['us'] / full['us']:.3f}x, energy {split['mJ'] / full['mJ']:.3f}x"
    log(f"{label} k={k}: winner={winner}{note}")
    del x, reference
    torch.cuda.empty_cache()
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="+", default=["8192x8192", "2048x32768"])
    parser.add_argument("--k", nargs="+", type=int, default=[2, 4, 8, 16, 24])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--window", type=float, default=2.0, help="Seconds per measurement window")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rel-tol", type=float, default=1e-4, help="Max |compiled - eager| / max|eager|")
    parser.add_argument("--output-dir", type=Path, default=Path("results/haripreeth/exp3_fanout_screen"))
    args = parser.parse_args()

    import torch
    import torch._inductor.config as inductor_config

    if not torch.cuda.is_available():
        sys.exit("CUDA is unavailable; this screen measures GPU energy only")
    inductor_config.triton.cudagraphs = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meter = EnergyMeter(0)
    progress = open(args.output_dir / "progress.log", "a")

    def log(line):
        stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
        print(stamped, flush=True)
        progress.write(stamped + "\n")
        progress.flush()

    environment = {"gpu": meter.name, "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                   "python": platform.python_version(), "argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
                   "inductor": {"triton.cudagraphs": False, "persistent_reductions": inductor_config.triton.persistent_reductions,
                                "realize_reads_threshold": inductor_config.realize_reads_threshold},
                   "measurement": "ordinary execution; synchronized wall-clock and NVML GPU-device energy over the same window"}
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    log(f"start: {meter.name}, torch {torch.__version__}, shapes={args.shapes}, k={args.k}")
    rows = []
    for shape_text in args.shapes:
        shape = tuple(int(v) for v in shape_text.split("x"))
        for k in args.k:
            try:
                screen(shape, k, args, meter, args.output_dir, rows, log)
            except Exception as error:
                log(f"{shape_text} k={k}: FAILED {type(error).__name__}: {error}")
            with open(args.output_dir / "summary.csv", "w", newline="") as handle:
                if rows:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    log("done")


if __name__ == "__main__":
    main()
