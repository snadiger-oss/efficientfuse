"""Build a line-by-line kernel trace for Experiment 3 from the generated Triton code.

Reads the code saved by exp3_inspect_kernels.py and writes, for one shape:

  KERNEL_TRACE_<shape>.md   launch order per grouping + every computation line of
                            every kernel, with the pass over the row it runs in and
                            what value it produces (x, y step i of K, row sum, out)
  KERNEL_TRACE_<shape>.csv  the same trace rows, machine-readable

With --chrome-traces it also re-runs each grouping once under the PyTorch
profiler and saves Chrome/Perfetto timelines (needs a GPU).
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

KERNEL = re.compile(r"async_compile\.triton\('(\w+)',\s*'''(.*?)'''", re.S)
RUN = re.compile(r"(\w+)\.run\(([^)]*)\)")
CANDIDATES = {
    "compiled_full": ("Full fusion", [(["x"], "out")]),
    "compiled_split_after_y": ("Partial: split after y", [(["x"], "y"), (["y"], "out")]),
    "compiled_separate": ("Separate (3 groups)", [(["x"], "y"), (["y"], "s"), (["y", "s"], "out")]),
}
KIND = {"triton_poi": "pointwise (1 pass over elements)", "triton_red": "looped row reduction",
        "triton_per": "persistent row reduction"}


def describe(value, k):
    if isinstance(value, tuple):
        step = value[1]
        return f"y step {step}/{k}" + ("  (= y)" if step == k else "")
    return value


def trace_kernel(body, ptr_names, k):
    """Return trace rows for one kernel body; ptr_names maps pointer params to tensors."""
    body = body[body.index("def triton"):]
    meaning, contents, rows = {}, {}, []
    current_pass = "-"
    passes = 0

    def val(token):
        return meaning.get(token.strip(), token.strip())

    for raw in body.splitlines()[1:]:
        line = raw.strip()
        if line.startswith("for r") and "offset in" in line:
            passes += 1
            current_pass = str(passes)
            rows.append((current_pass, line, f"── pass {passes} over the row starts (loop over row chunks)"))
            continue
        target, _, expr = line.partition(" = ")
        if line.startswith("tl.store("):
            ptr, value = re.match(r"tl\.store\((\w+)[^,]*,\s*(\w+)", line).groups()
            tensor = ptr_names.get(ptr, ptr)
            if isinstance(val(value), tuple) and tensor == "out":
                ptr_names[ptr] = tensor = "out buffer, reused to hold y"
            contents[ptr] = val(value)
            rows.append((current_pass, line, f"STORE {describe(val(value), k)} → memory [{tensor}]"))
        elif "tl.load(" in expr:
            ptr = re.search(r"tl\.load\((\w+)", expr).group(1)
            tensor = ptr_names.get(ptr, ptr)
            meaning[target] = contents.get(ptr, tensor)
            rows.append((current_pass, line, f"LOAD {describe(meaning[target], k)} ← memory [{tensor}]"))
        elif re.search(r"\.(sin|tanh)\(", expr):
            op = re.search(r"\.(sin|tanh)\(", expr).group(1)
            source = val(re.search(r"\.(?:sin|tanh)\((\w+)\)", expr).group(1))
            step = source[1] + 1 if isinstance(source, tuple) else 1
            meaning[target] = ("E", step)
            rows.append((current_pass, line, f"{op}  → {describe(meaning[target], k)}"))
        elif "tl.broadcast_to(" in expr:
            meaning[target] = val(re.search(r"broadcast_to\((\w+)", expr).group(1))
        elif "tl.sum(" in expr:
            meaning[target] = "row sum Σy"
            rows.append((f"end of {current_pass}", line, "finish row sum → Σy (one number per row)"))
        elif target.startswith("_tmp") and "tl.where" in expr:
            continue
        elif target.startswith("tmp") and re.fullmatch(r"_?tmp\d+ \+ _?tmp\d+", expr):
            left, right = [t.strip() for t in expr.split("+")]
            added = val(right if left.startswith("_") else left)
            meaning[target] = "partial sum"
            rows.append((current_pass, line, f"add {describe(added, k)} to running row sum"))
        elif re.fullmatch(r"_?tmp\d+ \* _?tmp\d+", expr):
            left, right = [val(t) for t in expr.split("*")]
            meaning[target] = "out = y × Σy"
            rows.append((current_pass, line, f"multiply {describe(left, k)} × {describe(right, k)} → out"))
        elif target.startswith("_tmp") and "tl.full" in expr:
            rows.append(("-", line, "initialise running row sum = 0"))
    return rows, passes


def wrapper_calls(source):
    call = source[source.index("def call(self, args)"):]
    call = call[:call.index("return (") + 40]
    returned = re.search(r"return \((\w+)", call).group(1)
    arg_names = re.search(r"\n\s+(.*?)= args\n", call).group(1).replace(" ", "").rstrip(",").split(",")
    return RUN.findall(call), returned, arg_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="8192x8192")
    parser.add_argument("--k", nargs="+", type=int, default=[4, 8, 12, 16, 24, 32])
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/haripreeth/exp3_fanout_screen/kernel_analysis"))
    parser.add_argument("--screen-csv", type=Path, default=Path("results/haripreeth/exp3_fanout_screen/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/haripreeth/exp3_fanout_screen/kernel_trace"))
    parser.add_argument("--chrome-traces", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    screen = {}
    if args.screen_csv.exists():
        for row in csv.DictReader(open(args.screen_csv)):
            if row["shape"] == args.shape:
                screen[(int(row["k_transcendentals"]), row["candidate"])] = row

    md, csv_rows, overview = [], [], []
    for k in args.k:
        section = [f"\n---\n\n## K = {k}\n"]
        stats = {}
        for cand, (title, groups) in CANDIDATES.items():
            folder = args.analysis_dir / f"{args.shape}_k{k}" / cand
            timings = {kern["kernel"]: kern["gpu_us_per_call"] for kern in json.loads((folder / "kernels.json").read_text())}
            launches, traces, total_evals, recompute = [], [], 0, False
            for index, (inputs, output) in enumerate(groups, start=1):
                source = (folder / f"group{index}_generated.py").read_text()
                bodies = dict(KERNEL.findall(source))
                runs, returned, arg_names = wrapper_calls(source)
                for kernel, run_args in runs:
                    run_args = [a.strip() for a in run_args.split(",")]
                    body = bodies[kernel]
                    params = re.search(r"def \w+\((.*?)\)", body).group(1).split(",")
                    params = [p.split(":")[0].strip() for p in params]
                    names = {}
                    for param, arg in zip(params, run_args):
                        if arg in arg_names:
                            names[param] = inputs[arg_names.index(arg)]
                        elif arg == returned:
                            names[param] = output
                        else:
                            names[param] = f"temp {arg}"
                    rows, passes = trace_kernel(body, names, k)
                    evals = sum(1 for r in rows if r[2].startswith(("sin", "tanh")))
                    total_evals += evals
                    stores = [r for r in rows if r[2].startswith("STORE")]
                    loads = [r for r in rows if r[2].startswith("LOAD")]
                    recompute |= sum(1 for r in loads if "x" in r[2].split("memory")[1]) > 1 and evals > k
                    kind = next((v for p, v in KIND.items() if kernel.startswith(p)), "other")
                    order = len(launches) + 1
                    launches.append(f"| {order} | group {index} | `{kernel}` | {kind} | {passes or 1} | "
                                    f"{', '.join(sorted({r[2].split('[')[1].rstrip(']') for r in loads}))} | "
                                    f"{', '.join(sorted({r[2].split('[')[1].rstrip(']') for r in stores}))} | "
                                    f"{evals} | {timings.get(kernel, '')} |")
                    trace_md = [f"\n**Launch {order}: `{kernel}`** ({kind}; group {index}: inputs {', '.join(inputs)} → output {output})\n",
                                "| # | Pass | Generated code (exact) | What it computes |", "|---|---|---|---|"]
                    for n, (pass_, code, meaning) in enumerate(rows, start=1):
                        trace_md.append(f"| {n} | {pass_} | `{code}` | {meaning} |")
                        csv_rows.append({"shape": args.shape, "k": k, "grouping": cand, "launch": order, "kernel": kernel,
                                         "step": n, "pass": pass_, "code": code, "meaning": meaning})
                    traces += trace_md
            s = screen.get((k, cand))
            stats[cand] = (len(launches), total_evals, recompute, s)
            section += [f"\n### {title} — {len(launches)} kernel launch(es), {total_evals} sin/tanh per element\n",
                        "| Order | Group | Kernel | Kind | Passes over row | Reads | Writes | sin/tanh per element | GPU µs/call |",
                        "|---|---|---|---|---|---|---|---|---|", *launches, *traces]
        full, split = stats["compiled_full"], stats["compiled_split_after_y"]
        winner = None
        if full[3] and split[3]:
            winner = "Partial" if float(split[3]["energy_mJ_median"]) < float(full[3]["energy_mJ_median"]) else "Full"
        fmt = lambda s: f"{float(s['runtime_us_median']):.0f} µs / {float(s['energy_mJ_median']):.0f} mJ" if s else "n/a"
        overview.append(f"| {k} | 1 kernel, {full[1]} sin/tanh {'(recomputes y)' if full[2] else '(computes y once, stores it)'} | "
                        f"{split[0]} kernels, {split[1]} sin/tanh (computes y once) | {fmt(full[3])} | {fmt(split[3])} | **{winner}** |")
        md += section

    header = [
        f"# Experiment 3 kernel trace — shape {args.shape}",
        "",
        "Every row below is copied from the Triton code TorchInductor generated for this exact run",
        "(`kernel_analysis/<shape>_k<K>/<grouping>/group*_generated.py`). The right-hand column says what each line computes.",
        "",
        "```python",
        "def workload(x):                       # x: FP32 matrix",
        "    y = x",
        "    for i in range(K):                 # K chained expensive ops",
        "        y = torch.sin(y) if i % 2 == 0 else torch.tanh(y)",
        "    return y * y.sum(dim=-1, keepdim=True)",
        "```",
        "",
        "- **Full fusion:** one compiled group `[sin/tanh chain + row sum + multiply]`",
        "- **Partial (split after y):** `[sin/tanh chain]` `[row sum + multiply]`",
        "- **Separate:** `[sin/tanh chain]` `[row sum]` `[multiply]`",
        "",
        "How to read a trace: a row is too long to keep on-chip, so a reduction kernel walks it twice:",
        "**pass 1** builds the row sum, **pass 2** multiplies and writes `out`. `y step i/K` is the value after i sin/tanh ops.",
        "If `LOAD x` and the sin/tanh chain appear in **both** passes, the kernel is recomputing `y`.",
        "",
        "## Overview",
        "",
        "| K | Full fusion | Partial (split after y) | Full: time / energy | Partial: time / energy | Lower energy |",
        "|---|---|---|---|---|---|",
        *overview,
        "",
        "Time and energy: screening run, medians of 5 rounds, ordinary execution, NVML GPU-device energy. GPU µs/call in the",
        "per-kernel tables comes from a separate profiler run and excludes launch overhead.",
    ]
    out_md = args.output_dir / f"KERNEL_TRACE_{args.shape}.md"
    out_md.write_text("\n".join(header + md) + "\n")
    with open(args.output_dir / f"KERNEL_TRACE_{args.shape}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Trace: {out_md}")

    if args.chrome_traces:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from exp3_screen_fanout import make_candidates

        shape = tuple(int(v) for v in args.shape.split("x"))
        folder = args.output_dir / f"chrome_traces_{args.shape}"
        folder.mkdir(exist_ok=True)
        for k in args.k:
            torch._dynamo.reset()
            x = torch.randn(shape, device="cuda")
            for cand, fn in make_candidates(k).items():
                for _ in range(3):
                    fn(x)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        fn(x)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(folder / f"k{k}_{cand}.json"))
            del x
            torch.cuda.empty_cache()
        print(f"Chrome traces: {folder} (open in https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
