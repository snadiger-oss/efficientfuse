"""Copy planner runs into readable experiment folders with descriptive candidate names.

Raw JSON is copied unchanged (internal IDs such as p00_launch are part of its
fingerprints); only the derived summary, plot, and README use readable names.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil

from greenfuse.results import load_report, summarize_candidate

GROUPINGS = {
    "p11": ("separate", "[bias add] [ReLU] [residual add]"),
    "p01": ("first_pair", "[bias add + ReLU] [residual add]"),
    "p10": ("last_pair", "[bias add] [ReLU + residual add]"),
    "p00": ("full", "[bias add + ReLU + residual add]"),
}
MODES = {"launch": "ordinary", "graph": "replay"}
# Default validated categorical slots 1 and 2; marker shape is the second cue for mode.
COLORS = {"full": "#2a78d6", "last_pair": "#eb6834", "first_pair": "#1baf7a", "separate": "#6250d6", "eager": "#52514e"}


def readable(candidate_id):
    if candidate_id == "eager_launch":
        return "eager_baseline"
    partition, mode = candidate_id.rsplit("_", 1)
    return f"compiled_{GROUPINGS[partition][0]}_{MODES[mode]}"


def grouping_of(candidate_id):
    return "eager" if candidate_id == "eager_launch" else GROUPINGS[candidate_id.rsplit("_", 1)[0]][0]


def summaries(report):
    gpu = report["scope"] == "gpu"
    return {cid: summarize_candidate(entry, report["config"]["rounds"], gpu_run=gpu)
            for cid, entry in report["candidates"].items()}


def write_rows(run_dir, decision, shape):
    rows = []
    for name in ("initial", "validation"):
        path = run_dir / f"{name}.json"
        if not path.exists():
            continue
        report = load_report(path)
        for cid, summary in summaries(report).items():
            time, energy = summary["timing"], summary["energy"]
            rows.append({
                "shape": shape, "pass": name, "candidate": readable(cid), "internal_id": cid,
                "role": report["candidates"][cid]["role"],
                "runtime_us_median": round(time["median"] * 1e6, 3) if time["count"] else "",
                "runtime_us_min": round(time["minimum"] * 1e6, 3) if time["count"] else "",
                "runtime_us_max": round(time["maximum"] * 1e6, 3) if time["count"] else "",
                "energy_mJ_median": round(energy["median"] * 1e3, 4) if energy["count"] else "",
                "energy_mJ_min": round(energy["minimum"] * 1e3, 4) if energy["count"] else "",
                "energy_mJ_max": round(energy["maximum"] * 1e3, 4) if energy["count"] else "",
                "rounds": time["count"],
                "selected": cid == decision.get("selected_id"),
            })
    return rows


def plot(run_dir, decision, shape, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = load_report(run_dir / "initial.json")
    stats = summaries(report)
    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for cid, summary in stats.items():
        time, energy = summary["timing"], summary["energy"]
        if not (time["count"] and energy["count"]):
            continue
        x, y = time["median"] * 1e6, energy["median"] * 1e3
        group, replay = grouping_of(cid), cid.endswith("_graph")
        color = COLORS[group]
        axis.errorbar(x, y, xerr=[[x - time["minimum"] * 1e6], [time["maximum"] * 1e6 - x]],
                      yerr=[[y - energy["minimum"] * 1e3], [energy["maximum"] * 1e3 - y]],
                      fmt="s" if group == "eager" else ("D" if replay else "o"), markersize=8,
                      color=color, mfc="white" if replay else color, mew=2, elinewidth=1, capsize=3,
                      label=readable(cid))
        if cid == decision.get("selected_id"):
            axis.scatter([x], [y], s=380, facecolors="none", edgecolors="#0b0b0b", linewidths=1.5, zorder=5)
            axis.annotate("selected", (x, y), xytext=(10, 10), textcoords="offset points", color="#0b0b0b", fontsize=9)
    selection = decision.get("selection") or {}
    if selection.get("runtime_limit_seconds"):
        limit = selection["runtime_limit_seconds"] * 1e6
        axis.axvline(limit / 1.05, color="#52514e", linewidth=1, linestyle=":", label="fastest candidate runtime")
        axis.axvline(limit, color="#52514e", linewidth=1, linestyle="--", label="1.05× runtime cutoff")
    axis.grid(color="#e5e4e0", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_xlabel("Synchronized wall-clock runtime per call (µs)")
    axis.set_ylabel("GPU-device energy per call (mJ)")
    axis.set_title(f"W1 {shape}: runtime vs GPU energy (initial search, median and observed range)", fontsize=11)
    axis.legend(fontsize=8, frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def export(run_dir, dest, shape):
    run_dir, dest = Path(run_dir), Path(dest)
    decision = json.loads((run_dir / "decision.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    shutil.copytree(run_dir, raw, ignore=shutil.ignore_patterns("summary.csv", "runtime_vs_energy.png"))
    rows = write_rows(run_dir, decision, shape)
    with open(dest / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if (run_dir / "initial.json").exists():
        plot(run_dir, decision, shape, dest / "runtime_vs_energy.png")
    return decision, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Folder holding initial.json, validation.json, decision.json")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--shape", required=True, help="Label such as 4096x4096")
    args = parser.parse_args()
    decision, _ = export(args.run_dir, args.dest, args.shape)
    print(f"{args.shape}: {decision['status']} -> {readable(decision['selected_id']) if decision.get('selected_id') else 'none'} | {args.dest}")


if __name__ == "__main__":
    raise SystemExit(main())
