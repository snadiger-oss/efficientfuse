"""Combine exported experiment summaries into one cross-workload table and figure.

Reads <experiment>/<shape>/summary.csv files written by export_experiment.py.
The figure uses the validation (fresh) pass and ordinary execution only: replay
numbers currently include per-call input copies (see results/README.md).
"""

import argparse
import csv
from pathlib import Path

GROUPINGS = ("full", "first_pair", "last_pair", "separate")
# Same slots as export_experiment.py so colors match across figures.
COLORS = {"full": "#2a78d6", "first_pair": "#1baf7a", "last_pair": "#eb6834", "separate": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def load(experiments):
    rows = []
    for label, folder, base_group in experiments:
        for path in sorted(Path(folder).glob("*/summary.csv")):
            for row in csv.DictReader(open(path)):
                rows.append({"experiment": Path(folder).name, "label": label, "base_group": base_group, **row})
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fresh = [r for r in rows if r["pass"] == "validation" and r["candidate"].endswith("_ordinary")]
    labels = list(dict.fromkeys(r["label"] for r in fresh))
    # Index energy to the best compiled ordinary option of the same workload (shared by W2's two GEMM policies).
    best = {}
    for r in fresh:
        best[r["base_group"]] = min(best.get(r["base_group"], float("inf")), float(r["energy_mJ_median"]))
    figure, axis = plt.subplots(figsize=(10, 5.2), layout="constrained")
    width = 0.2
    for gi, group in enumerate(GROUPINGS):
        for li, label in enumerate(labels):
            row = next(r for r in fresh if r["label"] == label and r["candidate"] == f"compiled_{group}_ordinary")
            value = float(row["energy_mJ_median"]) / best[row["base_group"]]
            x = li + (gi - 1.5) * width
            axis.bar(x, value, width=width - 0.03, color=COLORS[group], edgecolor="white",
                     linewidth=2, label=group if li == 0 else None)
            axis.annotate(f"{value:.2f}×", (x, value), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=8, color=MUTED)
    axis.axhline(1, color=MUTED, linewidth=1, linestyle="--")
    axis.set_ylim(0, 2.6)
    axis.set_xticks(range(len(labels)), labels, fontsize=9, color=INK)
    axis.set_ylabel("GPU energy per call vs best option (×)", color=INK)
    axis.set_title("Energy by grouping, 4096×4096 FP32, ordinary execution, fresh validation pass\n"
                   "1.00× = lowest-energy compiled option for that workload (W2: across both GEMM policies)",
                   fontsize=10, color=INK)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(title="grouping", fontsize=8, title_fontsize=8, frameon=False, ncols=4, loc="upper left")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/bhargav"))
    args = parser.parse_args()
    root = args.root
    experiments = [
        ("W2 linear+GELU\ncuBLAS GEMM", root / "exp3_w2_linear_gelu_cublas", "w2"),
        ("W2 linear+GELU\nTriton GEMM", root / "exp4_w2_linear_gelu_triton_gemm", "w2"),
        ("W3 add+LayerNorm\n+GELU", root / "exp5_w3_layernorm_reduction", "w3"),
        ("W4 fan-out\nh - mean(h)", root / "exp6_w4_fan_out", "w4"),
    ]
    rows = load(experiments)
    fields = ["experiment"] + [k for k in rows[0] if k not in ("experiment", "label", "base_group")]
    with open(root / "summary_all_experiments.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, root / "energy_by_workload.png")
    print(f"Wrote {root / 'summary_all_experiments.csv'} and {root / 'energy_by_workload.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
