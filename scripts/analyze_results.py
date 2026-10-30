"""Derive a CSV table and optional plot from saved, validated measurements."""

import argparse
from pathlib import Path

from greenfuse.results import load_report, summarize_candidate, write_summary_csv


def plot_results(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu = report["scope"] == "gpu"
    entries = list(report["candidates"].values())
    summaries = {c["id"]: summarize_candidate(c, report["config"]["rounds"], gpu_run=gpu) for c in entries}
    usable = [c for c in entries if summaries[c["id"]]["timing"]["count"]]
    if not usable:
        raise ValueError("No valid timing observations to plot")
    has_energy = any(summaries[c["id"]]["energy"]["count"] for c in usable)
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    for index, entry in enumerate(usable):
        summary = summaries[entry["id"]]
        time, energy = summary["timing"], summary["energy"]
        label = entry["id"] + (" (eager baseline)" if entry["role"] == "external_baseline" else "")
        if has_energy:
            if not energy["count"]:
                continue
            axis.errorbar(time["median"] * 1e6, energy["median"] * 1e6,
                          xerr=[[1e6 * (time["median"] - time["minimum"])], [1e6 * (time["maximum"] - time["median"])]],
                          yerr=[[1e6 * (energy["median"] - energy["minimum"])], [1e6 * (energy["maximum"] - energy["median"])]],
                          fmt="s" if entry["role"] == "external_baseline" else "o", label=label, capsize=3)
        else:
            axis.errorbar(time["median"] * 1e6, index,
                          xerr=[[1e6 * (time["median"] - time["minimum"])], [1e6 * (time["maximum"] - time["median"])]],
                          fmt="o", capsize=3)
    eligible = [summaries[c["id"]] for c in usable if c["role"] == "candidate"]
    complete = bool(report["complete"] and eligible and len(eligible) == sum(c["role"] == "candidate" for c in entries)
                    and all(s["eligible_for_selection"] for s in eligible))
    if has_energy and complete:
        cutoff = min(s["timing"]["median"] for s in eligible) * 1.05
        axis.axvline(cutoff * 1e6, linestyle="--", color="gray", label="5% runtime cutoff")
    if has_energy:
        axis.set_ylabel("GPU-device energy per call (µJ)")
        axis.legend(fontsize=8)
    else:
        axis.set_yticks(range(len(usable)), [c["id"] for c in usable])
        axis.set_ylabel("Execution variant; energy unavailable")
    axis.set_xlabel("Wall-clock runtime per call (µs)")
    axis.set_title(("GPU measurements" if gpu else "CPU development measurements — not GPU results")
                   + "\nMedians and full observed ranges; separate timing/energy passes")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path, help="Optional PNG/PDF; requires matplotlib")
    args = parser.parse_args()
    report = load_report(args.input)
    output = args.csv or args.input.with_suffix(".csv")
    write_summary_csv(report, output)
    print(f"Scope: {report['scope']}; requested coverage complete: {report['complete']}")
    for entry in report["candidates"].values():
        summary = summarize_candidate(entry, report["config"]["rounds"], gpu_run=report["scope"] == "gpu")
        print(f"{entry['id']}: seconds/call={summary['timing']['median']}, joules/call={summary['energy']['median']}, "
              f"eligible={summary['eligible_for_selection']}")
    print(f"CSV: {output}")
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        try:
            plot_results(report, args.plot)
        except (ImportError, ValueError) as error:
            parser.exit(1, f"Plot unavailable: {error}\n")
        print(f"Plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
