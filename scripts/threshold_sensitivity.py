"""Re-run selection on saved reports at several runtime allowances (no GPU).

For every exported run folder (<exp>/<shape>/raw/{initial,validation}.json), report the candidate
selected at each tolerance in both passes, and whether the two passes agree. Every run is listed,
wins and losses alike.
"""

import argparse
import csv
from pathlib import Path

from greenfuse.planner.select import select_report
from greenfuse.results import load_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/bhargav"))
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = parser.parse_args()
    rows = []
    for raw in sorted(args.root.glob("exp*/*/raw")):
        if not (raw / "initial.json").exists():
            continue
        reports = {p: load_report(raw / f"{p}.json") for p in ("initial", "validation") if (raw / f"{p}.json").exists()}
        for tolerance in args.tolerances:
            picks = {}
            row = {"experiment": raw.parent.parent.name, "shape": raw.parent.name, "tolerance": tolerance}
            for name, report in reports.items():
                result = select_report(report, latency_tolerance=tolerance)
                picks[name] = result.get("selected_id")
                row[f"{name}_status"] = result["status"]
                row[f"{name}_selected"] = result.get("selected_id") or ""
                row[f"{name}_fastest"] = result.get("fastest_id") or ""
            row["passes_agree"] = len(reports) == 2 and picks.get("initial") is not None and picks.get("initial") == picks.get("validation")
            rows.append(row)
    out = args.root / "threshold_sensitivity.csv"
    fields = ["experiment", "shape", "tolerance", "initial_status", "initial_selected", "initial_fastest",
              "validation_status", "validation_selected", "validation_fastest", "passes_agree"]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
