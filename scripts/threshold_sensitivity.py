"""Re-run selection on saved reports at several runtime allowances (no GPU).

For every exported run folder (<exp>/<shape>/raw/{initial,validation}.json), report the candidate
selected at each tolerance in both passes, and whether the two passes agree. Every run is listed,
wins and losses alike.

The planner refuses a tolerance other than the predeclared one, and reports are fingerprinted, so
editing them is detected (guards against tuning the rule after seeing data). Secondary tolerances
are therefore computed here from the per-candidate medians the planner validated at the primary
policy, applying the same rule: fastest -> cutoff -> lowest median energy -> runtime -> ID.
They are labelled as a secondary policy declared after the data (Oct 30 2026).
"""


def apply_rule(candidates, tolerance):
    fastest = min(candidates, key=lambda c: (candidates[c]["seconds"], c))
    limit = candidates[fastest]["seconds"] * (1 + tolerance)
    feasible = [c for c, v in candidates.items() if v["seconds"] <= limit]
    return fastest, min(feasible, key=lambda c: (candidates[c]["joules"], candidates[c]["seconds"], c))

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
            row["policy"] = "primary (predeclared)" if tolerance == 0.05 else "secondary (declared after data)"
            for name, report in reports.items():
                result = select_report(report)  # primary policy, full validation of the evidence
                if result["status"] == "selected" and tolerance != 0.05:
                    fastest, selected = apply_rule(result["candidates"], tolerance)
                    result = {**result, "fastest_id": fastest, "selected_id": selected}
                picks[name] = result.get("selected_id")
                row[f"{name}_status"] = result["status"]
                row[f"{name}_selected"] = result.get("selected_id") or ""
                row[f"{name}_fastest"] = result.get("fastest_id") or ""
            row["passes_agree"] = len(reports) == 2 and picks.get("initial") is not None and picks.get("initial") == picks.get("validation")
            rows.append(row)
    out = args.root / "threshold_sensitivity.csv"
    fields = ["experiment", "shape", "tolerance", "policy", "initial_status", "initial_selected", "initial_fastest",
              "validation_status", "validation_selected", "validation_fastest", "passes_agree"]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
