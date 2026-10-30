"""Portable measurement records; derived summaries never replace raw trials."""

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

SCHEMA_VERSION = 1


@dataclass
class Trial:
    kind: str
    requested_iterations: int
    candidate_id: str = ""
    round_index: int = -1
    order_index: int = -1
    status: str = "pending"
    reason: str | None = None
    submitted_iterations: int = 0
    completed_iterations: int = 0
    start_monotonic_seconds: float | None = None
    end_monotonic_seconds: float | None = None
    elapsed_seconds: float | None = None
    seconds_per_execution: float | None = None
    energy_start_mj: int | None = None
    energy_end_mj: int | None = None
    energy_joules_per_execution: float | None = None
    counter_read_intervals: list = field(default_factory=list)
    environment_before: dict = field(default_factory=dict)
    environment_after: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def portable(value):
    """Normalize tuples and other JSON containers before handoff or storage."""
    return json.loads(json.dumps(value, allow_nan=False))


def statistics_for(values):
    if not values:
        return {key: None for key in ("median", "minimum", "maximum", "mad", "cv")} | {"count": 0}
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Metrics must be finite and positive")
    middle = statistics.median(values)
    return {"count": len(values), "median": middle, "minimum": min(values), "maximum": max(values),
            "mad": statistics.median(abs(v - middle) for v in values),
            "cv": statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None}


def summarize_candidate(candidate, rounds, *, gpu_run):
    """Require every planned trial; retain invalid trials and partial summaries."""
    summaries = {}
    complete = True
    for kind, metric in (("timing", "seconds_per_execution"), ("energy", "energy_joules_per_execution")):
        trials = [t for t in candidate["trials"] if t["kind"] == kind]
        valid = [t for t in trials if t["status"] == "ok"]
        values = [t[metric] for t in valid]
        summaries[kind] = statistics_for(values)
        coverage = (len(trials) == rounds and len(valid) == rounds
                    and {t["round_index"] for t in valid} == set(range(rounds)))
        summaries[kind]["complete"] = coverage
        complete = complete and coverage
    summaries["all_trials_valid"] = complete
    summaries["eligible_for_selection"] = bool(
        complete and gpu_run and rounds >= 3 and candidate["role"] == "candidate"
        and not candidate.get("failures")
        and candidate.get("preparations")
        and all(p.get("ready_for_gpu_measurement") for p in candidate["preparations"])
    )
    summaries["variability_note"] = "Medians, full range, MAD and sample CV; no outliers removed; these are not confidence intervals"
    return summaries


def validate_report(report):
    """Reject malformed units/counts and nonfinite or invented missing values."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported measurement schema")
    rounds = report.get("config", {}).get("rounds")
    if type(rounds) is not int or rounds < 1 or type(report.get("complete")) is not bool:
        raise ValueError("Invalid round count or completion flag")
    if report.get("scope") not in ("gpu", "cpu_development", "unavailable"):
        raise ValueError("Unknown measurement scope")
    ids = report.get("requested_ids", [])
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(report.get("candidates", {})):
        raise ValueError("Requested candidate coverage is inconsistent")
    for key, candidate in report["candidates"].items():
        if candidate["id"] != key or candidate["role"] not in ("candidate", "external_baseline"):
            raise ValueError("Invalid candidate identity/role")
        seen = set()
        for trial in candidate["trials"]:
            slot = trial["kind"], trial["round_index"]
            if slot in seen or trial["kind"] not in ("timing", "energy") or trial["candidate_id"] != key:
                raise ValueError("Duplicate or inconsistent trial identity")
            seen.add(slot)
            if type(trial["round_index"]) is not int or not 0 <= trial["round_index"] < rounds:
                raise ValueError("Trial round is outside the requested range")
            counts = [trial[n] for n in ("requested_iterations", "submitted_iterations", "completed_iterations")]
            if any(type(n) is not int or n < 0 for n in counts) or not counts[0] >= counts[1] >= counts[2]:
                raise ValueError("Invalid execution counts")
            if trial["status"] == "ok":
                if counts[0] == 0 or counts[0] != counts[2]:
                    raise ValueError("Successful trials must complete every invocation")
                elapsed = trial["elapsed_seconds"]
                if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("Missing successful window duration")
                if trial["kind"] == "timing":
                    expected = elapsed / counts[2]
                    actual = trial["seconds_per_execution"]
                    if trial["energy_joules_per_execution"] is not None:
                        raise ValueError("Timing passes cannot supply energy")
                else:
                    first, last = trial["energy_start_mj"], trial["energy_end_mj"]
                    if any(type(n) is not int or n < 0 for n in (first, last)) or last <= first:
                        raise ValueError("Invalid successful energy counter delta")
                    expected = (last - first) / 1000 / counts[2]
                    actual = trial["energy_joules_per_execution"]
                    if trial["seconds_per_execution"] is not None:
                        raise ValueError("Energy-pass wall time is diagnostic, not a timing trial")
                if type(actual) not in (int, float) or not math.isclose(actual, expected, rel_tol=1e-9):
                    raise ValueError("Metric does not match raw units/counts")
            elif trial["status"] not in ("failed", "invalid", "unavailable") or not trial["reason"]:
                raise ValueError("Unsuccessful trials require an explicit status/reason")
            elif trial["seconds_per_execution"] is not None or trial["energy_joules_per_execution"] is not None:
                raise ValueError("Unsuccessful trials must have null metrics")
        if report["complete"] and seen != {(kind, index) for kind in ("timing", "energy") for index in range(rounds)}:
            raise ValueError("Completed coverage requires a record for every requested trial slot")
    json.dumps(report, allow_nan=False)


def save_report(report, path):
    validate_report(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a fully serialized checkpoint has reached disk.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".measurement-", suffix=".json", delete=False) as out:
        temporary = Path(out.name)
        try:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_report(path):
    report = json.loads(Path(path).read_text())
    validate_report(report)
    return report


def write_summary_csv(report, path):
    validate_report(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["candidate_id", "role", "timing_count", "seconds_median",
                                               "seconds_mad", "energy_count", "joules_median", "joules_mad", "eligible"])
        writer.writeheader()
        for candidate in report["candidates"].values():
            summary = summarize_candidate(candidate, report["config"]["rounds"], gpu_run=report["scope"] == "gpu")
            writer.writerow({"candidate_id": candidate["id"], "role": candidate["role"],
                             "timing_count": summary["timing"]["count"], "seconds_median": summary["timing"]["median"],
                             "seconds_mad": summary["timing"]["mad"], "energy_count": summary["energy"]["count"],
                             "joules_median": summary["energy"]["median"], "joules_mad": summary["energy"]["mad"],
                             "eligible": summary["eligible_for_selection"]})
