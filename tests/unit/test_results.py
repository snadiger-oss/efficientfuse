"""Synthetic records validate units, missing data, coverage and serialization."""

import copy
import csv
from pathlib import Path
import tempfile
import unittest

from greenfuse.results import Trial, load_report, save_report, statistics_for, summarize_candidate, validate_report, write_summary_csv


def sample_report():
    entry = {"id": "p00_launch", "role": "candidate", "preparations": [{"ready_for_gpu_measurement": True}],
             "failures": [], "trials": []}
    for kind in ("timing", "energy"):
        for index in range(3):
            trial = Trial(kind, 10, "p00_launch", index, status="ok", submitted_iterations=10,
                          completed_iterations=10, elapsed_seconds=4.0,
                          seconds_per_execution=0.4 if kind == "timing" else None,
                          energy_start_mj=1000 if kind == "energy" else None,
                          energy_end_mj=6000 if kind == "energy" else None,
                          energy_joules_per_execution=0.5 if kind == "energy" else None)
            entry["trials"].append(trial.to_dict())
    return {"schema_version": 1, "scope": "gpu", "complete": True, "config": {"rounds": 3},
            "requested_ids": ["p00_launch"], "candidates": {"p00_launch": entry}, "source": "SYNTHETIC TEST DATA"}


class ResultsTests(unittest.TestCase):
    def test_complete_candidate_and_baseline_eligibility(self):
        entry = sample_report()["candidates"]["p00_launch"]
        self.assertTrue(summarize_candidate(entry, 3, gpu_run=True)["eligible_for_selection"])
        self.assertFalse(summarize_candidate(entry, 3, gpu_run=False)["eligible_for_selection"])
        entry["role"] = "external_baseline"
        self.assertFalse(summarize_candidate(entry, 3, gpu_run=True)["eligible_for_selection"])

    def test_missing_trial_and_bad_profile_block_selection(self):
        entry = sample_report()["candidates"]["p00_launch"]
        entry["trials"].pop()
        self.assertFalse(summarize_candidate(entry, 3, gpu_run=True)["eligible_for_selection"])
        entry = sample_report()["candidates"]["p00_launch"]
        entry["preparations"][0]["ready_for_gpu_measurement"] = False
        self.assertFalse(summarize_candidate(entry, 3, gpu_run=True)["eligible_for_selection"])

    def test_nulls_survive_json_and_csv(self):
        report = sample_report()
        for trial in report["candidates"]["p00_launch"]["trials"]:
            if trial["kind"] == "energy":
                trial.update(status="unavailable", reason="SYNTHETIC missing counter", energy_joules_per_execution=None)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            save_report(report, path)
            restored = load_report(path)
            self.assertEqual(restored, report)
            write_summary_csv(restored, Path(root) / "summary.csv")
            with (Path(root) / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["joules_median"], "")
            self.assertEqual(row["eligible"], "False")

    def test_rejects_wrong_units_incomplete_counts_duplicate_and_nan(self):
        for mutation in (lambda t: t.update(seconds_per_execution=400),
                         lambda t: t.update(completed_iterations=9),
                         lambda t: t.update(elapsed_seconds=float("nan"))):
            report = sample_report()
            mutation(report["candidates"]["p00_launch"]["trials"][0])
            with self.assertRaises(ValueError):
                validate_report(report)
        report = sample_report()
        trials = report["candidates"]["p00_launch"]["trials"]
        trials.append(copy.deepcopy(trials[0]))
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_variability_preserves_outlier_and_no_fake_empty_statistics(self):
        summary = statistics_for([1, 2, 3, 100])
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["maximum"], 100)
        self.assertGreater(summary["cv"], 1)
        self.assertIsNone(statistics_for([])["median"])

    def test_complete_report_cannot_hide_missing_trial(self):
        report = sample_report()
        report["candidates"]["p00_launch"]["trials"].pop()
        with self.assertRaisesRegex(ValueError, "every requested"):
            validate_report(report)
        report["complete"] = False
        validate_report(report)


if __name__ == "__main__":
    unittest.main()
