"""SYNTHETIC full calibration handoff; no claimed hardware measurements."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from greenfuse.planner.cache import DecisionCache
from greenfuse.planner.calibrate import calibrate_context
from greenfuse.results import fingerprint
from tests.unit.planner_fixtures import synthetic_report


class CalibrationTests(unittest.TestCase):
    def test_full_repeat_before_cache_write(self):
        initial, fresh = synthetic_report(), synthetic_report(fresh=True)
        identity = initial["metadata"]["planner_identity"]
        context = SimpleNamespace(key=fingerprint(identity), identity=identity, environment=initial["environment"],
                                  config=SimpleNamespace(device="cuda:0"))
        with tempfile.TemporaryDirectory() as root, \
             patch("greenfuse.planner.calibrate.benchmark_context", side_effect=[initial, fresh]) as benchmark, \
             patch("greenfuse.planner.calibrate.environment_report", return_value=fresh["environment"]):
            result = calibrate_context(context, output_dir=Path(root) / "runs", cache_dir=Path(root) / "cache", exclusive_gpu=True)
            self.assertEqual(result["status"], "validated", result)
            self.assertEqual(benchmark.call_count, 2)
            self.assertTrue(benchmark.call_args.kwargs["validation"])
            self.assertTrue(result["cache_written"])
            self.assertEqual(DecisionCache(Path(root) / "cache").lookup(identity).status, "hit")

    def test_incomplete_initial_never_starts_validation_or_caches(self):
        initial = synthetic_report()
        initial["complete"] = False
        identity = initial["metadata"]["planner_identity"]
        context = SimpleNamespace(key=fingerprint(identity), identity=identity)
        with tempfile.TemporaryDirectory() as root, \
             patch("greenfuse.planner.calibrate.benchmark_context", return_value=initial) as benchmark:
            result = calibrate_context(context, output_dir=Path(root) / "runs", cache_dir=Path(root) / "cache")
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(benchmark.call_count, 1)
            self.assertFalse(result["cache_written"])
            self.assertEqual(DecisionCache(Path(root) / "cache").lookup(identity).status, "miss")

    def test_environment_change_prevents_validation_and_cache(self):
        initial = synthetic_report()
        identity = initial["metadata"]["planner_identity"]
        context = SimpleNamespace(key=fingerprint(identity), identity=identity, config=SimpleNamespace(device="cuda:0"))
        with tempfile.TemporaryDirectory() as root, \
             patch("greenfuse.planner.calibrate.benchmark_context", return_value=initial) as benchmark, \
             patch("greenfuse.planner.calibrate.environment_report", return_value={"status": "unavailable"}):
            result = calibrate_context(context, output_dir=Path(root) / "runs", cache_dir=Path(root) / "cache")
            self.assertEqual(result["status"], "inconclusive")
            self.assertEqual(benchmark.call_count, 1)
            self.assertFalse(result["cache_written"])
