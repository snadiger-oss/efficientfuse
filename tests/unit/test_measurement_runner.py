"""SYNTHETIC orchestration tests; these are never production measurements."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from greenfuse.execution.prepare import ExecutionRecord, PreparedExecution
from greenfuse.measurement.energy import NvmlEnergyMeter
from greenfuse.measurement.runner import MeasurementCandidate, MeasurementConfig, run_measurements
from greenfuse.results import Trial, load_report


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.device = SimpleNamespace(type="cpu")
        self.inputs = {"x": SimpleNamespace(device=self.device)}
        self.config = MeasurementConfig(rounds=2, allow_cpu=True, window_seconds=0.1)
        self.meter = NvmlEnergyMeter()
        self.meter.reason = "SYNTHETIC CPU TEST: no GPU energy"
        self.live = 0
        self.peak_live = 0
        self.calls = []

    def factory(self, candidate_id, *, fail_close=False):
        def prepare(path):
            self.live += 1
            self.peak_live = max(self.peak_live, self.live)
            executor = Mock(device=self.device)
            def close():
                self.live -= 1
                if fail_close:
                    raise RuntimeError("SYNTHETIC cleanup failure")
            executor.close.side_effect = close
            record = ExecutionRecord(candidate_id, "synthetic-graph", "launch", 1, status="cpu_validated")
            return PreparedExecution(record, executor)
        return prepare

    def injected_setup(self, stack):
        stack.enter_context(patch("greenfuse.measurement.runner.check_correctness", return_value={"status": "passed"}))
        stack.enter_context(patch("greenfuse.measurement.runner.synchronize"))
        stack.enter_context(patch("greenfuse.measurement.runner.calibrate_iterations", return_value={
            "status": "ready", "iterations": 10, "pilots": [], "reason": None}))
        def timing(executor, inputs, count):
            self.assertIs(inputs, self.inputs)
            return Trial("timing", count, status="ok", submitted_iterations=count, completed_iterations=count,
                         elapsed_seconds=1, seconds_per_execution=1 / count)
        stack.enter_context(patch("greenfuse.measurement.runner.measure_timing", side_effect=timing))

    def test_one_executor_at_a_time_all_slots_preserved_and_cpu_ineligible(self):
        candidates = [MeasurementCandidate(name, self.factory(name)) for name in ("a", "b")]
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            self.injected_setup(stack)
            report = run_measurements(candidates, self.inputs, None, config=self.config, energy_meter=self.meter,
                                      metadata={"source": "SYNTHETIC TEST"}, output_path=Path(root) / "result.json")
            restored = load_report(Path(root) / "result.json")
        self.assertEqual(self.peak_live, 1)
        self.assertEqual(self.live, 0)
        self.assertTrue(report["complete"])
        self.assertFalse(report["comparison_complete"])
        self.assertEqual(restored, report)
        for entry in report["candidates"].values():
            self.assertEqual(len(entry["trials"]), 4)
            self.assertTrue(entry["summary"]["timing"]["complete"])
            self.assertFalse(entry["summary"]["eligible_for_selection"])
            self.assertIsNone(entry["summary"]["energy"]["median"])

    def test_failed_preparation_remains_in_requested_coverage(self):
        record = ExecutionRecord("bad", "synthetic", "launch", 1, status="failed", reason="synthetic compile error")
        candidate = MeasurementCandidate("bad", lambda path: PreparedExecution(record))
        with tempfile.TemporaryDirectory() as root:
            report = run_measurements([candidate], self.inputs, None, config=self.config, energy_meter=self.meter,
                                      output_path=Path(root) / "result.json")
        entry = report["candidates"]["bad"]
        self.assertEqual(len(entry["trials"]), 4)
        self.assertTrue(all(t["status"] == "unavailable" for t in entry["trials"]))
        self.assertFalse(report["comparison_complete"])

    def test_cleanup_failure_invalidates_successful_window(self):
        candidate = MeasurementCandidate("a", self.factory("a", fail_close=True))
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            self.injected_setup(stack)
            report = run_measurements([candidate], self.inputs, None, config=self.config, energy_meter=self.meter,
                                      output_path=Path(root) / "result.json")
        for trial in report["candidates"]["a"]["trials"][:2]:
            self.assertEqual(trial["status"], "invalid")
            self.assertIsNone(trial["seconds_per_execution"])

    def test_interruption_closes_executor_and_keeps_incomplete_checkpoint(self):
        candidate = MeasurementCandidate("a", self.factory("a"))
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            self.injected_setup(stack)
            stack.enter_context(patch("greenfuse.measurement.runner.measure_timing", side_effect=KeyboardInterrupt))
            path = Path(root) / "result.json"
            with self.assertRaises(KeyboardInterrupt):
                run_measurements([candidate], self.inputs, None, config=self.config, energy_meter=self.meter, output_path=path)
            self.assertFalse(load_report(path)["complete"])
        self.assertEqual(self.live, 0)

    def test_short_windows_are_kept_but_invalid(self):
        candidate = MeasurementCandidate("a", self.factory("a"))
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            self.injected_setup(stack)
            stack.enter_context(patch("greenfuse.measurement.runner.measure_timing", return_value=Trial(
                "timing", 10, status="ok", submitted_iterations=10, completed_iterations=10,
                elapsed_seconds=0.01, seconds_per_execution=0.001)))
            report = run_measurements([candidate], self.inputs, None, config=self.config, energy_meter=self.meter,
                                      output_path=Path(root) / "result.json")
        self.assertFalse(report["candidates"]["a"]["summary"]["timing"]["complete"])

    def test_unverified_gpu_mode_cannot_be_benchmarked(self):
        self.device = SimpleNamespace(type="cuda")
        self.inputs = {"x": SimpleNamespace(device=self.device)}
        self.config = MeasurementConfig(rounds=1)
        candidate = MeasurementCandidate("a", self.factory("a"))
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            self.injected_setup(stack)
            profiler = stack.enter_context(patch("greenfuse.measurement.runner.profile_prepared"))
            timing = stack.enter_context(patch("greenfuse.measurement.runner.measure_timing"))
            report = run_measurements([candidate], self.inputs, None, config=self.config, energy_meter=self.meter,
                                      output_path=Path(root) / "result.json")
            profiler.assert_called_once()
            timing.assert_not_called()
        self.assertEqual(report["candidates"]["a"]["trials"][0]["status"], "failed")
        self.assertFalse(report["comparison_complete"])

    def test_synthetic_complete_gpu_handoff_excludes_eager_baseline(self):
        self.device = SimpleNamespace(type="cuda")
        self.inputs = {"x": SimpleNamespace(device=self.device)}
        self.config = MeasurementConfig(rounds=3)
        self.meter.available, self.meter.exclusive_gpu = True, True
        self.meter.snapshot = lambda: {"synthetic": True}
        self.meter.qualification_reason = lambda state: None
        candidates = [MeasurementCandidate("a", self.factory("a")),
                      MeasurementCandidate("eager", self.factory("eager"), role="external_baseline")]
        def verify(prepared, inputs, path):
            prepared.record.ready_for_gpu_measurement = True
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            self.injected_setup(stack)
            stack.enter_context(patch("greenfuse.measurement.runner.profile_prepared", side_effect=verify))
            stack.enter_context(patch("greenfuse.measurement.runner.measure_timing", side_effect=lambda *args: Trial(
                "timing", 10, status="ok", submitted_iterations=10, completed_iterations=10,
                elapsed_seconds=4, seconds_per_execution=0.4)))
            stack.enter_context(patch("greenfuse.measurement.runner.measure_energy", side_effect=lambda *args: Trial(
                "energy", 10, status="ok", submitted_iterations=10, completed_iterations=10,
                elapsed_seconds=4, energy_start_mj=1000, energy_end_mj=6000, energy_joules_per_execution=0.5)))
            report = run_measurements(candidates, self.inputs, None, config=self.config, energy_meter=self.meter,
                                      metadata={"source": "SYNTHETIC TEST ONLY"}, output_path=Path(root) / "result.json")
        self.assertTrue(report["comparison_complete"])
        self.assertTrue(report["candidates"]["a"]["summary"]["eligible_for_selection"])
        self.assertFalse(report["candidates"]["eager"]["summary"]["eligible_for_selection"])
        self.assertEqual(self.live, 0)


if __name__ == "__main__":
    unittest.main()
