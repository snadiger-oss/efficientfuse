"""Real tensors/Inductor; CPU timing only, never GPU energy validation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from greenfuse.execution.prepare import prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.measurement.baseline import EagerExecutor, prepare_eager
from greenfuse.measurement.correctness import check_correctness
from greenfuse.measurement.runner import MeasurementCandidate, MeasurementConfig, run_measurements
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.results import load_report
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Requires optional PyTorch")
class CpuMeasurementTests(unittest.TestCase):
    def test_real_compiled_and_eager_measurement_roundtrip(self):
        captured = capture_graph(workload_w1)
        inputs = make_inputs(w1_spec(3, 4))
        partition = enumerate_partitions(captured.description)[0]
        candidates = [MeasurementCandidate("p00_launch", lambda path: prepare_execution(
            build_partition(captured, partition), inputs, reference=workload_w1, allow_cpu=True)),
            MeasurementCandidate("eager_launch", lambda path: prepare_eager(
                workload_w1, inputs, captured.description.fingerprint, allow_cpu=True), role="external_baseline")]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "measurements.json"
            report = run_measurements(candidates, inputs, workload_w1,
                                      config=MeasurementConfig(rounds=2, window_seconds=0.01, allow_cpu=True), output_path=path)
            self.assertEqual(load_report(path), report)
        self.assertTrue(report["complete"])
        self.assertFalse(report["comparison_complete"])
        for entry in report["candidates"].values():
            self.assertTrue(entry["summary"]["timing"]["complete"])
            self.assertGreater(entry["summary"]["timing"]["median"], 0)
            self.assertIsNone(entry["summary"]["energy"]["median"])
            self.assertFalse(entry["summary"]["eligible_for_selection"])

    def test_stale_output_detected_by_changed_inputs(self):
        inputs = make_inputs(w1_spec(3, 4))
        stale = workload_w1(**inputs)
        executor = EagerExecutor(lambda **values: stale, inputs, allow_cpu=True)
        try:
            with self.assertRaises(AssertionError):
                check_correctness(executor, workload_w1, inputs)
        finally:
            executor.close()

    def test_input_mutation_detected(self):
        inputs = make_inputs(w1_spec(3, 4))
        def mutating(x, bias, residual):
            output = workload_w1(x, bias, residual)
            x.add_(1)
            return output
        executor = EagerExecutor(mutating, inputs, allow_cpu=True)
        try:
            with self.assertRaises(AssertionError):
                check_correctness(executor, workload_w1, inputs)
        finally:
            executor.close()

    def test_nonfinite_and_wrong_shape_rejected(self):
        import torch
        inputs = make_inputs(w1_spec(3, 4))
        for fn in (lambda **values: torch.full((3, 4), float("nan")), lambda **values: torch.zeros(4)):
            executor = EagerExecutor(fn, inputs, allow_cpu=True)
            try:
                with self.assertRaises(AssertionError):
                    check_correctness(executor, workload_w1, inputs)
            finally:
                executor.close()
