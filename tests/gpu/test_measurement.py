"""Actual GPU acceptance tests; energy additionally needs an exclusive GPU."""

import os
from pathlib import Path
import tempfile
import unittest

from tests.gpu import GPU_SKIP_REASON, gpu_tests_enabled
from greenfuse.execution.prepare import prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.measurement.energy import NvmlEnergyMeter, measure_energy
from greenfuse.measurement.timing import calibrate_iterations, measure_timing
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


@unittest.skipUnless(gpu_tests_enabled(), GPU_SKIP_REASON)
class GpuMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.inputs = make_inputs(w1_spec(), device="cuda:0")
        graph = capture_graph(workload_w1)
        self.built = build_partition(graph, enumerate_partitions(graph.description)[0])

    def test_completed_timing_for_launch_and_graph(self):
        with tempfile.TemporaryDirectory() as root:
            for mode in ("launch", "graph"):
                prepared = prepare_execution(self.built, self.inputs, mode=mode, reference=workload_w1,
                                             trace_path=Path(root) / f"{mode}.json")
                self.assertIsNotNone(prepared.executor, prepared.record.reason)
                try:
                    if not prepared.record.ready_for_gpu_measurement:
                        self.skipTest(prepared.record.profiling["reason"])
                    trial = measure_timing(prepared.executor, self.inputs, 1000)
                    self.assertEqual(trial.status, "ok", trial.reason)
                    self.assertEqual(trial.completed_iterations, 1000)
                    self.assertGreater(trial.seconds_per_execution, 0)
                finally:
                    prepared.close()

    def test_real_nvml_counter_window(self):
        if os.environ.get("GREENFUSE_EXCLUSIVE_GPU") != "1":
            self.skipTest("Set GREENFUSE_EXCLUSIVE_GPU=1 only in a whole-physical-GPU exclusive allocation")
        import torch
        meter = NvmlEnergyMeter.open(torch.device("cuda:0"), exclusive_gpu=True)
        prepared = None
        try:
            reason = meter.qualification_reason(meter.snapshot())
            if reason:
                self.skipTest(reason)
            prepared = prepare_execution(self.built, self.inputs, reference=workload_w1)
            self.assertIsNotNone(prepared.executor, prepared.record.reason)
            pilot = calibrate_iterations(prepared.executor, self.inputs, 3.0)
            self.assertEqual(pilot["status"], "ready", pilot["reason"])
            trial = measure_energy(prepared.executor, self.inputs, pilot["iterations"], meter)
            self.assertEqual(trial.status, "ok", trial.reason)
            self.assertGreater(trial.elapsed_seconds, 3.0)
            self.assertGreater(trial.energy_joules_per_execution, 0)
        finally:
            if prepared:
                prepared.close()
            meter.close()
