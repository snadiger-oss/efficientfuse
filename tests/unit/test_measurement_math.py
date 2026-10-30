"""Injected clocks/counters test software, not real GPU energy accuracy."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from greenfuse.measurement.energy import NvmlEnergyMeter, energy_per_execution, measure_energy, physical_uuid
from greenfuse.measurement.runner import MeasurementConfig, trial_schedule
from greenfuse.measurement.timing import calibrate_iterations, measure_timing
from greenfuse.results import Trial


class FakeMeter:
    available = True
    reason = None
    identity = {"source": "SYNTHETIC TEST COUNTER"}
    exclusive_gpu = True

    def __init__(self, readings=(1000, 7000), qualification=None):
        self.readings = iter(readings)
        self.qualification = qualification

    def snapshot(self):
        return {"test_only": True}

    def qualification_reason(self, snapshot):
        return self.qualification

    def read_mj(self):
        value = next(self.readings)
        if isinstance(value, Exception):
            raise value
        return value


class MeasurementMathTests(unittest.TestCase):
    def test_wall_clock_brackets_completed_work(self):
        events = []
        executor = Mock()
        executor.run.side_effect = lambda inputs: events.append("run")
        values = iter((10.0, 13.0))
        def clock():
            events.append("clock")
            return next(values)
        trial = measure_timing(executor, {}, 3, sync=lambda: events.append("sync"), clock=clock)
        self.assertEqual(events, ["sync", "clock", "run", "run", "run", "sync", "clock"])
        self.assertEqual(trial.completed_iterations, 3)
        self.assertEqual(trial.seconds_per_execution, 1)

    def test_failed_execution_never_divides_by_requested_count(self):
        executor = Mock()
        executor.run.side_effect = [None, RuntimeError("injected failure")]
        trial = measure_timing(executor, {}, 3, sync=lambda: None, clock=lambda: 1)
        self.assertEqual(trial.submitted_iterations, 1)
        self.assertEqual(trial.completed_iterations, 0)
        self.assertEqual(trial.status, "failed")
        self.assertIsNone(trial.seconds_per_execution)

    def test_failed_final_sync_is_not_completed_gpu_work(self):
        sync = Mock(side_effect=[None, RuntimeError("injected GPU failure")])
        trial = measure_timing(Mock(), {}, 3, sync=sync, clock=lambda: 1)
        self.assertEqual(trial.submitted_iterations, 3)
        self.assertEqual(trial.completed_iterations, 0)
        self.assertIsNone(trial.seconds_per_execution)

    def test_counter_units_and_invalid_counters(self):
        self.assertEqual(energy_per_execution(1000, 7000, 3), 2.0)
        for first, last, n in ((7, 7, 1), (7, 6, 1), (-1, 8, 1), (1, 8, 0), (True, 8, 1), (1, 1.5, 1)):
            with self.subTest(values=(first, last, n)), self.assertRaises(ValueError):
                energy_per_execution(first, last, n)

    def test_energy_has_separate_window_and_raw_readings(self):
        readings = iter((10.0, 10.1, 13.0, 13.1))
        trial = measure_energy(Mock(), {}, 3, FakeMeter(), sync=lambda: None, clock=lambda: next(readings))
        self.assertEqual(trial.status, "ok")
        self.assertEqual(trial.energy_joules_per_execution, 2.0)
        self.assertIsNone(trial.seconds_per_execution)
        self.assertEqual(trial.counter_read_intervals, [[10.0, 10.1], [13.0, 13.1]])
        self.assertEqual((trial.energy_start_mj, trial.energy_end_mj), (1000, 7000))

    def test_missing_or_unqualified_energy_is_never_zero(self):
        executor = Mock()
        trial = measure_energy(executor, {}, 3, FakeMeter(qualification="not exclusive"), sync=lambda: None)
        self.assertEqual(trial.status, "unavailable")
        self.assertIsNone(trial.energy_joules_per_execution)
        executor.run.assert_not_called()

    def test_counter_failure_preserves_completed_work_but_no_energy(self):
        trial = measure_energy(Mock(), {}, 3, FakeMeter((1000, RuntimeError("counter failed"))), sync=lambda: None)
        self.assertEqual(trial.completed_iterations, 3)
        self.assertEqual(trial.energy_start_mj, 1000)
        self.assertIsNone(trial.energy_joules_per_execution)
        self.assertEqual(trial.status, "failed")

    def test_reset_and_zero_delta_invalid(self):
        for readings in ((1000, 999), (1000, 1000)):
            trial = measure_energy(Mock(), {}, 3, FakeMeter(readings), sync=lambda: None)
            self.assertEqual(trial.status, "invalid")
            self.assertIsNone(trial.energy_joules_per_execution)

    def test_detected_contention_after_window_invalidates_energy(self):
        meter = FakeMeter()
        meter.qualification_reason = Mock(side_effect=[None, "other GPU process appeared"])
        trial = measure_energy(Mock(), {}, 3, meter, sync=lambda: None)
        self.assertEqual(trial.status, "invalid")
        self.assertEqual(trial.energy_end_mj, 7000)
        self.assertIsNone(trial.energy_joules_per_execution)

    def test_calibration_and_cap_failure(self):
        def synthetic_measure(executor, inputs, n):
            return Trial("timing", n, status="ok", submitted_iterations=n, completed_iterations=n,
                         elapsed_seconds=n * 0.01, seconds_per_execution=0.01)
        result = calibrate_iterations(None, {}, 3, measure=synthetic_measure)
        self.assertEqual(result["status"], "ready")
        self.assertGreaterEqual(result["pilots"][-1]["elapsed_seconds"], 3.75)
        failed = calibrate_iterations(None, {}, 3, max_iterations=20, measure=synthetic_measure)
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["iterations"])

    def test_schedule_is_repeatable_complete_and_shuffled(self):
        ids = [f"p{i}" for i in range(8)]
        schedule = list(trial_schedule(ids, 5, 42))
        self.assertEqual(schedule, list(trial_schedule(ids, 5, 42)))
        self.assertEqual(len(schedule), 80)
        for kind in ("timing", "energy"):
            for round_index in range(5):
                self.assertEqual({s[3] for s in schedule if s[:2] == (kind, round_index)}, set(ids))
        self.assertNotEqual([s[3] for s in schedule[:8]], ids)

    def test_config_rejects_short_gpu_windows_and_bad_numbers(self):
        for options in ({"window_seconds": 0.1}, {"rounds": True}, {"window_seconds": float("nan")}, {"rtol": -1}):
            with self.assertRaises(ValueError):
                MeasurementConfig(**options)
        self.assertEqual(MeasurementConfig(window_seconds=0.01, allow_cpu=True).window_seconds, 0.01)


class FakeNvml:
    """TEST ONLY: UUID mapping and unsupported-capability behavior."""
    class NVMLError_NotSupported(Exception):
        pass
    gpu_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    nvmlInit = Mock()
    nvmlShutdown = Mock()
    nvmlDeviceGetHandleByUUID = Mock(return_value="physical GPU 7")
    nvmlDeviceGetUUID = Mock(return_value=gpu_uuid)
    nvmlDeviceGetName = Mock(return_value="SYNTHETIC GPU")
    nvmlSystemGetDriverVersion = Mock(return_value="test driver")
    nvmlDeviceGetPciInfo = Mock(return_value=SimpleNamespace(busId="0000:07:00.0"))
    nvmlDeviceGetMigMode = Mock(return_value=(0, 0))
    nvmlDeviceGetVirtualizationMode = Mock(return_value=0)
    nvmlDeviceGetTotalEnergyConsumption = Mock(return_value=1000)


class NvmlMappingTests(unittest.TestCase):
    def test_exact_uuid_mapping_ignores_logical_device_index(self):
        meter = NvmlEnergyMeter(exclusive_gpu=True)
        nvml = FakeNvml()
        meter.connect(nvml, nvml.gpu_uuid)
        self.assertTrue(meter.available)
        nvml.nvmlDeviceGetHandleByUUID.assert_called_with(nvml.gpu_uuid)
        self.assertEqual(meter.identity["pci_bus_id"], "0000:07:00.0")
        meter.close()
        self.assertFalse(meter.initialized)

    def test_uuid_mismatch_and_mig_rejected(self):
        nvml = FakeNvml()
        nvml.nvmlDeviceGetUUID = Mock(return_value="GPU-ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee")
        meter = NvmlEnergyMeter()
        with self.assertRaisesRegex(ValueError, "differ"):
            meter.connect(nvml, nvml.gpu_uuid)
        meter.close()
        with self.assertRaisesRegex(ValueError, "MIG"):
            physical_uuid("MIG-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    def test_mig_mode_and_process_qualification(self):
        nvml = FakeNvml()
        nvml.nvmlDeviceGetMigMode = Mock(return_value=(1, 1))
        meter = NvmlEnergyMeter(exclusive_gpu=True)
        with self.assertRaisesRegex(ValueError, "MIG"):
            meter.connect(nvml, nvml.gpu_uuid)
        meter.close()
        meter.available = True
        self.assertIsNone(meter.qualification_reason({"compute_pids": [os.getpid()], "graphics_pids": []}))
        self.assertIn("Concurrent", meter.qualification_reason({"compute_pids": [os.getpid() + 99999], "graphics_pids": []}))
        self.assertIn("Cannot check", meter.qualification_reason({"compute_pids": None, "graphics_pids": []}))


if __name__ == "__main__":
    unittest.main()
