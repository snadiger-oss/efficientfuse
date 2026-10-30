"""Opt-in full research run. This test may take several minutes on a real GPU."""

import os
from pathlib import Path
import tempfile
import unittest

from tests.gpu import gpu_tests_enabled
from greenfuse.config import ExperimentConfig
from greenfuse.measurement.environment import environment_report
from greenfuse.planner.calibrate import calibrate_context, create_context
from greenfuse.planner.dispatch import CachedDispatcher


@unittest.skipUnless(gpu_tests_enabled() and os.environ.get("GREENFUSE_RUN_PLANNER_TESTS") == "1",
                     "Full GPU planner test needs GREENFUSE_RUN_GPU_TESTS=1 and GREENFUSE_RUN_PLANNER_TESTS=1")
class EndToEndGpuTests(unittest.TestCase):
    def test_real_calibration_fresh_validation_and_cached_reuse(self):
        if os.environ.get("GREENFUSE_EXCLUSIVE_GPU") != "1":
            self.skipTest("Full planner test requires an exclusive whole physical GPU")
        environment = environment_report("cuda:0", exclusive_gpu=True)
        reason = environment.get("energy", {}).get("reason")
        if environment["status"] != "cuda_available" or reason:
            self.skipTest(reason or environment.get("reason") or "Energy capability unavailable")
        context = create_context(ExperimentConfig(rounds=3), (32, 256), exclusive_gpu=True)
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "decisions"
            result = calibrate_context(context, output_dir=Path(root) / "runs", cache_dir=cache, exclusive_gpu=True)
            self.assertNotEqual(result["status"], "invalid", result)
            if result["status"] != "validated":
                self.assertFalse(result["cache_written"])
                self.skipTest(f"Real experiment did not establish a stable choice: {result.get('reason')}")
            import torch
            with CachedDispatcher(cache, Path(root) / "dispatch") as dispatcher:
                first, warm = dispatcher.acquire(context), dispatcher.acquire(context)
                self.assertFalse(first.live_executor_reused)
                self.assertTrue(warm.live_executor_reused)
                torch.testing.assert_close(warm.run(context.inputs), context.workload(**context.inputs), rtol=1e-5, atol=1e-6)
