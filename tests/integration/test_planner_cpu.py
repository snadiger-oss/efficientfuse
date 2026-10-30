"""Real CPU reconstruction and CLI; no GPU-energy selection claims."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from greenfuse.cli import main
from greenfuse.config import ExperimentConfig
from greenfuse.planner.calibrate import create_context
from greenfuse.planner.dispatch import ExecutorRegistry
from tests.unit.planner_fixtures import synthetic_report


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires optional PyTorch")
class PlannerCpuTests(unittest.TestCase):
    def test_real_compilation_reused_in_process_with_new_values(self):
        import torch
        config = ExperimentConfig(shapes=((3, 4),), device="cpu", modes=("launch",), window_seconds=0.01)
        context = create_context(config, (3, 4))
        with tempfile.TemporaryDirectory() as root, ExecutorRegistry() as registry:
            calls = []
            def factory():
                calls.append(True)
                return context.prepare_candidate("p01_launch", root)
            first, reused, _ = registry.acquire(context.key, factory, context_token="cpu", require_gpu=False)
            self.assertFalse(reused)
            second, reused, _ = registry.acquire(context.key, factory, context_token="cpu", require_gpu=False)
            self.assertTrue(reused)
            self.assertIs(first.executor, second.executor)
            self.assertEqual(len(calls), 1)
            changed = {name: tensor + 1 for name, tensor in context.inputs.items()}
            torch.testing.assert_close(second.executor.run(changed), context.workload(**changed))

    def test_inspect_and_midpoint_grid(self):
        with io.StringIO() as output, redirect_stdout(output):
            code = main(["inspect", "--device", "cpu"])
            evidence = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(len(evidence[0]["candidates"]), 4)


class OfflineCliTests(unittest.TestCase):
    def test_select_runs_without_pytorch_import_and_reports_cpu_limitation(self):
        report = synthetic_report()
        report["scope"] = "cpu_development"
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cpu.json"
            path.write_text(json.dumps(report))
            result = subprocess.run([sys.executable, "-S", "-m", "greenfuse", "select", str(path)],
                                    env={**os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "runtime_only")

    def test_unknown_config_is_reported_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.json"
            path.write_text('{"unknown_setting": 1}')
            result = subprocess.run([sys.executable, "-S", "-m", "greenfuse", "inspect", "--config", str(path)],
                                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown configuration keys", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
