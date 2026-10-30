"""Real CPU Inductor checks; these do not establish CUDA/GPU correctness."""

import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

from greenfuse.compiler.inductor import CompilationError, CompilationRecord, compile_partition
from greenfuse.execution.prepare import prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "CPU compilation tests require optional PyTorch and a C++ toolchain")
class CompiledCpuTests(unittest.TestCase):
    def setUp(self):
        self.captured = capture_graph(workload_w1)
        self.plans = enumerate_partitions(self.captured.description)
        self.inputs = make_inputs(w1_spec(3, 4))

    def test_all_partitions_compile_and_run_with_changed_inputs(self):
        import torch

        for plan in self.plans:
            with self.subTest(partition=plan.id):
                prepared = prepare_execution(build_partition(self.captured, plan), self.inputs, reference=workload_w1, allow_cpu=True)
                self.assertEqual(prepared.record.status, "cpu_validated", prepared.record.reason)
                self.assertFalse(prepared.record.ready_for_gpu_measurement)
                self.assertIsNone(prepared.record.profiling["kernel_launches_per_invocation"])
                self.assertEqual(len(prepared.record.compilation["groups"]), len(plan.groups))
                self.assertFalse(prepared.record.compilation["options"]["triton.cudagraphs"])
                changed = {name: value + 2 for name, value in self.inputs.items()}
                try:
                    torch.testing.assert_close(prepared.executor.run(changed), workload_w1(**changed))
                    json.dumps(prepared.record.to_dict(), allow_nan=False)
                finally:
                    prepared.close()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    prepared.executor.run(self.inputs)
                prepared.close()

    def test_cuda_graph_on_cpu_is_unsupported_without_fallback(self):
        prepared = prepare_execution(build_partition(self.captured, self.plans[0]), self.inputs, mode="graph", allow_cpu=True)
        self.assertEqual(prepared.record.status, "unsupported")
        self.assertEqual(prepared.record.mode, "graph")
        self.assertIsNone(prepared.executor)
        self.assertIsNone(prepared.record.compilation)

    def test_compile_failure_stays_visible(self):
        failure = CompilationError("injected test failure", CompilationRecord())
        with patch("greenfuse.execution.prepare.compile_partition", side_effect=failure):
            prepared = prepare_execution(build_partition(self.captured, self.plans[0]), self.inputs, allow_cpu=True)
        self.assertEqual(prepared.record.status, "failed")
        self.assertEqual(prepared.record.failure_stage, "compilation")
        self.assertIn("injected test failure", prepared.record.reason)
        self.assertIsNone(prepared.executor)

    def test_incorrect_output_is_not_ready(self):
        def deliberately_wrong_reference(x, bias, residual):
            return workload_w1(x, bias, residual) + 100

        prepared = prepare_execution(build_partition(self.captured, self.plans[0]), self.inputs,
                                     allow_cpu=True, reference=deliberately_wrong_reference)
        self.assertEqual(prepared.record.failure_stage, "correctness")
        self.assertEqual(prepared.record.correctness["status"], "failed")
        self.assertIsNone(prepared.executor)

    def test_failed_group_preserves_partial_compile_status(self):
        built = build_partition(self.captured, self.plans[-1])
        # Deliberately injected compiler failure: this tests status handling only.
        with patch("torch.compile", side_effect=[built.groups[0].module, RuntimeError("injected second-group failure")]):
            with self.assertRaises(CompilationError) as failure:
                compile_partition(built, self.inputs, allow_cpu=True)
        self.assertEqual([item["status"] for item in failure.exception.record.groups], ["compiled", "failed", "not_attempted"])

    def test_capture_failure_is_not_relabelled_as_ordinary_execution(self):
        import torch

        # Synthetic CUDA contract only for exercising this error path. No GPU
        # correctness or capture claim is made by this test.
        fake_contract = Mock(device=torch.device("cuda:0"))
        fake_compiled = Mock()
        fake_compiled.record = CompilationRecord()
        fake_compiled.contract.device = torch.device("cpu")
        with patch("greenfuse.execution.prepare.InputContract.from_inputs", return_value=fake_contract), \
             patch("greenfuse.execution.prepare.compile_partition", return_value=fake_compiled), \
             patch("greenfuse.execution.prepare.GraphExecutor.prepare", side_effect=RuntimeError("injected capture failure")), \
             patch("greenfuse.execution.prepare.LaunchExecutor") as ordinary:
            prepared = prepare_execution(build_partition(self.captured, self.plans[0]), self.inputs, mode="graph")
        self.assertEqual(prepared.record.mode, "graph")
        self.assertEqual(prepared.record.failure_stage, "capture")
        self.assertEqual(prepared.record.status, "failed")
        self.assertIsNone(prepared.executor)
        ordinary.assert_not_called()
        fake_compiled.close.assert_called_once()

    def test_invalid_tolerances_rejected_before_compilation(self):
        for rtol in (float("nan"), float("inf"), -1, True):
            prepared = prepare_execution(build_partition(self.captured, self.plans[0]), self.inputs,
                                         allow_cpu=True, rtol=rtol)
            self.assertEqual(prepared.record.failure_stage, "support")
            self.assertIsNone(prepared.record.compilation)


if __name__ == "__main__":
    unittest.main()
