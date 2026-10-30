import tempfile
from pathlib import Path
import unittest

from tests.gpu import GPU_SKIP_REASON, gpu_tests_enabled
from greenfuse.execution.prepare import prepare_execution
from greenfuse.graph.capture import capture_graph
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


@unittest.skipUnless(gpu_tests_enabled(), GPU_SKIP_REASON)
class CompiledGpuTests(unittest.TestCase):
    def test_all_ordinary_candidates_and_kernel_evidence(self):
        import torch

        captured = capture_graph(workload_w1)
        inputs = make_inputs(w1_spec(), device="cuda:0")
        with tempfile.TemporaryDirectory() as root:
            for spec in enumerate_partitions(captured.description):
                with self.subTest(partition=spec.id):
                    prepared = prepare_execution(
                        build_partition(captured, spec), inputs, reference=workload_w1,
                        trace_path=Path(root) / f"{spec.id}.json",
                    )
                    self.assertIn(prepared.record.status, ("ready", "correct_profile_unverified"), prepared.record.reason)
                    try:
                        changed = {name: value + 1 for name, value in inputs.items()}
                        torch.testing.assert_close(prepared.executor.run(changed), workload_w1(**changed), rtol=1e-5, atol=1e-6)
                        evidence = prepared.record.profiling
                        self.assertNotEqual(evidence["replay_check"], "failed", evidence)
                        if evidence["status"] == "observed":
                            self.assertGreater(evidence["kernel_launches_per_invocation"], 0)
                        else:
                            self.assertIsNone(evidence["kernel_launches_per_invocation"])
                            self.assertTrue(evidence["reason"])
                    finally:
                        prepared.close()

    def test_changed_signature_and_closed_executor(self):
        import torch

        captured = capture_graph(workload_w1)
        built = build_partition(captured, enumerate_partitions(captured.description)[0])
        inputs = make_inputs(w1_spec(), device="cuda:0")
        prepared = prepare_execution(built, inputs, reference=workload_w1)
        self.assertIsNotNone(prepared.executor, prepared.record.reason)
        try:
            with self.assertRaises(ValueError):
                prepared.executor.run({**inputs, "x": inputs["x"].double()})
            with torch.cuda.stream(torch.cuda.Stream()), self.assertRaisesRegex(ValueError, "stream"):
                prepared.executor.run(inputs)
        finally:
            prepared.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            prepared.executor.run(inputs)
