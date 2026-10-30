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
class GraphReplayGpuTests(unittest.TestCase):
    def test_all_partitions_replay_changed_inputs_and_borrowed_outputs(self):
        import torch

        captured = capture_graph(workload_w1)
        inputs = make_inputs(w1_spec(), device="cuda:0")
        with tempfile.TemporaryDirectory() as root:
            for spec in enumerate_partitions(captured.description):
                with self.subTest(partition=spec.id):
                    prepared = prepare_execution(
                        build_partition(captured, spec), inputs, mode="graph", reference=workload_w1,
                        trace_path=Path(root) / f"{spec.id}.json",
                    )
                    self.assertIn(prepared.record.status, ("ready", "correct_profile_unverified"), prepared.record.reason)
                    executor = prepared.executor
                    try:
                        addresses = executor.buffers.addresses
                        borrowed = executor.run(inputs)
                        retained = borrowed.clone()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(retained, workload_w1(**inputs), rtol=1e-5, atol=1e-6)
                        changed = {**inputs, "residual": inputs["residual"] + 3}
                        updated = executor.run(changed)
                        torch.cuda.synchronize()
                        self.assertEqual(borrowed.data_ptr(), updated.data_ptr())
                        self.assertEqual(executor.buffers.addresses, addresses)
                        torch.testing.assert_close(updated, workload_w1(**changed), rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(retained, workload_w1(**inputs), rtol=1e-5, atol=1e-6)
                        evidence = prepared.record.profiling
                        if evidence["replay_check"] == "passed":
                            self.assertGreater(evidence["graph_launch_api_events"], 0)
                            self.assertGreater(evidence["kernel_launches_per_invocation"], 0)
                        else:
                            self.assertFalse(prepared.record.ready_for_gpu_measurement)
                            self.assertTrue(evidence["reason"])
                    finally:
                        prepared.close()
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        executor.run(inputs)

    def test_graph_rejects_wrong_shape_before_replay(self):
        captured = capture_graph(workload_w1)
        built = build_partition(captured, enumerate_partitions(captured.description)[0])
        inputs = make_inputs(w1_spec(), device="cuda:0")
        prepared = prepare_execution(built, inputs, mode="graph", reference=workload_w1)
        self.assertIsNotNone(prepared.executor, prepared.record.reason)
        try:
            with self.assertRaises(ValueError):
                prepared.executor.run(make_inputs(w1_spec(2, 4), device="cuda:0"))
        finally:
            prepared.close()
