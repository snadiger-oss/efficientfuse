import importlib.util
import unittest

from greenfuse.graph.capture import capture_graph
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "CPU graph tests require optional PyTorch")
class CpuEquivalenceTests(unittest.TestCase):
    def test_all_w1_partitions_shapes_and_seeds(self):
        import torch

        captured = capture_graph(workload_w1)
        built = [build_partition(captured, plan) for plan in enumerate_partitions(captured.description)]
        for m, n in [(1, 1), (3, 4), (32, 256)]:
            for seed in (0, 7, 42):
                spec = w1_spec(m, n, seed=seed)
                inputs = make_inputs(spec)
                originals = {key: value.clone() for key, value in inputs.items()}
                expected = workload_w1(**inputs)
                for partition in built:
                    with self.subTest(m=m, n=n, seed=seed, partition=partition.spec.id):
                        actual = partition.run(inputs)
                        torch.testing.assert_close(actual, expected, rtol=spec.rtol, atol=spec.atol)
                        for key, original in originals.items():
                            torch.testing.assert_close(inputs[key], original, rtol=0, atol=0)

    def test_relu_boundaries_and_bias_broadcast(self):
        import torch

        inputs = {
            "x": torch.tensor([[-3., 0., 3.], [3., 0., -3.]]),
            "bias": torch.tensor([1., 0., -1.]),
            "residual": torch.tensor([[0.5, 1., 2.], [3., 4., 5.]]),
        }
        expected = torch.tensor([[0.5, 1., 4.], [7., 4., 5.]])
        captured = capture_graph(workload_w1)
        for plan in enumerate_partitions(captured.description):
            torch.testing.assert_close(build_partition(captured, plan).run(inputs), expected, rtol=0, atol=0)

    def test_generic_builder_with_other_chain_lengths(self):
        import torch

        # These are builder tests, not additional completed research workloads.
        for fn, expected_count in [
            (lambda x: torch.relu(x), 1),
            (lambda x: torch.relu(x + 2), 2),
            (lambda x: torch.relu(torch.relu(x + 2) + 3), 8),
        ]:
            captured = capture_graph(fn)
            plans = enumerate_partitions(captured.description)
            self.assertEqual(len(plans), expected_count)
            inputs = {"x": torch.tensor([-5., -2., 0., 3.])}
            for plan in plans:
                torch.testing.assert_close(build_partition(captured, plan).run(inputs), fn(**inputs))


if __name__ == "__main__":
    unittest.main()
