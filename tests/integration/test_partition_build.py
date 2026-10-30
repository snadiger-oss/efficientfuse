import importlib.util
import json
import unittest

from greenfuse.graph.capture import capture_graph
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "CPU graph tests require optional PyTorch")
class PartitionBuildTests(unittest.TestCase):
    def test_residual_binding_and_changed_input_values(self):
        import torch

        captured = capture_graph(workload_w1)
        plans = {plan.id: plan for plan in enumerate_partitions(captured.description)}
        built = build_partition(captured, plans["p01"])
        self.assertEqual(built.groups[0].bindings.input_ids, ("input_0", "input_1"))
        self.assertEqual(built.groups[1].bindings.input_ids, ("op_1", "input_2"))
        inputs = make_inputs(w1_spec(3, 4))
        first = built.run(inputs)
        changed = {**inputs, "residual": inputs["residual"] + 7}
        second = built.run(changed)
        torch.testing.assert_close(second, first + 7)
        torch.testing.assert_close(second, workload_w1(**changed))
        self.assertEqual(json.loads(json.dumps(built.descriptor())), built.descriptor())

    def test_external_values_reused_in_later_groups(self):
        import torch

        def reuse(x, bias):
            return torch.relu(x + bias) + bias

        inputs = {"x": torch.randn(2, 4), "bias": torch.randn(4)}
        captured = capture_graph(reuse)
        for plan in enumerate_partitions(captured.description):
            built = build_partition(captured, plan)
            torch.testing.assert_close(built.run(inputs), reuse(**inputs))

    def test_builder_preserves_positional_and_keyword_arguments(self):
        import torch
        import torch.nn.functional as functional

        def kwargs(x, bias):
            a = torch.add(input=x, other=bias, alpha=2.5)
            return functional.relu(a, inplace=False) + 0.75

        inputs = {"x": torch.randn(2, 4), "bias": torch.randn(4)}
        captured = capture_graph(kwargs)
        for plan in enumerate_partitions(captured.description):
            torch.testing.assert_close(build_partition(captured, plan).run(inputs), kwargs(**inputs))

    def test_reject_foreign_partition_and_stale_capture(self):
        captured = capture_graph(workload_w1)
        other = capture_graph(lambda x: x + 1)
        with self.assertRaisesRegex(ValueError, "different graph"):
            build_partition(captured, enumerate_partitions(other.description)[0])
        plan = enumerate_partitions(captured.description)[0]
        node = next(node for node in captured.module.graph.nodes if node.name == captured.node_names["op_0"])
        node.args = (node.args[0], 2)
        with self.assertRaisesRegex(ValueError, "changed"):
            build_partition(captured, plan)

    def test_run_rejects_missing_or_extra_input(self):
        captured = capture_graph(workload_w1)
        built = build_partition(captured, enumerate_partitions(captured.description)[0])
        inputs = make_inputs(w1_spec())
        with self.assertRaisesRegex(ValueError, "Expected inputs"):
            built.run({"x": inputs["x"]})
        with self.assertRaisesRegex(ValueError, "Expected inputs"):
            built.run({**inputs, "extra": inputs["x"]})

    def test_building_leaves_original_graph_intact(self):
        captured = capture_graph(workload_w1)
        before = str(captured.module.graph)
        for plan in enumerate_partitions(captured.description):
            build_partition(captured, plan)
        self.assertEqual(str(captured.module.graph), before)


if __name__ == "__main__":
    unittest.main()
