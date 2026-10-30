import importlib.util
import json
import unittest

from greenfuse.graph.capture import capture_graph
from greenfuse.graph.ir import ValueRef
from greenfuse.workloads.definitions import workload_w1
from greenfuse.workloads.inputs import make_inputs, validate_inputs
from greenfuse.workloads.specs import w1_spec

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "CPU graph tests require optional PyTorch")
class GraphCaptureTests(unittest.TestCase):
    def test_w1_capture_metadata_and_execution(self):
        import torch

        captured = capture_graph(workload_w1)
        graph = captured.description
        self.assertEqual([op.kind for op in graph.operations], ["add", "relu", "add"])
        self.assertEqual([item.name for item in graph.inputs], ["x", "bias", "residual"])
        self.assertEqual(graph.operations[-1].dependencies, ("op_1", "input_2"))
        self.assertEqual(graph.output, ValueRef("op_2"))
        self.assertEqual(json.loads(json.dumps(graph.to_dict())), graph.to_dict())
        inputs = make_inputs(w1_spec())
        torch.testing.assert_close(captured.module(**inputs), workload_w1(**inputs))
        self.assertEqual(graph.fingerprint, capture_graph(workload_w1).description.fingerprint)

    def test_scalar_and_keyword_constants_affect_identity(self):
        import torch

        def first(x, bias):
            return torch.add(x, bias, alpha=2.0).relu()

        def second(x, bias):
            return torch.add(x, bias, alpha=3.0).relu()

        a, b = capture_graph(first), capture_graph(second)
        self.assertEqual(dict(a.description.operations[0].kwargs)["alpha"], 2.0)
        self.assertNotEqual(a.description.fingerprint, b.description.fingerprint)

    def test_seeded_inputs_do_not_change_global_rng(self):
        import torch

        state = torch.random.get_rng_state().clone()
        a, b, c = make_inputs(w1_spec()), make_inputs(w1_spec()), make_inputs(w1_spec(seed=1))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
            self.assertFalse(torch.equal(a[key], c[key]))
            self.assertTrue(a[key].is_contiguous())

    def test_input_contract_validation(self):
        import torch

        spec = w1_spec(3, 4)
        original = make_inputs(spec)
        bad_inputs = [
            {"x": original["x"]},
            {**original, "x": torch.zeros(4, 3)},
            {**original, "x": original["x"].double()},
            {**original, "x": torch.zeros(4, 3).t()},
            {**original, "x": original["x"].clone().requires_grad_(True)},
            {**original, "x": torch.full((3, 4), float("nan"))},
        ]
        for inputs in bad_inputs:
            with self.subTest(keys=list(inputs)), self.assertRaises(ValueError):
                validate_inputs(spec, inputs)


if __name__ == "__main__":
    unittest.main()
