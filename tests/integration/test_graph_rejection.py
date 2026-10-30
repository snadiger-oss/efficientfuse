import importlib.util
import unittest

from greenfuse.graph.capture import capture_graph
from greenfuse.graph.validate import UnsupportedGraphError


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "CPU graph tests require optional PyTorch")
class GraphRejectionTests(unittest.TestCase):
    def test_reject_unsupported_operator_and_aliasing(self):
        for fn in [lambda x: x * 2, lambda x: x.view(-1), lambda x: x]:
            with self.assertRaises(UnsupportedGraphError):
                capture_graph(fn)

    def test_reject_branching_and_skipped_dependencies(self):
        import torch

        def two_branches(x, bias):
            a = torch.relu(x)
            b = torch.relu(bias)
            return a + b

        def reuse_old_intermediate(x):
            a = x + 1
            b = torch.relu(a)
            return a + b

        for fn in (two_branches, reuse_old_intermediate):
            with self.assertRaisesRegex(UnsupportedGraphError, "straight compute chain"):
                capture_graph(fn)

    def test_reject_inplace_methods_and_functions(self):
        import torch
        import torch.nn.functional as functional

        def out_variant(x, bias):
            return torch.add(x, bias, out=x)

        for fn in [lambda x: x.relu_(), lambda x: x.add_(1), lambda x: functional.relu(x, inplace=True), lambda x: functional.relu(x, True), out_variant]:
            with self.assertRaises(UnsupportedGraphError):
                capture_graph(fn)

    def test_reject_output_containers_and_constant_tensor_state(self):
        import torch

        class Stateful(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.ones(3))

            def forward(self, x):
                return x + self.bias

        for fn in [lambda x: (torch.relu(x),), lambda x: {"y": torch.relu(x)}, Stateful(), torch.nn.ReLU()]:
            # Root nn.ReLU traces to functional.relu and is supported; a leaf
            # module call inside Sequential is the unsupported case.
            if isinstance(fn, torch.nn.ReLU):
                fn = torch.nn.Sequential(fn)
            with self.assertRaises(UnsupportedGraphError):
                capture_graph(fn)

    def test_reject_data_dependent_control_flow(self):
        def branch(x):
            if x.sum() > 0:
                return x + 1
            return x + 2

        with self.assertRaisesRegex(UnsupportedGraphError, "FX capture failed"):
            capture_graph(branch)

    def test_reject_default_input_values(self):
        def default_input(x, bias=1):
            return x + bias

        with self.assertRaisesRegex(UnsupportedGraphError, "Default"):
            capture_graph(default_input)


if __name__ == "__main__":
    unittest.main()
