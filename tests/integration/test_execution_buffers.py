import importlib.util
import unittest

from greenfuse.execution.buffers import InputContract, StaticBuffers, UnsupportedExecutionError
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.specs import w1_spec


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Execution buffer tests require optional PyTorch")
class BufferTests(unittest.TestCase):
    def setUp(self):
        self.inputs = make_inputs(w1_spec(3, 4))
        self.names = ("x", "bias", "residual")
        self.contract = InputContract.from_inputs(self.inputs, self.names, allow_cpu=True)

    def test_cpu_needs_explicit_opt_in(self):
        with self.assertRaises(UnsupportedExecutionError):
            InputContract.from_inputs(self.inputs, self.names)

    def test_owned_buffers_keep_addresses_and_copy_new_values(self):
        import torch

        buffers = StaticBuffers(self.contract, self.inputs)
        addresses = buffers.addresses
        changed = {key: value + 5 for key, value in self.inputs.items()}
        buffers.update(changed)
        self.assertEqual(tuple(value.data_ptr() for value in buffers.values), addresses)
        for name, target in zip(self.names, buffers.values):
            self.assertNotEqual(target.data_ptr(), self.inputs[name].data_ptr())
            torch.testing.assert_close(target, changed[name])
            self.assertFalse(target.is_inference())
        self.assertEqual(buffers.nbytes, sum(value.numel() * value.element_size() for value in self.inputs.values()))
        buffers.close()
        self.assertEqual(buffers.values, ())

    def test_changed_contract_rejected_before_copy(self):
        import torch

        invalid = [
            {**self.inputs, "x": torch.zeros(4, 3)},
            {**self.inputs, "x": self.inputs["x"].double()},
            {**self.inputs, "x": torch.zeros(4, 3).t()},
            {**self.inputs, "x": self.inputs["x"].clone().requires_grad_(True)},
            {"x": self.inputs["x"]},
        ]
        for inputs in invalid:
            with self.assertRaises(ValueError):
                self.contract.validate(inputs)

    def test_autocast_and_inference_input_rejected(self):
        import torch

        with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, "Autocast"):
            self.contract.validate(self.inputs)
        with torch.inference_mode():
            inference_x = self.inputs["x"].clone()
        with self.assertRaises(ValueError):
            self.contract.validate({**self.inputs, "x": inference_x})


if __name__ == "__main__":
    unittest.main()


class OnChangeRefreshTests(unittest.TestCase):
    """CPU-only checks of the on_change input policy used by CUDA Graph replay."""

    def setUp(self):
        import torch
        from greenfuse.execution.buffers import InputContract, StaticBuffers

        self.torch = torch
        self.inputs = {"x": torch.ones(4), "y": torch.zeros(4)}
        contract = InputContract.from_inputs(self.inputs, ("x", "y"), allow_cpu=True)
        self.buffers = StaticBuffers(contract, self.inputs)

    def test_same_unmodified_tensors_are_not_recopied(self):
        self.assertTrue(self.buffers.update(self.inputs, on_change=True))
        self.assertFalse(self.buffers.update(self.inputs, on_change=True))
        self.assertTrue(self.buffers.update(self.inputs))  # default policy always copies

    def test_new_or_modified_tensors_are_copied(self):
        self.buffers.update(self.inputs, on_change=True)
        self.inputs["x"].add_(1)  # in-place change bumps the version counter
        self.assertTrue(self.buffers.update(self.inputs, on_change=True))
        self.assertEqual(self.buffers.values[0][0].item(), 2.0)
        changed = {"x": self.torch.full((4,), 5.0), "y": self.inputs["y"]}
        self.assertTrue(self.buffers.update(changed, on_change=True))
        self.assertEqual(self.buffers.values[0][0].item(), 5.0)
