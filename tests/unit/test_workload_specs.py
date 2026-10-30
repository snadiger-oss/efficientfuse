import json
import unittest

from greenfuse.workloads.registry import get_workload, get_workload_spec
from greenfuse.workloads.specs import TensorSpec, WorkloadSpec, w1_spec


class WorkloadSpecTests(unittest.TestCase):
    def test_w1_defaults_and_identity(self):
        spec = w1_spec()
        self.assertEqual([(item.name, item.shape) for item in spec.inputs], [("x", (32, 256)), ("bias", (256,)), ("residual", (32, 256))])
        self.assertEqual(json.loads(json.dumps(spec.to_dict())), spec.to_dict())
        self.assertEqual(spec.fingerprint, w1_spec().fingerprint)
        self.assertNotEqual(spec.fingerprint, w1_spec(seed=1).fingerprint)
        self.assertNotEqual(spec.fingerprint, w1_spec(64, 256).fingerprint)

    def test_invalid_specs(self):
        for shape in [(0, 1), (-1,), (True,), ()]:
            with self.assertRaises(ValueError):
                TensorSpec("x", shape)
        with self.assertRaises(ValueError):
            TensorSpec("x", (2,), dtype="float16")
        with self.assertRaises(ValueError):
            WorkloadSpec("w1", (TensorSpec("x", (2,)), TensorSpec("x", (2,))))
        for seed in (-1, True, 2**63):
            with self.assertRaises(ValueError):
                w1_spec(seed=seed)
        for tolerance in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                WorkloadSpec("test", (TensorSpec("x", (2,)),), rtol=tolerance)

    def test_registry(self):
        self.assertTrue(callable(get_workload("w1")))
        self.assertEqual(get_workload_spec("w1", m=2, n=3), w1_spec(2, 3))
        with self.assertRaisesRegex(ValueError, "Unknown workload"):
            get_workload("w2")


if __name__ == "__main__":
    unittest.main()
