import importlib.util
from pathlib import Path
import tempfile
import unittest

from greenfuse.config import ExperimentConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_grid_expansion_and_case_identity(self):
        config = ExperimentConfig.from_dict({"grid": {"m": [32, 1024], "n": [256, 1024, 4096]}})
        self.assertEqual(len(config.shapes), 6)
        self.assertEqual(config.for_shape((32, 256)).shapes, ((32, 256),))
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)

    def test_invalid_or_unsupported_config_is_not_silently_accepted(self):
        for data in ({"round": 5}, {"shapes": [[True, 256]]}, {"modes": ["launch", "launch"]},
                     {"workload": "w2"}, {"dtype": "float16"}, {"device": "cuda"}, {"latency_tolerance": float("nan")},
                     {"rounds": True}, {"seed": -1}, {"eager_baseline": 1}, {"grid": {"m": [], "n": [2]}},
                     {"shapes": [[1, 2]], "grid": {"m": [1], "n": [2]}}, {"shapes": [[1, 2], [1, 2]]}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                ExperimentConfig.from_dict(data)

    def test_cpu_development_explicit_and_short_windows_allowed_only_there(self):
        config = ExperimentConfig.from_dict({"device": "cpu", "modes": ["launch"], "window_seconds": 0.01})
        self.assertTrue(config.measurement().allow_cpu)
        with self.assertRaises(ValueError):
            ExperimentConfig(window_seconds=0.01)

    def test_json_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text('{"shapes": [[32, 256]]}')
            self.assertEqual(load_config(path), ExperimentConfig())
            path.write_text('{"rounds": 3, "rounds": 5}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "YAML needs optional PyYAML")
    def test_yaml_examples_and_duplicate_keys(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(len(load_config(root / "configs/midpoint.yaml").shapes), 6)
        self.assertEqual(load_config(root / "configs/smoke.yaml"), ExperimentConfig())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("rounds: 3\nrounds: 5\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)
