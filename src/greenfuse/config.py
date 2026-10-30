"""Strict experiment configuration. Loading JSON needs only the standard library."""

from dataclasses import asdict, dataclass, fields, replace
from itertools import product
import json
import math
from pathlib import Path
import re

from .measurement.runner import MeasurementConfig
from .results import portable


@dataclass(frozen=True)
class ExperimentConfig:
    workload: str = "w1"
    shapes: tuple = ((32, 256),)
    device: str = "cuda:0"
    modes: tuple = ("launch", "graph")
    dtype: str = "float32"
    layout: str = "contiguous"
    seed: int = 0
    rtol: float = 1e-5
    atol: float = 1e-6
    rounds: int = 5
    window_seconds: float = 3.0
    warmup_iterations: int = 3
    max_iterations: int = 10_000_000
    max_partitions: int = 256
    latency_tolerance: float = 0.05
    exclusion_policy: str = "none"
    eager_baseline: bool = True
    compiler_artifacts: bool = False
    partitions: tuple | None = None  # Optional subset of partition IDs, e.g. ["p10", "p00"]; None means all

    def __post_init__(self):
        if self.workload != "w1" or self.dtype != "float32" or self.layout != "contiguous":
            raise ValueError("The initial supported experiment is W1 with contiguous float32 inputs")
        if not isinstance(self.shapes, tuple) or not self.shapes:
            raise ValueError("shapes must be a nonempty sequence of [M, N] pairs")
        for shape in self.shapes:
            if not isinstance(shape, tuple) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError("Every shape must contain positive integers M and N")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("Duplicate shapes are not allowed")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", self.device):
            raise ValueError("device must be cpu or an explicit cuda:N")
        if not isinstance(self.modes, tuple) or not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must be a nonempty sequence without duplicates")
        if any(mode not in ("launch", "graph") for mode in self.modes):
            raise ValueError("Supported modes are launch and graph")
        if self.device == "cpu" and self.modes != ("launch",):
            raise ValueError("CPU development supports launch mode only")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.max_partitions) is not int or self.max_partitions < 1:
            raise ValueError("max_partitions must be a positive integer")
        if type(self.latency_tolerance) not in (int, float) or not math.isfinite(self.latency_tolerance) or not 0 <= self.latency_tolerance <= 1:
            raise ValueError("latency_tolerance must be finite and between 0 and 1")
        if self.exclusion_policy not in ("none", "preparation_failures"):
            raise ValueError("Unknown exclusion policy")
        if any(type(v) is not bool for v in (self.eager_baseline, self.compiler_artifacts)):
            raise ValueError("Baseline and artifact switches must be booleans")
        if self.partitions is not None and (not isinstance(self.partitions, tuple) or not self.partitions
                                            or len(set(self.partitions)) != len(self.partitions)
                                            or any(not isinstance(v, str) for v in self.partitions)):
            raise ValueError("partitions must be a nonempty list of unique partition IDs")
        self.measurement()

    def select_partitions(self, partitions):
        if self.partitions is None:
            return tuple(partitions)
        known = {p.id: p for p in partitions}
        missing = [pid for pid in self.partitions if pid not in known]
        if missing:
            raise ValueError(f"Unknown partition IDs {missing}; available: {sorted(known)}")
        return tuple(known[pid] for pid in self.partitions)

    def measurement(self, *, validation=False):
        return MeasurementConfig(rounds=self.rounds, window_seconds=self.window_seconds,
                                 seed=self.seed + int(validation), warmup_iterations=self.warmup_iterations,
                                 max_iterations=self.max_iterations, allow_cpu=self.device == "cpu",
                                 rtol=self.rtol, atol=self.atol)

    def to_dict(self):
        return portable(asdict(self))

    def for_shape(self, shape):
        if tuple(shape) not in self.shapes:
            raise ValueError("Shape is outside the configured experiment")
        return replace(self, shapes=(tuple(shape),))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Configuration must be an object")
        data = dict(data)
        unknown = set(data) - {f.name for f in fields(cls)} - {"grid"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        if "grid" in data:
            if "shapes" in data:
                raise ValueError("Choose shapes or grid, not both")
            grid = data.pop("grid")
            if not isinstance(grid, dict) or set(grid) != {"m", "n"} or any(
                not isinstance(grid[key], list) or not grid[key] for key in ("m", "n")
            ):
                raise ValueError("grid must contain nonempty m and n lists")
            data["shapes"] = list(product(grid["m"], grid["n"]))
        if "shapes" in data:
            if not isinstance(data["shapes"], (list, tuple)) or any(not isinstance(v, (list, tuple)) for v in data["shapes"]):
                raise ValueError("shapes must be a list of pairs")
            data["shapes"] = tuple(tuple(v) for v in data["shapes"])
        if "modes" in data:
            if not isinstance(data["modes"], (list, tuple)) or any(not isinstance(v, str) for v in data["modes"]):
                raise ValueError("modes must be a list of strings")
            data["modes"] = tuple(data["modes"])
        if data.get("partitions") is not None:
            if not isinstance(data["partitions"], (list, tuple)):
                raise ValueError("partitions must be a list of partition IDs")
            data["partitions"] = tuple(data["partitions"])
        return cls(**data)


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def load_config(path):
    text = Path(path).read_text()
    if Path(path).suffix.lower() == ".json":
        data = json.loads(text, object_pairs_hook=_unique_mapping)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError('YAML configuration requires pip install -e ".[config]"; JSON works without PyYAML') from error
        class UniqueLoader(yaml.SafeLoader):
            pass
        def mapping(loader, node):
            return _unique_mapping((loader.construct_object(k), loader.construct_object(v)) for k, v in node.value)
        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        data = yaml.load(text, Loader=UniqueLoader)
    return ExperimentConfig.from_dict(data)
