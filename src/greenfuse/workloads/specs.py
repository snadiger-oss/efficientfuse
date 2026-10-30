"""Input metadata; no tensors, device handles, or PyTorch imports live here."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    layout: str = "contiguous"

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError("Tensor name must be a Python identifier")
        if not isinstance(self.shape, tuple) or not self.shape or any(
            type(size) is not int or size <= 0 for size in self.shape
        ):
            raise ValueError("Shape must be a nonempty tuple of positive integers")
        if self.dtype != "float32" or self.layout != "contiguous":
            raise ValueError("The initial workload contract requires contiguous float32")


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    inputs: tuple[TensorSpec, ...]
    seed: int = 0
    rtol: float = 1e-5
    atol: float = 1e-6

    def __post_init__(self):
        if not self.name or not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("A workload needs a name and a nonempty tuple of inputs")
        names = [item.name for item in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("Input names must be unique")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Seed must be an integer in [0, 2**63)")
        if any(not math.isfinite(value) or value < 0 for value in (self.rtol, self.atol)):
            raise ValueError("Comparison tolerances must be finite and nonnegative")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def w1_spec(m: int = 32, n: int = 256, *, seed: int = 0) -> WorkloadSpec:
    return WorkloadSpec(
        name="w1",
        inputs=(TensorSpec("x", (m, n)), TensorSpec("bias", (n,)), TensorSpec("residual", (m, n))),
        seed=seed,
    )
