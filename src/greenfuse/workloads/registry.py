"""Resolve workload names without storing manually written partition variants."""

from .definitions import workload_w1
from .specs import w1_spec


def get_workload(name: str):
    if name != "w1":
        raise ValueError(f"Unknown workload {name!r}; supported workloads: w1")
    return workload_w1


def get_workload_spec(name: str, *, m: int = 32, n: int = 256, seed: int = 0):
    get_workload(name)
    return w1_spec(m, n, seed=seed)
