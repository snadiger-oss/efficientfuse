"""Resolve workload names without storing manually written partition variants."""

from .definitions import make_workload_w3, make_workload_w6, workload_w1, workload_w2, workload_w4, workload_w5
from .specs import w1_spec, w2_spec, w3_spec, w4_spec, w5_spec, w6_spec

WORKLOADS = {
    "w1": "relu(x + bias) + residual",
    "w2": "gelu(x @ weight + bias)",
    "w3": "gelu(layer_norm(x + residual))",
    "w4": "h = x + bias; h - h.mean(-1, keepdim=True)",
    "w5": "softmax(gelu(x + bias), -1)",
    "w6": "gelu(layer_norm(x + residual) @ weight + bias)",
}
_SPECS = {"w1": w1_spec, "w2": w2_spec, "w3": w3_spec, "w4": w4_spec, "w5": w5_spec, "w6": w6_spec}


def get_workload(name: str, *, n: int | None = None):
    """n (row width) is needed only by workloads whose graph depends on it (w3's LayerNorm)."""
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload {name!r}; supported workloads: {', '.join(WORKLOADS)}")
    if name in ("w3", "w6"):
        if n is None:
            raise ValueError(f"{name} needs the row width n to build its LayerNorm")
        return make_workload_w3(n) if name == "w3" else make_workload_w6(n)
    return {"w1": workload_w1, "w2": workload_w2, "w4": workload_w4, "w5": workload_w5}[name]


def get_workload_spec(name: str, *, m: int = 32, n: int = 256, seed: int = 0):
    get_workload(name, n=n)
    return _SPECS[name](m, n, seed=seed)
