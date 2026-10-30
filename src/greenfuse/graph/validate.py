"""Validate the supported compute chain (with optional reuse of earlier results) using metadata only."""

import math

from .ir import NormalizedGraph, ValueRef


class UnsupportedGraphError(ValueError):
    """A graph is outside the explicit supported subset."""


SUPPORTED_TARGETS = {
    "operator.add": "add",
    "torch.add": "add",
    "Tensor.add": "add",
    "torch.relu": "relu",
    "functional.relu": "relu",
    "Tensor.relu": "relu",
    "operator.sub": "sub",
    "torch.sub": "sub",
    "Tensor.sub": "sub",
    "operator.matmul": "matmul",
    "torch.matmul": "matmul",
    "Tensor.matmul": "matmul",
    "functional.gelu": "gelu",
    "functional.layer_norm": "layer_norm",
    "torch.mean": "mean",
    "Tensor.mean": "mean",
    "torch.softmax": "softmax",
    "functional.softmax": "softmax",
    "Tensor.softmax": "softmax",
}


def validate_graph(graph: NormalizedGraph) -> None:
    if graph.schema_version != 1:
        raise UnsupportedGraphError("Unsupported graph schema version")
    if not graph.inputs or not graph.operations:
        raise UnsupportedGraphError("Expected inputs and a nonempty compute chain")
    ids = [item.id for item in graph.inputs] + [op.id for op in graph.operations]
    names = [item.name for item in graph.inputs]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise UnsupportedGraphError("Graph IDs and input names must be unique")
    known = {item.id for item in graph.inputs}
    compute_ids = {op.id for op in graph.operations}
    for index, op in enumerate(graph.operations):
        if SUPPORTED_TARGETS.get(op.target) != op.kind:
            raise UnsupportedGraphError(f"{op.id}: unsupported operation {op.target}")
        if not set(op.dependencies) <= known:
            raise UnsupportedGraphError(f"{op.id}: unknown or forward dependency")
        if not op.dependencies:
            raise UnsupportedGraphError(f"{op.id}: expected an operation on an input tensor")
        computed_dependencies = set(op.dependencies) & compute_ids
        # A chain backbone is required: each operation consumes the previous one. It may also reuse
        # earlier results (fan-out, e.g. h - mean(h)). Independent branches are still deferred.
        if index == 0:
            if computed_dependencies:
                raise UnsupportedGraphError(f"{op.id}: unknown or forward dependency")
        elif graph.operations[index - 1].id not in computed_dependencies:
            raise UnsupportedGraphError(f"{op.id}: only a straight compute chain is supported; independent branches are deferred")
        kw = dict(op.kwargs)
        if len(kw) != len(op.kwargs):
            raise UnsupportedGraphError(f"{op.id}: duplicate keyword argument")
        if "out" in kw:
            raise UnsupportedGraphError(f"{op.id}: out= mutation is not supported")
        if op.kind in ("gelu", "layer_norm") and any(isinstance(v, str) for v in (*op.args, *kw.values())):
            raise UnsupportedGraphError(f"{op.id}: string options are not supported")
        if op.target == "functional.relu":
            inplace = op.args[1] if len(op.args) > 1 else kw.get("inplace", False)
            if inplace is not False:
                raise UnsupportedGraphError(f"{op.id}: in-place ReLU is not supported")
        for value in (*op.args, *kw.values()):
            if isinstance(value, ValueRef):
                continue
            if type(value) is tuple and value and all(type(item) is int for item in value):
                continue
            if value is not None and type(value) not in (int, float, bool):
                raise UnsupportedGraphError(f"{op.id}: unsupported argument structure")
            if type(value) is float and not math.isfinite(value):
                raise UnsupportedGraphError(f"{op.id}: nonfinite constants are not supported")
        known.add(op.id)
    if graph.output != ValueRef(graph.operations[-1].id):
        raise UnsupportedGraphError("Expected one tensor output from the last operation")
