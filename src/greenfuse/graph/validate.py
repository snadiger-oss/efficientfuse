"""Validate the initially supported straight compute chain using metadata only."""

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
        expected = set() if index == 0 else {graph.operations[index - 1].id}
        if computed_dependencies != expected:
            raise UnsupportedGraphError(f"{op.id}: only a straight compute chain is supported; branching/merges are deferred")
        kw = dict(op.kwargs)
        if len(kw) != len(op.kwargs):
            raise UnsupportedGraphError(f"{op.id}: duplicate keyword argument")
        if "out" in kw:
            raise UnsupportedGraphError(f"{op.id}: out= mutation is not supported")
        if op.target == "functional.relu":
            inplace = op.args[1] if len(op.args) > 1 else kw.get("inplace", False)
            if inplace is not False:
                raise UnsupportedGraphError(f"{op.id}: in-place ReLU is not supported")
        for value in (*op.args, *kw.values()):
            if isinstance(value, ValueRef):
                continue
            if value is not None and type(value) not in (int, float, bool):
                raise UnsupportedGraphError(f"{op.id}: unsupported argument structure")
            if type(value) is float and not math.isfinite(value):
                raise UnsupportedGraphError(f"{op.id}: nonfinite constants are not supported")
        known.add(op.id)
    if graph.output != ValueRef(graph.operations[-1].id):
        raise UnsupportedGraphError("Expected one tensor output from the last operation")
