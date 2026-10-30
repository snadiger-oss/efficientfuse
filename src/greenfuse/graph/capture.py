"""FX adapter: keep live executable objects separate from portable metadata."""

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING

from .ir import InputBinding, NormalizedGraph, Operation, ValueRef
from .validate import UnsupportedGraphError, validate_graph

if TYPE_CHECKING:
    from torch.fx import GraphModule


@dataclass
class CapturedGraph:
    module: "GraphModule"
    description: NormalizedGraph
    node_names: dict[str, str]


def describe_graph(module) -> tuple[NormalizedGraph, dict[str, str]]:
    """Normalize a live FX module and reject operations outside our small subset."""
    import torch
    import torch.nn.functional as functional
    from torch.fx import Node

    targets = {
        operator.add: ("add", "operator.add"),
        torch.add: ("add", "torch.add"),
        torch.relu: ("relu", "torch.relu"),
        functional.relu: ("relu", "functional.relu"),
    }
    refs = {}
    names = {}
    inputs = []
    operations = []
    output = None

    def argument(value):
        if isinstance(value, Node):
            if value not in refs:
                raise UnsupportedGraphError("Unknown or forward graph reference")
            return ValueRef(refs[value])
        if value is None or type(value) in (int, float, bool):
            if type(value) is float and not math.isfinite(value):
                raise UnsupportedGraphError("Nonfinite constants are not supported")
            return value
        raise UnsupportedGraphError(f"Unsupported argument type: {type(value).__name__}")

    for node in module.graph.nodes:
        if node.op == "placeholder":
            if node.args or not str(node.target).isidentifier():
                raise UnsupportedGraphError("Default, variadic, and specialized input arguments are not supported")
            node_id = f"input_{len(inputs)}"
            inputs.append(InputBinding(node_id, str(node.target)))
        elif node.op in ("call_function", "call_method"):
            if node.op == "call_function":
                definition = targets.get(node.target)
            else:
                definition = {"add": ("add", "Tensor.add"), "relu": ("relu", "Tensor.relu")}.get(node.target)
            if definition is None:
                raise UnsupportedGraphError(f"Unsupported operation {node.op}: {node.target}; only addition and ReLU are supported")
            node_id = f"op_{len(operations)}"
            kind, target = definition
            operations.append(Operation(
                node_id, kind, target,
                tuple(argument(value) for value in node.args),
                tuple((key, argument(value)) for key, value in sorted(node.kwargs.items())),
            ))
        elif node.op == "output":
            if output is not None or len(node.args) != 1 or not isinstance(node.args[0], Node):
                raise UnsupportedGraphError("Expected one tensor output, not a tuple/list/dict")
            output = argument(node.args[0])
            continue
        else:
            raise UnsupportedGraphError(f"Unsupported node {node.op}: {node.target}; module calls and captured parameters/tensor constants are deferred")
        refs[node] = node_id
        names[node_id] = node.name
    if output is None:
        raise UnsupportedGraphError("Graph has no output")
    description = NormalizedGraph(tuple(inputs), tuple(operations), output)
    validate_graph(description)
    return description, names


def capture_graph(workload) -> CapturedGraph:
    """Trace a pure callable. Tensor-dependent Python control flow is unsupported."""
    from torch.fx import symbolic_trace

    try:
        module = symbolic_trace(workload)
        module.graph.lint()
    except Exception as error:
        raise UnsupportedGraphError(f"FX capture failed: {error}") from error
    description, names = describe_graph(module)
    return CapturedGraph(module, description, names)
