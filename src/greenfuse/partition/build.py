"""Build independent FX group callables and an explicit ordered binding plan."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from greenfuse.graph.capture import CapturedGraph, describe_graph
from greenfuse.graph.ir import InputBinding
from .enumerate import PartitionSpec, validate_partition

if TYPE_CHECKING:
    from torch.fx import GraphModule


@dataclass(frozen=True)
class GroupBindings:
    node_ids: tuple[str, ...]
    input_ids: tuple[str, ...]
    output_id: str

    def to_dict(self) -> dict:
        return {"node_ids": list(self.node_ids), "input_ids": list(self.input_ids), "output_id": self.output_id}


@dataclass
class BuiltGroup:
    bindings: GroupBindings
    module: "GraphModule"


@dataclass
class BuiltPartition:
    """Live handoff to Person 2. The descriptor below contains no live objects."""

    spec: PartitionSpec
    inputs: tuple[InputBinding, ...]
    groups: tuple[BuiltGroup, ...]
    output_id: str

    def descriptor(self) -> dict:
        return {
            "partition": self.spec.to_dict(),
            "inputs": [{"id": item.id, "name": item.name} for item in self.inputs],
            "bindings": [group.bindings.to_dict() for group in self.groups],
            "output_id": self.output_id,
        }

    def run(self, inputs: dict):
        """Uncompiled reference execution only; the GPU adapter owns timed execution."""
        expected = {item.name for item in self.inputs}
        if set(inputs) != expected:
            raise ValueError(f"Expected inputs {sorted(expected)}, got {sorted(inputs)}")
        values = {item.id: inputs[item.name] for item in self.inputs}
        for group in self.groups:
            args = [values[key] for key in group.bindings.input_ids]
            values[group.bindings.output_id] = group.module(*args)
        return values[self.output_id]


def build_partition(captured: CapturedGraph, spec: PartitionSpec) -> BuiltPartition:
    from torch.fx import Graph, GraphModule

    # Detect stale metadata if someone edited the live graph after capture.
    current, node_names = describe_graph(captured.module)
    if current != captured.description or node_names != captured.node_names:
        raise ValueError("Captured FX graph changed; capture it again before building")
    validate_partition(current, spec)
    fx_by_name = {node.name: node for node in captured.module.graph.nodes}
    original = {key: fx_by_name[name] for key, name in node_names.items()}
    operations = {op.id: op for op in current.operations}
    groups = []

    for index, node_ids in enumerate(spec.groups):
        members = set(node_ids)
        # First-use order is deterministic. External inputs include residual/bias.
        input_ids = tuple(dict.fromkeys(
            dependency
            for node_id in node_ids
            for dependency in operations[node_id].dependencies
            if dependency not in members
        ))
        graph = Graph()
        remap = {original[key]: graph.placeholder(key) for key in input_ids}
        for node_id in node_ids:
            source = original[node_id]
            remap[source] = graph.node_copy(source, lambda node: remap[node])
        output_id = node_ids[-1]
        graph.output(remap[original[output_id]])
        graph.lint()
        module = GraphModule(captured.module, graph, class_name=f"Partition_{spec.id}_Group_{index}")
        module.graph.lint()
        groups.append(BuiltGroup(GroupBindings(node_ids, input_ids, output_id), module))
    return BuiltPartition(spec, current.inputs, tuple(groups), current.output.id)
