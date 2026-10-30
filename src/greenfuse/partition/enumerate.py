"""Enumerate every contiguous partition. This module does not import PyTorch."""

from dataclasses import dataclass
from itertools import product

from greenfuse.graph.ir import NormalizedGraph
from greenfuse.graph.validate import validate_graph


class SearchSpaceTooLarge(ValueError):
    """Fail explicitly instead of silently truncating an exhaustive search."""


@dataclass(frozen=True)
class PartitionSpec:
    graph_fingerprint: str
    cuts: tuple[int, ...]
    groups: tuple[tuple[str, ...], ...]

    @property
    def id(self) -> str:
        return "p" + "".join(str(bit) for bit in self.cuts) if self.cuts else "p_single"

    @property
    def key(self) -> str:
        return f"{self.graph_fingerprint}:{self.id}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "graph_fingerprint": self.graph_fingerprint,
            "cuts": list(self.cuts),
            "groups": [list(group) for group in self.groups],
        }


def groups_from_cuts(node_ids: tuple[str, ...], cuts: tuple[int, ...]) -> tuple[tuple[str, ...], ...]:
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError("Expected a nonempty sequence of unique operation IDs")
    if len(cuts) != len(node_ids) - 1 or any(type(bit) is not int or bit not in (0, 1) for bit in cuts):
        raise ValueError("Expected one 0/1 cut bit per operation boundary")
    groups = [[node_ids[0]]]
    for cut, node_id in zip(cuts, node_ids[1:]):
        if cut:
            groups.append([node_id])
        else:
            groups[-1].append(node_id)
    return tuple(tuple(group) for group in groups)


def validate_partition(graph: NormalizedGraph, partition: PartitionSpec) -> None:
    validate_graph(graph)
    if partition.graph_fingerprint != graph.fingerprint:
        raise ValueError("Partition belongs to a different graph")
    expected = groups_from_cuts(tuple(op.id for op in graph.operations), partition.cuts)
    if partition.groups != expected:
        raise ValueError("Partition groups do not match their cuts and graph order")


def enumerate_partitions(graph: NormalizedGraph, *, max_partitions: int = 256) -> tuple[PartitionSpec, ...]:
    validate_graph(graph)
    if type(max_partitions) is not int or max_partitions < 1:
        raise ValueError("max_partitions must be a positive integer")
    node_ids = tuple(op.id for op in graph.operations)
    gaps = len(node_ids) - 1
    if gaps >= max_partitions.bit_length():
        raise SearchSpaceTooLarge(f"This graph requires 2**{gaps} partitions; limit is {max_partitions}")
    count = 1 << gaps
    if count > max_partitions:
        raise SearchSpaceTooLarge(f"This graph requires {count} partitions; limit is {max_partitions}")
    fingerprint = graph.fingerprint
    return tuple(
        PartitionSpec(fingerprint, cuts, groups_from_cuts(node_ids, cuts))
        for cuts in product((0, 1), repeat=gaps)
    )
