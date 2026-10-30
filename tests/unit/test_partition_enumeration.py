import json
import unittest

from greenfuse.graph.ir import InputBinding, NormalizedGraph, Operation, ValueRef
from greenfuse.graph.validate import UnsupportedGraphError, validate_graph
from greenfuse.partition.enumerate import (
    PartitionSpec, SearchSpaceTooLarge, enumerate_partitions, groups_from_cuts, validate_partition,
)


def chain(length):
    operations = tuple(
        Operation(f"op_{i}", "relu", "torch.relu", (ValueRef("input_0" if i == 0 else f"op_{i-1}"),))
        for i in range(length)
    )
    return NormalizedGraph((InputBinding("input_0", "x"),), operations, ValueRef(f"op_{length-1}"))


class PartitionEnumerationTests(unittest.TestCase):
    def test_counts_unique_coverage_and_order(self):
        for n in (1, 2, 3, 4):
            with self.subTest(n=n):
                graph = chain(n)
                plans = enumerate_partitions(graph)
                self.assertEqual(len(plans), 2 ** (n - 1))
                self.assertEqual(len({plan.groups for plan in plans}), len(plans))
                self.assertEqual(len({plan.key for plan in plans}), len(plans))
                for plan in plans:
                    validate_partition(graph, plan)
                    self.assertEqual(tuple(node for group in plan.groups for node in group), tuple(op.id for op in graph.operations))
                    self.assertEqual(len(plan.groups), sum(plan.cuts) + 1)
                self.assertEqual(plans, enumerate_partitions(graph))

    def test_w1_boundary_meaning(self):
        plans = {plan.id: plan.groups for plan in enumerate_partitions(chain(3))}
        self.assertEqual(plans, {
            "p00": (("op_0", "op_1", "op_2"),),
            "p01": (("op_0", "op_1"), ("op_2",)),
            "p10": (("op_0",), ("op_1", "op_2")),
            "p11": (("op_0",), ("op_1",), ("op_2",)),
        })
        self.assertEqual(enumerate_partitions(chain(1))[0].id, "p_single")

    def test_search_limit_rejects_instead_of_truncating(self):
        for limit in (1, 3):
            with self.assertRaises(SearchSpaceTooLarge):
                enumerate_partitions(chain(3), max_partitions=limit)
        self.assertEqual(len(enumerate_partitions(chain(3), max_partitions=4)), 4)
        with self.assertRaises(SearchSpaceTooLarge):
            enumerate_partitions(chain(12))
        for invalid in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                enumerate_partitions(chain(2), max_partitions=invalid)

    def test_empty_and_invalid_cuts(self):
        with self.assertRaises(UnsupportedGraphError):
            enumerate_partitions(chain(0))
        for nodes, cuts in [((), ()), (("a", "a"), (1,)), (("a", "b"), ()), (("a", "b"), (2,)), (("a", "b"), (True,))]:
            with self.subTest(nodes=nodes, cuts=cuts), self.assertRaises(ValueError):
                groups_from_cuts(nodes, cuts)

    def test_reject_wrong_graph_and_malformed_partition(self):
        graph = chain(3)
        with self.assertRaisesRegex(ValueError, "different graph"):
            validate_partition(graph, enumerate_partitions(chain(2))[0])
        bad = PartitionSpec(graph.fingerprint, (1, 1), (("op_0", "op_1"), ("op_2",)))
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_partition(graph, bad)

    def test_metadata_json_and_deterministic_fingerprint(self):
        graph = chain(3)
        encoded = json.dumps(graph.to_dict(), allow_nan=False)
        self.assertEqual(json.loads(encoded), graph.to_dict())
        self.assertEqual(graph.fingerprint, chain(3).fingerprint)
        self.assertNotEqual(graph.fingerprint, chain(2).fingerprint)
        for plan in enumerate_partitions(graph):
            self.assertEqual(json.loads(json.dumps(plan.to_dict())), plan.to_dict())

    def test_reject_unknown_dependency(self):
        graph = NormalizedGraph((InputBinding("i", "x"),), (Operation("a", "relu", "torch.relu", (ValueRef("missing"),)),), ValueRef("a"))
        with self.assertRaisesRegex(UnsupportedGraphError, "dependency"):
            validate_graph(graph)


if __name__ == "__main__":
    unittest.main()
