import unittest

from greenfuse.partition.enumerate import enumerate_partitions, select_partitions
from tests.unit.test_partition_enumeration import chain


class PartitionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.plans = enumerate_partitions(chain(3))

    def test_default_keeps_all(self):
        self.assertEqual(select_partitions(self.plans, None), self.plans)

    def test_aliases_map_to_expected_groupings(self):
        separate, first_pair = select_partitions(self.plans, ["separate", "first_pair"])
        self.assertEqual(separate.groups, (("op_0",), ("op_1",), ("op_2",)))
        self.assertEqual(first_pair.groups, (("op_0", "op_1"), ("op_2",)))

    def test_ids_and_duplicates(self):
        self.assertEqual([p.id for p in select_partitions(self.plans, ["p11", "separate", "p01"])], ["p11", "p01"])

    def test_unknown_name_fails(self):
        with self.assertRaises(ValueError):
            select_partitions(self.plans, ["middle"])


if __name__ == "__main__":
    unittest.main()
