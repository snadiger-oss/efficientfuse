"""Synthetic portable decisions and process-local ownership checks."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from greenfuse.execution.prepare import ExecutionRecord, PreparedExecution
from greenfuse.planner.cache import DecisionCache
from greenfuse.planner.dispatch import ExecutorRegistry
from greenfuse.results import fingerprint
from tests.unit.planner_fixtures import synthetic_report


class DecisionCacheTests(unittest.TestCase):
    def setUp(self):
        self.initial = synthetic_report()
        self.fresh = synthetic_report(fresh=True)
        self.identity = self.initial["metadata"]["planner_identity"]
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = DecisionCache(self.directory.name)

    def test_miss_store_and_verified_hit(self):
        self.assertEqual(self.cache.lookup(self.identity).status, "miss")
        stored = self.cache.store(self.identity, self.initial, self.fresh)
        lookup = self.cache.lookup(self.identity)
        self.assertEqual(lookup.status, "hit", lookup.reason)
        self.assertEqual(lookup.decision["selected_id"], "p01_launch")
        self.assertEqual(lookup.decision, stored)

    def test_graph_shape_dtype_device_software_and_policy_invalidate(self):
        self.cache.store(self.identity, self.initial, self.fresh)
        for mutate in (
            lambda i: i.update(graph_fingerprint="different graph"),
            lambda i: i["workload"]["inputs"][0].update(shape=[64, 256]),
            lambda i: i["workload"]["inputs"][0].update(dtype="float64"),
            lambda i: i["environment"].update(physical_gpu_uuid="another GPU"),
            lambda i: i["environment"]["versions"].update(torch="another version"),
            lambda i: i["policy"].update(latency_tolerance=0.1),
            lambda i: i.update(implementation="new implementation"),
        ):
            changed = copy.deepcopy(self.identity)
            mutate(changed)
            self.assertEqual(self.cache.lookup(changed).status, "miss")

    def test_corrupt_schema_checksum_or_evidence_is_invalid(self):
        self.cache.store(self.identity, self.initial, self.fresh)
        path = Path(self.directory.name) / f"{fingerprint(self.identity)}.json"
        original = path.read_text()
        path.write_text("broken JSON")
        self.assertEqual(self.cache.lookup(self.identity).status, "invalid")
        envelope = json.loads(original)
        envelope["schema_version"] = 99
        path.write_text(json.dumps(envelope))
        self.assertEqual(self.cache.lookup(self.identity).status, "invalid")
        path.write_text(original)
        evidence = next((Path(self.directory.name) / "evidence").glob("*.json"))
        evidence.write_text("{}")
        self.assertEqual(self.cache.lookup(self.identity).status, "invalid")

    def test_tampered_plan_rejected_even_if_envelope_checksum_recomputed(self):
        self.cache.store(self.identity, self.initial, self.fresh)
        path = Path(self.directory.name) / f"{fingerprint(self.identity)}.json"
        envelope = json.loads(path.read_text())
        envelope["payload"]["selected_id"] = "p10_launch"
        envelope["checksum"] = fingerprint(envelope["payload"])
        path.write_text(json.dumps(envelope))
        self.assertEqual(self.cache.lookup(self.identity).status, "invalid")

    def test_incomplete_or_unvalidated_or_mixed_evidence_cannot_be_saved(self):
        self.fresh["complete"] = False
        with self.assertRaisesRegex(ValueError, "not validated"):
            self.cache.store(self.identity, self.initial, self.fresh)
        with self.assertRaises(ValueError):
            self.cache.store(self.identity, self.initial, self.initial)
        self.fresh = synthetic_report(fresh=True)
        self.fresh["metadata"]["planner_key"] = "another identity"
        with self.assertRaisesRegex(ValueError, "different"):
            self.cache.store(self.identity, self.initial, self.fresh)


class ExecutorRegistryTests(unittest.TestCase):
    def factory(self):
        record = ExecutionRecord("test", "graph", "launch", 1, correctness={"status": "passed"})
        return PreparedExecution(record, Mock(device=SimpleNamespace(type="cpu")))

    def test_same_process_reuses_executor_and_context_changes_rebuild(self):
        factory = Mock(side_effect=self.factory)
        registry = ExecutorRegistry()
        first, reused, _ = registry.acquire("key", factory, context_token="stream1", require_gpu=False)
        second, reused, seconds = registry.acquire("key", factory, context_token="stream1", require_gpu=False)
        self.assertIs(first, second)
        self.assertTrue(reused)
        self.assertEqual(seconds, 0)
        registry.acquire("key", factory, context_token="stream2", require_gpu=False)
        self.assertEqual(factory.call_count, 2)
        registry.close()
        first.executor.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            registry.acquire("key", factory, context_token="stream1", require_gpu=False)

    def test_cpu_never_substitutes_for_cached_gpu_executor(self):
        prepared = self.factory()
        registry = ExecutorRegistry()
        with self.assertRaisesRegex(RuntimeError, "GPU"):
            registry.acquire("key", lambda: prepared, context_token=0)
        prepared.executor.close.assert_called_once()
        self.assertEqual(registry.entries, {})
