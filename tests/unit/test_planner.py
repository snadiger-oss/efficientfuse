import copy
import unittest

from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.planner.candidates import combine_candidates
from greenfuse.planner.select import select_report, validate_repeat
from greenfuse.results import fingerprint
from tests.unit.planner_fixtures import make_failed_candidate, synthetic_report
from tests.unit.test_partition_enumeration import chain


class PlannerTests(unittest.TestCase):
    def test_four_partitions_produce_eight_distinct_candidates(self):
        partitions = enumerate_partitions(chain(3))
        candidates = combine_candidates(partitions, ("launch", "graph"), "experiment1")
        self.assertEqual(len(candidates), 8)
        self.assertEqual(len({c.key for c in candidates}), 8)
        changed = combine_candidates(partitions, ("launch", "graph"), "experiment2")
        self.assertNotEqual(candidates[0].key, changed[0].key)

    def test_exact_five_percent_boundary_and_slower_cheaper_candidate(self):
        result = select_report(synthetic_report())
        self.assertEqual(result["status"], "selected", result)
        self.assertEqual(result["selected_id"], "p01_launch")
        self.assertEqual(result["fastest_id"], "p00_launch")
        self.assertIn("p01_graph", result["feasible_ids"])  # exactly 105 us
        self.assertNotIn("p10_launch", result["feasible_ids"])  # 107 us, cheaper
        self.assertNotIn("eager_launch", result["candidates"])

    def test_tie_break_runtime_then_stable_id(self):
        costs = {name: (100, 40) for name in synthetic_report()["requested_ids"] if name != "eager_launch"}
        result = select_report(synthetic_report(costs=costs))
        self.assertEqual(result["selected_id"], "p00_graph")
        costs["p00_launch"] = (99, 40)
        self.assertEqual(select_report(synthetic_report(costs=costs))["selected_id"], "p00_launch")

    def test_missing_energy_blocks_entire_comparison(self):
        report = synthetic_report()
        trial = report["candidates"]["p10_launch"]["trials"][-1]
        trial.update(status="unavailable", reason="SYNTHETIC missing energy", energy_joules_per_execution=None)
        self.assertEqual(select_report(report)["status"], "incomplete")
        self.assertIsNone(select_report(report, exclusion_policy="preparation_failures")["selected_id"])

    def test_incomplete_and_cpu_reports_never_select_energy_winner(self):
        report = synthetic_report()
        report["complete"] = False
        self.assertEqual(select_report(report)["status"], "incomplete")
        report["scope"] = "cpu_development"
        self.assertEqual(select_report(report)["status"], "runtime_only")

    def test_exclusions_require_explicit_policy_and_preparation_evidence(self):
        report = synthetic_report()
        make_failed_candidate(report, "p11_graph")
        self.assertEqual(select_report(report)["status"], "incomplete")
        self.assertEqual(select_report(report, exclusion_policy="preparation_failures")["status"], "invalid")
        report = synthetic_report(exclusion_policy="preparation_failures")
        make_failed_candidate(report, "p11_graph")
        result = select_report(report)
        self.assertEqual(result["status"], "selected", result)
        self.assertIn("p11_graph", result["exclusions"])
        report["candidates"]["p11_graph"]["preparations"][0]["failure_stage"] = "profiling"
        self.assertEqual(select_report(report, exclusion_policy="preparation_failures")["status"], "incomplete")

    def test_all_excluded_has_no_winner(self):
        report = synthetic_report(exclusion_policy="preparation_failures")
        for name in report["requested_ids"]:
            if name != "eager_launch":
                make_failed_candidate(report, name)
        self.assertEqual(select_report(report, exclusion_policy="preparation_failures")["status"], "empty")

    def test_nan_nonpositive_mixed_graph_and_wrong_compiler_rejected(self):
        for mutation in (
            lambda e: e["trials"][0].update(seconds_per_execution=float("nan")),
            lambda e: e["trials"][0].update(seconds_per_execution=0),
            lambda e: e["preparations"][0].update(graph_fingerprint="another-graph"),
            lambda e: e["preparations"][0]["compilation"]["options"].update({"triton.cudagraphs": True}),
            lambda e: e["preparations"][0]["buffers"]["tensors"][0].update(shape=[1, 2]),
        ):
            report = synthetic_report()
            mutation(report["candidates"]["p00_launch"])
            self.assertEqual(select_report(report)["status"], "invalid")

    def test_manifest_and_metadata_mismatch_rejected(self):
        report = synthetic_report()
        del report["candidates"]["p11_launch"]
        report["requested_ids"].remove("p11_launch")
        self.assertEqual(select_report(report)["status"], "invalid")
        report = synthetic_report()
        report["metadata"]["workload"]["seed"] += 1
        self.assertEqual(select_report(report)["status"], "invalid")

    def test_fresh_validation_requires_stable_winner_and_distinct_windows(self):
        initial, fresh = synthetic_report(), synthetic_report(fresh=True)
        self.assertEqual(validate_repeat(initial, fresh)["status"], "validated")
        self.assertEqual(validate_repeat(initial, copy.deepcopy(initial))["status"], "inconclusive")
        reused = copy.deepcopy(initial)
        reused["run_id"] = "new-id-same-windows"
        self.assertIn("overlap", validate_repeat(initial, reused)["reason"])
        costs = {name: (100, 40) for name in initial["requested_ids"] if name != "eager_launch"}
        self.assertEqual(validate_repeat(initial, synthetic_report(fresh=True, costs=costs))["status"], "inconclusive")

    def test_changed_environment_cannot_validate(self):
        initial, fresh = synthetic_report(), synthetic_report(fresh=True)
        fresh["environment"]["versions"]["torch"] = "different software"
        fresh["comparison_fingerprint"] = fingerprint({k: fresh[k] for k in ("metadata", "config", "environment", "energy_identity")})
        self.assertEqual(validate_repeat(initial, fresh)["status"], "inconclusive")
