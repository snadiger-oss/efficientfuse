"""Synthetic trace fixtures test the parser, not actual GPU behavior."""

import json
import unittest

from greenfuse.compiler.inductor import CompilerConfig
from greenfuse.compiler.inspect import INVOCATION_MARKER, summarize_kernel_trace


def trace(*events):
    return {"traceEvents": [
        {"name": INVOCATION_MARKER, "ph": "X", "cat": "user_annotation", "ts": 0, "dur": 100},
        *events,
    ]}


def kernel(name="same_kernel", device=0, ts=20):
    return {"name": name, "ph": "X", "cat": "kernel", "ts": ts, "dur": 5, "args": {"device": device}}


def launch(name="cudaLaunchKernel"):
    return {"name": name, "ph": "X", "cat": "cuda_runtime", "ts": 10, "dur": 1}


class KernelEvidenceTests(unittest.TestCase):
    def test_count_launch_events_not_unique_names(self):
        evidence = summarize_kernel_trace(trace(launch(), kernel(), kernel(ts=30)), mode="launch", device_index=0)
        self.assertEqual(evidence.kernel_launches_per_invocation, 2)
        self.assertEqual(evidence.replay_check, "passed")
        self.assertEqual(evidence.graph_launch_api_events, 0)
        json.dumps(evidence.to_dict(), allow_nan=False)

    def test_graph_submission_can_contain_multiple_kernels(self):
        evidence = summarize_kernel_trace(trace(launch("cudaGraphLaunch"), kernel(), kernel(ts=30)), mode="graph", device_index=0)
        self.assertEqual(evidence.kernel_launches_per_invocation, 2)
        self.assertEqual(evidence.graph_launch_api_events, 1)
        self.assertEqual(evidence.replay_check, "passed")

    def test_ordinary_mode_rejects_hidden_graph_replay(self):
        evidence = summarize_kernel_trace(trace(launch("cuGraphLaunch"), kernel()), mode="launch", device_index=0)
        self.assertEqual(evidence.status, "invalid")
        self.assertEqual(evidence.replay_check, "failed")

    def test_missing_kernel_data_is_none_not_zero(self):
        evidence = summarize_kernel_trace(trace(launch("cudaGraphLaunch")), mode="graph", device_index=0)
        self.assertIsNone(evidence.kernel_launches_per_invocation)
        self.assertEqual(evidence.status, "unavailable")

    def test_memcopies_and_work_outside_scope_are_not_kernels(self):
        copy = {"name": "Memcpy DtoD", "ph": "X", "cat": "gpu_memcpy", "ts": 15, "dur": 2}
        evidence = summarize_kernel_trace(trace(launch(), copy, kernel(), kernel(ts=200)), mode="launch", device_index=0)
        self.assertEqual(evidence.kernel_launches_per_invocation, 1)

    def test_require_complete_unambiguous_scope_and_device(self):
        bad = [
            {}, {"traceEvents": []}, trace(kernel(device=1)),
            trace({**kernel(), "ph": "B"}), trace({**kernel(), "dur": -1}),
            trace(kernel(ts=98)), trace({**kernel(), "args": {}}),
        ]
        for fixture in bad:
            with self.subTest(fixture=fixture):
                evidence = summarize_kernel_trace(fixture, mode="launch", device_index=0)
                self.assertIsNone(evidence.kernel_launches_per_invocation)

    def test_kernel_count_does_not_prove_replay_policy(self):
        evidence = summarize_kernel_trace(trace(kernel()), mode="graph", device_index=0)
        self.assertEqual(evidence.kernel_launches_per_invocation, 1)
        self.assertEqual(evidence.replay_check, "unavailable")

    def test_compiler_policy_requires_supported_disable_option(self):
        self.assertEqual(CompilerConfig().options(["triton.cudagraphs"]), {"triton.cudagraphs": False})
        with self.assertRaisesRegex(RuntimeError, "lacks required options"):
            CompilerConfig().options([])


if __name__ == "__main__":
    unittest.main()
