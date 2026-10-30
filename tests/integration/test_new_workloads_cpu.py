"""CPU checks for W2-W4: capture, four partitions, fan-out exports, uncompiled equivalence."""

import importlib.util
import unittest

from greenfuse.graph.capture import capture_graph
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions, select_partitions
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.registry import get_workload, get_workload_spec


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "CPU graph tests require optional PyTorch")
class NewWorkloadTests(unittest.TestCase):
    KINDS = {"w2": ["matmul", "add", "gelu"], "w3": ["add", "layer_norm", "gelu"], "w4": ["add", "mean", "sub"],
             "w5": ["add", "gelu", "softmax"], "w6": ["add", "layer_norm", "matmul", "add", "gelu"]}

    def test_capture_and_all_partitions_match_eager(self):
        import torch

        for name, kinds in self.KINDS.items():
            with self.subTest(workload=name):
                workload = get_workload(name, n=16)
                captured = capture_graph(workload)
                self.assertEqual([op.kind for op in captured.description.operations], kinds)
                partitions = enumerate_partitions(captured.description)
                self.assertEqual(len(partitions), 2 ** (len(kinds) - 1))
                for seed in (0, 1):
                    inputs = make_inputs(get_workload_spec(name, m=8, n=16, seed=seed))
                    expected = workload(**inputs)
                    if name != "w5":  # softmax outputs are always positive
                        self.assertTrue((expected < 0).any() and (expected > 0).any())
                    for partition in partitions:
                        actual = build_partition(captured, partition).run(inputs)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_fan_out_group_exports_reused_value(self):
        captured = capture_graph(get_workload("w4"))
        partitions = enumerate_partitions(captured.description)
        first_pair, = select_partitions(partitions, ["first_pair"])
        first, second = build_partition(captured, first_pair).groups
        self.assertEqual(first.bindings.output_ids, ("op_0", "op_1"))
        self.assertEqual(second.bindings.input_ids, ("op_0", "op_1"))
        separate, = select_partitions(partitions, ["separate"])
        self.assertEqual([g.bindings.output_ids for g in build_partition(captured, separate).groups],
                         [("op_0",), ("op_1",), ("op_2",)])

    def test_w3_requires_row_width(self):
        for name in ("w3", "w6"):
            with self.assertRaises(ValueError):
                get_workload(name)

    def test_fuse_epilogue_uses_triton_only_for_gemm_plus_epilogue_groups(self):
        from greenfuse.compiler.inductor import CompilerConfig
        from greenfuse.naming import grouping_name

        captured = capture_graph(get_workload("w6", n=16))
        kinds = [op.kind for op in captured.description.operations]
        config = CompilerConfig(gemm_backend="fuse_epilogue")
        base = {"triton.cudagraphs": False}
        by_name = {grouping_name(p, kinds): p for p in enumerate_partitions(captured.description)}
        groups = build_partition(captured, by_name["add+ln_mm_add+gelu"]).groups
        self.assertEqual([config.group_options(base, g.module).get("max_autotune_gemm_backends") for g in groups],
                         [None, None, None])  # GEMM alone keeps cuBLAS
        groups = build_partition(captured, by_name["add+ln_mm+add+gelu"]).groups
        self.assertEqual([config.group_options(base, g.module).get("max_autotune_gemm_backends") for g in groups],
                         [None, "TRITON"])  # fused epilogue needs a Triton GEMM
        self.assertEqual(CompilerConfig().group_options(base, groups[1].module), base)

    def test_compiled_fan_out_partitions_on_cpu(self):
        import torch
        from greenfuse.compiler.inductor import compile_partition

        captured = capture_graph(get_workload("w4"))
        inputs = make_inputs(get_workload_spec("w4", m=8, n=16))
        expected = get_workload("w4")(**inputs)
        for partition in enumerate_partitions(captured.description):
            with self.subTest(partition=partition.id):
                compiled = compile_partition(build_partition(captured, partition), inputs, allow_cpu=True)
                with torch.inference_mode():
                    output, _ = compiled.evaluate(tuple(inputs[k] for k in ("x", "bias")))
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
