"""CPU-only Person 1 demo. Run after installation or with PYTHONPATH=src."""

import argparse
import json
from pathlib import Path

from greenfuse.graph.capture import capture_graph
from greenfuse.partition.build import build_partition
from greenfuse.partition.enumerate import enumerate_partitions
from greenfuse.workloads.inputs import make_inputs
from greenfuse.workloads.registry import get_workload, get_workload_spec


def inspect_w1(*, m: int = 32, n: int = 256, seed: int = 0, max_partitions: int = 256) -> dict:
    """Return real CPU correctness evidence; no kernel/time/energy estimates."""
    import torch

    spec = get_workload_spec("w1", m=m, n=n, seed=seed)
    workload = get_workload("w1")
    inputs = make_inputs(spec)
    captured = capture_graph(workload)
    records = []
    with torch.inference_mode():
        expected = workload(**inputs)
        torch.testing.assert_close(captured.module(**inputs), expected, rtol=spec.rtol, atol=spec.atol)
        for partition in enumerate_partitions(captured.description, max_partitions=max_partitions):
            built = build_partition(captured, partition)
            actual = built.run(inputs)
            if not torch.isfinite(actual).all().item():
                raise AssertionError(f"{partition.id}: nonfinite CPU output")
            torch.testing.assert_close(actual, expected, rtol=spec.rtol, atol=spec.atol)
            records.append({
                **built.descriptor(),
                "correctness": {
                    "passed": True,
                    "device": "cpu",
                    "rtol": spec.rtol,
                    "atol": spec.atol,
                    "max_absolute_error": (actual - expected).abs().max().item(),
                },
            })
    return {
        "schema_version": 1,
        "scope": "CPU graph capture, partition construction, and correctness only",
        "torch_version": str(torch.__version__),
        "workload": spec.to_dict(),
        "input_spec_fingerprint": spec.fingerprint,
        "graph": captured.description.to_dict(),
        "graph_fingerprint": captured.description.fingerprint,
        "partitions": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-partitions", type=int, default=256)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file, e.g. results/person1_cpu.json")
    args = parser.parse_args()
    report = inspect_w1(m=args.m, n=args.n, seed=args.seed, max_partitions=args.max_partitions)
    print(f"W1 | shape=({args.m}, {args.n}) | seed={args.seed} | PyTorch {report['torch_version']} | CPU")
    print("Graph: add -> relu -> add")
    for record in report["partitions"]:
        partition = record["partition"]
        groups = " ".join("[" + ", ".join(group) + "]" for group in partition["groups"])
        print(f"{partition['id']}: {groups} | CPU correctness PASS")
    print("GPU compilation, observed kernel counts, runtime, and energy are not measured here.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Evidence: {args.output}")


if __name__ == "__main__":
    main()
