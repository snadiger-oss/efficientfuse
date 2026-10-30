"""CLI: inspect, benchmark, select, calibrate, run, and report."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import uuid

from .config import ExperimentConfig, load_config
from .planner.cache import DecisionCache, write_json_atomic
from .planner.select import select_report
from .results import load_report, summarize_candidate, write_summary_csv


def _experiment_arguments(parser):
    parser.add_argument("--config", type=Path, help="Strict JSON/YAML experiment; defaults to one W1 case")
    parser.add_argument("--device", help="cuda:N, or cpu for explicit development")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--exclusive-gpu", action="store_true", help="Assert a reserved whole physical GPU")
    parser.add_argument("--output-dir", type=Path, default=Path("results/planner"))


def _config(args):
    config = load_config(args.config) if args.config else ExperimentConfig()
    updates = {}
    if args.device is not None:
        updates["device"] = args.device
        if args.device == "cpu":
            updates["modes"] = ("launch",)
    if args.rounds is not None:
        updates["rounds"] = args.rounds
    if args.window_seconds is not None:
        updates["window_seconds"] = args.window_seconds
    return replace(config, **updates)


def _print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="greenfuse", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "benchmark", "calibrate", "run"):
        command = subparsers.add_parser(name)
        _experiment_arguments(command)
        if name in ("calibrate", "run"):
            command.add_argument("--cache-dir", type=Path, default=Path("results/decisions"))
        if name == "run":
            command.add_argument("--calibrate-on-miss", action="store_true", help="Allow a full search and validation on a cache miss")
            command.add_argument("--iterations", type=int, default=1000, help="Warm execution diagnostic timing; not fresh energy validation")
    select_parser = subparsers.add_parser("select", help="Read saved measurements without loading a GPU")
    select_parser.add_argument("input", type=Path)
    select_parser.add_argument("--output", type=Path)
    select_parser.add_argument("--latency-tolerance", type=float)
    select_parser.add_argument("--exclusion-policy", choices=["none", "preparation_failures"])
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("input", type=Path)
    report_parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ("select", "report"):
            report = load_report(args.input)
            if args.command == "select":
                result = select_report(report, latency_tolerance=args.latency_tolerance, exclusion_policy=args.exclusion_policy)
                if args.output:
                    write_json_atomic(args.output, result)
                _print_json(result)
                return 0 if result["status"] == "selected" else 2
            output = args.csv or args.input.with_suffix(".csv")
            write_summary_csv(report, output)
            _print_json({"scope": report["scope"], "coverage_complete": report["complete"], "csv": str(output),
                         "candidates": {name: summarize_candidate(entry, report["config"]["rounds"], gpu_run=report["scope"] == "gpu")
                                        for name, entry in report["candidates"].items()}})
            return 0
        config = _config(args)
        if args.command == "run" and (type(args.iterations) is not int or args.iterations < 1):
            raise ValueError("iterations must be a positive integer")
        # GPU-dependent modules are imported only by commands that execute code.
        from .planner.calibrate import benchmark_context, calibrate_context, create_context
        session = args.output_dir / uuid.uuid4().hex[:12]
        outcomes = []
        success = True
        for shape in config.shapes:
            context = create_context(config, shape, exclusive_gpu=args.exclusive_gpu)
            directory = session / f"{config.workload}_{shape[0]}x{shape[1]}"
            if args.command == "inspect":
                result = {"shape": list(shape), "graph": context.captured.description.to_dict(),
                          "planner_key": context.key, "candidates": [c.to_dict() for c in context.candidates],
                          "note": "Requested operation groups are not observed GPU kernel counts"}
            elif args.command == "benchmark":
                path = directory / "measurements.json"
                report = benchmark_context(context, path, exclusive_gpu=args.exclusive_gpu, progress=print)
                result = {"shape": list(shape), "scope": report["scope"], "report": str(path.resolve()),
                          "selection": select_report(report)}
                valid = all(c["summary"]["timing"]["complete"] for c in report["candidates"].values()) if config.device == "cpu" else result["selection"]["status"] == "selected"
                success = success and valid
            elif args.command == "calibrate":
                result = calibrate_context(context, output_dir=directory, cache_dir=args.cache_dir,
                                           exclusive_gpu=args.exclusive_gpu, progress=print)
                success = success and result["status"] == "validated"
            else:
                from .execution.buffers import synchronize
                from .measurement.timing import measure_timing
                from .planner.dispatch import CachedDispatcher
                lookup = DecisionCache(args.cache_dir).lookup(context.identity)
                if lookup.status != "hit" and args.calibrate_on_miss:
                    calibration = calibrate_context(context, output_dir=directory, cache_dir=args.cache_dir,
                                                    exclusive_gpu=args.exclusive_gpu, progress=print)
                    if calibration["status"] != "validated":
                        raise RuntimeError(f"Calibration did not produce a validated decision: {calibration.get('reason')}")
                with CachedDispatcher(args.cache_dir, directory / "dispatch") as dispatcher:
                    first = dispatcher.acquire(context)
                    output = first.run(context.inputs)
                    synchronize(first.prepared.executor.device)
                    warm = dispatcher.acquire(context)
                    trial = measure_timing(warm.prepared.executor, context.inputs, args.iterations)
                    if trial.status != "ok":
                        raise RuntimeError(trial.reason)
                    result = {"status": "cached_execution", "first_acquisition": first.to_dict(),
                              "warm_acquisition": warm.to_dict(), "diagnostic_timing": trial.to_dict(),
                              "output_shape": list(output.shape),
                              "note": "One warm timing window; no new energy, significance, or break-even claim"}
                write_json_atomic(directory / "cached_run.json", result)
                result["report"] = str((directory / "cached_run.json").resolve())
            outcomes.append(result)
            if args.command != "inspect":
                write_json_atomic(session / "outcomes.json", outcomes)
        _print_json(outcomes)
        return 0 if success else 2
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(f"greenfuse: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
