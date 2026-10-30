"""Synchronized wall-clock windows, including the entire executor call."""

import math
from time import perf_counter

from greenfuse.execution.buffers import synchronize
from greenfuse.results import Trial


def measure_timing(executor, inputs, iterations, *, sync=None, clock=perf_counter):
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    sync = sync or (lambda: synchronize(executor.device))
    trial = Trial("timing", iterations)
    try:
        sync()
        trial.start_monotonic_seconds = clock()
        for _ in range(iterations):
            executor.run(inputs)
            trial.submitted_iterations += 1
        sync()
        trial.end_monotonic_seconds = clock()
        trial.completed_iterations = iterations
        trial.elapsed_seconds = trial.end_monotonic_seconds - trial.start_monotonic_seconds
        if not math.isfinite(trial.elapsed_seconds) or trial.elapsed_seconds <= 0:
            raise ValueError("Clock did not produce a positive finite elapsed duration")
        trial.seconds_per_execution = trial.elapsed_seconds / iterations
        trial.status = "ok"
    except Exception as error:
        trial.status, trial.reason = "failed", f"{type(error).__name__}: {error}"
        trial.seconds_per_execution = None
    return trial


def calibrate_iterations(executor, inputs, window_seconds, *, max_iterations=10_000_000, measure=measure_timing):
    """Record setup pilots; aim 25% above the minimum to allow ordinary jitter."""
    if type(window_seconds) not in (int, float) or not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    pilots = []
    iterations = min(16, max_iterations)
    for _ in range(12):
        trial = measure(executor, inputs, iterations)
        pilots.append(trial.to_dict())
        if trial.status != "ok":
            return {"status": "failed", "reason": trial.reason, "pilots": pilots, "iterations": None}
        if trial.elapsed_seconds >= window_seconds * 1.25:
            return {"status": "ready", "reason": None, "pilots": pilots, "iterations": iterations}
        if iterations >= max_iterations:
            break
        estimate = math.ceil(iterations * window_seconds * 1.25 / trial.elapsed_seconds)
        iterations = min(max_iterations, max(iterations + 1, min(iterations * 10, estimate)))
    return {"status": "failed", "reason": "Pilot limit reached before a sufficiently long window",
            "pilots": pilots, "iterations": None}
