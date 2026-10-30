"""Keep profiling separate from timing/energy; reuse Person 2's trace parser."""

from time import perf_counter

from greenfuse.compiler.inspect import profile_executor


def profile_prepared(prepared, inputs, trace_path):
    start = perf_counter()
    record = prepared.record
    evidence = profile_executor(prepared.executor, inputs, trace_path)
    record.profiling = evidence.to_dict()
    record.setup_seconds["profiling"] = perf_counter() - start
    record.ready_for_gpu_measurement = bool(
        prepared.executor.device.type == "cuda" and record.correctness.get("status") == "passed"
        and evidence.replay_check == "passed"
    )
    if evidence.replay_check == "failed":
        record.status, record.failure_stage, record.reason = "failed", "profiling", evidence.reason
    elif record.ready_for_gpu_measurement:
        record.status = "ready"
    return evidence
