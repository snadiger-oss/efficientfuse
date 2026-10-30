"""Pure selection from raw, compatible evidence. No tensor/GPU dependencies."""

from decimal import Decimal
import math

from greenfuse.measurement.energy import physical_uuid
from greenfuse.measurement.runner import MeasurementConfig
from greenfuse.results import fingerprint, portable, summarize_candidate, validate_report
from .identity import stable_environment


def _policy(report, latency_tolerance, exclusion_policy):
    saved = report.get("metadata", {}).get("planner_identity", {}).get("policy", {})
    tolerance = saved.get("latency_tolerance", 0.05) if latency_tolerance is None else latency_tolerance
    exclusion = saved.get("exclusion_policy", "none") if exclusion_policy is None else exclusion_policy
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("Invalid latency tolerance")
    if exclusion not in ("none", "preparation_failures"):
        raise ValueError("Unknown exclusion policy")
    if saved and (tolerance != saved["latency_tolerance"] or exclusion != saved["exclusion_policy"]):
        raise ValueError("Selection policy differs from the predeclared planner configuration")
    return {"latency_tolerance": tolerance, "exclusion_policy": exclusion}


def _preparation_exclusion(entry):
    allowed = {"support", "compilation", "capture", "correctness"}
    preparations = entry.get("preparations", [])
    if not preparations or any(t["submitted_iterations"] or t["status"] == "ok" for t in entry["trials"]):
        return None
    if any(f.get("stage") not in allowed for f in entry.get("failures", [])):
        return None
    if all(p.get("status") in ("failed", "unsupported") and p.get("failure_stage") in allowed
           and p.get("reason") and not p.get("ready_for_gpu_measurement") for p in preparations):
        return sorted({f"{p['failure_stage']}: {p['reason']}" for p in preparations})
    return None


def _identity_checks(report):
    metadata, environment = report["metadata"], report["environment"]
    if metadata["input_spec_fingerprint"] != fingerprint(metadata["workload"]):
        raise ValueError("Input specification fingerprint differs from workload metadata")
    energy = report["energy_identity"]
    if energy.get("exclusive_physical_gpu_asserted") is not True:
        raise ValueError("Energy evidence lacks an exclusive physical GPU assertion")
    if physical_uuid(energy["cuda_uuid"]) != physical_uuid(energy["nvml_uuid"]) or physical_uuid(
        environment["device"]["uuid"]
    ) != physical_uuid(energy["nvml_uuid"]):
        raise ValueError("CUDA, NVML and environment GPU identities differ")
    expected_comparison = fingerprint({name: report[name] for name in ("metadata", "config", "environment", "energy_identity")})
    if report["comparison_fingerprint"] != expected_comparison:
        raise ValueError("Mixed or modified experiment metadata")
    identity = metadata.get("planner_identity")
    if identity is not None:
        if metadata.get("planner_key") != fingerprint(identity):
            raise ValueError("Planner identity key mismatch")
        if identity["graph_fingerprint"] != metadata["graph_fingerprint"] or identity["workload"] != metadata["workload"]:
            raise ValueError("Planner graph/input identity mismatch")
        if identity["environment"] != stable_environment(environment):
            raise ValueError("Current measurement environment differs from the planner identity")
        experiment = identity["experiment"]
        for key in ("rounds", "window_seconds", "warmup_iterations", "max_iterations", "rtol", "atol"):
            if report["config"][key] != experiment[key]:
                raise ValueError("Measurement configuration differs from the planner identity")
        if report["config"]["seed"] not in (experiment["seed"], experiment["seed"] + 1):
            raise ValueError("Measurement shuffle seed differs from calibration/validation policy")
        expected = {item["id"] for item in identity["candidates"]}
        if identity["experiment"]["eager_baseline"]:
            expected.add("eager_launch")
        if expected != set(report["requested_ids"]):
            raise ValueError("Requested candidates differ from the planner manifest")


def _candidate_checks(report, entry, partition, mode):
    metadata, config = report["metadata"], report["config"]
    rounds = config["rounds"]
    preparations = entry["preparations"]
    slots = {(p.get("measurement_pass"), p.get("round_index")) for p in preparations}
    if len(preparations) != 2 * rounds or slots != {(k, r) for k in ("timing", "energy") for r in range(rounds)}:
        raise ValueError(f"{entry['id']}: missing preparation evidence for a trial")
    device = f"cuda:{report['environment']['device']['logical_index']}"
    expected_inputs = metadata["workload"]["inputs"]
    for prep in preparations:
        if (prep["id"] != entry["id"] or prep["partition_id"] != partition["id"] or prep["mode"] != mode
                or prep["graph_fingerprint"] != metadata["graph_fingerprint"] or prep["device"] != device
                or prep["requested_groups"] != len(partition["groups"])):
            raise ValueError(f"{entry['id']}: mixed candidate/graph/device identity")
        compilation = prep["compilation"]
        if (compilation["backend"] != "inductor" or compilation["fullgraph"] is not True
                or compilation["dynamic"] is not False or compilation["options"].get("triton.cudagraphs") is not False
                or compilation["device"] != device
                or compilation["torch_version"] != report["environment"]["versions"]["torch"]
                or compilation["cuda_build"] != report["environment"]["cuda_build"]):
            raise ValueError(f"{entry['id']}: incompatible compiler/software policy")
        groups = compilation["groups"]
        if len(groups) != len(partition["groups"]) or any(
            g.get("status") != "compiled" or g.get("node_ids") != nodes for g, nodes in zip(groups, partition["groups"])
        ):
            raise ValueError(f"{entry['id']}: incomplete group compilation")
        correctness = prep["correctness"]
        if (correctness["status"] != "passed" or correctness["rtol"] != config["rtol"]
                or correctness["atol"] != config["atol"]
                or {c["inputs"] for c in correctness["checks"]} != {"original", "changed"}):
            raise ValueError(f"{entry['id']}: incompatible correctness evidence")
        buffers = prep["buffers"]
        if (buffers.get("forward_only") is not True or buffers.get("autocast") is not False
                or len(buffers["tensors"]) != len(expected_inputs)):
            raise ValueError(f"{entry['id']}: incompatible input/buffer contract")
        for actual, expected in zip(buffers["tensors"], expected_inputs):
            if (actual["name"] != expected["name"] or actual["shape"] != expected["shape"]
                    or actual["dtype"] != "torch." + expected["dtype"] or actual["device"] != device):
                raise ValueError(f"{entry['id']}: mixed input specialization")
        profiling = prep["profiling"]
        if profiling.get("replay_check") != "passed" or type(profiling.get("kernel_launches_per_invocation")) is not int or profiling["kernel_launches_per_invocation"] <= 0:
            raise ValueError(f"{entry['id']}: missing actual kernel/mode evidence")
    for trial in entry["trials"]:
        if trial["elapsed_seconds"] < config["window_seconds"]:
            raise ValueError(f"{entry['id']}: window is shorter than the experiment policy")
        start, end = trial["start_monotonic_seconds"], trial["end_monotonic_seconds"]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or not math.isclose(
            end - start, trial["elapsed_seconds"], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"{entry['id']}: invalid window boundaries")


def select_report(report, *, latency_tolerance=None, exclusion_policy=None):
    """Return selected/incomplete/runtime_only/invalid, never a silent fallback."""
    result = {"status": "invalid", "reason": None, "selected_id": None, "exclusions": {}, "candidates": {},
              "scope": "Lowest observed median GPU energy within the median-runtime bound; no significance claim"}
    try:
        policy = _policy(report, latency_tolerance, exclusion_policy)
        result["policy"] = policy
        validate_report(report)
        result["requested_ids"] = report["requested_ids"]
        if report["scope"] != "gpu":
            result.update(status="runtime_only" if report["scope"] == "cpu_development" else "incomplete",
                          reason="GPU energy selection requires real, complete GPU measurements")
            return result
        if not report["complete"]:
            result.update(status="incomplete", reason="Measurement run did not finish every requested slot")
            return result
        config = MeasurementConfig(**report["config"])
        if config.allow_cpu or config.rounds < 3:
            raise ValueError("Selection requires GPU settings and at least three rounds")
        # Missing energy is an incomplete comparison, not an exclusion.
        entries = [entry for entry in report["candidates"].values() if entry["role"] == "candidate"]
        incomplete = []
        for entry in entries:
            summary = summarize_candidate(entry, config.rounds, gpu_run=True)
            if not summary["eligible_for_selection"]:
                reasons = _preparation_exclusion(entry) if policy["exclusion_policy"] == "preparation_failures" else None
                if reasons:
                    result["exclusions"][entry["id"]] = reasons
                else:
                    incomplete.append(entry["id"])
        if incomplete:
            result.update(status="incomplete", reason=f"Missing/invalid time, energy, or preparation evidence: {', '.join(incomplete)}")
            return result
        if not entries or len(result["exclusions"]) == len(entries):
            result.update(status="empty", reason="No eligible compiled candidates remain")
            return result
        _identity_checks(report)
        partitions = {p["id"]: p for p in report["metadata"]["partitions"]}
        if len(partitions) != len(report["metadata"]["partitions"]) or any(
            p["graph_fingerprint"] != report["metadata"]["graph_fingerprint"] for p in partitions.values()
        ):
            raise ValueError("Mixed partition graph identities")
        modes = set()
        for entry in report["candidates"].values():
            if entry["role"] == "external_baseline":
                if entry["id"] != "eager_launch":
                    raise ValueError("Only the original eager workload may be an external baseline")
                continue
            partition_id, mode = entry["id"].rsplit("_", 1)
            if partition_id not in partitions or mode not in ("launch", "graph"):
                raise ValueError("Candidate does not match the partition/mode manifest")
            modes.add(mode)
            if entry["id"] in result["exclusions"]:
                continue
            _candidate_checks(report, entry, partitions[partition_id], mode)
            summary = summarize_candidate(entry, config.rounds, gpu_run=True)
            result["candidates"][entry["id"]] = {"seconds": summary["timing"]["median"],
                "joules": summary["energy"]["median"], "timing": summary["timing"], "energy": summary["energy"],
                "partition_id": partition_id, "mode": mode}
        if {e["id"] for e in entries} != {f"{pid}_{mode}" for pid in partitions for mode in modes}:
            raise ValueError("Partition/mode combinations are missing from requested coverage")
        rows = result["candidates"]
        fastest = min(rows, key=lambda name: (rows[name]["seconds"], name))
        limit = Decimal(str(rows[fastest]["seconds"])) * (Decimal(1) + Decimal(str(policy["latency_tolerance"])))
        feasible = sorted(name for name in rows if Decimal(str(rows[name]["seconds"])) <= limit)
        selected = min(feasible, key=lambda name: (rows[name]["joules"], rows[name]["seconds"], name))
        energy_overlap = [name for name in feasible if name != selected and
                          rows[selected]["energy"]["maximum"] >= rows[name]["energy"]["minimum"]]
        result.update(status="selected", selected_id=selected, fastest_id=fastest,
                      runtime_limit_seconds=float(limit), feasible_ids=feasible,
                      runtime_ratio=rows[selected]["seconds"] / rows[fastest]["seconds"],
                      energy_ratio_to_fastest=rows[selected]["joules"] / rows[fastest]["joules"],
                      energy_range_overlaps=energy_overlap,
                      uncertainty="Observed ranges are not confidence intervals; fresh validation is required before caching",
                      evidence_fingerprint=fingerprint(report), comparison_fingerprint=report["comparison_fingerprint"])
        return portable(result)
    except (ValueError, TypeError, KeyError, ArithmeticError, AttributeError) as error:
        result.update(status="invalid", reason=f"{type(error).__name__}: {error}", selected_id=None)
        return result


def validate_repeat(initial_report, fresh_report, *, latency_tolerance=None, exclusion_policy=None):
    """Full-set repeat: require a fresh run and the same observed median winner."""
    initial = select_report(initial_report, latency_tolerance=latency_tolerance, exclusion_policy=exclusion_policy)
    fresh = select_report(fresh_report, latency_tolerance=latency_tolerance, exclusion_policy=exclusion_policy)
    result = {"status": "inconclusive", "reason": None, "initial": initial, "fresh": fresh,
              "selected_id": None, "statistical_significance": "not established",
              "setup_energy_and_break_even": "not measured; no net-energy savings claim"}
    if initial["status"] != "selected" or fresh["status"] != "selected":
        result["reason"] = "Both runs must contain a complete selectable comparison"
    elif not initial_report.get("run_id") or initial_report["run_id"] == fresh_report.get("run_id"):
        result["reason"] = "Validation must use a distinct measurement run"
    elif (initial_report["metadata"] != fresh_report["metadata"]
          or {k: v for k, v in initial_report["config"].items() if k != "seed"} !=
             {k: v for k, v in fresh_report["config"].items() if k != "seed"}
          or initial_report["energy_identity"] != fresh_report["energy_identity"]
          or stable_environment(initial_report["environment"]) != stable_environment(fresh_report["environment"])
          or initial["requested_ids"] != fresh["requested_ids"] or initial["exclusions"] != fresh["exclusions"]):
        result["reason"] = "Validation changed the experiment, requested set, GPU, protocol, or exclusions"
    elif initial["selected_id"] != fresh["selected_id"]:
        result["reason"] = "Fresh measurements chose a different winner; rerun or increase measurement windows"
    else:
        # Monotonic clock times are comparable because validation happens in
        # this process. Also reject overlapping/reused trials with a new run ID.
        first_ends = [t["end_monotonic_seconds"] for e in initial_report["candidates"].values()
                      for t in e["trials"] if t["status"] == "ok"]
        fresh_starts = [t["start_monotonic_seconds"] for e in fresh_report["candidates"].values()
                        for t in e["trials"] if t["status"] == "ok"]
        if not first_ends or not fresh_starts or min(fresh_starts) <= max(first_ends):
            result["reason"] = "Validation windows overlap the original run or reuse its trials"
        else:
            result.update(status="validated", selected_id=initial["selected_id"],
                          reason="Same median-energy winner meets the runtime bound in a fresh full-set repeat")
    return result
