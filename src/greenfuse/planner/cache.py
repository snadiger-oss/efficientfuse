"""Portable decisions and content-checked evidence; no executable deserialization."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

from greenfuse.results import fingerprint, portable
from .select import validate_repeat


def write_json_atomic(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".greenfuse-", delete=False) as out:
            temporary = Path(out.name)
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CacheLookup:
    status: str
    key: str
    reason: str | None = None
    decision: dict | None = None


class DecisionCache:
    def __init__(self, directory="results/decisions"):
        self.directory = Path(directory)

    def store(self, identity, initial_report, validation_report):
        identity = portable(identity)
        key = fingerprint(identity)
        if identity.get("schema_version") != 1 or identity.get("scope") != "gpu":
            raise ValueError("Only matching GPU identities can receive an energy decision")
        for report in (initial_report, validation_report):
            if report.get("metadata", {}).get("planner_identity") != identity or report["metadata"].get("planner_key") != key:
                raise ValueError("Evidence belongs to a different planner identity")
        policy = identity["policy"]
        validation = validate_repeat(initial_report, validation_report,
                                     latency_tolerance=policy["latency_tolerance"], exclusion_policy=policy["exclusion_policy"])
        if validation["status"] != "validated":
            raise ValueError(f"Decision is not validated: {validation['reason']}")
        selected_id = validation["selected_id"]
        plans = {c["id"]: c for c in identity["candidates"]}
        if selected_id not in plans:
            raise ValueError("Winner is absent from the requested plan manifest")
        evidence = {}
        for label, report in (("initial", initial_report), ("validation", validation_report)):
            digest = fingerprint(report)
            write_json_atomic(self.directory / "evidence" / f"{digest}.json", report)
            evidence[label] = digest
        payload = {"key": key, "identity": identity, "selected_id": selected_id, "plan": plans[selected_id],
                   "evidence": evidence, "validation": validation,
                   "created_utc": datetime.now(timezone.utc).isoformat()}
        write_json_atomic(self.directory / f"{key}.json", {"schema_version": 1, "checksum": fingerprint(payload), "payload": payload})
        return payload

    def lookup(self, identity):
        key = fingerprint(identity)
        path = self.directory / f"{key}.json"
        if not path.exists():
            return CacheLookup("miss", key, "No decision for this graph, input specialization, device, software and policy")
        try:
            envelope = json.loads(path.read_text())
            if envelope["schema_version"] != 1 or envelope["checksum"] != fingerprint(envelope["payload"]):
                raise ValueError("Cache schema/checksum mismatch")
            payload = envelope["payload"]
            if payload["key"] != key or payload["identity"] != portable(identity) or identity.get("scope") != "gpu":
                raise ValueError("Cache identity mismatch")
            reports = {}
            for label in ("initial", "validation"):
                digest = payload["evidence"][label]
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid evidence digest")
                reports[label] = json.loads((self.directory / "evidence" / f"{digest}.json").read_text())
                if fingerprint(reports[label]) != digest or reports[label]["metadata"]["planner_identity"] != portable(identity):
                    raise ValueError("Cached evidence has changed or belongs to another experiment")
            policy = identity["policy"]
            validation = validate_repeat(reports["initial"], reports["validation"],
                                         latency_tolerance=policy["latency_tolerance"], exclusion_policy=policy["exclusion_policy"])
            plans = {p["id"]: p for p in identity["candidates"]}
            if (validation["status"] != "validated" or validation != payload["validation"]
                    or validation["selected_id"] != payload["selected_id"]
                    or payload["plan"] != plans[payload["selected_id"]]):
                raise ValueError("Cached decision is inconsistent with its validated raw evidence")
            return CacheLookup("hit", key, decision=payload)
        except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as error:
            return CacheLookup("invalid", key, f"{type(error).__name__}: {error}")
