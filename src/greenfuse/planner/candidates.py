"""Combine Person 1's partitions with the requested execution modes."""

from dataclasses import dataclass

from greenfuse.partition.enumerate import PartitionSpec
from greenfuse.results import fingerprint


@dataclass(frozen=True)
class CandidateSpec:
    partition: PartitionSpec
    mode: str
    experiment_fingerprint: str

    @property
    def id(self):
        return f"{self.partition.id}_{self.mode}"

    @property
    def key(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return {"id": self.id, "partition": self.partition.to_dict(), "mode": self.mode,
                "experiment_fingerprint": self.experiment_fingerprint}


def combine_candidates(partitions, modes, experiment_fingerprint):
    partitions, modes = tuple(partitions), tuple(modes)
    if not partitions or len({p.id for p in partitions}) != len(partitions):
        raise ValueError("Require nonempty unique partitions")
    if len({p.graph_fingerprint for p in partitions}) != 1:
        raise ValueError("Cannot mix partitions from different graphs")
    if not modes or len(set(modes)) != len(modes) or any(m not in ("launch", "graph") for m in modes):
        raise ValueError("Require unique launch/graph modes")
    if not isinstance(experiment_fingerprint, str) or not experiment_fingerprint:
        raise ValueError("Experiment identity is required")
    return tuple(CandidateSpec(p, mode, experiment_fingerprint) for p in partitions for mode in modes)
