"""Portable graph records. Live FX modules belong in capture.py, not here."""

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class ValueRef:
    """A reference to an input or an earlier operation, not its tensor value."""

    id: str


Argument = ValueRef | int | float | bool | None


def argument_to_dict(value: Argument):
    if isinstance(value, ValueRef):
        return {"ref": value.id}
    return value


@dataclass(frozen=True)
class InputBinding:
    id: str
    name: str


@dataclass(frozen=True)
class Operation:
    id: str
    kind: str
    target: str
    args: tuple[Argument, ...]
    kwargs: tuple[tuple[str, Argument], ...] = ()

    @property
    def dependencies(self) -> tuple[str, ...]:
        values = (*self.args, *(value for _, value in self.kwargs))
        return tuple(dict.fromkeys(value.id for value in values if isinstance(value, ValueRef)))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "args": [argument_to_dict(value) for value in self.args],
            "kwargs": {name: argument_to_dict(value) for name, value in self.kwargs},
        }


@dataclass(frozen=True)
class NormalizedGraph:
    inputs: tuple[InputBinding, ...]
    operations: tuple[Operation, ...]
    output: ValueRef
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "inputs": [{"id": item.id, "name": item.name} for item in self.inputs],
            "operations": [operation.to_dict() for operation in self.operations],
            "output": argument_to_dict(self.output),
        }

    @property
    def fingerprint(self) -> str:
        """Graph identity only; input/device/compiler identity is separate."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()
