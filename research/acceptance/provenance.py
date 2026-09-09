"""Every number carries what kind of number it is, as a field rather than as a sentence.

Three kinds, and the difference between them is the whole point:

``measured-here``   taken by this run, on this device, from the thing named.
``laptop-basis``    measured on the laptop and carried forward for comparison. On the device it
                    is history, not a prediction, and it is expected to be re-measured.
``expected``        neither measured nor carried: a declaration made before the run, such as a
                    projected peak or a gate's target. A projection must also say what it rests
                    on, so ``basis`` is required for this kind and refused as empty.

The kinds exist because a record that mixes them reads as though every figure were equally
solid. This programme has already spent a run on a projection quoted as a measurement, and a
comparison against a figure taken under conditions the run was not in. Both are invisible once
the sentence around the number is gone, and a field survives being copied into a table where
the prose does not.

Construction refuses an unknown kind and there is **no default**. A number whose kind nobody
chose is the failure this module exists to prevent, so it cannot be written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MEASURED_HERE", "LAPTOP_BASIS", "EXPECTED", "Measured", "head"]

MEASURED_HERE = "measured-here"
LAPTOP_BASIS = "laptop-basis"
EXPECTED = "expected"

_KINDS = (MEASURED_HERE, LAPTOP_BASIS, EXPECTED)


@dataclass(frozen=True)
class Measured:
    """One number and what kind of number it is."""

    value: Any
    kind: str
    #: What it is a number *of*, and for a projection what it rests on. Required for
    #: ``expected``, because a projection without its basis cannot be learned from, only failed.
    basis: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; one of {_KINDS}")
        if self.kind == EXPECTED and not self.basis.strip():
            raise ValueError(
                "an expected value must say what it rests on: a projection without its basis "
                "cannot be learned from, only failed"
            )
        if self.kind == LAPTOP_BASIS and not self.basis.strip():
            raise ValueError(
                "a carried-forward laptop figure must say what it was measured on, or a later "
                "reader cannot tell whether it is comparable"
            )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"value": self.value, "basis": self.kind}
        if self.basis:
            payload["basis_note"] = self.basis
        if self.unit:
            payload["unit"] = self.unit
        return payload


def head(identity: Any, **extra: Any) -> dict[str, Any]:
    """The record's head: what ran, where, and against which weights.

    Taken from the resume identity rather than gathered again, so the thing a record is bound
    to and the thing resume keys on cannot drift apart.
    """
    return {
        "tree_hash": identity.tree_hash,
        "source_commit": identity.source_commit,
        "diff_stat": identity.diff_stat,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "device": identity.device,
        "input_digest": identity.input_digest,
        "smoke": identity.smoke,
        **extra,
    }
