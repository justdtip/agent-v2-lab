"""A number that does not say what kind of number it is cannot be written.

Three kinds: measured here, carried from the laptop, or declared before the run. A record that
mixes them reads as though every figure were equally solid, and this programme has already
spent a run on a projection quoted as a measurement and a comparison against a figure taken
under conditions the run was not in. The field survives being copied into a table; the sentence
around the number does not.

This covers the acceptance kit's own provenance helper, which is a different thing from
``local_llm_lab.provenance`` and is tested separately for that reason.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_ACCEPTANCE = Path(__file__).resolve().parents[1] / "research" / "acceptance"
if str(_ACCEPTANCE) not in sys.path:
    sys.path.insert(0, str(_ACCEPTANCE))

from provenance import EXPECTED, LAPTOP_BASIS, MEASURED_HERE, Measured, head  # noqa: E402


def test_a_measurement_taken_here_needs_no_basis_note() -> None:
    measured = Measured(7.88, MEASURED_HERE, unit="GiB")
    assert measured.as_dict() == {"value": 7.88, "basis": "measured-here", "unit": "GiB"}


def test_a_projection_must_say_what_it_rests_on() -> None:
    """The rule this repository paid for twice, as a constructor that refuses."""
    with pytest.raises(ValueError, match="cannot be learned from, only failed"):
        Measured(7.56, EXPECTED, unit="GiB")
    with pytest.raises(ValueError, match="cannot be learned from"):
        Measured(7.56, EXPECTED, basis="   ")

    stated = Measured(7.56, EXPECTED, basis="7.23 GiB text tensors + 256 MiB ranking + 80 MiB KV")
    assert stated.as_dict()["basis_note"].startswith("7.23 GiB text tensors")


def test_a_carried_laptop_figure_must_say_where_it_came_from() -> None:
    with pytest.raises(ValueError, match="whether it is comparable"):
        Measured(83.5, LAPTOP_BASIS, unit="s")
    carried = Measured(83.5, LAPTOP_BASIS, basis="calculate-0158 on an idle laptop", unit="s")
    assert carried.as_dict()["basis"] == "laptop-basis"


def test_a_kind_nobody_chose_cannot_be_written() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        Measured(1, "")
    with pytest.raises(ValueError, match="unknown kind"):
        Measured(1, "probably-fine")
    with pytest.raises(TypeError):
        Measured(1)  # no default: the kind is always a choice


def test_the_three_kinds_are_distinguishable_in_the_record() -> None:
    """The property that matters once the prose is gone: the kinds do not collapse."""
    rows = [
        Measured(7.88, MEASURED_HERE, unit="GiB").as_dict(),
        Measured(7.56, EXPECTED, basis="header + ranking block + KV", unit="GiB").as_dict(),
        Measured(83.5, LAPTOP_BASIS, basis="idle laptop", unit="s").as_dict(),
    ]
    assert [row["basis"] for row in rows] == ["measured-here", "expected", "laptop-basis"]
    assert all("value" in row for row in rows)


@dataclass
class _Identity:
    tree_hash: str = "a" * 40
    source_commit: str = "c" * 40
    diff_stat: str = "1 file changed"
    checkpoint_sha256: str = "cafe" * 8
    device: str = '{"device": "cuda"}'
    input_digest: str = "full"
    smoke: bool = False


def test_the_head_is_taken_from_the_resume_identity_and_not_gathered_again() -> None:
    """So what a record is bound to and what resume keys on cannot drift apart."""
    identity = _Identity()
    written = head(identity, records="stage2")
    assert written["tree_hash"] == identity.tree_hash
    assert written["checkpoint_sha256"] == identity.checkpoint_sha256
    assert written["device"] == identity.device
    assert written["diff_stat"] == "1 file changed", "the reader wants to know what was edited"
    assert written["records"] == "stage2"


def test_the_head_carries_the_smoke_flag_for_the_reader() -> None:
    assert head(_Identity(smoke=True))["smoke"] is True
