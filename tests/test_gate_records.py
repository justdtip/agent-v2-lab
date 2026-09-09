"""Resume is keyed on what was measured, and every refusal names the field that differs.

A paid hour that dies at gate five must not re-run gates one to four, and must not accept a
gate three that was measured on another tree, another checkpoint or another device. The second
half is the one worth testing hardest: a resume that is too generous silently reports a gate as
passed on evidence from somewhere else, and nothing downstream can tell.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_ACCEPTANCE = Path(__file__).resolve().parents[1] / "research" / "acceptance"
if str(_ACCEPTANCE) not in sys.path:
    sys.path.insert(0, str(_ACCEPTANCE))

from gate_records import GateRecords, Identity  # noqa: E402


def _identity(**overrides) -> Identity:
    base = {
        "source_commit": "a" * 40,
        "tree_dirty": False,
        "checkpoint_sha256": '{"model.safetensors": "cafe"}',
        "device": '{"backend": "torch", "device": "cpu"}',
    }
    return Identity(**{**base, **overrides})


def _result(status: str = "pass") -> dict:
    return {"status": status, "saw": "15 episodes identical", "expected": "15 identical"}


def test_a_matching_record_is_reused(tmp_path: Path) -> None:
    records = GateRecords(tmp_path, _identity())
    records.write(5, _result())
    stored = GateRecords(tmp_path, _identity()).completed(5)
    assert stored is not None and stored["status"] == "pass" and stored["gate"] == 5


def test_a_gate_with_no_record_is_simply_absent(tmp_path: Path) -> None:
    records = GateRecords(tmp_path, _identity())
    assert records.completed(5) is None
    assert 5 not in records.refusals, "no record is not a refusal and needs no explanation"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("source_commit", "b" * 40, "source commit"),
        ("checkpoint_sha256", '{"model.safetensors": "beef"}', "checkpoint"),
        ("device", '{"backend": "torch", "device": "cuda"}', "device"),
    ],
)
def test_a_record_measuring_something_else_is_refused_by_name(
    tmp_path: Path, field: str, value: str, expected: str
) -> None:
    GateRecords(tmp_path, _identity()).write(3, _result())
    later = GateRecords(tmp_path, _identity(**{field: value}))
    assert later.completed(3) is None
    assert expected in later.refusals[3], (
        "'not resumable' tells an operator nothing at minute fifty of a paid hour; the field "
        "that differs tells them whether to re-run or to check out the other commit"
    )


def test_a_record_taken_on_a_modified_tree_is_never_reused(tmp_path: Path) -> None:
    """The commit does not describe the code, so the key does not identify what ran."""
    GateRecords(tmp_path, _identity(tree_dirty=True)).write(2, _result())
    later = GateRecords(tmp_path, _identity(tree_dirty=False))
    assert later.completed(2) is None
    assert "working tree was modified" in later.refusals[2]


def test_a_run_on_a_modified_tree_resumes_nothing(tmp_path: Path) -> None:
    GateRecords(tmp_path, _identity()).write(2, _result())
    later = GateRecords(tmp_path, _identity(tree_dirty=True))
    assert later.completed(2) is None
    assert "cannot resume anything" in later.refusals[2]


def test_an_unreadable_tree_state_is_refused_rather_than_assumed_clean(tmp_path: Path) -> None:
    GateRecords(tmp_path, _identity()).write(2, _result())
    later = GateRecords(tmp_path, _identity(tree_dirty=None))
    assert later.completed(2) is None
    assert "could not be read" in later.refusals[2]


def test_an_unknown_commit_is_refused(tmp_path: Path) -> None:
    GateRecords(tmp_path, _identity(source_commit="unknown")).write(1, _result())
    later = GateRecords(tmp_path, _identity())
    assert later.completed(1) is None
    assert "not reusable" in later.refusals[1]


def test_a_corrupt_record_is_refused_and_says_so(tmp_path: Path) -> None:
    records = GateRecords(tmp_path, _identity())
    records.write(4, _result())
    (tmp_path / "gate-04.json").write_text("{ this is not json")
    assert records.completed(4) is None
    assert "could not be read" in records.refusals[4]


def test_each_gate_is_written_as_it_completes(tmp_path: Path) -> None:
    """Written and flushed one at a time; an interrupt keeps what finished."""
    records = GateRecords(tmp_path, _identity())
    records.write(1, _result())
    records.write(2, _result("fail"))
    on_disk = sorted(path.name for path in tmp_path.glob("gate-*.json"))
    assert on_disk == ["gate-01.json", "gate-02.json"]
    assert json.loads((tmp_path / "gate-02.json").read_text())["status"] == "fail"


def test_the_identity_reports_every_difference_not_only_the_first() -> None:
    mine = _identity()
    theirs = replace(mine, source_commit="c" * 40, device='{"device": "cuda"}')
    differences = mine.differences(theirs)
    assert len(differences) == 2, "an operator fixing one field should not discover the next"
