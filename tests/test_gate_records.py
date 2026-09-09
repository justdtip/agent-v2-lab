"""Resume is keyed on what was measured, and every refusal names the field that differs.

A paid hour that dies at gate five must not re-run gates one to four, and must not accept a
gate three measured on another tree, checkpoint, device or input. The second half is the one to
test hardest: a resume that is too generous silently reports a gate as passed on evidence from
somewhere else, and nothing downstream can tell.

Two corrections from the Chief are pinned here as tests rather than as prose. The key describes
the working tree instead of refusing a modified one, because a one-line edit on the device is
the expected case and voiding every record for it recreates the waste resume exists to prevent.
And the gate's input is part of the key, so one store holds both smoke and full records without
either resuming past the other.
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

from gate_records import GateRecords, Identity, input_digest  # noqa: E402


def _identity(**overrides) -> Identity:
    base = {
        "tree_hash": "a" * 40,
        "checkpoint_sha256": "cafe" * 8,
        "device": '{"backend": "torch", "device": "cpu"}',
        "input_digest": "full-fifteen",
        "source_commit": "c" * 40,
    }
    return Identity(**{**base, **overrides})


def _result(status: str = "pass") -> dict:
    return {"status": status, "saw": "15 episodes identical", "expected": "15 identical"}


def test_a_matching_record_is_reused(tmp_path: Path) -> None:
    GateRecords(tmp_path, _identity()).write(5, _result())
    stored = GateRecords(tmp_path, _identity()).completed(5)
    assert stored is not None and stored["status"] == "pass" and stored["gate"] == 5


def test_a_gate_with_no_record_is_simply_absent(tmp_path: Path) -> None:
    records = GateRecords(tmp_path, _identity())
    assert records.completed(5) is None
    assert 5 not in records.refusals, "no record is not a refusal and needs no explanation"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("tree_hash", "b" * 40, "working tree"),
        ("checkpoint_sha256", "beef" * 8, "checkpoint"),
        ("device", '{"backend": "torch", "device": "cuda"}', "device"),
        ("input_digest", "smoke-one", "gate input"),
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
        "that differs tells them whether to re-run or to check out the other tree"
    )


def test_a_modified_tree_resumes_its_own_records_and_not_another_tree_s(tmp_path: Path) -> None:
    """Describe the tree, do not refuse it.

    Refusing every record on a modified tree voids the operator's work for a one-line edit,
    which on the device is the expected case rather than the exception. Keying on the tree's
    content keeps what they measured and still cannot confuse two trees.
    """
    GateRecords(tmp_path, _identity(tree_hash="e" * 40)).write(2, _result())

    same_edit = GateRecords(tmp_path, _identity(tree_hash="e" * 40))
    assert same_edit.completed(2) is not None, "the same modified tree resumes its own record"

    other_tree = GateRecords(tmp_path, _identity(tree_hash="a" * 40))
    assert other_tree.completed(2) is None
    assert "working tree" in other_tree.refusals[2]


def test_an_unreadable_tree_is_refused_rather_than_assumed(tmp_path: Path) -> None:
    GateRecords(tmp_path, _identity()).write(2, _result())
    later = GateRecords(tmp_path, _identity(tree_hash="unknown"))
    assert later.completed(2) is None
    assert "could not be read" in later.refusals[2]


def test_a_record_written_by_an_unidentifiable_tree_is_never_reused(tmp_path: Path) -> None:
    GateRecords(tmp_path, _identity(tree_hash="unknown")).write(1, _result())
    later = GateRecords(tmp_path, _identity())
    assert later.completed(1) is None
    assert "not reusable" in later.refusals[1]


def test_a_full_pass_never_resumes_past_a_smoke_record(tmp_path: Path) -> None:
    """One store, with the input in the key, so this cannot happen by accident."""
    smoke = _identity(input_digest="smoke-one-episode", smoke=True)
    GateRecords(tmp_path, smoke).write(5, _result())

    full = GateRecords(tmp_path, _identity(input_digest="full-fifteen-episodes"))
    assert full.completed(5) is None
    assert "gate input" in full.refusals[5]

    again = GateRecords(tmp_path, smoke)
    assert again.completed(5) is not None, "a smoke pass still resumes its own smoke records"


def test_a_gate_whose_smallest_input_is_its_full_input_resumes_across_both(tmp_path: Path) -> None:
    """The other half of putting the input in the key: identical inputs are one key."""
    reads_records_only = _identity(input_digest="reads-the-records-only")
    GateRecords(tmp_path, reads_records_only).write(6, _result())
    assert GateRecords(tmp_path, replace(reads_records_only, smoke=True)).completed(6) is not None


def test_the_input_digest_is_stable_and_order_independent() -> None:
    assert input_digest(episodes=["a", "b"], layers=34) == input_digest(
        layers=34, episodes=["a", "b"]
    )
    assert input_digest(episodes=["a", "b"]) != input_digest(episodes=["a"])


def test_a_corrupt_record_is_refused_and_says_so(tmp_path: Path) -> None:
    records = GateRecords(tmp_path, _identity())
    records.write(4, _result())
    (tmp_path / "gate-04.json").write_text("{ this is not json")
    assert records.completed(4) is None
    assert "could not be read" in records.refusals[4]


def test_a_record_from_another_identity_shape_does_not_crash_the_reader(tmp_path: Path) -> None:
    """Fields this version does not know are ignored rather than raising mid-hour."""
    records = GateRecords(tmp_path, _identity())
    records.write(7, _result())
    stored = json.loads((tmp_path / "gate-07.json").read_text())
    stored["identity"]["some_field_from_a_later_version"] = True
    (tmp_path / "gate-07.json").write_text(json.dumps(stored))
    assert GateRecords(tmp_path, _identity()).completed(7) is not None


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
    theirs = replace(mine, tree_hash="c" * 40, device='{"device": "cuda"}')
    assert len(mine.differences(theirs)) == 2, (
        "an operator fixing one field should not then discover the next"
    )


def test_the_reader_only_fields_never_split_a_key() -> None:
    """Two runs on the same tree agree whatever the commit label or diff stat says."""
    mine = _identity()
    relabelled = replace(mine, source_commit="d" * 40, diff_stat="one file changed")
    assert mine.differences(relabelled) == []
