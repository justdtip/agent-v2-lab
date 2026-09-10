"""The read gate refuses, and each of its refusals is shown to be the one doing the work.

Every test that breaks something is paired with the intact case, because a gate that refuses
everything is as useless as one that refuses nothing (`METHOD-2026-09-08`, entries 23 and 27).
"""

from __future__ import annotations

import hashlib
import json
import pytest

from local_llm_lab import spawn

from local_llm_lab.pipeline.state_programme.read_gate import (
    NotSealed,
    SealBroken,
    load_seal,
    require_seal,
    seal_digest,
    verify_files,
)

CHECKER = '''
def check(document, *, verbose=True):
    return [] if "SOUND" in document else ["the document lost its claim"]
'''


def build(directory, *, baseline=None, document="SOUND\n"):
    """A minimal sealed record: a checker, a document, and a seal that fixes both."""
    (directory / "check_prereg.py").write_text(CHECKER)
    (directory / "PREREGISTRATION.md").write_text(document)
    (directory / "capture-set.jsonl").write_text('{"task_id": "t", "step": 0}\n')
    files = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("check_prereg.py", "PREREGISTRATION.md", "capture-set.jsonl")
    }
    seal = {
        "schema_version": 1,
        "commit": "0" * 40,
        "baseline_commit": baseline,
        "seed": 20260910,
        "folds": {"k": 5},
        "capture_set": {"lines": 1},
        "tolerances": {"m": 12},
        "files": files,
    }
    (directory / "seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    return seal


def test_an_intact_tree_passes_and_says_what_it_verified(tmp_path):
    build(tmp_path)
    got = require_seal(tmp_path)
    assert got["seed"] == 20260910
    assert got["verified"]["seal_sha256"] == seal_digest(tmp_path)
    assert "3 file(s)" in got["verified"]["files"]
    assert got["verified"]["document"] == "passes its own checker as sealed"


def test_no_seal_refuses_by_name(tmp_path):
    (tmp_path / "PREREGISTRATION.md").write_text("SOUND\n")
    with pytest.raises(NotSealed, match="precondition 5"):
        require_seal(tmp_path)


def test_a_partial_seal_is_no_seal(tmp_path):
    build(tmp_path)
    payload = json.loads((tmp_path / "seal.json").read_text())
    del payload["folds"]
    (tmp_path / "seal.json").write_text(json.dumps(payload))
    with pytest.raises(NotSealed, match="folds"):
        require_seal(tmp_path)


def test_a_seal_over_nothing_is_no_seal(tmp_path):
    build(tmp_path)
    payload = json.loads((tmp_path / "seal.json").read_text())
    payload["files"] = {}
    (tmp_path / "seal.json").write_text(json.dumps(payload))
    with pytest.raises(NotSealed, match="fixes no files"):
        require_seal(tmp_path)


def test_unreadable_seal_refuses(tmp_path):
    build(tmp_path)
    (tmp_path / "seal.json").write_text("{not json")
    with pytest.raises(NotSealed, match="not readable as JSON"):
        require_seal(tmp_path)


def test_a_different_seal_is_a_different_study(tmp_path):
    build(tmp_path)
    require_seal(tmp_path, expected_digest=seal_digest(tmp_path))  # the intact case
    with pytest.raises(SealBroken, match="different seal"):
        require_seal(tmp_path, expected_digest="0" * 64)


def test_a_moved_sealed_file_refuses_and_names_itself(tmp_path):
    build(tmp_path)
    (tmp_path / "capture-set.jsonl").write_text('{"task_id": "t", "step": 1}\n')
    with pytest.raises(SealBroken, match="capture-set.jsonl has moved"):
        require_seal(tmp_path)


def test_a_deleted_sealed_file_refuses(tmp_path):
    build(tmp_path)
    (tmp_path / "capture-set.jsonl").unlink()
    with pytest.raises(SealBroken, match="capture-set.jsonl is gone"):
        require_seal(tmp_path)


def test_every_broken_file_is_named_not_only_the_first(tmp_path):
    seal = build(tmp_path)
    (tmp_path / "capture-set.jsonl").write_text("moved\n")
    (tmp_path / "PREREGISTRATION.md").write_text("SOUND but moved\n")
    broken = verify_files(tmp_path, seal)
    assert len(broken) == 2, broken


def test_a_document_that_no_longer_passes_its_checker_refuses(tmp_path):
    build(tmp_path, document="the claim is gone\n")
    # The document's own digest still matches the seal, so only the checker can catch this.
    assert verify_files(tmp_path, load_seal(tmp_path)) == []
    with pytest.raises(SealBroken, match="no longer passes its checker"):
        require_seal(tmp_path)


def test_a_checker_that_cannot_run_is_a_refusal_not_a_pass(tmp_path):
    build(tmp_path)
    (tmp_path / "check_prereg.py").write_text("raise ImportError('no torch on this box')\n")
    payload = json.loads((tmp_path / "seal.json").read_text())
    payload["files"]["check_prereg.py"] = hashlib.sha256(
        (tmp_path / "check_prereg.py").read_bytes()).hexdigest()
    (tmp_path / "seal.json").write_text(json.dumps(payload))
    with pytest.raises(SealBroken, match="did not run"):
        require_seal(tmp_path)


def test_outside_a_work_tree_the_baseline_is_reported_not_silently_dropped(tmp_path):
    build(tmp_path, baseline="a" * 40)
    got = require_seal(tmp_path)
    assert "not a git work tree" in got["verified"]["baseline"]
    assert "was not re-derived" in got["verified"]["baseline"]


def test_no_baseline_named_says_so(tmp_path):
    build(tmp_path, baseline=None)
    got = require_seal(tmp_path)
    assert got["verified"]["baseline"] == "no baseline commit named in the seal; not checked"


def test_a_baseline_this_repository_does_not_have_refuses(tmp_path):
    spawn.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    build(tmp_path, baseline="b" * 40)
    with pytest.raises(SealBroken, match="does not have"):
        require_seal(tmp_path)


def test_the_baseline_is_a_second_path_to_the_bytes(tmp_path):
    spawn.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    spawn.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    spawn.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    build(tmp_path)
    spawn.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    spawn.run(["git", "-C", str(tmp_path), "commit", "-qm", "seal"], check=True)
    head = spawn.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    payload = json.loads((tmp_path / "seal.json").read_text())
    payload["baseline_commit"] = head
    (tmp_path / "seal.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    got = require_seal(tmp_path)  # the intact case: both paths agree
    assert got["verified"]["baseline"].startswith("re-derived against")

    # Now move a file and restore the seal's own digest for it, so ONLY the baseline path can catch it.
    (tmp_path / "capture-set.jsonl").write_text("moved after the commit\n")
    payload["files"]["capture-set.jsonl"] = hashlib.sha256(
        (tmp_path / "capture-set.jsonl").read_bytes()).hexdigest()
    (tmp_path / "seal.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assert verify_files(tmp_path, load_seal(tmp_path)) == []
    with pytest.raises(SealBroken, match="differs from its bytes at"):
        require_seal(tmp_path)
