from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

# `tests/` is not a package, so pytest puts it on `sys.path` and the suite's own conftest
# imports as a top-level module.
from conftest import PRE_EXISTING_FORK_SITES

import local_llm_lab.provenance as provenance
from local_llm_lab import runlock
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION
from local_llm_lab.provenance import source_tree_hashes, write_provenance


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_source_tree_hashes_include_only_deterministic_source_config_and_lock_files(
    tmp_path: Path,
) -> None:
    """The provenance source manifest is ordered and excludes unrelated or non-code files."""
    expected = {
        "configs/model.yaml": _write(tmp_path / "configs" / "model.yaml", b"model: fake\n"),
        "configs/nested/run.json": _write(tmp_path / "configs" / "nested" / "run.json", b"{}\n"),
        "src/pkg/a.py": _write(tmp_path / "src" / "pkg" / "a.py", b"VALUE = 1\n"),
        "uv.lock": _write(tmp_path / "uv.lock", b"lock\n"),
    }
    _write(tmp_path / "src" / "pkg" / "ignored.txt", b"ignore\n")
    _write(tmp_path / "unrelated.txt", b"ignore\n")

    hashes = source_tree_hashes(tmp_path)

    assert hashes == expected
    assert list(hashes) == sorted(expected)


@dataclass
class _Spec:
    name: str


class _Resolved:
    def as_dict(self) -> dict[str, str]:
        return {"name": "resolved"}


def test_write_provenance_is_deterministic_and_prefers_resolved_model_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    """The durable schema uses patched metadata seams and stable JSON bytes."""
    root = tmp_path / "project"
    root.mkdir()
    expected_hashes = {"src/pkg/a.py": "c" * 64, "uv.lock": "d" * 64}
    expected_git = {
        "branch": "codex/agent-v2-specs",
        "commit": "a" * 40,
        "dirty_patch_sha256": "b" * 64,
    }
    versions = {"mlx": "1.0", "mlx-lm": "2.0", "numpy": "3.0", "transformers": "4.0"}
    monkeypatch.setattr(provenance, "PROJECT_ROOT", root)
    monkeypatch.setattr(provenance, "source_tree_hashes", lambda root: expected_hashes)
    monkeypatch.setattr(provenance, "_git_metadata", lambda root: expected_git)
    monkeypatch.setattr(provenance, "version", lambda package: versions[package])
    monkeypatch.setattr(provenance.sys, "argv", ["agent-pipeline", "data"])

    target = write_provenance(
        tmp_path / "run", resolved=_Resolved(), spec=_Spec("fallback"), extra={"stage": "data"}
    )
    first = target.read_bytes()
    write_provenance(
        tmp_path / "run", resolved=_Resolved(), spec=_Spec("fallback"), extra={"stage": "data"}
    )

    assert target.read_bytes() == first
    assert json.loads(first) == {
        "command": ["agent-pipeline", "data"],
        "extra": {"stage": "data"},
        "generator_version": GENERATOR_VERSION,
        "git": expected_git,
        "model": {"name": "resolved"},
        "packages": versions,
        "source_tree_hashes": expected_hashes,
    }


def test_package_versions_use_none_only_for_absent_packages(monkeypatch) -> None:
    """Metadata lookup failures have a stable, explicit null representation."""
    versions = {"mlx-lm": "2.0", "numpy": "3.0", "transformers": "4.0"}

    def fake_version(package: str) -> str:
        if package == "mlx":
            raise PackageNotFoundError
        return versions[package]

    monkeypatch.setattr(provenance, "version", fake_version)

    assert provenance._package_versions() == {"mlx": None, **versions}


def test_git_metadata_names_its_root_in_argv_and_keeps_binary_diff_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    """The exact commands, the root carried as ``-C`` rather than ``cwd``, and raw patch bytes.

    The `cwd` half is the point of the assertion, not incidental tidiness: `cwd` is one of
    CPython's disqualifiers for `posix_spawn`, and `write_provenance` runs after the model load,
    so a call that passes it forks with Metal up and aborts the interpreter. Pinned on the argv
    here so that a revert to `cwd=root` fails on a line that says why, as well as tripping the
    fork guard in `test_a_provenance_write_after_a_load_does_not_fork`.
    """
    root = tmp_path / "project"
    raw_diff = b"diff --git a/a b/a\n\x00binary\xffpatch\n"
    calls: list[tuple[list[str], dict[str, object]]] = []
    responses = [b"topic\n", b"a" * 40 + b"\n", raw_diff]

    def fake_run(argv, **kwargs):
        calls.append(([str(part) for part in argv], kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=responses.pop(0))

    monkeypatch.setattr(provenance, "spawn_run", fake_run)

    assert provenance._git_metadata(root) == {
        "branch": "topic",
        "commit": "a" * 40,
        "dirty_patch_sha256": hashlib.sha256(raw_diff).hexdigest(),
    }
    assert [argv for argv, _ in calls] == [
        ["git", "-C", str(root), "branch", "--show-current"],
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
    ]
    # `check_output`'s contract, unchanged by the migration: raise on a non-zero status, and
    # leave stderr alone so a git error still reaches the run log.
    assert [kwargs for _, kwargs in calls] == [{"stdout": subprocess.PIPE, "check": True}] * 3


def test_a_provenance_write_after_a_load_does_not_fork(monkeypatch, tmp_path: Path) -> None:
    """Issue 84 item 1: the real git calls, in the production ordering, under the fork guard.

    Ordering is the whole hazard. `write_provenance` is the last thing every probe and `cli`
    stage does, so it runs with the model resident, Metal up, and the model-run lock held; a
    fork there is `SIGABRT`, `atexit` never runs, and the lock is orphaned -- then correctly
    reported stale and correctly never deleted, so one aborted probe locks the machine until a
    person clears it.

    So the test reproduces that ordering rather than calling `_git_metadata` on its own. The
    load goes through `runlock.load_weights`, the repository's only door to `mlx_lm.load`, with
    the loader itself faked -- no checkpoint is read and no weights exist -- because what has to
    be real here is the lock being held and the git calls being the genuine ones. Metal needs no
    help: pytest imports MLX at collection, so this interpreter is already the dangerous one,
    which is why the conftest guard is the right mechanism and a second one would be worse.

    The guard turns a fork in `provenance.py` into `ForkInThisInterpreter` on the line that
    caused it, so this test fails outright if the migration is reverted -- it does not need to
    assert on the exception. It asserts instead on what a reader can check: git really ran (a
    40-character commit that was not stubbed), and the lock survived the write.
    """
    assert "src/local_llm_lab/provenance.py" not in PRE_EXISTING_FORK_SITES, (
        "with the exemption back the guard lets provenance fork and this test proves nothing"
    )
    loader = types.ModuleType("mlx_lm")
    loader.load = lambda hf_id, **kwargs: (f"model:{hf_id}", "tokenizer")
    monkeypatch.setitem(sys.modules, "mlx_lm", loader)

    model, _tokenizer = runlock.load_weights("fake/checkpoint")
    lock = runlock.default_lock_path()
    assert model == "model:fake/checkpoint"
    assert lock.is_file(), "the faked load must still take the lock, or the ordering is not real"

    target = write_provenance(
        tmp_path / "run", resolved=None, spec=_Spec("fallback"), extra={"stage": "after-load"}
    )

    git = json.loads(target.read_text(encoding="utf-8"))["git"]
    assert len(git["commit"]) == 40 and set(git["commit"]) <= set("0123456789abcdef")
    assert len(git["dirty_patch_sha256"]) == 64
    assert lock.is_file(), "the write orphaned or dropped the lock the load was holding"


def test_a_provenance_record_is_whole_or_absent_when_the_write_is_interrupted(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue 92: the write is the failure `git_commit`'s contract does not cover.

    "Provenance is a nice-to-have and must never be the thing that kills a run" is about
    *collecting* it -- no git, not a repository, a timeout -- and those answer "unknown". A
    failure in the write is different in kind: it leaves a record that parses as nothing and
    reads on a listing as complete.
    """
    from test_branch import _interrupt_the_manifest_writer

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = '{"kept": true}\n'
    (run_dir / "provenance.json").write_text(original, encoding="utf-8")

    _interrupt_the_manifest_writer(monkeypatch, 14)
    with pytest.raises(OSError, match="no space left on device"):
        write_provenance(run_dir, resolved=None, spec=_Spec("fallback"), extra={"stage": "x"})
    monkeypatch.undo()

    assert (run_dir / "provenance.json").read_text(encoding="utf-8") == original
    leftovers = sorted(path.name for path in run_dir.iterdir() if path.name.startswith("."))
    assert leftovers == [], "no partial temporary file may survive at the destination"
