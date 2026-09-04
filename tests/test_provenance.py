from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import local_llm_lab.provenance as provenance
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


def test_git_metadata_uses_exact_commands_root_and_binary_diff_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    """Git provenance preserves exact binary patch bytes at its supplied root."""
    root = tmp_path / "project"
    raw_diff = b"diff --git a/a b/a\n\x00binary\xffpatch\n"
    calls = []
    responses = [b"topic\n", b"a" * 40 + b"\n", raw_diff]

    def fake_check_output(command, *, cwd):
        calls.append((command, cwd))
        return responses.pop(0)

    monkeypatch.setattr(provenance.subprocess, "check_output", fake_check_output)

    assert provenance._git_metadata(root) == {
        "branch": "topic",
        "commit": "a" * 40,
        "dirty_patch_sha256": hashlib.sha256(raw_diff).hexdigest(),
    }
    assert calls == [
        (["git", "branch", "--show-current"], root),
        (["git", "rev-parse", "HEAD"], root),
        (["git", "diff", "--binary", "HEAD", "--"], root),
    ]
