from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from local_llm_lab.models import ModelSpec, ResolvedSpec
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION
from local_llm_lab.project import PROJECT_ROOT

_PACKAGES = ("mlx", "mlx-lm", "numpy", "transformers")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hashes(root: Path) -> dict[str, str]:
    """Hash the deterministic source, configuration, and dependency-lock inputs."""
    candidates = []
    src = root / "src"
    if src.is_dir():
        candidates.extend(path for path in src.rglob("*.py") if path.is_file())
    configs = root / "configs"
    if configs.is_dir():
        candidates.extend(path for path in configs.rglob("*") if path.is_file())
    lock = root / "uv.lock"
    if lock.is_file():
        candidates.append(lock)
    ordered = sorted(candidates, key=lambda path: path.relative_to(root).as_posix())
    return {path.relative_to(root).as_posix(): _sha256(path) for path in ordered}


def _git_metadata(root: Path) -> dict[str, str]:
    """Read the revision, branch, and complete working-tree patch from one repository root."""

    def command(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=root)

    return {
        "branch": command("branch", "--show-current").decode().strip(),
        "commit": command("rev-parse", "HEAD").decode().strip(),
        "dirty_patch_sha256": hashlib.sha256(command("diff", "--binary", "HEAD", "--")).hexdigest(),
    }


def _package_versions() -> dict[str, str | None]:
    """Return installed package versions, with absent packages represented explicitly."""
    packages = {}
    for package in _PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def write_provenance(
    run_dir: Path,
    *,
    resolved: ResolvedSpec | None,
    spec: ModelSpec,
    extra: dict[str, Any],
) -> Path:
    payload = {
        "command": list(sys.argv),
        "extra": extra,
        "generator_version": GENERATOR_VERSION,
        "git": _git_metadata(PROJECT_ROOT),
        "model": resolved.as_dict() if resolved is not None else asdict(spec),
        "packages": _package_versions(),
        "source_tree_hashes": source_tree_hashes(PROJECT_ROOT),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "provenance.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
