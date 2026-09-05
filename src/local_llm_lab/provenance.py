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
from local_llm_lab.spawn import run as spawn_run

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
    """Read the revision, branch, and complete working-tree patch from one repository root.

    ``root`` is passed as ``git -C <root>`` **in argv, never as ``cwd``**, and ``git`` is
    resolved to an absolute path by ``spawn.run``. Both spellings matter and both are load-
    bearing: CPython takes ``posix_spawn`` only when ``cwd`` is ``None`` *and* the program has a
    directory component, so either shortcut on its own returns this call to the ``fork`` path
    (R45; ``local_llm_lab.spawn`` carries the mechanism and the crash report).

    Why that is fatal here rather than merely untidy: ``write_provenance`` runs at the *end* of
    every probe and ``cli`` stage, after ``load_policy``, so Metal is up and the model-run lock
    is held. A fork in that interpreter aborts in libplatform -- ``SIGABRT``, which means
    ``atexit`` never runs, which means the lock is orphaned, correctly reported stale, and
    correctly never deleted, so one aborted probe leaves the machine locked until a person
    clears it. ``cwd=root`` is the shorter spelling and reads like a simplification; the stack
    trace it buys names ``AGX::PooledAllocator::shrink`` and does not mention this file.
    """

    def command(*args: str) -> bytes:
        # `stdout=PIPE, check=True` is `subprocess.check_output`'s exact contract, kept so the
        # migration changes the spawn path and nothing else: stderr still reaches the run log.
        argv = ["git", "-C", str(root), *args]
        return spawn_run(argv, stdout=subprocess.PIPE, check=True).stdout

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
