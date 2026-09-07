"""Pure preflight and one process-scoped loader for standalone lens tools.

Resolve a cached snapshot before importing MLX and pass that exact directory through
load_policy. The original repository ID and immutable revision remain in provenance.
No adapter entry point is exposed before requirements 3.1–3.6 land.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from local_llm_lab.models import ModelSpec
from local_llm_lab.pipeline.lens_fitting.artifacts import validate_output
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus
from local_llm_lab.pipeline.live_lens.instruments import file_sha256
from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.spawn import run

# The installed mlx_lm.utils._download asset classes. Offline completeness must not
# require unrelated Hub documentation; the same set defines hashed runtime assets.
SNAPSHOT_ALLOW_PATTERNS = (
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)


def primary_worktree() -> Path:
    """Git's first worktree owns the shared lock, including from linked checkouts."""
    result = run(
        ["git", "-C", str(PROJECT_ROOT), "worktree", "list", "--porcelain", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    first = result.stdout.split("\0", 1)[0]
    if not first.startswith("worktree "):
        raise ValueError("cannot resolve primary worktree for the model-run lock")
    root = Path(first.removeprefix("worktree "))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("invalid primary worktree path")
    return root


def snapshot_identity(directory: Path, *, hf_id: str) -> dict:
    """Hash the exact MLX config/shards plus optional config/index/code assets.

    MLX loads every model*.safetensors file; index references must also all exist.
    An explicit local model has no invented Hub revision. Cached Hub models require
    a repository-matching snapshots/<commit> path; no newest-snapshot inference.
    """
    directory = Path(directory).absolute()
    revision = None
    if directory.parent.name == "snapshots":
        if directory.parent.parent.name != "models--" + hf_id.replace("/", "--"):
            raise ValueError("snapshot repository does not match declared model")
        revision = directory.name
        if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError("snapshot revision must be an immutable commit")
    elif not Path(hf_id).is_dir() or Path(hf_id).resolve() != directory.resolve():
        raise ValueError("Hub model requires a repository-matching immutable snapshot path")
    config = directory / "config.json"
    data = json.loads(config.read_bytes())
    shards = sorted(directory.glob("model*.safetensors"))
    if not shards or any(not path.is_file() for path in shards):
        raise ValueError("model snapshot has no complete weight shards")
    assets = {config, *shards}
    for index in directory.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_bytes()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("invalid weight shard index")
        names = set(mapping.values())
        if not names <= {path.name for path in shards}:
            raise ValueError("snapshot is missing index-referenced weight shards")
        assets.add(index)
    # Include tokenizer, generation settings, quantization overrides and custom model code.
    assets.update(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and any(
            fnmatch(path.relative_to(directory).as_posix(), pattern)
            for pattern in SNAPSHOT_ALLOW_PATTERNS
        )
    )
    if data.get("model_file"):
        model_file = directory / data["model_file"]
        if model_file.parent != directory or not model_file.is_file():
            raise ValueError("custom model config references a missing/outside model file")
        assets.add(model_file)
    files = [
        {
            "name": path.relative_to(directory).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(assets)
    ]
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "hf_id": hf_id,
        "resolved_revision": revision,
        "snapshot_path": str(directory),
        "snapshot_sha256": digest,
        "files": files,
        "identity_basis": "exact config, model*.safetensors, indices and runtime assets",
    }


def resolve_snapshot(spec: ModelSpec, *, revision: str = "main") -> dict:
    """Offline only: a lens run never downloads or guesses a different checkpoint."""
    if Path(spec.hf_id).is_dir():
        directory = Path(spec.hf_id)
    else:
        from huggingface_hub import snapshot_download

        cache_home = Path(os.environ.get("HF_HOME", primary_worktree() / ".cache/huggingface"))
        cache_dir = Path(os.environ.get("HF_HUB_CACHE", cache_home / "hub"))
        directory = Path(
            snapshot_download(
                spec.hf_id,
                revision=revision,
                local_files_only=True,
                cache_dir=str(cache_dir),
                allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS),
            )
        )
    return snapshot_identity(directory, hf_id=spec.hf_id)


@dataclass(frozen=True)
class PreparedFit:
    spec: ModelSpec
    rows: list[dict]
    manifest: dict
    corpus_manifest_sha256: str
    output: Path
    snapshot: dict


def prepare_fit(
    corpus_path: Path,
    spec: ModelSpec,
    output: Path,
    *,
    kind: str = "regression",
    revision: str = "main",
) -> PreparedFit:
    """Validate immutable inputs/output before any loader or Metal import."""
    output = validate_output(output)
    before = file_sha256(Path(corpus_path))
    rows = read_corpus(corpus_path)
    manifest = json.loads(Path(corpus_path).read_bytes())
    if before != file_sha256(Path(corpus_path)):
        raise ValueError("corpus manifest changed during validation")
    if manifest["model_hf_id"] != spec.hf_id:
        raise ValueError("corpus model does not match requested model")
    if not rows or {row["split"] for row in rows} != {"fit", "held"}:
        raise ValueError("corpus needs both fit and held selection sequences")
    name = spec.name.replace("/", "--")
    if any(part not in output.stem for part in (name, manifest["domain"], kind)):
        raise ValueError("output name must include model, domain and kind")
    identity = resolve_snapshot(spec, revision=revision)
    config = json.loads((Path(identity["snapshot_path"]) / "config.json").read_bytes())
    text_config = config.get("text_config", config)
    vocab = text_config.get("vocab_size")
    if vocab is not None and any(token >= vocab for row in rows for token in row["ids"]):
        raise ValueError("corpus token exceeds the model vocabulary")
    return PreparedFit(spec, rows, manifest, before, output, identity)


def _load_policy(spec: ModelSpec, adapter: None) -> tuple:
    from local_llm_lab.pipeline.evaluate import load_policy

    return load_policy(spec, adapter)


@dataclass(frozen=True)
class LoadedRuntime:
    model: Any
    tokenizer: Any
    view: Any
    resolved: Any
    spec: ModelSpec
    snapshot: dict
    lock_path: Path


def load_runtime(prepared: Any, *, capture: bool = False) -> LoadedRuntime:
    """Keep the primary lock until process exit; capture fixes none before loading.

    runlock imported PROJECT_ROOT by value. Bind only its process-local lock root so
    its subsequent load_weights acquisition is idempotent at the primary path.
    project.PROJECT_ROOT and ordinary data/output paths remain worktree-relative.
    """
    from local_llm_lab import runlock

    root = primary_worktree()
    runlock.PROJECT_ROOT = root
    lock_path = root / runlock.LOCK_RELATIVE_PATH
    runlock.hold_model_run_lock(path=lock_path)
    identity = prepared.snapshot
    directory = Path(identity["snapshot_path"])
    if snapshot_identity(directory, hf_id=prepared.spec.hf_id) != identity:
        raise ValueError("snapshot changed since preflight")
    spec = replace(prepared.spec, cache_strategy="none") if capture else prepared.spec
    local_spec = replace(spec, hf_id=str(directory))
    model, tokenizer, view, resolved = _load_policy(local_spec, None)
    if snapshot_identity(directory, hf_id=spec.hf_id) != identity:
        raise ValueError("snapshot changed during loading")
    resolved = replace(resolved, spec=spec, snapshot_revision=identity["resolved_revision"])
    return LoadedRuntime(model, tokenizer, view, resolved, spec, identity, lock_path)


def configure_allocator_cache(*, set_limit=None) -> dict:
    """Match fit and preflight allocation caching; this does not bound live buffers."""
    if set_limit is None:
        import mlx.core as mx

        set_limit = mx.set_cache_limit
    previous = set_limit(0)
    return {"previous_limit_bytes": previous, "limit_bytes": 0}


def resource_snapshot(*, array_api=None) -> dict:
    """R46 process peak (including load) against the runtime's device working set."""
    if array_api is None:
        import mlx.core as array_api
    peak = array_api.get_peak_memory()
    working = array_api.device_info()["max_recommended_working_set_size"]
    return {"peak_memory_gib": peak / 2**30, "working_set_share": peak / working}
