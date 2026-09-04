"""Deterministic, pre-load compatibility evidence for registered base models."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from math import ceil
from pathlib import Path
from typing import Any

from local_llm_lab.models import ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

__all__ = [
    "artifact_path",
    "cached_revision",
    "require_preflight",
    "run_preflight",
    "write_report",
]

_SCHEMA_VERSION = 1
_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "preflight"
_FIXED_PROMPT = "Explain why a model preflight check protects a local training run."
_FIXED_MESSAGES = [{"role": "user", "content": _FIXED_PROMPT}]
_THINKING_MODES = ("unsupported", "off", "inference", "trained")


def artifact_path(spec: ModelSpec, output_root: Path | None = None) -> Path:
    """Return the direct, stable evidence location for one registered model."""
    return (output_root or _OUTPUT_DIRECTORY) / f"{spec.name}.json"


def cached_revision(spec: ModelSpec) -> str | None:
    """Read the pinned local Hub revision without importing or loading a model."""
    configure_local_cache()
    safe_id = spec.hf_id.replace("/", "--")
    path = Path(os.environ["HF_HOME"]) / "hub" / f"models--{safe_id}" / "refs" / "main"
    try:
        revision = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return revision or None


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    """Persist stable preflight metadata without the array-report writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_preflight(
    model_name: str,
    *,
    loader: Callable[..., tuple[Any, Any]] | None = None,
    output_root: Path | None = None,
    spec_loader: Callable[[str], ModelSpec] | None = None,
    view_factory: Callable[[Any], Any] | None = None,
    resolver: Callable[[ModelSpec, Any, Any], ResolvedSpec] | None = None,
    jvp: Callable[..., Any] | None = None,
    revision_reader: Callable[[ModelSpec], str | None] | None = None,
    array_api: Any | None = None,
) -> Path:
    """Inspect one registered model and write a deterministic compatibility artifact.

    The optional collaborators make this model-loading stage wholly fake-testable.  The default
    loader and numerical runtime remain late imports so all ordinary CLI paths stay lightweight.
    """
    if spec_loader is None:
        spec_loader = load_model_spec
    spec = spec_loader(model_name)
    if loader is None:
        loader = _default_loader
    if view_factory is None:
        from local_llm_lab.arch import ArchitectureView

        view_factory = ArchitectureView.from_model
    if resolver is None:
        def resolver(declared: ModelSpec, model: Any, tokenizer: Any) -> ResolvedSpec:
            return declared.resolve(model, tokenizer)
    if jvp is None:
        from local_llm_lab.pipeline.jlens import jacobian_vector_product

        jvp = jacobian_vector_product
    if revision_reader is None:
        revision_reader = cached_revision
    if array_api is None:
        import mlx.core as mx

        array_api = mx

    model, tokenizer = loader(spec.hf_id, lazy=True)
    view = view_factory(model)
    resolved = resolver(spec, model, tokenizer)
    revision = resolved.snapshot_revision or revision_reader(spec)
    if not revision:
        raise SystemExit(
            f"no cached revision for {spec.name}; run agent-pipeline preflight --model {spec.name} "
            "only after the local snapshot is available"
        )

    token_ids = _fixed_token_ids(tokenizer, array_api)
    residual = _residual_equivalence(view, token_ids, array_api)
    jvp_result = _jvp_result(view, token_ids, jvp, array_api)
    prompts = _thinking_prompts(spec, tokenizer)
    parameter_bytes = _parameter_tree_bytes(model)
    activation_bytes = _activation_bytes(spec, view)
    total_bytes = parameter_bytes + activation_bytes
    total_gib = total_bytes / 1024**3
    cache = list(view.make_cache())
    layer_counts = _layer_counts(view)

    report = {
        "architecture": {
            "hidden_size": int(view.hidden_size),
            "layer_counts": layer_counts,
            "num_layers": int(view.num_layers),
            "tie_word_embeddings": bool(view.tie_word_embeddings),
            "vocab_size": int(view.vocab_size),
        },
        "cache": {
            "entry_types": [type(entry).__name__ for entry in cache],
            "strategy": resolved.cache_strategy,
            "strategy_reason": resolved.cache_strategy_reason,
        },
        "hf_id": spec.hf_id,
        "jvp": jvp_result,
        "lora": {
            "keys": list(resolved.lora_keys),
            "trainable_parameters": int(resolved.trainable_parameters),
        },
        "memory": {
            "activation_bytes": activation_bytes,
            "budget_gib": spec.memory_budget_gib,
            "parameter_bytes": parameter_bytes,
            "total_bytes": total_bytes,
            "total_gib": total_gib,
            "within_budget": total_gib <= spec.memory_budget_gib,
        },
        "model_name": spec.name,
        "passed": False,
        "residual_equivalence": residual,
        "schema_version": _SCHEMA_VERSION,
        "snapshot_revision": revision,
        "thinking_prompts": prompts,
    }
    report["passed"] = bool(
        residual["passed"] and jvp_result["finite"] and report["memory"]["within_budget"]
    )
    return write_report(report, artifact_path(spec, output_root))


def require_preflight(
    spec: ModelSpec,
    *,
    skip: bool = False,
    output_root: Path | None = None,
    revision_reader: Callable[[ModelSpec], str | None] | None = None,
    action: Callable[[], Any] | None = None,
) -> dict[str, Any] | None:
    """Reject stale, malformed, or failed evidence before callers can load a model.

    ``action`` is deliberately called only after validation and exists solely as a narrow test
    seam proving the guard's order.  Production callers leave it unset.
    """
    if skip:
        return None
    if revision_reader is None:
        revision_reader = cached_revision
    path = artifact_path(spec, output_root)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _reject(spec, "preflight artifact is missing")
    except (OSError, json.JSONDecodeError):
        _reject(spec, "preflight artifact is malformed or unreadable")
    if not isinstance(record, dict):
        _reject(spec, "preflight artifact is not a JSON object")
    if record.get("model_name") != spec.name:
        _reject(spec, "preflight artifact model name does not match the registered model")
    if record.get("hf_id") != spec.hf_id:
        _reject(spec, "preflight artifact hf_id does not match the registered model")
    revision = revision_reader(spec)
    if not revision:
        _reject(spec, "cached snapshot revision is absent")
    if record.get("snapshot_revision") != revision:
        _reject(spec, "preflight artifact snapshot revision is stale")
    if record.get("passed") is not True:
        _reject(spec, "preflight did not pass")
    memory = record.get("memory")
    if not isinstance(memory, dict) or memory.get("within_budget") is not True:
        _reject(spec, "preflight memory budget did not pass")
    if action is not None:
        action()
    return record


def _default_loader(hf_id: str, *, lazy: bool) -> tuple[Any, Any]:
    configure_local_cache()
    from mlx_lm import load

    return load(hf_id, lazy=lazy)


def _fixed_token_ids(tokenizer: Any, array_api: Any) -> Any:
    try:
        ids = list(tokenizer.encode(_FIXED_PROMPT, add_special_tokens=False))
    except TypeError:
        ids = list(tokenizer.encode(_FIXED_PROMPT))
    if not ids:
        raise ValueError("the preflight prompt tokenized to no token ids")
    ids = (ids * ceil(64 / len(ids)))[:64]
    return array_api.array(ids).astype(array_api.int32)[None, :]


def _residual_equivalence(view: Any, ids: Any, array_api: Any) -> dict[str, Any]:
    manual = view.embed(ids)
    masks = view.masks(manual, None)
    for index in range(view.num_layers):
        manual = view.run_block(index, manual, masks, None)
    manual = view.final_norm(manual)
    native = _native_final_residual(view, ids)
    error = _scalar(array_api.max(array_api.abs(manual - native)))
    return {"max_abs_error": error, "passed": error == 0.0, "token_count": 64}


def _native_final_residual(view: Any, ids: Any) -> Any:
    native = view.text_module(ids)
    if hasattr(native, "last_hidden_state"):
        return native.last_hidden_state
    if isinstance(native, tuple):
        return native[0]
    return native


def _jvp_result(view: Any, ids: Any, jvp: Callable[..., Any], array_api: Any) -> dict[str, Any]:
    layer = view.num_layers // 2
    primal = view.residuals(ids, [layer])[layer].astype(array_api.float32)
    tangent = array_api.ones_like(primal).astype(array_api.float32)
    method = "forward"
    try:
        result = jvp(view, layer, primal, tangent, method=method)
        finite = _is_finite(result, array_api)
    except Exception:
        finite = False
    if not finite:
        method = "finite_difference"
        result = jvp(view, layer, primal, tangent, method=method)
        finite = _is_finite(result, array_api)
    if not finite:
        raise SystemExit("preflight JVP is non-finite after finite-difference fallback")
    return {"finite": True, "layer": layer, "method": method}


def _thinking_prompts(spec: ModelSpec, tokenizer: Any) -> list[dict[str, Any]]:
    prompts = []
    for mode in _THINKING_MODES:
        kwargs = dict(spec.chat.template_kwargs)
        if mode == "unsupported":
            kwargs.pop("enable_thinking", None)
        else:
            kwargs["enable_thinking"] = mode in {"inference", "trained"}
        prompt = tokenizer.apply_chat_template(
            _FIXED_MESSAGES, add_generation_prompt=True, tokenize=False, **kwargs
        )
        try:
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
        except TypeError:
            count = len(tokenizer.encode(prompt))
        prompts.append({"mode": mode, "prompt": prompt, "token_count": count})
    return prompts


def _parameter_tree_bytes(model: Any) -> int:
    """Count materialized parameter buffers without evaluating or dequantizing them."""
    parameters = model.parameters() if callable(getattr(model, "parameters", None)) else model
    return sum(_leaf_nbytes(value) for value in _tree_values(parameters))


def _tree_values(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _tree_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tree_values(child)
    else:
        yield value


def _leaf_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    size = getattr(value, "size", None)
    itemsize = getattr(value, "itemsize", None)
    if size is not None and itemsize is not None:
        return int(size) * int(itemsize)
    raise ValueError(f"parameter leaf {type(value).__name__} has no byte size")


def _activation_bytes(spec: ModelSpec, view: Any) -> int:
    batch_size = int(spec.train["batch_size"])
    max_seq_length = int(spec.train["max_seq_length"])
    return batch_size * max_seq_length * int(view.hidden_size) * int(view.num_layers) * 4


def _layer_counts(view: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in range(view.num_layers):
        kind = str(view.layer_kind(index))
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _is_finite(value: Any, array_api: Any) -> bool:
    return bool(_scalar(array_api.all(array_api.isfinite(value))))


def _scalar(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _reject(spec: ModelSpec, reason: str) -> None:
    raise SystemExit(
        f"{reason} for {spec.name}; run agent-pipeline preflight --model {spec.name} "
        "or use --skip-preflight-check to override"
    )
