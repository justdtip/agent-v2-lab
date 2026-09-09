"""Load the text decoder of an HF checkpoint on torch: text-only, model-agnostic, failing closed.

Two checkpoint shapes exist. A plain causal LM keeps its tensors at the root. A multimodal wrapper
keeps its text tower under ``language_model.`` beside other towers (a vision encoder, a projector),
and its ``config.json`` carries the text decoder's config under ``text_config``. The wrapper case
is detected from the checkpoint's own headers and config, never from a model name, and no model
class is named here: the text config goes through ``AutoConfig`` and the model through
``AutoModelForCausalLM``, so what loads is whatever the checkpoint says it is (§16.3).

Every check fails closed, in the order a silent defect would otherwise pass: a text tensor the
model did not take, a shape that changed in transit, an unexpected key that is not one of the
checkpoint's own non-text towers, a tied weight that came back as two tensors, a parameter on the
wrong device or in the wrong dtype. The report is a reading of the loaded object, not an echo of
the arguments.

Lifted from Codex's WS-A record companion (``cpu_gates.py``), which fixed Gemma 3's classes and a
listed vision-prefix set; both are forbidden in the package.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

__all__ = ["TEXT_PREFIX", "checkpoint_metadata", "load_text_causal_lm"]

#: Where HF's multimodal wrappers keep the text decoder's tensors.
TEXT_PREFIX = "language_model."

_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}  # fmt: skip


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _headers(shard: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """One shard's tensor entries and its ``__metadata__``, without reading a tensor."""
    with shard.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError(f"truncated safetensors header: {shard.name}")
        header = json.loads(stream.read(int.from_bytes(size, "little")))
    metadata = header.pop("__metadata__", None) or {}
    return header, metadata


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Read the checkpoint's config and safetensors headers; load no tensor.

    Returns the config, the text config, whether the checkpoint is a wrapper, every tensor's
    header entry, the text tensors keyed as the text model names them, the non-text keys, the
    storage dtypes seen, the safetensors format, the text tower's stored bytes, and a sha256 per
    file. A format other than PyTorch's is refused here, by name.
    """
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint must be an existing local directory: {root}")
    config = json.loads((root / "config.json").read_text())
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"checkpoint has no local safetensors shards: {root}")
    tensors: dict[str, dict] = {}
    hashes = {"config.json": _sha256(root / "config.json")}
    formats: set[str] = set()
    for shard in shards:
        hashes[shard.name] = _sha256(shard)
        header, metadata = _headers(shard)
        formats.add(str(metadata.get("format", "")))
        for name, entry in header.items():
            if name in tensors:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            tensors[name] = entry
    if formats - {"pt", ""}:
        # The repository's own MLX conversion declares the same architecture and text_config
        # as the official snapshot and keeps its tensors under the same prefix, so nothing in
        # the config distinguishes them; the safetensors metadata does. It is refused by name,
        # not by a missing-keys puzzle, because the torch path loads the registry entry whose
        # hf_id is the official snapshot, never a conversion made for another runtime.
        raise ValueError(
            f"checkpoint safetensors declare format {sorted(formats - {'pt', ''})}, not a "
            "PyTorch checkpoint; the torch path loads the official snapshot, not a conversion "
            f"made for another runtime: {root}"
        )
    wrapper = "text_config" in config
    if wrapper:
        text = {
            name.removeprefix(TEXT_PREFIX): entry
            for name, entry in tensors.items()
            if name.startswith(TEXT_PREFIX)
        }
        other = {name for name in tensors if not name.startswith(TEXT_PREFIX)}
        if not text:
            raise ValueError(
                f"config declares a text_config but no tensor is under {TEXT_PREFIX!r}: {root}"
            )
    else:
        text, other = dict(tensors), set()
    dtypes = sorted({entry["dtype"] for entry in text.values()})
    unknown = [dtype for dtype in dtypes if dtype not in _BYTES]
    if unknown:
        raise ValueError(f"unknown safetensors dtype(s) in the text tower: {unknown}")
    return {
        "path": str(root),
        "config": config,
        "text_config": config["text_config"] if wrapper else config,
        "wrapper": wrapper,
        "tensors": tensors,
        "text": text,
        "other": other,
        "other_prefixes": sorted({name.split(".", 1)[0] for name in other}),
        "storage_dtypes": dtypes,
        "safetensors_format": sorted(formats),
        "text_bytes": sum(
            math.prod(entry["shape"]) * _BYTES[entry["dtype"]] for entry in text.values()
        ),
        "sha256": hashes,
    }


def load_text_causal_lm(
    path: str | Path,
    *,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load the text decoder through HF's own loading, then fail closed on every gap.

    ``dtype`` and ``attn_implementation`` are what the model is asked for; the report says what
    it got. The model comes back in eval mode with gradients off, on ``device``. Loading goes
    through the CPU and then moves, which doubles the transient for a CUDA load; the memory rung
    for that is a ``device_map`` under ``accelerate`` and is deliberately not taken here.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    meta = checkpoint_metadata(path)
    torch_dtype = getattr(torch, dtype)
    if meta["wrapper"]:
        text_config = dict(meta["text_config"])
        model_type = text_config.pop("model_type")
        config = AutoConfig.for_model(model_type, **text_config)
    else:
        config = AutoConfig.from_pretrained(meta["path"], local_files_only=True)
    key_mapping = {rf"^{re.escape(TEXT_PREFIX)}": ""} if meta["wrapper"] else None
    kwargs: dict[str, Any] = dict(
        config=config,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        output_loading_info=True,
    )
    if key_mapping:
        kwargs["key_mapping"] = key_mapping
    try:
        model, info = AutoModelForCausalLM.from_pretrained(meta["path"], **kwargs)
    except RuntimeError as error:
        # transformers 5 raises on a size mismatch before it can report it; one type out.
        raise ValueError(
            "text checkpoint load failed, a tensor did not match the model it was loaded into "
            f"(shape or dtype mismatch; HF's load report names it): {error}"
        ) from error

    for field in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors"):
        if info.get(field):
            raise ValueError(f"text checkpoint load failed: {field}={info[field]}")
    unexpected = set(info.get("unexpected_keys", ()))
    if unexpected != meta["other"]:
        raise ValueError(
            "unexpected keys are not exactly the checkpoint's non-text tensors: "
            f"extra={sorted(unexpected - meta['other'])} "
            f"missing={sorted(meta['other'] - unexpected)}"
        )
    state = model.state_dict()
    for key, entry in meta["text"].items():
        if key not in state or tuple(state[key].shape) != tuple(entry["shape"]):
            raise ValueError(f"text tensor missing or shape changed: {key}")
    tied = dict(getattr(model, "all_tied_weights_keys", None) or {})
    uncovered = set(state) - set(meta["text"])
    if any(key not in tied or tied[key] not in meta["text"] for key in uncovered):
        raise ValueError(f"model holds text tensors the checkpoint does not: {sorted(uncovered)}")
    for target, source in tied.items():
        if state[target].data_ptr() != state[source].data_ptr():
            raise ValueError(f"weight tying was not restored: {target} is not {source}")
    if device != "cpu":
        model.to(device)
    devices = {p.device.type for p in model.parameters()}
    dtypes = {str(p.dtype).removeprefix("torch.") for p in model.parameters()}
    if devices != {torch.device(device).type} or dtypes != {dtype}:
        raise ValueError(f"loaded parameters are on {sorted(devices)} in {sorted(dtypes)}")
    model.eval().requires_grad_(False)
    # `save_pretrained(save_original_format=True)`, the default, reverts the conversions the
    # load applied, read from this attribute; a wrapper loaded under the prefix mapping would
    # save `language_model.model.*` tensors under a config that names the causal LM, an artefact
    # nothing reads (SWE-2, 7f5c6e4). Cleared, the save recomputes only the architecture's own
    # conversions and drops the prefix change: the model saves in its own layout, and what it
    # saves loads back through this function as a plain checkpoint.
    model._weight_conversions = None

    report = {
        "path": meta["path"],
        "wrapper": meta["wrapper"],
        "model_type": config.model_type,
        "architecture": type(model).__name__,
        "key_mapping": key_mapping,
        "other_prefixes": meta["other_prefixes"],
        "unexpected_keys": sorted(unexpected),
        "storage_dtypes": meta["storage_dtypes"],
        "safetensors_format": meta["safetensors_format"],
        "text_bytes": meta["text_bytes"],
        "checkpoint_text_tensors": len(meta["text"]),
        "model_state_tensors": len(state),
        "tied_keys": tied,
        "device": sorted(devices)[0],
        "dtype": sorted(dtypes)[0],
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "save_layout": "the model's own; the load-time prefix mapping is not reverted on save",
        "sha256": meta["sha256"],
        "loading_info": {
            key: sorted(value) if isinstance(value, (set, list, tuple)) else value
            for key, value in info.items()
        },
    }
    return model, report
