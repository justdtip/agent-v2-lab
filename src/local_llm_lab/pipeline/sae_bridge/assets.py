"""Read exact bridge matrices as data without importing a model runtime.

The BF16 conversion is a bit shift, not a model conversion. Safetensors validates
the container before we inspect the selected tensor's offsets. Only bounded raw
chunks coexist with the float32 result; no whole BF16 tensor copy is needed.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from safetensors import safe_open

from local_llm_lab.pipeline.live_lens.instruments import file_sha256

COORDINATE_CONVENTION = "gemma_scope2_raw_resid_post_no_rescale"


def checked_hash(path: Path, expected: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("expected SHA256 must be explicitly supplied")
    actual = file_sha256(Path(path))
    if actual != expected:
        raise ValueError(f"asset hash mismatch: {path}")
    return actual


def validate_reference_config(config: dict) -> tuple[int, int, int]:
    """A1's BF16 precondition is stricter than generic lens lineage compatibility."""
    text = config.get("text_config", config)
    if any(
        key in level for level in (config, text) for key in ("quantization", "quantization_config")
    ):
        raise ValueError("A1 requires unquantised BF16 weights")
    if config.get("torch_dtype", text.get("torch_dtype")) != "bfloat16":
        raise ValueError("A1 reference config must declare bfloat16")
    shape = tuple(text.get(key) for key in ("num_hidden_layers", "hidden_size", "vocab_size"))
    if any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("reference config needs positive depth, width and vocabulary")
    return shape


def read_bf16_matrix(
    path: Path,
    key: str,
    *,
    expected_sha256: str,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Return selected BF16 bits widened exactly to float32; no MLX or Torch."""
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    checked_hash(path, expected_sha256)
    with safe_open(str(path), framework="np") as archive:
        selected = archive.get_slice(key)
        shape = tuple(selected.get_shape())
        if selected.get_dtype() != "BF16":
            raise ValueError("reference readout tensor must be BF16")
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("readout must be a nonempty matrix")
    with Path(path).open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        metadata = json.loads(stream.read(header_length))[key]
        start, end = metadata["data_offsets"]
        if metadata["dtype"] != "BF16" or tuple(metadata["shape"]) != shape:
            raise ValueError("tensor metadata changed after container validation")
        if end - start != shape[0] * shape[1] * 2:
            raise ValueError("BF16 tensor byte extent differs from shape")
        stream.seek(8 + header_length + start)
        result = np.empty(shape, dtype=np.float32)
        for begin in range(0, shape[0], chunk_rows):
            count = min(chunk_rows, shape[0] - begin)
            raw = stream.read(count * shape[1] * 2)
            if len(raw) != count * shape[1] * 2:
                raise ValueError("truncated BF16 tensor")
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            bits <<= 16
            values = bits.view(np.float32).reshape(count, shape[1])
            if not np.isfinite(values).all():
                raise ValueError("nonfinite BF16 readout")
            result[begin : begin + count] = values
    checked_hash(path, expected_sha256)
    return result


def decoder_from_files(
    config_path: Path,
    params_path: Path,
    *,
    expected_config_sha256: str,
    expected_params_sha256: str,
    base: str,
    probe_layer: int,
    hidden_size: int,
    width: int,
    l0: int,
    coordinate_convention: str,
) -> np.ndarray:
    """Gemma Scope 2 residual decoder: stored feature rows become D columns.

    SAELens' pinned Gemma 3 loader uses raw w_dec and normalize_activations=none.
    A bias belongs to the later activation decomposition, not direction readout.
    """
    checked_hash(config_path, expected_config_sha256)
    checked_hash(params_path, expected_params_sha256)
    config = json.loads(Path(config_path).read_bytes())
    if coordinate_convention != COORDINATE_CONVENTION:
        raise ValueError("unverified decoder coordinate scaling")
    hook = f"model.layers.{probe_layer - 1}.output"
    if any(config.get(key) != hook for key in ("hf_hook_point_in", "hf_hook_point_out")):
        raise ValueError("decoder hook does not match lens layer")
    expected = {
        "model_name": base,
        "width": width,
        "l0": l0,
        "type": "sae",
        "architecture": "jump_relu",
        "affine_connection": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"decoder precondition mismatch: {key}")
    with safe_open(str(params_path), framework="np") as archive:
        dec = archive.get_slice("w_dec")
        enc = archive.get_slice("w_enc")
        if dec.get_shape() != [width, hidden_size] or enc.get_shape() != [hidden_size, width]:
            raise ValueError("decoder/encoder shape does not match declared coordinates")
        if dec.get_dtype() != "F32":
            raise ValueError("Gemma Scope decoder must have its native F32 dtype")
        result = archive.get_tensor("w_dec")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite dictionary decoder")
    checked_hash(params_path, expected_params_sha256)
    return result.T


def example_topk_from_file(
    path: Path, *, expected_sha256: str, width: int, k: int, vocabulary: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read only the shipped top-token companion, never its activation corpus.

    Full-file hashes bind these small arrays to the exact selected dictionary's
    examples artifact. Equal dimensions or byte lengths cannot establish identity.
    """
    if any(type(n) is not int or n <= 0 for n in (width, k, vocabulary)):
        raise ValueError("examples geometry must be positive integers")
    checked_hash(path, expected_sha256)
    with safe_open(str(path), framework="np") as archive:
        for key, dtype in (("top_tokens", "I32"), ("top_logits", "F32")):
            tensor = archive.get_slice(key)
            if tensor.get_shape() != [width, k] or tensor.get_dtype() != dtype:
                raise ValueError(f"examples {key} shape/dtype differs from registration")
        tokens = archive.get_tensor("top_tokens")
        logits = archive.get_tensor("top_logits")
    if np.any(tokens < 0) or np.any(tokens >= vocabulary):
        raise ValueError("examples token ID outside readout vocabulary")
    if np.any(np.diff(np.sort(tokens, axis=1), axis=1) == 0):
        raise ValueError("examples top tokens contain duplicate IDs")
    if not np.isfinite(logits).all() or np.any(np.diff(logits, axis=1) > 0):
        raise ValueError("examples top logits must be finite and nonincreasing")
    checked_hash(path, expected_sha256)
    return tokens, logits
