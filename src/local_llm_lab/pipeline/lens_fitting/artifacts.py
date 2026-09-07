"""Exclusive all-layer lens artifacts, compatible with unchanged LensMaps.load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from local_llm_lab.pipeline.live_lens.instruments import LensMaps, file_sha256

MAX_NPZ_BYTES = 1_000_000_000


def validate_output(path: Path) -> Path:
    path = Path(path).absolute()
    if path.suffix != ".npz":
        raise ValueError("lens output must have .npz suffix")
    for candidate in (path, path.with_suffix(".json")):
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f"immutable lens output already exists: {candidate}")
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError("lens output parent must be a directory")
    return path


def write_lens(
    path: Path, maps: dict[int, np.ndarray], *, hidden_size: int, num_layers: int, metadata: dict
) -> dict:
    """Validate complete maps before exclusive creation; partial writes remain evidence."""
    path = validate_output(path)
    expected = set(range(1, num_layers))
    if not expected or set(maps) != expected:
        raise ValueError("lens must contain all and only nonfinal layers")
    arrays = {}
    for layer in sorted(expected):
        a = np.asarray(maps[layer], dtype=np.float32)
        if a.shape != (hidden_size, hidden_size) or not np.isfinite(a).all():
            raise ValueError(f"invalid finite square lens map for layer {layer}")
        arrays[f"J{layer - 1}"] = a
    # Reserve enough overhead for NumPy headers and ZIP directory entries.
    if sum(a.nbytes + 1024 for a in arrays.values()) >= MAX_NPZ_BYTES:
        raise ValueError("float32 lens archive must stay below 1 GB")
    json.dumps(metadata, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream, path.with_suffix(".json").open("x") as sidecar:
        np.savez(stream, **arrays)
        stream.flush()
        if path.stat().st_size >= MAX_NPZ_BYTES:
            raise ValueError("lens archive exceeds 1 GB")
        sha = file_sha256(path)
        loaded = LensMaps.load(
            path, expected_sha256=sha, hidden_size=hidden_size, num_layers=num_layers
        )
        if set(loaded.maps) != expected or any(
            not np.array_equal(loaded.maps[layer], arrays[f"J{layer - 1}"]) for layer in expected
        ):
            raise ValueError("LensMaps round-trip differs from fitted maps")
        result = metadata | {
            "npz_sha256": sha,
            "layers": sorted(expected),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dtype": "float32",
            "npz_bytes": path.stat().st_size,
            "layer_convention": "J{L-1}; h @ J.T = h @ W; final layer is identity",
        }
        json.dump(result, sidecar, indent=2, allow_nan=False)
        sidecar.write("\n")
    return result
