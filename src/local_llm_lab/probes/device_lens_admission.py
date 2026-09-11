"""Admit immutable device-fit files through the existing LensMaps identity format.

This is file work only: no model constructor, device use, or download. Admission certifies
serialization and provenance; it does not certify a fit/capture pairing or extrapolation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from local_llm_lab.hf_text import checkpoint_metadata
from local_llm_lab.pipeline.live_lens.instruments import (
    LensIdentity,
    LensMaps,
    file_sha256,
    resolve_base,
)


def declaration_sha256(value: dict) -> str:
    """Bind a declaration independently of JSON whitespace, matching instrument digests."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required file absent: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    declaration_sha256(value)  # Refuse non-finite JSON numbers in every nested declaration.
    return value


def _int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _rows(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty population")
    rows = [_int(row, name, minimum=0) for row in value]
    if len(set(rows)) != len(rows):
        raise ValueError(f"{name} contains duplicate or overlapping rows")
    return rows


def _hashes(value: object) -> dict:
    hashes = _mapping(value, "checkpoint load_report_sha256")
    if "config.json" not in hashes or len(hashes) < 2:
        raise ValueError("checkpoint hash manifest needs config.json and every weight shard")
    for name, sha in hashes.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("checkpoint hash manifest contains an invalid filename or sha256")
    return hashes


@dataclass
class _Source:
    root: Path
    manifest: dict
    nu: dict
    maps: dict[str, np.ndarray]
    width_rows: list[dict]

    @property
    def depth(self) -> int:
        return self.manifest["n_layers"]

    @property
    def hidden(self) -> int:
        return self.manifest["d_model"]


def _source_layers(manifest: dict) -> list[int]:
    """Legacy sources cover every interior layer; partial sources must say so."""
    depth = _int(manifest.get("n_layers"), "manifest n_layers", minimum=2)
    if "source_layers_repo" not in manifest:
        return list(range(1, depth))
    layers = _rows(manifest["source_layers_repo"], "source_layers_repo")
    if layers != sorted(layers) or any(not 1 <= layer < depth for layer in layers):
        raise ValueError("source_layers_repo must be sorted interior repository layers")
    return layers


def _single_declaration(manifest: dict, nu: dict) -> list[dict]:
    depth = _int(manifest.get("n_layers"), "manifest n_layers", minimum=2)
    _int(manifest.get("d_model"), "manifest d_model")
    if nu.get("decoder_depth") != depth:
        raise ValueError("declaration decoder_depth disagrees with manifest n_layers")
    if nu.get("estimator") != "upstream-exact-autograd":
        raise ValueError("device admission requires the declared upstream-exact-autograd estimator")
    end = _mapping(nu.get("endpoint"), "nu endpoint")
    layers = _source_layers(manifest)
    for key, expected in {
        "source_layers_repo": layers,
        "source_layers_upstream": [layer - 1 for layer in layers],
        "target_layer_repo": depth,
        "target_layer_upstream": depth - 1,
        "target_is_pre_final_norm": True,
    }.items():
        if end.get(key) != expected:
            raise ValueError(f"endpoint {key} must be {expected!r}")
    corpus = _mapping(nu.get("corpus"), "nu corpus")
    for field in ("manifest", "split"):
        if not isinstance(corpus.get(field), str) or not corpus[field]:
            raise ValueError(f"corpus {field} must be declared")
    rows = _rows(corpus.get("rows"), "corpus rows")
    if manifest.get("corpus") != corpus:
        raise ValueError("manifest and nu corpus populations disagree")
    pair = _mapping(nu.get("pair_weighting"), "pair_weighting")
    count = _int(manifest.get("n_rows"), "manifest n_rows")
    if pair.get("n_prompts") != count or len(rows) != count:
        raise ValueError("n_prompts, n_rows and corpus population disagree")
    if pair.get("n_skipped") != 0 or pair.get("skipped") != []:
        raise ValueError("skipped fit rows cannot receive requested-population weights")
    for field in ("per_prompt", "source_target_pairs"):
        if not isinstance(pair.get(field), str) or not pair[field]:
            raise ValueError(f"pair_weighting {field} must be declared")
    pos = _mapping(nu.get("position_weighting"), "position_weighting")
    context = _int(pos.get("max_seq_len"), "max_seq_len")
    if manifest.get("max_seq_len") != context:
        raise ValueError("manifest max_seq_len disagrees with the fit declaration")
    _int(pos.get("skip_first"), "skip_first", minimum=0)
    for field in ("source_reduction", "target_reduction"):
        if not isinstance(pos.get(field), str) or not pos[field]:
            raise ValueError(f"position_weighting {field} must be declared")
    if not isinstance(pos.get("probe"), list) or not pos["probe"]:
        raise ValueError("position_weighting probe must declare the selected positions")
    for probe in pos["probe"]:
        probe = _mapping(probe, "position probe")
        first = _int(probe.get("first"), "position first", minimum=0)
        last = _int(probe.get("last"), "position last", minimum=0)
        length = _int(probe.get("seq_len"), "position seq_len")
        if not first <= last < length <= context:
            raise ValueError("selected position probe is outside the declared context")
    precision = _mapping(nu.get("precision"), "precision")
    for key in ("dtype", "declared_dtype", "stored_dtype"):
        if precision.get(key) != "float32":
            raise ValueError(f"precision {key} must explicitly declare float32")
    if manifest.get("precision") != "float32":
        raise ValueError("manifest precision disagrees with the float32 declaration")
    if "fit_dtype" in precision and precision["fit_dtype"] != precision["dtype"]:
        raise ValueError("precision fit_dtype contradicts the declared dtype")
    widths = {
        "dim_batch": _int(pos.get("dim_batch"), "dim_batch"),
        "forward_batch": _int(precision.get("forward_batch"), "forward_batch"),
        "anchor_batch": _int(precision.get("anchor_batch"), "anchor_batch"),
    }
    for key in ("dim_batch", "forward_batch", "anchor_batch"):
        # Earlier as-run manifests omitted anchor_batch; nu itself must always carry it.
        if (key != "anchor_batch" or key in manifest) and manifest.get(key) != widths[key]:
            raise ValueError(f"manifest {key} disagrees with nu")
    schedule = manifest.get("estimator_schedule", "sequential")
    if schedule != nu.get("estimator_schedule", "sequential"):
        raise ValueError("manifest estimator_schedule disagrees with nu")
    if schedule == "graph-once":
        if widths["forward_batch"] != 1 or widths["anchor_batch"] != 1:
            raise ValueError("graph-once requires forward_batch and anchor_batch equal to one")
    elif schedule == "sequential":
        if len(set(widths.values())) != 1:
            raise ValueError("sequential requires dim_batch, forward_batch and anchor_batch equal")
    else:
        raise ValueError("estimator_schedule must be sequential or graph-once")
    _hashes(manifest.get("load_report_sha256"))
    return [widths | {"n_prompts": count, "rows": rows, "weight": 1.0}]


def _maps(path: Path, layers: list[int], hidden: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"source archive absent: {path}")
    expected = {f"J{i}" for i in layers}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected or len(archive.files) != len(expected):
            raise ValueError(f"source map keys must be declared repository layers {layers}")
        arrays = {}
        for key in sorted(expected, key=lambda k: int(k[1:])):
            value = archive[key]
            if value.shape != (hidden, hidden):
                raise ValueError(f"{key} shape disagrees with declared hidden size {hidden}")
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain finite float32 values")
            arrays[key] = value
    return arrays


def _read_source(root: Path, *, allow_merge: bool = True) -> _Source:
    root = Path(root).resolve(strict=True)
    man, nu = _read_json(root / "manifest.json"), _read_json(root / "nu.json")
    if (
        type(man.get("schema_version")) is not int
        or man["schema_version"] != 1
        or type(nu.get("schema_version")) is not int
        or nu["schema_version"] != 1
    ):
        raise ValueError("manifest and nu must declare supported schema_version 1")
    if "merge" in nu:
        if not allow_merge:
            raise ValueError("nested chunk merges must be flattened before admission")
        recorded = _mapping(nu["merge"], "merge").get("chunks")
        if not isinstance(recorded, list) or not recorded:
            raise ValueError("merged declaration requires all chunk declarations")
        chunks = [_read_source(Path(row["chunk"]), allow_merge=False) for row in recorded]
        expected_nu, expected_man, widths = _merge_declarations(chunks)
        if nu != expected_nu:
            raise ValueError("merged nu declaration/hash provenance disagrees with source chunks")
        for key, value in expected_man.items():
            if man.get(key) != value:
                raise ValueError(f"merged manifest {key} disagrees with source chunk provenance")
    else:
        widths = _single_declaration(man, nu)
    arrays = _maps(root / "exact-maps.npz", _source_layers(man), man["d_model"])
    recorded_sha = man.get("exact_maps_sha256")
    if recorded_sha is not None and recorded_sha != file_sha256(root / "exact-maps.npz"):
        raise ValueError("source archive hash disagrees with manifest")
    if "merge" in nu:
        for key, matrix in arrays.items():
            if not np.array_equal(matrix, _weighted_map(chunks, key)):
                raise ValueError(f"merged weighted map {key} differs from its source chunks")
    return _Source(root, man, nu, arrays, widths)


def _common_nu(nu: dict) -> dict:
    """Only measured population sizes and batch schedules may vary across chunks."""
    common = copy.deepcopy(nu)
    common["corpus"].pop("rows")
    for key in ("n_prompts", "n_skipped", "skipped"):
        common["pair_weighting"].pop(key)
    for key in ("dim_batch", "n_valid_positions", "seq_len"):
        common["position_weighting"].pop(key, None)
    for key in ("forward_batch", "anchor_batch"):
        common["precision"].pop(key)
    return common


def _merge_declarations(chunks: list[_Source]) -> tuple[dict, dict, list[dict]]:
    if not chunks:
        raise ValueError("at least one source chunk is required")
    first = chunks[0]
    baseline = _common_nu(first.nu)
    seen: set[int] = set()
    total = sum(chunk.manifest["n_rows"] for chunk in chunks)
    records, widths, population = [], [], []
    for chunk in chunks:
        if chunk.depth != first.depth or chunk.hidden != first.hidden:
            raise ValueError("chunk model dimensions disagree")
        if chunk.manifest["load_report_sha256"] != first.manifest["load_report_sha256"]:
            raise ValueError("chunk checkpoint hash manifests disagree")
        other = _common_nu(chunk.nu)
        if other != baseline:
            differing = [
                key for key in set(baseline) | set(other) if baseline.get(key) != other.get(key)
            ]
            raise ValueError(f"chunk declarations disagree in {', '.join(sorted(differing))}")
        rows = chunk.nu["corpus"]["rows"]
        if seen.intersection(rows):
            raise ValueError("chunk corpus populations overlap")
        seen.update(rows)
        population.extend(rows)
        weight = len(rows) / total
        widths.append(chunk.width_rows[0] | {"weight": weight})
        records.append(
            {
                "chunk": str(chunk.root),
                "n_prompts": len(rows),
                "rows": rows,
                "weight": weight,
                "archive_sha256": file_sha256(chunk.root / "exact-maps.npz"),
                "manifest_sha256": file_sha256(chunk.root / "manifest.json"),
                "nu_sha256": declaration_sha256(chunk.nu),
                "nu_file_sha256": file_sha256(chunk.root / "nu.json"),
                "nu": chunk.nu,
            }
        )
    merged = copy.deepcopy(first.nu)
    merged["corpus"]["rows"] = population
    merged["pair_weighting"]["n_prompts"] = total
    for section, key in (
        ("position_weighting", "dim_batch"),
        ("precision", "forward_batch"),
        ("precision", "anchor_batch"),
    ):
        merged[section][key] = [width[key] for width in widths]
    for key in ("n_valid_positions", "seq_len"):
        values = [chunk.nu["position_weighting"].get(key) for chunk in chunks]
        if any(value is None for value in values):
            raise ValueError(f"every chunk must declare position_weighting {key}")
        merged["position_weighting"][key] = {
            "min": min(value["min"] for value in values),
            "max": max(value["max"] for value in values),
        }
        if key == "n_valid_positions":
            merged["position_weighting"][key]["total"] = sum(value["total"] for value in values)
    merged["merge"] = {"weighting": "n_prompts / total_n_prompts", "chunks": records}
    manifest = {
        "schema_version": 1,
        "stage": "merge of chunked float32 exact lenses",
        "n_rows": total,
        "n_prompts": total,
        "n_layers": first.depth,
        "d_model": first.hidden,
        "layers": _source_layers(first.manifest),
        "checkpoint": first.manifest["checkpoint"],
        "load_report_sha256": first.manifest["load_report_sha256"],
        "corpus": merged["corpus"],
        "precision": "float32",
        "max_seq_len": first.manifest["max_seq_len"],
        "chunks": [{k: v for k, v in row.items() if k != "nu"} for row in records],
    }
    if "source_layers_repo" in first.manifest:
        manifest["source_layers_repo"] = _source_layers(first.manifest)
    if "estimator_schedule" in first.manifest:
        manifest["estimator_schedule"] = first.manifest["estimator_schedule"]
    return merged, manifest, widths


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _weighted_map(chunks: list[_Source], key: str) -> np.ndarray:
    """Retain upstream's stored-float32 multiply, input-order sum, then divide."""
    total = sum(chunk.manifest["n_rows"] for chunk in chunks)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            accumulator = np.zeros_like(chunks[0].maps[key], dtype=np.float32)
            for chunk in chunks:
                accumulator += chunk.maps[key] * chunk.manifest["n_rows"]
            return accumulator / total
    except FloatingPointError as error:
        raise ValueError(f"merged weighted map {key} is not finite") from error


def merge_device_lens_chunks(output: Path, chunk_dirs: list[Path]) -> dict:
    """Merge disjoint compatible populations, preserving every complete fit declaration.

    NumPy retains upstream's float32 accumulation and input order. No torch/upstream import
    or GPU is needed for this already-fitted matrix operation.
    """
    output = Path(output)
    chunks = [_read_source(Path(root), allow_merge=False) for root in chunk_dirs]
    nu, manifest, _ = _merge_declarations(chunks)
    arrays = {key: _weighted_map(chunks, key) for key in chunks[0].maps}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "exact-maps.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest["exact_maps_sha256"] = file_sha256(output / "exact-maps.npz")
    _write_json_exclusive(output / "nu.json", nu)
    _write_json_exclusive(output / "manifest.json", manifest)
    return manifest


def _checkpoint_base(path: Path, config: dict) -> str:
    candidates = []
    for value in (config.get("_name_or_path"), config.get("text_config", {}).get("_name_or_path")):
        if isinstance(value, str) and value:
            candidates.append(resolve_base(value))
    for part in path.parts:
        if part.startswith("models--"):
            candidates.append(part.removeprefix("models--").replace("--", "/"))
    if not candidates or len(set(candidates)) != 1:
        raise ValueError("checkpoint base is absent or contradictory; use a named HF snapshot")
    return candidates[0]


def _verify_checkpoint(source: _Source, checkpoint: Path, model_base: str) -> dict:
    checkpoint = Path(checkpoint).resolve(strict=True)
    config = _read_json(checkpoint / "config.json")
    base = resolve_base(model_base)
    if _checkpoint_base(checkpoint, config) != base:
        raise ValueError("checkpoint base disagrees with requested model identity")
    meta = checkpoint_metadata(checkpoint)
    if meta["sha256"] != source.manifest["load_report_sha256"]:
        raise ValueError("checkpoint hash manifest disagrees with the source fit checkpoint")
    text = meta["text_config"]
    if text.get("num_hidden_layers") != source.depth or text.get("hidden_size") != source.hidden:
        raise ValueError("checkpoint dimensions disagree with source fit declaration")
    return {"base": base, "num_layers": source.depth, "endpoint": "identity"}


def _admission_metadata(source: _Source, identity: dict) -> dict:
    archive = source.root / "exact-maps.npz"
    # declare_nu writes dtype/declared_dtype; the bridge consumes fit_dtype. Preserve the
    # complete source declaration below and bind the normalized declaration separately.
    admitted_nu = copy.deepcopy(source.nu)
    fit_dtype = source.nu["precision"]["dtype"]
    if fit_dtype != source.manifest["precision"]:
        raise ValueError("source precision dtype and manifest precision disagree")
    admitted_nu["precision"]["fit_dtype"] = fit_dtype
    return {
        "model": identity,
        "source_archive_sha256": file_sha256(archive),
        "source_archive": {"path": str(archive), "key_convention": "repository-layer"},
        "source_manifest": {
            "path": str(source.root / "manifest.json"),
            "sha256": file_sha256(source.root / "manifest.json"),
            "content": source.manifest,
        },
        "source_nu": {
            "path": str(source.root / "nu.json"),
            "sha256": file_sha256(source.root / "nu.json"),
            "content": source.nu,
        },
        "nu": admitted_nu,
        "nu_sha256": declaration_sha256(admitted_nu),
        "fit_widths": source.width_rows,
        "checkpoint_files_sha256": source.manifest["load_report_sha256"],
    }


def admit_device_lens(
    fit_dir: Path, *, checkpoint: Path, model_base: str, output: Path | None = None
) -> dict:
    """Write a new loader-form archive and hash-bound identity sidecar, never the originals."""
    source = _read_source(Path(fit_dir))
    identity = _verify_checkpoint(source, checkpoint, model_base)
    target = Path(output) if output is not None else source.root / "admitted-maps.npz"
    if target.suffix != ".npz":
        raise ValueError("admitted output must use the .npz suffix")
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(f"admitted output already exists: {target}")
    existing = source.root / "exact-maps.json"
    if existing.exists():
        sidecar = _read_json(existing)
        if sidecar.get("npz_sha256") != file_sha256(source.root / "exact-maps.npz"):
            raise ValueError("source sidecar names a different file hash")
        expected = LensIdentity.from_dict(identity)
        if LensIdentity.from_dict(sidecar.get("model")) != expected:
            raise ValueError("source sidecar has the wrong model identity")
    payload = _admission_metadata(source, identity)
    arrays = {f"J{int(key[1:]) - 1}": value for key, value in source.maps.items()}
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    payload["npz_sha256"] = file_sha256(target)
    _write_json_exclusive(target.with_suffix(".json"), payload)
    return payload


def load_admitted_device_lens(
    archive: Path, *, checkpoint: Path, model_base: str, expected_sha256: str | None = None
) -> tuple[LensMaps, dict]:
    """Revalidate source bindings and actual checkpoint before returning the real loader object."""
    archive = Path(archive)
    payload = _read_json(archive.with_suffix(".json"))
    actual_sha = file_sha256(archive)
    if payload.get("npz_sha256") != actual_sha or (
        expected_sha256 is not None and expected_sha256 != actual_sha
    ):
        raise ValueError("admitted archive hash mismatch")
    source_info = _mapping(payload.get("source_archive"), "source_archive provenance")
    if not isinstance(source_info.get("path"), str):
        raise ValueError("source archive provenance path is absent")
    source = _read_source(Path(source_info["path"]).parent)
    identity = _verify_checkpoint(source, checkpoint, model_base)
    expected = _admission_metadata(source, identity)
    if {key: payload.get(key) for key in expected} != expected:
        raise ValueError("admitted identity, declaration or source hash provenance mismatch")
    lens = LensMaps.load(
        archive,
        expected_sha256=actual_sha,
        hidden_size=source.hidden,
        num_layers=source.depth,
        identity=LensIdentity.from_dict(identity),
    )
    if set(lens.maps) != set(_source_layers(source.manifest)):
        raise ValueError("admitted map keys differ from declared source layers")
    for layer, matrix in lens.maps.items():
        if not np.array_equal(matrix, source.maps[f"J{layer}"]):
            raise ValueError(f"admitted matrix at repository layer {layer} differs from source")
    return lens, payload
