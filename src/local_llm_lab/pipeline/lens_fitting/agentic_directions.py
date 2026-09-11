"""Explicit frozen-direction experiment on original model forwards, through TorchCapture.

This does not draw controls or choose error vectors. Supply a frozen, hash-bound
direction bundle for every requested cell/layer and each declared arm. Responses
are pre-final-normalization vectors, not claims about complete-call behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("dictionary_error", "norm_random", "angle_random")
PROVENANCE_KEYS = ("corpus_sha256", "positions_sha256", "checkpoint_sha256", "files_sha256")
CONTROL_SEED = 20260911


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _base_sha(h):
    value = np.asarray(h)
    if value.ndim != 1 or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("base residual must be a finite float32 vector")
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_directions(path, cells, layers, provenance):
    """Refuse changed, incomplete, duplicated, or incorrectly matched frozen arms."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("arms") != list(ARMS):
        raise ValueError("direction manifest must explicitly declare all three supported arms")
    if (
        not isinstance(manifest.get("control_definition"), str)
        or not manifest["control_definition"]
    ):
        raise ValueError("frozen control definition is required")
    if manifest.get("provenance") != {key: provenance[key] for key in PROVENANCE_KEYS}:
        raise ValueError("direction corpus/positions/capture/checkpoint provenance mismatch")
    dictionaries = manifest.get("dictionary_files")
    if not isinstance(dictionaries, dict) or not dictionaries:
        raise ValueError("dictionary files and digests are required")
    for name, digest in dictionaries.items():
        if _sha(path.parent / name) != digest:
            raise ValueError("dictionary file digest mismatch")
    archive_path = path.parent / manifest["npz"]
    if _sha(archive_path) != manifest["npz_sha256"]:
        raise ValueError("direction archive digest mismatch")
    bases = {
        (c["cell"]["row"], c["cell"]["position"], layer): c["residuals"][layer]
        for c in cells
        for layer in layers
    }
    if not bases or len(bases) != len(cells) * len(layers):
        raise ValueError("requested direction population must be nonempty and unique")
    expected = {(*key, arm) for key in bases for arm in ARMS}
    directions, used = {}, set()
    with np.load(archive_path, allow_pickle=False) as archive:
        for row in manifest["records"]:
            key = (row["row"], row["position"], row["layer"], row["arm"])
            if key not in expected or key in directions or row["array"] in used:
                raise ValueError("direction coverage has an extra or duplicate row/array")
            base = bases[key[:3]]
            if row["base_residual_sha256"] != _base_sha(base):
                raise ValueError("direction source residual digest mismatch")
            value = archive[row["array"]]
            if (
                value.dtype != np.float32
                or value.shape != base.shape
                or not np.isfinite(value).all()
            ):
                raise ValueError("frozen direction must match the finite float32 source shape")
            directions[key] = value.copy()
            used.add(row["array"])
        if (
            set(directions) != expected
            or set(archive.files) != used
            or len(archive.files) != len(used)
        ):
            raise ValueError("direction coverage must exactly match every requested cell/layer/arm")
    # Matching is checked rather than inferred from the names on producer receipts.
    for key, base in bases.items():
        base = base.astype(np.float64)
        error = directions[(*key, "dictionary_error")].astype(np.float64)
        norm = np.linalg.norm(error)
        for arm in ("norm_random", "angle_random"):
            control = directions[(*key, arm)].astype(np.float64)
            if norm == 0:
                if np.any(control != 0):
                    raise ValueError("zero dictionary error requires zero matched controls")
                continue
            if not np.isclose(np.linalg.norm(control), norm, rtol=1e-5, atol=0):
                raise ValueError("frozen control is not norm matched")
            if arm == "angle_random" and np.linalg.norm(base) > 0:
                difference = abs(np.dot(control - error, base) / (norm * np.linalg.norm(base)))
                if difference > 1e-5:
                    raise ValueError("frozen angle control is not angle matched")
    return directions, {"manifest_sha256": _sha(path), "content": manifest}


def matched_controls(base, error, key):
    """One deterministic Gaussian draw per arm, keyed independently of iteration order."""
    h, error = np.asarray(base, dtype=np.float64), np.asarray(error, dtype=np.float64)
    norm, hn = np.linalg.norm(error), np.linalg.norm(h)
    if (
        h.shape != error.shape
        or h.ndim != 1
        or not np.isfinite(h).all()
        or not np.isfinite(error).all()
    ):
        raise ValueError("control vectors must be finite matching vectors")
    controls = {}
    for arm in ("norm_random", "angle_random"):
        seed_material = json.dumps([CONTROL_SEED, *key, arm], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:16], "little")
        rng = np.random.Generator(np.random.PCG64(seed))
        if norm == 0:
            value = np.zeros_like(error)
        else:
            gaussian = rng.standard_normal(len(error))
            if arm == "norm_random" or hn == 0:
                value = gaussian * (norm / np.linalg.norm(gaussian))
            else:
                unit = h / hn
                along = float(np.dot(error, unit))
                perpendicular_norm = np.linalg.norm(error - along * unit)
                perpendicular = gaussian - np.dot(gaussian, unit) * unit
                if perpendicular_norm <= np.finfo(np.float64).eps * norm:
                    value = along * unit
                else:
                    pn = np.linalg.norm(perpendicular)
                    if pn == 0:
                        raise ValueError("degenerate Gaussian orthogonal draw; no silent redraw")
                    value = along * unit + perpendicular * (perpendicular_norm / pn)
        controls[arm] = value.astype(np.float32)
    return controls


def freeze_directions(cells, layers, dictionaries, manifest_path, provenance, dictionary_files):
    """Materialize reconstruction errors and declared single-draw controls, without a model."""
    from local_llm_lab.probes import sae_bridge as B

    from .agentic import atomic_json

    manifest_path = Path(manifest_path)
    archive_path = manifest_path.with_suffix(".npz")
    if manifest_path.exists() or archive_path.exists():
        raise FileExistsError("direction outputs must be new")
    arrays, records, seen = {}, [], set()
    for cell in sorted(cells, key=lambda x: (x["cell"]["row"], x["cell"]["position"])):
        for layer in sorted(layers):
            key = (cell["cell"]["row"], cell["cell"]["position"], layer)
            if key in seen:
                raise ValueError("duplicate frozen-direction cell/layer")
            seen.add(key)
            h = cell["residuals"][layer]
            base_sha = _base_sha(h)
            error = (B.decode(dictionaries[layer], B.encode(dictionaries[layer], h)) - h).astype(
                np.float32
            )
            if not np.isfinite(error).all():
                raise ValueError("dictionary reconstruction produced a nonfinite error")
            controls = {"dictionary_error": error, **matched_controls(h, error, key)}
            if any(not np.isfinite(value).all() for value in controls.values()):
                raise ValueError("generated control is not finite in float32")
            for arm in ARMS:
                name = f"r{key[0]}_{key[1]}_L{layer}_{arm}"
                arrays[name] = controls[arm]
                records.append(
                    {
                        "row": key[0],
                        "position": key[1],
                        "layer": layer,
                        "arm": arm,
                        "array": name,
                        "base_residual_sha256": base_sha,
                        "geometry": "zero_error"
                        if not np.any(error)
                        else "angle_undefined_zero_base"
                        if not np.any(h)
                        else "nondegenerate",
                    }
                )
    if not records:
        raise ValueError("cannot freeze an empty direction population")
    manifest = {
        "schema_version": 1,
        "npz": archive_path.name,
        "provenance": {key: provenance[key] for key in PROVENANCE_KEYS},
        "dictionary_files": {
            str(Path(path).resolve()): digest for path, digest in dictionary_files.items()
        },
        "arms": list(ARMS),
        "seed": CONTROL_SEED,
        "draws_per_control_per_cell": 1,
        "generator": "NumPy PCG64; first 128 SHA256 bits of compact JSON [seed,row,site,layer,arm], little-endian",
        "numpy_version": np.__version__,
        "control_definition": "Single-draw pilot, not the historical eight-draw analysis. Error=decode(encode(h))-h. Norm control scales a Gaussian vector to error norm. Angle control preserves error projection on h and rotates its orthogonal component using a Gaussian projection. Geometry computed float64; frozen vectors float32. Zero errors give zero controls; zero bases have undefined angles.",
        "records": records,
    }
    # Publish the declared algorithm before any device job can consume this bundle.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("xb") as stream:
        np.savez(stream, **arrays)
    manifest["npz_sha256"] = _sha(archive_path)
    atomic_json(manifest_path, manifest)
    load_directions(manifest_path, cells, layers, provenance)
    return manifest


def prepare(args):
    """File-only capture/dictionary admission and direction freeze using the prose reference."""
    from local_llm_lab.probes import device_bridge as D
    from local_llm_lab.probes import measure_pairings as P
    from local_llm_lab.probes import sae_bridge as B

    paths = {
        key: Path(args[key])
        for key in ("model_snapshot", "checkpoint_manifest", "capture_dir", "positions", "corpus")
    }
    paths.update({key: Path(args["prose"][key]) for key in ("lens_archive", "lens_sidecar")})
    paths["layers"] = args["layers"]
    cells, metadata, provenance = P._admit(paths)
    dictionaries, files = {}, {}
    for layer in args["layers"]:
        spec = args["dictionaries"][str(layer)]
        directory, receipt = Path(spec["directory"]), Path(spec["receipt"])
        conf = {
            "dictionary_repo": spec["repo"],
            "dictionary_folder": spec["folder"],
            "model_base": metadata["model"]["base"],
        }
        view, _ = D._dictionary(directory, receipt, conf, cells[0]["residuals"][layer].size)
        if B.layer_for_hook(view.hook_point) != layer:
            raise ValueError("dictionary hook differs from requested direction layer")
        dictionaries[layer] = B.load_dictionary(
            directory / "params.safetensors", directory / "config.json"
        )
        for path in (receipt, directory / "config.json", directory / "params.safetensors"):
            files[str(path.resolve())] = _sha(path)
    return freeze_directions(
        cells, args["layers"], dictionaries, args["directions_manifest"], provenance, files
    )


class _Sink:
    def __init__(self):
        self.values = {}

    def residual(self, layer, offset, h):
        if offset != 0:
            raise ValueError("directional experiment requires fresh full-input forwards")
        self.values[layer] = h

    def output(self, offset, ids, logits):
        pass


def run_directions(view, cells, lenses, directions, progress=None):
    """Run shared exact/actual responses and read both maps along identical vectors.

    No new attention mask, cache policy, padding or batch is supplied. Native model
    defaults and the original capture's attention/kept-logit flags are preserved.
    Every replacement checks its unmodified source against the saved capture.
    """
    import torch

    from local_llm_lab.torch_capture import TorchCapture

    from .agentic_compare import _vector_comparison, directional_check

    if set(lenses) != {"prose", "agentic"}:
        raise ValueError("both prose and agentic lenses are required")
    if any(p.dtype != torch.float32 for p in view.model.parameters()):
        raise ValueError("directional model must be entirely float32")
    lookup = {(item["cell"]["row"], item["cell"]["position"]): item for item in cells}
    if len(lookup) != len(cells) or not directions:
        raise ValueError("directional cells must be unique and directions nonempty")
    output = []
    for (row, site, layer, arm), vector in directions.items():
        item = lookup[(row, site)]
        if not 1 <= layer < view.num_layers or arm not in ARMS:
            raise ValueError("direction layer/arm outside supported experiment")
        pos = item["cell"]["token_index"]
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=view.input_device)
        kwargs = {
            "output_attentions": item["capture_forward"]["output_attentions"],
            "logits_to_keep": torch.tensor(
                item["capture_forward"]["logits_to_keep"], device=view.input_device
            ),
        }
        base = item["residuals"][layer]
        expected_sha = _base_sha(base)

        def check_source(hidden, expected_sha=expected_sha, row=row, site=site, layer=layer):
            array = hidden.detach().cpu().contiguous().numpy()
            if _base_sha(array) != expected_sha:
                raise ValueError(f"source residual mismatch at row {row}, {site}, layer {layer}")

        sink = _Sink()
        with TorchCapture(view, sink, layers=[layer, view.num_layers]), torch.no_grad():
            view.model(input_ids=ids, **kwargs)
            check_source(sink.values[layer][0, pos])
            baseline = sink.values[view.num_layers][0, pos].detach().clone()

        def tail(replacement, layer=layer, pos=pos, ids=ids, kwargs=kwargs, verify=check_source):
            sink = _Sink()
            with TorchCapture(view, sink, layers=[layer, view.num_layers]) as capture:

                def replace(hidden):
                    verify(hidden)
                    return replacement

                capture.intervene(layer, pos, replace)
                view.model(input_ids=ids, **kwargs)
                return sink.values[view.num_layers][0, pos]

        h = torch.tensor(base, device=view.input_device)
        delta = torch.tensor(vector, device=view.input_device)
        measured = directional_check(tail, h, delta, lenses["prose"].maps[layer], baseline=baseline)
        exact, actual = np.asarray(measured["exact"]), np.asarray(measured["actual"])
        row_result = {
            "row": row,
            "position": site,
            "layer": layer,
            "arm": arm,
            "episode": item["prompt_identity"]["task_id"],
            "source_residual_sha256": expected_sha,
            "direction_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            "forward_batch": 1,
            "precision": "float32",
            "endpoint": "pre-final-norm",
            "same_state_equal": measured["same_state_equal"],
            "exact": measured["exact"],
            "actual": measured["actual"],
            "exact_vs_actual": measured["exact_vs_actual"],
        }
        for name, lens in lenses.items():
            prediction = (
                (torch.tensor(lens.maps[layer], device=view.input_device) @ delta)
                .detach()
                .cpu()
                .numpy()
            )
            row_result[name] = {
                "prediction": prediction.tolist(),
                "lens_vs_exact": _vector_comparison(prediction, exact),
                "lens_vs_actual": _vector_comparison(prediction, actual),
            }
        output.append(row_result)
        if progress:
            progress(
                {
                    "event": "direction",
                    "completed": len(output),
                    "of": len(directions),
                    "row": row,
                    "position": site,
                    "layer": layer,
                    "arm": arm,
                }
            )
    return output


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--prepare", action="store_true", help="freeze error/control vectors without a model load"
    )
    parsed = parser.parse_args(argv)
    args = json.loads(parsed.config.read_text())
    if parsed.prepare:
        prepare(args)
        return 0
    from local_llm_lab import runlock
    from local_llm_lab.arch_torch import TorchArchitectureView
    from local_llm_lab.probes import measure_pairings as P
    from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens

    from .agentic import atomic_json
    from .agentic_compare import validate_cell_pairings

    output = Path(args["directions_output"])
    if output.exists():
        raise FileExistsError(output)
    lenses, admissions, population = {}, {}, None
    for name in ("prose", "agentic"):
        paths = {
            key: Path(args[key])
            for key in (
                "model_snapshot",
                "checkpoint_manifest",
                "capture_dir",
                "positions",
                "corpus",
            )
        }
        paths.update({key: Path(args[name][key]) for key in ("lens_archive", "lens_sidecar")})
        paths.update(
            capture_reference_archive=Path(args["prose"]["lens_archive"]),
            capture_reference_sidecar=Path(args["prose"]["lens_sidecar"]),
            layers=args["layers"],
        )
        cells, metadata, provenance = P._admit(paths)
        keys = [(c["cell"]["row"], c["cell"]["position"]) for c in cells]
        if population is not None and keys != population:
            raise ValueError("prose and agentic direction populations differ")
        population = keys
        validate_cell_pairings(
            json.loads(Path(args[name]["pairings"]).read_text()),
            cells,
            metadata,
            provenance,
            args["layers"],
            args["maximum_pairing_relative"],
        )
        lenses[name], _ = load_admitted_device_lens(
            paths["lens_archive"],
            checkpoint=paths["model_snapshot"],
            model_base=metadata["model"]["base"],
        )
        admissions[name] = provenance
    directions, receipt = load_directions(
        args["directions_manifest"], cells, args["layers"], provenance
    )
    runlock.hold_model_run_lock(command="agentic frozen directions", session="codex-agentic")
    path, runtime = P._load_path(Path(args["model_snapshot"]), provenance["manifest"])
    if runtime["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
        raise ValueError("loaded directional checkpoint differs from admitted checkpoint")
    view = TorchArchitectureView.from_model(path.capture_model)
    rows = run_directions(
        view, cells, lenses, directions, progress=lambda event: print(json.dumps(event), flush=True)
    )
    result = {
        "schema_version": 1,
        "endpoint": "pre-final-norm",
        "runtime": runtime,
        "admissions": admissions,
        "directions": receipt,
        "rows": rows,
        "claim_scope": "local vector response; complete-call effects not measured",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
