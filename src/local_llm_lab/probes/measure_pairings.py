"""Measure WS-D cross-path displacement on sealed workspace capture cells.

File admission precedes model loading. The replay is the historical workspace capture's
eager decoder forward: bf16 checkpoint load, promotion to float32, no padding or new BOS,
no attention mask, and replicated inputs at the admitted fit widths. This is a measurement
producer; it grants no domain-of-validity statement or bridge interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from local_llm_lab import hf_text, spawn
from local_llm_lab.probes import device_bridge as D
from local_llm_lab.probes import sae_bridge as B
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens


class PairingRefusal(ValueError):
    """Stable refusal name followed by the exact failed check."""

    def __init__(self, name, detail):
        self.name = name
        super().__init__(f"{name}: {detail}")


class TorchResidualPath:
    """Read block outputs at repository layers, retaining only the requested token.

    The wrapped object is the fitter/capture HFLensModel (or a tiny CPU fixture with the
    same layers/forward interface). Observe width and dtype at the hook, before selecting
    row zero. Refuse nonidentical replicas rather than silently choosing a batch member.
    """

    def __init__(self, wrapped, device="cpu", *, capture_model=None):
        self.wrapped = wrapped
        self.device = device
        self.capture_model = capture_model

    def read(self, input_ids, width, layers, position, *, capture_forward=None):
        import torch

        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device).reshape(1, -1)
        batch = ids.expand(width, -1)
        values, handles = {}, []

        def hook(layer):
            def record(_module, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                if (
                    h.ndim != 3
                    or h.dtype != torch.float32
                    or tuple(h.shape[:2]) != tuple(batch.shape)
                ):
                    raise PairingRefusal("path_mismatch", f"layer {layer}: dtype/width/context")
                selected = h[:, position, :].detach().cpu().contiguous().numpy()
                if not np.isfinite(selected).all():
                    raise PairingRefusal("non_finite_values", f"forward layer {layer}")
                if any(row.tobytes() != selected[0].tobytes() for row in selected[1:]):
                    raise PairingRefusal("path_mismatch", f"layer {layer}: replicas differ")
                values[layer] = selected[0].copy()

            return record

        try:
            for layer in layers:
                handles.append(self.wrapped.layers[layer - 1].register_forward_hook(hook(layer)))
            with torch.no_grad():
                if capture_forward is None:
                    self.wrapped.forward(batch)
                else:
                    if self.capture_model is None:
                        raise PairingRefusal("path_mismatch", "capture model callable is absent")
                    # The as-run capture uses the outer LM, its default cache setting,
                    # per-row attention request, and two kept logits. The fit uses the
                    # bare decoder through HFLensModel.forward(use_cache=False).
                    self.capture_model(
                        input_ids=batch,
                        output_attentions=capture_forward["output_attentions"],
                        logits_to_keep=torch.tensor(
                            capture_forward["logits_to_keep"], device=self.device
                        ),
                    )
        finally:
            for handle in handles:
                handle.remove()
        if set(values) != set(layers):
            raise PairingRefusal("path_mismatch", "requested block did not execute")
        return values


def _load_path(snapshot, manifest):
    from local_llm_lab import device

    device.pin(seed=0)
    import torch

    from local_llm_lab.pipeline.lens_fitting.upstream import load_upstream

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, report = hf_text.load_text_causal_lm(
        snapshot, dtype="bfloat16", attn_implementation="eager", device=manifest["device"]
    )
    model.requires_grad_(False)
    model.eval().to(torch.float32)
    upstream = load_upstream()
    from jlens.hf import HFLensModel

    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise PairingRefusal("path_mismatch", "model is not entirely float32")
    return TorchResidualPath(
        HFLensModel(model, tokenizer=None), manifest["device"], capture_model=model
    ), {
        "checkpoint_sha256": report["sha256"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "attention": report["attn_implementation"],
        "device": report["device"],
        "flags": device.describe(),
        "kernel_flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "upstream": upstream.provenance(),
    }


def _admit(args):
    snapshot, archive, sidecar = args["model_snapshot"], args["lens_archive"], args["lens_sidecar"]
    try:
        if sidecar.resolve(strict=True) != archive.with_suffix(".json").resolve(strict=True):
            raise ValueError("explicit sidecar is not the admitted archive's sidecar")
        metadata = D._json(sidecar)
        checkpoint = hf_text.checkpoint_metadata(snapshot)
        expected = D._json(args["checkpoint_manifest"])
        D._equal(checkpoint["sha256"], expected, "checkpoint hash manifest")
        index_sha = D._checkpoint_index(snapshot, checkpoint)
        lens, metadata = load_admitted_device_lens(
            archive, checkpoint=snapshot, model_base=metadata["model"]["base"]
        )
        reference_archive = args.get("capture_reference_archive")
        reference_sidecar = args.get("capture_reference_sidecar")
        if (reference_archive is None) != (reference_sidecar is None):
            raise ValueError("capture reference archive and sidecar must be supplied together")
        reference_lens, reference_metadata = lens, metadata
        if reference_archive is not None:
            reference_archive, reference_sidecar = Path(reference_archive), Path(reference_sidecar)
            if reference_sidecar.resolve(strict=True) != reference_archive.with_suffix(
                ".json"
            ).resolve(strict=True):
                raise ValueError("capture reference sidecar is not its admitted archive's sidecar")
            reference_lens, reference_metadata = load_admitted_device_lens(
                reference_archive, checkpoint=snapshot, model_base=metadata["model"]["base"]
            )
    except (ValueError, KeyError, OSError) as exc:
        raise PairingRefusal("hash_mismatch", str(exc)) from exc
    if (
        not args["layers"]
        or len(set(args["layers"])) != len(args["layers"])
        or any(type(layer) is not int or layer not in lens.maps for layer in args["layers"])
    ):
        raise PairingRefusal(
            "cell_outside_capture", "requested layers must be unique admitted source layers"
        )
    try:
        # Check the seal before parsing: damaged JSON is still a capture hash mismatch.
        seal = D._json(args["positions"])
        D._equal(
            D._hash(args["capture_dir"] / "manifest.json"),
            seal["capture_files_sha256"]["manifest.json"],
            "capture manifest.json hash",
        )
    except (ValueError, KeyError, OSError) as exc:
        raise PairingRefusal("hash_mismatch", str(exc)) from exc
    try:
        manifest = D._json(args["capture_dir"] / "manifest.json")
        D._equal(manifest.get("load_report_sha256"), expected, "capture checkpoint hash manifest")
        cells, provenance = D._capture_cells(
            args["capture_dir"],
            args["positions"],
            args["corpus"],
            snapshot,
            reference_metadata,
            reference_lens,
            {"checkpoint_sha256": expected, "capture_model": manifest.get("model")},
            args["layers"],
        )
    except (ValueError, OSError) as exc:
        detail = str(exc)
        name = (
            "hash_mismatch"
            if any(word in detail for word in ("hash", "sha256"))
            else "non_finite_values"
            if "not finite" in detail
            else "cell_outside_capture"
        )
        raise PairingRefusal(name, detail) from exc
    # The historical lens verifies the capture event. The new lens identifies the
    # proposed pairing; substituting its digest into the old event would forge history.
    cells = [
        {
            **item,
            "identity": B.reading_identity(
                metadata["nu"],
                lens_sha256=lens.sha256,
                capture_dtype=manifest["precision"],
                capture_batch=manifest["width"],
                reading=item["reading"],
            ),
        }
        for item in cells
    ]
    if metadata["nu"]["precision"]["fit_dtype"] != "float32" or manifest["width"] != 1:
        raise PairingRefusal("path_mismatch", "T5 requires float32 fit and float32 width-1 capture")
    if not isinstance(manifest.get("device"), str) or not manifest["device"]:
        raise PairingRefusal("path_mismatch", "capture device is absent")
    for item in cells:
        sampled = item["capture_forward"]["output_attentions"]
        if (
            type(sampled) is not bool
            or not isinstance(manifest.get("sample"), list)
            or sampled != (item["cell"]["row"] in manifest["sample"])
        ):
            raise PairingRefusal("path_mismatch", "capture attention sample/index disagree")
    loaded = next(
        e for e in D._rows(args["capture_dir"] / "progress.jsonl") if e.get("event") == "loaded"
    )
    if loaded.get("attn", "eager") != "eager":
        raise PairingRefusal("path_mismatch", "capture attention must be eager")
    provenance.pop("domain_of_validity")  # Measurement does not issue an interpretation license.
    provenance.update(
        {
            "schema_version": 1,
            "producer": "measure_pairings",
            "checkpoint_sha256": expected,
            "checkpoint_manifest_sha256": D._hash(args["checkpoint_manifest"]),
            "checkpoint_index_sha256": index_sha,
            "lens_sha256": lens.sha256,
            "lens_sidecar_sha256": D._hash(sidecar),
            "nu_sha256": metadata["nu_sha256"],
            "fit_widths": metadata["fit_widths"],
        }
    )
    if reference_archive is not None:
        provenance["capture_reference"] = {
            "archive": str(reference_archive),
            "lens_sha256": reference_lens.sha256,
            "sidecar_sha256": D._hash(reference_sidecar),
            "nu_sha256": reference_metadata["nu_sha256"],
            "source_archive_sha256": reference_metadata["source_archive_sha256"],
        }
    return cells, metadata, provenance


def _source():
    root = Path(__file__).resolve().parents[3]
    result = spawn.run(
        ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [
        Path(__file__),
        Path(D.__file__),
        Path(B.__file__),
        root / "scripts/measure_pairings.py",
    ]
    return result.stdout.strip(), {str(p.relative_to(root)): D._hash(p) for p in paths}


def measure_pairings(
    *,
    model_snapshot,
    checkpoint_manifest,
    lens_archive,
    lens_sidecar,
    capture_dir,
    positions,
    corpus,
    layers,
    output,
    capture_reference_archive=None,
    capture_reference_sidecar=None,
):
    """Write a new model-keyed registration table after every requested cell succeeds."""
    args = {
        key: Path(value)
        for key, value in {
            "model_snapshot": model_snapshot,
            "checkpoint_manifest": checkpoint_manifest,
            "lens_archive": lens_archive,
            "lens_sidecar": lens_sidecar,
            "capture_dir": capture_dir,
            "positions": positions,
            "corpus": corpus,
            "output": output,
        }.items()
    }
    output = args["output"]
    if output.exists() or output.is_symlink():
        raise PairingRefusal("existing_output", str(output))
    args["layers"] = list(layers)
    args["capture_reference_archive"] = capture_reference_archive
    args["capture_reference_sidecar"] = capture_reference_sidecar
    layers = args["layers"]
    cells, metadata, provenance = _admit(args)
    commit, source_hashes = _source()
    path, runtime = _load_path(args["model_snapshot"], provenance["manifest"])
    if runtime["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
        raise PairingRefusal("hash_mismatch", "loaded model checkpoint receipt")
    widths = [row["forward_batch"] for row in metadata["fit_widths"]]
    registrations = {str(layer): [] for layer in layers}
    measurements = {str(layer): [] for layer in layers}
    for item in cells:
        cell = item["cell"]
        capture = path.read(
            item["input_ids"],
            1,
            layers,
            cell["token_index"],
            capture_forward=item["capture_forward"],
        )
        for layer in layers:
            h = capture[layer]
            if not np.isfinite(h).all():
                raise PairingRefusal("non_finite_values", f"width-1 layer {layer}, cell {cell}")
            if h.dtype != np.float32 or h.tobytes() != item["residuals"][layer].tobytes():
                raise PairingRefusal("capture_residual_mismatch", f"layer {layer}, cell {cell}")
        fitted = [
            path.read(item["input_ids"], width, layers, cell["token_index"]) for width in widths
        ]
        for layer in layers:
            denominator = float(np.linalg.norm(capture[layer].astype(np.float64)))
            if denominator == 0 or not np.isfinite(denominator):
                raise PairingRefusal(
                    "non_finite_values", f"undefined relative term at layer {layer}"
                )
            terms = [
                float(
                    np.linalg.norm(h[layer].astype(np.float64) - capture[layer].astype(np.float64))
                    / denominator
                )
                for h in fitted
            ]
            if not np.isfinite(terms).all():
                raise PairingRefusal("non_finite_values", f"fit displacement at layer {layer}")
            # The ordered terms stay explicit; a scalar cannot describe a mixed path by itself.
            relative = max(terms)
            basis = B.T5_PAIRING_BASIS_PREFIX + (
                f"Measured WS-D residual displacement on corpus row {cell['row']} "
                f"({item['prompt_identity']!r}), prompt sha256 {item['prompt_sha256']}, "
                f"{cell['position']} token {cell['token_index']}, context {cell['context_tokens']}; "
                f"float32 fit forward widths {widths} versus float32 capture width 1, "
                f"eager pre-final-norm block output at repository layer {layer}. "
                f"Width-1 bytes reproduce the sealed stored residual. "
                f"Each term is ||h_fit-h_capture||_2/||h_capture||_2: {terms}; "
                "relative is the maximum across the ordered chunks, with no cancellation."
            )
            registrations[str(layer)].append(
                {"pair": item["identity"], "relative": relative, "basis": basis}
            )
            measurements[str(layer)].append(
                {
                    "cell": cell,
                    "capture_forward": item["capture_forward"],
                    "prompt_identity": item["prompt_identity"],
                    "prompt_sha256": item["prompt_sha256"],
                    "input_ids_sha256": hashlib.sha256(
                        json.dumps(item["input_ids"], separators=(",", ":")).encode()
                    ).hexdigest(),
                    "capture_residual_sha256": hashlib.sha256(capture[layer].tobytes()).hexdigest(),
                    "chunk_terms": [
                        {"forward_batch": width, "relative": term}
                        for width, term in zip(widths, terms, strict=True)
                    ],
                }
            )
    result = {
        metadata["model"]["base"]: {"measured_pairings": registrations},
        "_provenance": {
            **provenance,
            "source_commit": commit,
            "implementation_sha256": source_hashes,
            "runtime": runtime,
            "measurements": measurements,
        },
    }
    B._supplied_pairing_table(result)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise PairingRefusal("existing_output", str(output)) from exc
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-snapshot",
        "checkpoint-manifest",
        "lens-archive",
        "lens-sidecar",
        "capture-dir",
        "positions",
        "corpus",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=int, nargs="+")
    parser.add_argument("--capture-reference-archive", type=Path)
    parser.add_argument("--capture-reference-sidecar", type=Path)
    try:
        result = measure_pairings(**vars(parser.parse_args(argv)))
    except PairingRefusal as exc:
        parser.exit(2, f"{exc}\n")
    print(
        json.dumps(
            {
                "verdict": "measured",
                "cells": sum(
                    len(records) for records in result["_provenance"]["measurements"].values()
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
