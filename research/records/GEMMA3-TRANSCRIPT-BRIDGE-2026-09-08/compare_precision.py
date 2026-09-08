"""Compare the paired transcript regression maps after both fits; no model load.

The primary relative denominator is the BF16 map's Frobenius norm. The second
map's norm and the reverse-normalized difference are reported beside it. This
measures the fitted maps, not downstream task performance or causal influence.
"""

import argparse
import json
import math
import runpy
from pathlib import Path

import numpy as np


def compare_arrays(reference, quantized):
    a, b = np.asarray(reference, np.float64), np.asarray(quantized, np.float64)
    if a.ndim != 2 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("map comparison requires equal finite matrices")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    nd = float(np.linalg.norm(b - a))
    return {
        "bf16_frobenius_norm": na,
        "fourbit_frobenius_norm": nb,
        "difference_frobenius_norm": nd,
        "relative_difference_to_bf16": nd / na if na else None,
        "relative_difference_to_fourbit": nd / nb if nb else None,
        "matrix_cosine": float(np.vdot(a, b)) / (na * nb) if na and nb else None,
        "zero_norm_status": "undefined_relative_metric" if not na or not nb else None,
    }


def self_check():
    result = compare_arrays([[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]])
    assert math.isclose(result["bf16_frobenius_norm"], math.sqrt(30))
    assert math.isclose(result["difference_frobenius_norm"], math.sqrt(30))
    assert math.isclose(result["relative_difference_to_bf16"], 1)
    assert math.isclose(result["relative_difference_to_fourbit"], 0.5)
    assert math.isclose(result["matrix_cosine"], 1)
    assert compare_arrays([[0]], [[1]])["relative_difference_to_bf16"] is None
    print("analytic two-map comparison and zero-norm checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path)
    parser.add_argument("--fourbit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.bf16, args.fourbit, args.output)) or args.output.exists():
        parser.error("both lens paths and a new output are required")
    root = Path(__file__).resolve().parents[3]
    # Matrix scanning is CPU work. It shares the combined fitting window.
    runpy.run_path(str(root / "scripts/sae_bridge.py"))["require_window"]()
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps, file_sha256

    metadata = [json.loads(p.with_suffix(".json").read_bytes()) for p in (args.bf16, args.fourbit)]
    for meta, name in zip(metadata, ("gemma3-4b-bf16", "gemma3-4b"), strict=True):
        if meta.get("model_name") != name or meta.get("kind") != "regression":
            raise ValueError("each input must identify the intended precision's regression fit")
        if meta.get("residual_source") != "native" or not meta.get("snapshot"):
            raise ValueError("paired fits require native residuals and snapshot provenance")
    for key in (
        "model",
        "corpus_manifest_sha256",
        "corpus_sequences_sha256",
        "lambda_grid",
        "penalty_rule",
        "hidden_size",
        "num_layers",
        "counts",
    ):
        if metadata[0].get(key) is None or metadata[0][key] != metadata[1].get(key):
            raise ValueError(f"paired fit contract differs: {key}")
    identity = LensIdentity.from_dict(metadata[0]["model"])
    d, depth = metadata[0]["hidden_size"], metadata[0]["num_layers"]
    maps = [
        LensMaps.load(
            path,
            expected_sha256=meta["npz_sha256"],
            hidden_size=d,
            num_layers=depth,
            identity=identity,
        )
        for path, meta in zip((args.bf16, args.fourbit), metadata, strict=True)
    ]
    if any(set(lens.maps) != set(range(1, depth)) for lens in maps):
        raise ValueError("both fits must include every nonfinal layer")
    rows = [
        {"probe_layer": layer, **compare_arrays(maps[0].maps[layer], maps[1].maps[layer])}
        for layer in range(1, depth)
    ]
    result = {
        "finding": "per-layer fitted-map difference under checkpoint quantization",
        "technique": (
            "same captured IDs, scored positions, native forwards and ridge grid; "
            "paired Frobenius differences"
        ),
        "interpretation": "not a performance measure or a validation of a Jacobian lens",
        "model_identity": identity.as_dict(),
        "corpus_manifest_sha256": metadata[0]["corpus_manifest_sha256"],
        "inputs": [
            {
                "path": str(path),
                "npz_sha256": meta["npz_sha256"],
                "metadata_sha256": file_sha256(path.with_suffix(".json")),
                "snapshot": meta["snapshot"],
            }
            for path, meta in zip((args.bf16, args.fourbit), metadata, strict=True)
        ],
        "per_layer": rows,
        "final_layer": {
            "probe_layer": depth,
            "status": "analytic identity in both lenses; not fitted",
        },
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
