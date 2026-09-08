"""Registered matrix comparison; no model runtime, derivatives or token-readout claims."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))


def metrics(candidate, reference):
    if (
        candidate.shape != reference.shape
        or candidate.ndim != 2
        or candidate.shape[0] != candidate.shape[1]
    ):
        raise ValueError("matching square maps required")
    d = candidate.shape[0]
    if d < 1 or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("nonempty finite maps required")
    aa = bb = ab = diff = aid = bid = 0.0
    for start in range(0, d, 128):
        a = candidate[start : start + 128].astype(np.float64)
        b = reference[start : start + 128].astype(np.float64)
        aa += float(np.sum(a * a))
        bb += float(np.sum(b * b))
        ab += float(np.sum(a * b))
        diff += float(np.sum((a - b) ** 2))
        identity = np.zeros_like(a)
        identity[np.arange(a.shape[0]), np.arange(start, start + a.shape[0])] = 1
        aid += float(np.sum((a - identity) ** 2))
        bid += float(np.sum((b - identity) ** 2))
    return {
        "regression_frobenius": math.sqrt(aa),
        "hosted_frobenius": math.sqrt(bb),
        "cosine": None if aa == 0 or bb == 0 else ab / math.sqrt(aa * bb),
        "relative_difference_to_hosted": None if bb == 0 else math.sqrt(diff / bb),
        "regression_identity_distance": math.sqrt(aid / d),
        "hosted_identity_distance": math.sqrt(bid / d),
    }


def self_check():
    m = metrics(np.eye(2), np.eye(2))
    assert m["cosine"] == 1 and m["relative_difference_to_hosted"] == 0
    assert m["regression_identity_distance"] == m["hosted_identity_distance"] == 0
    m = metrics(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert m["cosine"] == 0 and abs(m["relative_difference_to_hosted"] - math.sqrt(2)) < 1e-12
    assert m["regression_identity_distance"] == math.sqrt(1.5)
    assert metrics(2 * np.eye(3), np.eye(3))["relative_difference_to_hosted"] == 1
    assert metrics(np.zeros((2, 2)), np.eye(2))["cosine"] is None
    # Cross the accumulation block boundary with a nonsymmetric, exactly known map.
    a = np.eye(130)
    a[129, 0] = 2
    assert metrics(a, np.eye(130))["relative_difference_to_hosted"] == math.sqrt(4 / 130)


def main():
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps, file_sha256

    root = Path(__file__).resolve().parent
    output = root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    self_check()
    fitted = PROJECT / "models/jlens/gemma3-4b-bf16-prose-regression-native.npz"
    hosted = Path(
        "/Users/daniel.tipton/Desktop/An app/models/jlens/gemma-3-4b-it_jacobian_lens.npz"
    )
    meta = json.loads(fitted.with_suffix(".json").read_bytes())
    hmeta = json.loads(hosted.with_suffix(".json").read_bytes())
    depth, d = meta["num_layers"], meta["hidden_size"]
    expected = LensIdentity(
        "gemma3-4b-bf16", "/Users/daniel.tipton/Desktop/An app/models/gemma-3-4b-it-bf16", depth
    )
    reg = LensMaps.load(
        fitted,
        expected_sha256=meta["npz_sha256"],
        hidden_size=d,
        num_layers=depth,
        identity=expected,
    )
    ref = LensMaps.load(
        hosted,
        expected_sha256="a14ab264fc47de9c59b3183a8783a60ca6f1dee2ea635e88223d5b2d67b03193",
        hidden_size=d,
        num_layers=depth,
        identity=LensIdentity("gemma3-4b", "google/gemma-3-4b-it", depth),
    )
    if set(reg.maps) != set(ref.maps) or set(reg.maps) != set(range(1, depth)):
        raise ValueError("all nonfinal layers must be present in both instruments")
    rows = [
        {
            "layer": layer,
            "normalized_depth": layer / depth,
            **metrics(reg.maps[layer], ref.maps[layer]),
        }
        for layer in sorted(reg.maps)
    ]
    result = {
        "purpose": "descriptive comparison of two fitting objectives; not a validation gate",
        "regression_identity": expected.as_dict(),
        "hosted_identity": ref.identity.as_dict(),
        "regression_sha256": reg.sha256,
        "hosted_sha256": ref.sha256,
        "regression_sidecar_sha256": file_sha256(fitted.with_suffix(".json")),
        "hosted_sidecar_sha256": file_sha256(hosted.with_suffix(".json")),
        "hosted_source_sha256": hmeta.get("source_sha256"),
        "precision": "our official BF16 conversion versus hosted BF16 fit, stored FP16",
        "context": "both fits at 128 tokens; no sliding/global contrast claim",
        "corpus": "our wikitext validation versus hosted wikitext train",
        "arithmetic": (
            "float64 reductions of stored float32 maps; hosted storage promoted from float16"
        ),
        "rows": rows,
        "final_layer_baseline": {
            "layer": depth,
            "normalized_depth": 1.0,
            **metrics(np.eye(d), np.eye(d)),
        },
        "scope": "matrix geometry only; no token predictions or causal effects measured",
    }
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (root / "comparison.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"layers": len(rows), "comparison": str(output)}))


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
        print("matrix metric self-check passed")
    else:
        main()
