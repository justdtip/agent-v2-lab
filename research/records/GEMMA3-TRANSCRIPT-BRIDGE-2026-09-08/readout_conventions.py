"""Compare declared direct-readout conventions during an owned CPU window.

Descriptive diagnosis only. This does not replace or relax the raw-W A1 gate.
The same predeclared feature IDs are used for all three conventions. No model
forward, nonlinear RMS denominator or full vocabulary-by-feature matrix is used.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    root = Path(__file__).resolve().parents[3]
    driver = runpy.run_path(str(root / "scripts/sae_bridge.py"))
    from local_llm_lab.pipeline.sae_bridge.assets import checked_hash
    from local_llm_lab.pipeline.sae_bridge.readout import identity_baseline

    driver["require_window"]()
    registration_bytes = args.registration.read_bytes()
    registration_sha = hashlib.sha256(registration_bytes).hexdigest()
    reg = json.loads(registration_bytes)
    meta = reg["final_norm_diagnostic"]
    norm_path = args.registration.parent / meta["artifact"]
    checked_hash(norm_path, meta["sha256"])
    with np.load(norm_path, allow_pickle=False) as arrays:
        stored = arrays["stored_weight"]
        gain = arrays["effective_gain"]
    if (
        stored.shape != (reg["hidden_size"],)
        or stored.dtype != np.float32
        or not np.isfinite(stored).all()
        or not np.array_equal(gain, 1 + stored)
    ):
        raise ValueError("normalization-vector artifact violates its declared convention")
    w, _j, _wrong_j, d, _tokenizer, tokens, logits = driver["load_inputs"](reg)
    results = []
    for name, multiplier in (
        ("raw_W", None),
        ("tutorial_stored_weight", stored),
        ("native_channel_gain", gain),
    ):
        driver["require_window"]()
        # Multiplying feature coordinates avoids a second full readout matrix.
        # Rounding can differ from first scaling W; this is declared, not bit equality.
        directions = d if multiplier is None else multiplier[:, None] * d
        result = identity_baseline(
            w,
            directions,
            tokens,
            logits,
            feature_ids=reg["identity_baseline"]["feature_ids"],
            k=reg["k"],
        )
        results.append({"convention": name, "comparison": result})
    driver["require_window"]()
    driver["write_json"](
        args.output,
        {
            "status": "descriptive_comparison",
            "instrument": reg,
            "registration_sha256": checked_hash(args.registration, registration_sha),
            "normalization_artifact_sha256": meta["sha256"],
            "results": results,
            "technique": "same W and feature directions, three declared diagonal channel gains",
            "implementation": "FP32 W @ (gain * d); gain scales D to bound memory",
            "limits": (
                "No nonlinear RMS denominator. Neither the tutorial nor native convention "
                "is established as the shipped artifact generator. Compatibility alone "
                "does not validate J or its layer indexing."
            ),
            "model_forward_count": 0,
        },
    )


if __name__ == "__main__":
    main()
