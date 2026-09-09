"""The negative control at every lens layer: how a fixed dictionary's readout moves with depth.

Two predictions of the three-layer control's ordering failed -- each map's distance to the
identity, then each map's distance to layer 18's -- so rather than fit a third, this measures the
curve. The same 512 features (the exact draw the A1 record made, so its three points must
reproduce) are read through every lens layer and their top-10 sets compared with layer 18's.
Both distance statistics are recorded beside the overlaps so the non-relationship is on one table.

    PYTHONPATH=src python research/records/SAE-J-BRIDGE-2026-09-09/control_sweep.py --block 17
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = "google/gemma-scope-2-4b-it"
SUITE = "resid_post_all"
WIDTH_TAG = "16k"
L0_TAG = "l0_small"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("a1"))
    args = parser.parse_args()

    from local_llm_lab import runlock
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps
    from local_llm_lab.probes import sae_bridge as B
    from local_llm_lab.project import configure_local_cache

    configure_local_cache()
    from huggingface_hub import hf_hub_download

    primary = runlock.primary_checkout_root()
    folder = f"{SUITE}/layer_{args.block}_width_{WIDTH_TAG}_{L0_TAG}"
    params = Path(hf_hub_download(REPO, f"{folder}/params.safetensors", local_files_only=True))
    config = Path(hf_hub_download(REPO, f"{folder}/config.json", local_files_only=True))
    lens_npz = primary / "models" / "jlens" / "gemma-3-4b-it_jacobian_lens.npz"
    ckpt = primary / "models" / "gemma-3-4b-it-bf16"
    side = json.loads(lens_npz.with_suffix(".json").read_text())
    conf = json.loads((ckpt / "config.json").read_text())
    t = conf.get("text_config", conf)
    lens = LensMaps.load(
        lens_npz, expected_sha256=side["npz_sha256"], hidden_size=int(t["hidden_size"]),
        num_layers=int(t["num_hidden_layers"]), identity=LensIdentity.from_dict(side["model"]),
    )
    d = B.load_dictionary(params, config, source=f"{REPO}/{folder}")
    u = B.load_unembedding(ckpt)
    layer = B.hook_alignment(d, lens)
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    # This script runs before it is committed, so HEAD alone would name a commit that does not
    # contain it. Its own content hash is the provenance that survives that.
    import hashlib

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    # The A1 record's exact draw: six named features first, then the control subset.
    rng = np.random.default_rng(20260909)
    _ = rng.choice(d.width, 6, replace=False)
    sub = rng.choice(d.width, args.features, replace=False)

    identity = np.eye(lens.hidden_size, dtype=np.float32)
    scale = float(np.sqrt(lens.hidden_size))  # the sidecar's identity_distance is ||.||_F / sqrt(d)
    t0 = time.time()
    rows = []
    for other in sorted(lens.maps):
        c = B.negative_control(d, u, lens, layer, other, k=args.k, features=sub)
        rows.append({
            "layer": other,
            "mean_overlap": c["mean_overlap"],
            "share_identical": c["share_identical"],
            "dist_to_identity": float(np.linalg.norm(lens.maps[other] - identity) / scale),
            "dist_to_layer": float(np.linalg.norm(lens.maps[other] - lens.maps[layer]) / scale),
        })
    # And the raw direction with the gain, no lens at all: what "no map" reads as.
    here, _ = B.feature_scores(B._subset(d, sub), u, lens.maps[layer], k=args.k)
    none, _ = B.feature_scores(B._subset(d, sub), u, None, k=args.k)
    rows.append({
        "layer": None,
        "mean_overlap": float(B.overlap_at_k(here, none).mean()),
        "share_identical": float((B.overlap_at_k(here, none) == 1.0).mean()),
        "dist_to_identity": 0.0,
        "dist_to_layer": float(np.linalg.norm(identity - lens.maps[layer]) / scale),
        "note": "no lens: the decoder direction read raw through the gain",
    })
    elapsed = time.time() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "source_commit": commit,
        "source_script_sha256": script_sha,
        "readout_layer": layer,
        "k": args.k,
        "features": int(args.features),
        "feature_draw": "default_rng(20260909): six named features drawn first, then this subset",
        "elapsed_s": round(elapsed, 1),
        "rows": rows,
    }
    (args.out / "control_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{'layer':>5} {'overlap@10':>10} {'identical':>9} "
        f"{'|J-I|/sqrt(d)':>14} {'|J-J18|/sqrt(d)':>16}"
    )
    for r in rows:
        name = "none" if r["layer"] is None else r["layer"]
        print(
            f"{name!s:>5} {r['mean_overlap']:>10.3f} {r['share_identical']:>9.3f} "
            f"{r['dist_to_identity']:>14.2f} {r['dist_to_layer']:>16.2f}"
        )
    print(
        f"{len(rows)} readouts of {args.features} features in {elapsed:.0f}s; "
        "wrote control_sweep.json"
    )


if __name__ == "__main__":
    main()
