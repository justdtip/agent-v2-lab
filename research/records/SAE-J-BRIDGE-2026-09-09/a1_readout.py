"""Stage A1 on the real matrices: the dictionary readout at one layer, no model loaded.

Reads the every-layer Gemma Scope 2 residual dictionary at one block, the Jacobian lens at the
layer that block's output is, and the unembedding and final-norm gain straight from the bf16
checkpoint's safetensors. Computes, for every feature, the top-k tokens of ``L d_i`` through the
lens, and runs the order's acceptance checks: the two-ways agreement, the negative control across
layers with its deliberate-mismatch failure, and the second amendment's raw-versus-gain
discriminator against the dictionary's own shipped ``top_logits``. The raw arm is run and recorded
before the gain arm is looked at.

CPU BLAS only. No model forward, no model lock. The box state at the time of the run is recorded.

    PYTHONPATH=src python research/records/SAE-J-BRIDGE-2026-09-09/a1_readout.py --block 17 --k 10
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = "google/gemma-scope-2-4b-it"
SUITE = "resid_post_all"
WIDTH_TAG = "16k"
L0_TAG = "l0_small"


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block", type=int, required=True, help="Gemma Scope block; reads layer block+1"
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--control-layers", type=int, nargs="+", default=[2, 10, 30])
    parser.add_argument("--control-features", type=int, default=512)
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

    box = {
        "window": runlock.read_window().raw if runlock.read_window() else None,
        "lock_held": runlock.default_lock_path().exists(),
    }

    t0 = time.time()
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
    J = lens.maps[layer]
    t_load = time.time() - t0
    print(
        f"loaded: dictionary {d.width}x{d.hidden_size}, lens layer {layer}, "
        f"vocab {u.vocab_size}; {t_load:.1f}s, rss {_rss_gib():.2f} GiB"
    )

    root = Path(__file__).resolve().parents[3]

    def git(*argv: str) -> str:
        done = subprocess.run(["git", *argv], cwd=root, capture_output=True, text=True)
        return done.stdout.strip()

    args.out.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "provenance": B.bridge_provenance(
            d, lens, u, layer, dictionary_repo=REPO, dictionary_folder=folder
        ),
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain", "--", str(Path(__file__).resolve()))),
        "box_state_at_run": box,
        "k": args.k,
        "chunk": args.chunk,
        "timing_s": {"load": round(t_load, 1)},
    }

    # --- the second amendment's discriminator: raw arm first, recorded before the gain arm ----
    shipped_path = Path(__file__).with_name(f"shipped_examples_layer{args.block}_{L0_TAG}.npz")
    if shipped_path.is_file():
        shipped = np.load(shipped_path)["top_tokens"][:, : args.k]
        t1 = time.time()
        raw_idx, _ = B.feature_scores(d, u, None, k=args.k, gain=False, chunk=args.chunk)
        raw_overlap = float(B.overlap_at_k(shipped, raw_idx).mean())
        result["convention_raw_arm"] = {"overlap": raw_overlap, "recorded_before_gain_arm": True}
        print(
            f"raw arm overlap@{args.k} with shipped top_tokens: {raw_overlap:.3f}  "
            f"({time.time()-t1:.0f}s)"
        )
        t1 = time.time()
        gain_idx, _ = B.feature_scores(d, u, None, k=args.k, gain=True, chunk=args.chunk)
        check = B.convention_check(shipped, raw_idx, gain_idx)
        result["convention_check"] = check
        print(f"gain arm overlap@{args.k}: {check['gain_overlap']:.3f}  ({time.time()-t1:.0f}s)")
        print("conclusion:", check["conclusion"])
        if check["stop"]:
            result["stopped"] = True
            (args.out / f"layer{layer}.json").write_text(json.dumps(result, indent=2) + "\n")
            raise SystemExit(
                "discriminator says orientation or indexing: stopping before the lens readout"
            )
    else:
        result["convention_check"] = {"skipped": f"no shipped readout at {shipped_path.name}"}

    # --- A1 through the lens ---------------------------------------------------------------
    t1 = time.time()
    idx, val = B.feature_scores(d, u, J, k=args.k, gain=True, chunk=args.chunk)
    t_a1 = time.time() - t1
    result["timing_s"]["a1_lens_readout"] = round(t_a1, 1)
    print(
        f"A1 through the lens: {d.width} features x top-{args.k} in {t_a1:.0f}s, "
        f"rss {_rss_gib():.2f} GiB"
    )
    np.savez_compressed(
        args.out / f"layer{layer}_top{args.k}.npz", top_tokens=idx, top_scores=val,
        layer=np.array(layer), block=np.array(args.block),
    )

    # --- two ways ---------------------------------------------------------------------------
    rng = np.random.default_rng(20260909)
    freq = np.load(shipped_path)["feature_frequencies"] if shipped_path.is_file() else None
    named = sorted(set(rng.choice(d.width, 6, replace=False).tolist()
                       + ([int(i) for i in np.argsort(-freq)[:2]] if freq is not None else [])))
    worst, order_differs, set_differs = 0.0, [], []
    for i in named:
        direct = B.feature_score_column(d, u, J, i)
        di, dv = B.top_tokens(direct, args.k)
        if set(idx[i].tolist()) != set(di[0].tolist()):
            set_differs.append(int(i))
        elif idx[i].tolist() != di[0].tolist():
            order_differs.append(int(i))  # a near-tie reordered between two float32 paths
        gap = np.abs(np.sort(val[i])[::-1] - np.sort(dv[0])[::-1]).max()
        worst = max(worst, float(gap / (np.abs(dv[0]).max() + 1e-12)))
    result["two_ways"] = {
        "features": named,
        "top_set_mismatches": set_differs,
        "order_only_differences": order_differs,
        "worst_relative_score_gap": worst,
    }
    assert not set_differs, f"two ways disagree on membership for features {set_differs}"
    print(
        f"two ways on {len(named)} features: top-{args.k} sets identical, "
        f"{len(order_differs)} order-only differences, worst relative score gap {worst:.2e}"
    )

    # --- negative control --------------------------------------------------------------------
    sub = rng.choice(d.width, args.control_features, replace=False)
    controls = []
    for other in args.control_layers:
        if other not in lens.maps or other == layer:
            continue
        c = B.negative_control(d, u, lens, layer, other, k=args.k, features=sub)
        controls.append(c)
        print(
            f"control vs layer {other}: mean overlap {c['mean_overlap']:.3f}, "
            f"identical {c['share_identical']:.3f}, distinguishes {c['distinguishes']}"
        )
    same = B.negative_control(d, u, lens, layer, layer, k=args.k, features=sub)
    assert same["mean_overlap"] == 1.0 and not same["distinguishes"]
    controls.append(same)
    print(
        f"control vs itself: overlap {same['mean_overlap']:.1f}, "
        f"distinguishes {same['distinguishes']} (the check shown to fail)"
    )
    result["negative_control"] = controls

    # --- readable table, headed as ours -----------------------------------------------------
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        table = []
        for i in named:
            table.append({
                "feature": int(i),
                "ours_read_from_layer": layer,
                "top_tokens": [tok.decode([int(x)]) for x in idx[i]],
            })
        result["readout_sample_ours"] = table
    except Exception as error:  # readability only; the numbers above are the artefact
        result["readout_sample_ours"] = {"unavailable": str(error)[:120]}

    result["timing_s"]["total"] = round(time.time() - t0, 1)
    result["peak_rss_gib"] = round(_rss_gib(), 2)
    (args.out / f"layer{layer}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"wrote {args.out / f'layer{layer}.json'}; total {result['timing_s']['total']}s, "
        f"peak rss {result['peak_rss_gib']} GiB (macOS ru_maxrss is bytes)"
    )


if __name__ == "__main__":
    main()
