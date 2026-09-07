"""The three gated-delta forms on random inputs at the model's own dimensions, no checkpoint.

Inference runs ``gated_delta_kernel`` (the Metal kernel, eval mode); training mode runs the
``gated_delta_ops`` reference loop, which the unified run replaced with the chunkwise form at
chunk 256 (``install_chunkwise_gated_delta``). The chunkwise test suite stops at 257 tokens and
chunks 16/64/128; arm A's rows ran to 2,688 tokens at chunk 256. This script reports the worst
absolute and relative error between every pair of forms at production shapes and lengths, and
whether the chunkwise form meets the R32 stage-2 tolerance (rtol 1e-4, atol 1e-5) against the
reference loop there. Keys are rms-normalised and scaled as the block does.

    .venv/bin/python scripts/recurrence_forms_check.py --out <json>
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops  # noqa: E402
from local_llm_lab.training.gated_delta_chunkwise import gated_delta_chunkwise_ops  # noqa: E402

HK, HV, DK, DV = 16, 32, 128, 128   # Qwen3.5-4B linear attention (config.json)
RTOL, ATOL = 1e-4, 1e-5             # R32 stage 2


def inputs(tokens: int, seed: int, dtype):
    mx.random.seed(seed)
    key = mx.random.normal((1, tokens, HK, DK))
    key = (DK ** -0.5) * mx.fast.rms_norm(key, None, 1e-6)
    q = (DK ** -1.0) * mx.fast.rms_norm(mx.random.normal((1, tokens, HK, DK)), None, 1e-6)
    d = {"q": q.astype(dtype), "k": key.astype(dtype), "v": mx.random.normal((1, tokens, HV, DV)).astype(dtype),
         "g": mx.random.uniform(low=0.9, high=1.0, shape=(1, tokens, HV)).astype(mx.float32),
         "beta": mx.random.uniform(shape=(1, tokens, HV)).astype(mx.float32)}
    mx.eval(*d.values())
    return d


def worst(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    diff = mx.abs(a - b); scale = mx.maximum(mx.abs(b).max(), mx.array(1.0))
    return {"max_abs": float(diff.max()), "max_rel_to_scale": float(diff.max() / scale),
            "mean_abs": float(diff.mean()), "allclose_r32": bool(mx.allclose(a, b, rtol=RTOL, atol=ATOL).item())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", nargs="*", type=int, default=[257, 1024, 2688, 3072])
    ap.add_argument("--chunks", nargs="*", type=int, default=[128, 256])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args(); t0 = time.time()
    dtype = getattr(mx, args.dtype)
    results = {"parameters": {"heads_k": HK, "heads_v": HV, "dim_k": DK, "dim_v": DV, "dtype": args.dtype, "rtol": RTOL, "atol": ATOL, "seed": 7}, "rows": []}
    for tokens in args.tokens:
        d = inputs(tokens, 7, dtype)
        state0 = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
        y_k, s_k = gated_delta_kernel(d["q"], d["k"], d["v"], d["g"], d["beta"], state0, None); mx.eval(y_k, s_k)
        y_o, s_o = gated_delta_ops(d["q"], d["k"], d["v"], d["g"], d["beta"], state0, None); mx.eval(y_o, s_o)
        row = {"tokens": tokens, "kernel_vs_ops": {"y": worst(y_k, y_o), "state": worst(s_k, s_o)}, "chunkwise": {}}
        for chunk in args.chunks:
            y_c, s_c = gated_delta_chunkwise_ops(d["q"], d["k"], d["v"], d["g"], d["beta"], state0, None, chunk=chunk); mx.eval(y_c, s_c)
            row["chunkwise"][str(chunk)] = {"n_chunks": -(-tokens // chunk), "vs_ops": {"y": worst(y_c, y_o), "state": worst(s_c, s_o)}, "vs_kernel": {"y": worst(y_c, y_k), "state": worst(s_c, s_k)}}
        results["rows"].append(row)
        print(json.dumps({"tokens": tokens, "kernel_vs_ops_y": row["kernel_vs_ops"]["y"], **{f"chunk{c}_vs_ops_y": row["chunkwise"][c]["vs_ops"]["y"] for c in row["chunkwise"]}, **{f"chunk{c}_vs_kernel_y": row["chunkwise"][c]["vs_kernel"]["y"]["max_abs"] for c in row["chunkwise"]}}), flush=True)
    results["parameters"]["elapsed_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True); json.dump(results, open(args.out, "w"), indent=1)
    print(json.dumps({"event": "done", "elapsed_s": results["parameters"]["elapsed_s"]}))


if __name__ == "__main__":
    main()
