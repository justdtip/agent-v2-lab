"""Issue #90's second item: does the chunkwise recurrence share the unrolled path's superlinear
backward?

Pre-registered at research/records/RECURRENCE-EXPONENT-2026-09-07/PRE-REGISTRATION.md before this
was run. Both forms in one harness on the same tensors and the same clock, forward and backward
timed separately, because the question is the backward's exponent and the training-step evidence
cannot separate the recurrence from the attention term.

The unrolled path is re-fitted here too: its 1.57 comes from 125-1,000 tokens in another harness,
so if it does not reproduce over these sizes the premise is range-dependent and gets restated
before the conclusion.

    .venv/bin/python scripts/recurrence_exponent_sweep.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mlx_lm.models.gated_delta import gated_delta_ops  # noqa: E402

from local_llm_lab.runlock import hold_model_run_lock  # noqa: E402
from local_llm_lab.training.gated_delta_chunkwise import gated_delta_chunkwise_ops  # noqa: E402

HK, HV, DK, DV = 16, 32, 128, 128          # Qwen3.5-4B linear attention (config.json)
WINDOW_GIB = 0.6 * 17.76                   # R47: above this a run needs a declared window


def inputs(tokens: int, seed: int) -> dict[str, mx.array]:
    """The shapes the block hands the recurrence, built as
    ``scripts/recurrence_forms_check.py`` builds them."""
    mx.random.seed(seed)
    key = (DK**-0.5) * mx.fast.rms_norm(mx.random.normal((1, tokens, HK, DK)), None, 1e-6)
    q = (DK**-1.0) * mx.fast.rms_norm(mx.random.normal((1, tokens, HK, DK)), None, 1e-6)
    d = {
        "q": q.astype(mx.float32),
        "k": key.astype(mx.float32),
        "v": mx.random.normal((1, tokens, HV, DV)).astype(mx.float32),
        "g": mx.random.uniform(low=0.9, high=1.0, shape=(1, tokens, HV)).astype(mx.float32),
        "beta": mx.random.uniform(shape=(1, tokens, HV)).astype(mx.float32),
    }
    mx.eval(*d.values())
    return d


def _call(form: str, d: dict[str, mx.array], state0: mx.array, chunk: int):
    if form == "chunkwise":
        return gated_delta_chunkwise_ops(
            d["q"], d["k"], d["v"], d["g"], d["beta"], state0, None, chunk=chunk
        )
    return gated_delta_ops(d["q"], d["k"], d["v"], d["g"], d["beta"], state0, None)


def timed(form: str, d, state0, chunk: int, repeats: int) -> dict[str, float]:
    """Forward and backward medians, each after a discarded warm-up.

    The warm-up is not politeness: the first call at any shape pays kernel compilation, which
    contaminated the 4,096 row of the CTX-EFFICIENCY sweep and was caught only because the
    resulting curve stopped being monotone. Every timed region ends in ``mx.eval`` so nothing is
    left lazy for the next measurement to pay for.
    """
    def forward():
        y, s = _call(form, d, state0, chunk)
        mx.eval(y, s)
        return y

    def loss(v):
        y, _ = (
            gated_delta_chunkwise_ops(
                d["q"], d["k"], v, d["g"], d["beta"], state0, None, chunk=chunk
            )
            if form == "chunkwise"
            else gated_delta_ops(d["q"], d["k"], v, d["g"], d["beta"], state0, None)
        )
        return (y.astype(mx.float32) ** 2).sum()

    grad = mx.grad(loss)
    forward()                                   # warm-up, discarded
    mx.eval(grad(d["v"]))                       # warm-up, discarded
    mx.reset_peak_memory()

    fwd, bwd = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        forward()
        fwd.append(time.perf_counter() - start)
        start = time.perf_counter()
        mx.eval(grad(d["v"]))
        bwd.append(time.perf_counter() - start)
    fwd.sort()
    bwd.sort()
    return {
        "forward_s": fwd[len(fwd) // 2],
        "backward_s": bwd[len(bwd) // 2],
        "forward_spread_s": fwd[-1] - fwd[0],
        "backward_spread_s": bwd[-1] - bwd[0],
        "peak_gib": mx.get_peak_memory() / 2**30,
    }


def fit(sizes: list[int], values: list[float]) -> dict:
    """Log-log least squares overall, and per adjacent segment."""
    n = len(sizes)
    lx = [math.log(t) for t in sizes]
    ly = [math.log(v) for v in values]
    mean_x, mean_y = sum(lx) / n, sum(ly) / n
    denom = sum((x - mean_x) ** 2 for x in lx)
    covar = sum((x - mean_x) * (y - mean_y) for x, y in zip(lx, ly, strict=True))
    slope = covar / denom if denom else float("nan")
    segments = [
        {
            "from": sizes[i - 1],
            "to": sizes[i],
            "exponent": math.log(values[i] / values[i - 1]) / math.log(sizes[i] / sizes[i - 1]),
        }
        for i in range(1, n)
    ]
    return {"exponent": slope, "segments": segments}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", nargs="*", type=int, default=[512, 1024, 2048, 2688, 4096, 8192])
    ap.add_argument("--chunks", nargs="*", type=int, default=[128, 256])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    started = time.time()
    rows: list[dict] = []
    stopped: str | None = None
    # Not a context manager: it takes the lock for the rest of the process and releases at
    # interpreter exit, including on SIGTERM, an uncaught exception or a KeyboardInterrupt.
    lock = hold_model_run_lock(
        command=" ".join(sys.argv), session="D-CRO recurrence-exponent sweep"
    )
    print(f"lock held at {lock}", flush=True)
    for tokens in args.tokens:
        d = inputs(tokens, 7)
        state0 = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
        for form in ("chunkwise", "unrolled"):
            for chunk in (args.chunks if form == "chunkwise" else [0]):
                row = {"tokens": tokens, "form": form, "chunk": chunk}
                row.update(timed(form, d, state0, chunk, args.repeats))
                rows.append(row)
                print(
                    f"{form:9s} chunk {chunk:3d} T={tokens:5d}  "
                    f"fwd {row['forward_s']:.4f}s  bwd {row['backward_s']:.4f}s  "
                    f"peak {row['peak_gib']:.2f} GiB",
                    flush=True,
                )
                # R47: stop before a size that would need a declared window, do not run it first
                if row["peak_gib"] > WINDOW_GIB:
                    stopped = (
                        f"peak {row['peak_gib']:.2f} GiB at {tokens} tokens exceeds R47's "
                        f"{WINDOW_GIB:.2f} GiB; a declared window is needed"
                    )
                    break
            if stopped:
                break
        if stopped:
            break

    fits = {}
    for form in ("chunkwise", "unrolled"):
        for chunk in sorted({r["chunk"] for r in rows if r["form"] == form}):
            got = [r for r in rows if r["form"] == form and r["chunk"] == chunk]
            if len(got) < 3:
                continue
            sizes = [r["tokens"] for r in got]
            fits[f"{form}-{chunk}"] = {
                "forward": fit(sizes, [r["forward_s"] for r in got]),
                "backward": fit(sizes, [r["backward_s"] for r in got]),
            }

    out = {
        "parameters": {"heads_k": HK, "heads_v": HV, "dim_k": DK, "dim_v": DV,
                       "dtype": "float32", "seed": 7, "repeats": args.repeats,
                       "tokens": args.tokens, "chunks": args.chunks,
                       "window_threshold_gib": WINDOW_GIB},
        "rows": rows, "fits": fits, "stopped": stopped, "elapsed_s": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}" + (f"\nSTOPPED: {stopped}" if stopped else ""))


if __name__ == "__main__":
    main()
