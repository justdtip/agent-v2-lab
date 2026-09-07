"""Fit a regression lens at every layer from the model's own forward passes, without storing features.

For each sequence the residual at every layer L = 1..31 and the final pre-norm residual (layer 32)
are taken from one uncached forward; per layer the sufficient statistics XᵀX, XᵀY and tr(YᵀY) are
accumulated (2,560 × 2,560 float32 each), separately for a fitting set and a held-out set of
sequences. One ridge solve per layer, W_L = (XᵀX + λI)⁻¹ XᵀY, with λ chosen per layer on the
held-out reconstruction error from a grid fixed here; nothing is stored per position. The output is
an ``.npz`` in the hosted lens's convention (key ``J{L-1}`` reads repo layer ``L``; ``W`` is stored
so that ``residual @ W.T`` maps to the final space, i.e. ``J = W``), loadable by ``LensMaps``.

Corpora: ``--corpus agentic`` rebuilds every step's prompt plus the model's own recorded turn from
evaluation JSONs (the model doing the tasks); ``--corpus prose`` chunks text files. Both may be
given; each is fitted separately and named in the sidecar.

    .venv/bin/python scripts/fit_regression_lens.py --corpus agentic --evals <json ...> --out <npz>
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from local_llm_lab.agent_protocol import Action  # noqa: E402
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT, assistant_message, build_prompt, tool_message  # noqa: E402

LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)   # multiples of mean diag(XᵀX)/N; fixed before any fit


def agentic_sequences(evals: list[Path], tok, spec, *, max_tokens: int) -> list[dict]:
    """Every step of every trajectory: the runner's prompt at that step plus the model's raw turn."""
    seqs = []
    for path in evals:
        for t in json.load(open(path))["trajectories"]:
            steps = [s for s in t["steps"] if "action" in s]
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["prompt"]}]
            for i, s in enumerate(steps):
                prompt = build_prompt(tok, messages, spec=spec, keep_last=2, generation=True)
                text = prompt + s["raw"]
                ids = list(tok.encode(text))
                if len(ids) <= max_tokens:
                    seqs.append({"source": f"{path.name}:{t['task_id']}:{i}", "ids": ids, "n_prompt": len(tok.encode(prompt))})
                a = s["action"]
                messages.append(assistant_message(s["thought"], Action(a["name"], a["arguments"])))
                messages.append(tool_message(a["name"], s["observation"]))
    return seqs


def prose_sequences(files: list[Path], tok, *, chunk: int) -> list[dict]:
    seqs = []
    for f in files:
        ids = list(tok.encode(f.read_text()))
        for k in range(0, len(ids) - chunk + 1, chunk):
            seqs.append({"source": f"{f.name}:{k}", "ids": ids[k:k + chunk], "n_prompt": 0})
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["agentic", "prose"], required=True)
    ap.add_argument("--evals", nargs="*", type=Path, default=[])
    ap.add_argument("--files", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--prose-chunk", type=int, default=1024)
    ap.add_argument("--held-out-every", type=int, default=5, help="every k-th sequence is held out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="qwen35-4b")
    args = ap.parse_args(); t0 = time.time()
    import mlx.core as mx
    from local_llm_lab.pipeline.evaluate import load_policy

    spec = load_model_spec(args.model)
    model, tok, view, resolved = load_policy(spec, None); model.eval()
    if args.corpus == "agentic":
        seqs = agentic_sequences(args.evals, tok, spec, max_tokens=args.max_tokens)
    else:
        seqs = prose_sequences(args.files, tok, chunk=args.prose_chunk)
    if args.limit:
        seqs = seqs[: args.limit]
    d, n_layers = view.hidden_size, view.num_layers
    layers = list(range(1, n_layers + 1))
    # accumulators per layer, for the fit set and the held-out set
    acc = {split: {L: {"xtx": mx.zeros((d, d), dtype=mx.float32), "xty": mx.zeros((d, d), dtype=mx.float32), "yty": mx.zeros((), dtype=mx.float32), "n": 0} for L in layers[:-1]} for split in ("fit", "held")}
    positions = {"fit": 0, "held": 0}
    print(json.dumps({"event": "corpus", "corpus": args.corpus, "sequences": len(seqs), "tokens": sum(len(s["ids"]) for s in seqs)}), flush=True)
    for k, s in enumerate(seqs):
        split = "held" if (k % args.held_out_every == 0) else "fit"
        res = view.residuals(s["ids"], layers)
        Y = res[n_layers][0] if res[n_layers].ndim == 3 else res[n_layers]   # (T, d) final pre-norm residual
        for L in layers[:-1]:
            X = res[L][0] if res[L].ndim == 3 else res[L]
            a = acc[split][L]
            a["xtx"] = a["xtx"] + X.T @ X
            a["xty"] = a["xty"] + X.T @ Y
            a["yty"] = a["yty"] + mx.sum(Y * Y)
            a["n"] += X.shape[0]
        mx.eval(*[acc[split][L][key] for L in layers[:-1] for key in ("xtx", "xty", "yty")])
        positions[split] += len(s["ids"])
        del res
        if (k + 1) % 20 == 0:
            print(json.dumps({"event": "progress", "sequences": k + 1, "positions": positions, "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    # solve per layer on the fit set; choose λ on the held-out set
    arrays, stats = {}, {}
    I = mx.eye(d, dtype=mx.float32)
    for L in layers[:-1]:
        f, h = acc["fit"][L], acc["held"][L]
        scale = float(mx.mean(mx.diag(f["xtx"])).item()) / max(f["n"], 1)
        best = None
        for lam in LAMBDA_GRID:
            W = mx.linalg.solve(f["xtx"] + (lam * scale * f["n"]) * I, f["xty"], stream=mx.cpu)   # (d, d): Y ≈ X W
            mx.eval(W)
            # held-out error: tr(YᵀY) − 2 tr(Wᵀ XᵀY) + tr(Wᵀ XᵀX W), relative to tr(YᵀY)
            err = float((h["yty"] - 2 * mx.sum(W * h["xty"]) + mx.sum((h["xtx"] @ W) * W)).item())
            rel = err / max(float(h["yty"].item()), 1e-9)
            if best is None or rel < best[1]:
                best = (lam, rel, W)
        lam, rel, W = best
        arrays[f"J{L - 1}"] = np.array(W.T).astype(np.float16)   # residual @ J.T == residual @ W
        stats[str(L)] = {"lambda_multiple": lam, "held_out_relative_error": round(rel, 5), "held_out_r2": round(1 - rel, 5), "fit_positions": f["n"], "held_positions": h["n"]}
        print(json.dumps({"event": "layer", "layer": L, **stats[str(L)]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(args.out)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    meta = {"kind": "regression lens (ridge, same-position, final pre-norm residual target)", "model": spec.hf_id, "corpus": args.corpus,
            "sources": [str(p) for p in (args.evals or args.files)], "sequences": len(seqs), "positions": positions, "held_out_every": args.held_out_every,
            "lambda_grid_multiples_of_mean_diag": LAMBDA_GRID, "layer_convention": "npz key J<l> reads repo layer l+1; residual @ J.T maps to the final pre-norm residual; layer 32 is the identity and is not stored",
            "per_layer": stats, "npz_sha256": sha, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(meta, open(args.out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps({"event": "done", "npz": str(args.out), "sha256": sha, "elapsed_s": meta["elapsed_s"]}))


if __name__ == "__main__":
    main()
