"""R3c source-set overlap test for the long-gate heads.

For each candidate head (gate constant > 1000 tokens on both contexts), at each sampled read
position t, take the top-10 source positions by |alpha[t, s]| among sources at gap >= 257. Compare
the overlap of those sets across read positions within the head to two nulls:
  (a) hypergeometric: 10*10/n expected overlap for n eligible sources;
  (b) query-randomised: keep the head's real beta_s and keys, replace q_t by random unit vectors
      (scaled to the real ||q||), recompute alpha via the same backward scan, take top-10, overlap.
Memory heads should exceed (b); random-alignment heads sit at (b).
"""
from __future__ import annotations
# RECORD, NOT A LAUNCHER (2026-09-05). This script produced figures cited in
# design_specifications/pending/WP3-TRANSPORT-WEIGHTS-DESIGN-2026-09-05.md and the hosted-lens results memo.
# It loads the model directly with an ad hoc process check and must not be started by hand again: the
# one operating constraint is one model load at a time, enforced by the loader wrapper and model-run lock
# of issue 83. Runnable versions arrive as package entry points under issues 79 (hosted lens) and 81 (transport).
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit("refusing to run: this file is a record of the 2026-09-05 measurements, not a launcher; "
              "re-run through the package entry points of issues 79 and 81 once the issue 83 lock has landed")

import json, sys, time, subprocess
from pathlib import Path
import numpy as np
REPO = Path("/Users/daniel.tipton/Desktop/An app"); sys.path.insert(0, str(REPO / "src"))
import mlx.core as mx
mx.set_cache_limit(2 * 1024 ** 3)
import os as _os
def _other_model_processes():
    """Pids running a model-loading entry point, excluding this process and the shell that launched it."""
    pids = subprocess.run(["pgrep", "-f", "agent-v2-|run_hosted_lens|contrast_reactions|calibrate_alpha|overlap_test|query_cosine|mlx_lm convert"], capture_output=True, text=True).stdout.split()
    mine = {str(_os.getpid()), str(_os.getppid())}
    found = []
    for pid in pids:
        if pid in mine: continue
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout.strip().replace("\n", " ")
        if not cmd or "pgrep" in cmd or "ps -o" in cmd: continue
        if any(tag in cmd.split("<<")[0] for tag in ("agent-v2-", "run_hosted_lens", "contrast_reactions", "calibrate_alpha.py", "overlap_test.py", "query_cosine.py", "mlx_lm convert")):
            found.append(f"{pid}: {cmd[:120]}")
    return found
others = _other_model_processes()
if others: print("REFUSED: another model-loading process is alive:\n" + "\n".join(others)); sys.exit(2)
from mlx_lm.models import qwen3_5 as q35, gated_delta as gd
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.evaluate import load_policy
from local_llm_lab.pipeline.jlens import encode
from local_llm_lab.pipeline.tasks import make_jspace_tasks
from local_llm_lab.probes.jspace_sweep import select_cases

captures = []
_orig = q35.gated_delta_update
def hooked(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    out, st = _orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel); mx.eval(out)
    captures.append({"q": q, "k": k, "a": a, "b": b, "A_log": A_log, "dt_bias": dt_bias}); return out, st
q35.gated_delta_update = hooked
t0 = time.time()
spec = load_model_spec("qwen35-4b"); model, tokenizer, view, resolved = load_policy(spec, None)
tasks = make_jspace_tasks("jsweep", 720, 20260902); cases = select_cases(tasks, tokenizer, spec=spec, probe_step=3)
contexts = [encode(tokenizer, cases[i]["prompt"]) for i in (0, 21)]
calib = json.load(open(REPO / "outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/alpha_calibration.json"))
tau = {0: np.zeros((24, 32)), 1: np.zeros((24, 32))}
for hi in calib["head_info"]: tau[hi["context"]][hi["block"]] = np.array(hi["tau_head"])
long_heads = [(b, h) for b in range(24) for h in range(32) if tau[0][b, h] > 1000 and tau[1][b, h] > 1000]
print("candidate long-gate heads (both contexts > 1000):", len(long_heads))

def alpha_row(q_t, kk, g, beta, t):
    """alpha[t, s] for s <= t for one head. kk: (T, Dk) keys of this head; q_t: (Dk,)"""
    w = q_t.copy(); out = np.zeros(t + 1)
    for r in range(t, -1, -1):
        kr = kk[r]; out[r] = beta[r] * (kr @ w); w = g[r] * (w - beta[r] * kr * (kr @ w))
    return out

def alpha_rows_batch(Q, kk, g, beta, t):
    """Same scan for many queries at once. Q: (n, Dk) -> (t+1, n)."""
    W = Q.copy(); out = np.zeros((t + 1, Q.shape[0]))
    for r in range(t, -1, -1):
        kr = kk[r]; d = W @ kr; out[r] = beta[r] * d; W = g[r] * (W - beta[r] * d[:, None] * kr[None, :])
    return out

rng = np.random.default_rng(0)
GAP = 257; K = 10; NDRAW = 200
records = []
for ci, ids in enumerate(contexts):
    captures.clear(); view.residuals(ids, [view.num_layers]); T = len(ids)
    positions = sorted(set([T - 1] + [int(round(T * f)) - 1 for f in (0.5, 0.75, 0.9)]))   # read positions with >= 257-gap sources
    for (b, h) in long_heads:
        cap = captures[b]
        q = np.array(cap["q"].astype(mx.float32))[0]; k = np.array(cap["k"].astype(mx.float32))[0]
        Hk = q.shape[1]; kk = k[:, h // (32 // Hk)]; qq = q[:, h // (32 // Hk)]
        beta = np.array(mx.sigmoid(cap["b"]).astype(mx.float32))[0][:, h]; g = np.array(gd.compute_g(cap["A_log"], cap["a"], cap["dt_bias"]).astype(mx.float32))[0][:, h]
        qnorm = np.linalg.norm(qq, axis=1).mean()
        real_sets = {}; null_sets = {}
        for t in positions:
            n_elig = t - GAP + 1
            if n_elig < 50: continue
            A = np.abs(alpha_row(qq[t], kk, g, beta, t))[: n_elig]          # sources with gap >= 257: s <= t - 257
            real_sets[t] = set(np.argsort(A)[-K:].tolist())
            Qr = rng.standard_normal((NDRAW, kk.shape[1])); Qr *= qnorm / np.linalg.norm(Qr, axis=1, keepdims=True)
            Ar = np.abs(alpha_rows_batch(Qr, kk, g, beta, t))[: n_elig]          # (n_elig, NDRAW)
            null_sets[t] = [set(np.argsort(Ar[:, d])[-K:].tolist()) for d in range(NDRAW)]
        ts = sorted(real_sets)
        if len(ts) < 2: continue
        # overlap across read positions within the head, restricted to sources eligible for both (s <= min(t) - 257)
        pair_stats = []
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                t1, t2 = ts[i], ts[j]; n_common = t1 - GAP + 1
                r1 = {s for s in real_sets[t1] if s < n_common}; r2 = {s for s in real_sets[t2] if s < n_common}
                real_ov = len(r1 & r2)
                hyper = len(r1) * len(r2) / n_common
                nulls = []
                for d in range(NDRAW):
                    a1 = {s for s in null_sets[t1][d] if s < n_common}; a2 = {s for s in null_sets[t2][d] if s < n_common}
                    nulls.append(len(a1 & a2))
                pair_stats.append({"t1": t1, "t2": t2, "n_common": n_common, "real_overlap": real_ov, "hypergeometric": hyper, "query_null_mean": float(np.mean(nulls)),
                                   "query_null_p90": float(np.percentile(nulls, 90)), "query_null_p99": float(np.percentile(nulls, 99)), "null_draws": NDRAW,
                                   "real_exceeds_null_fraction": float(np.mean(np.array(nulls) < real_ov))})
        records.append({"context": ci, "block": b, "head": h, "pairs": pair_stats,
                        "mean_real": float(np.mean([p["real_overlap"] for p in pair_stats])), "mean_hyper": float(np.mean([p["hypergeometric"] for p in pair_stats])),
                        "mean_query_null": float(np.mean([p["query_null_mean"] for p in pair_stats]))})
    print(f"context {ci} done, heads {len([r for r in records if r['context']==ci])}, elapsed {time.time()-t0:.0f}s")
real = np.array([r["mean_real"] for r in records]); hyp = np.array([r["mean_hyper"] for r in records]); qn = np.array([r["mean_query_null"] for r in records])
summary = {"heads_tested": len(records), "K": K, "gap_min": GAP,
           "mean_real_overlap": float(real.mean()), "mean_hypergeometric": float(hyp.mean()), "mean_query_null": float(qn.mean()),
           "heads_real_above_query_null": int((real > qn).sum()), "heads_real_above_2x_query_null": int((real > 2 * qn).sum()),
           "heads_real_at_or_below_query_null": int((real <= qn).sum()),
           "real_overlap_percentiles": {q: float(np.percentile(real, q)) for q in (10, 50, 90)}, "query_null_percentiles": {q: float(np.percentile(qn, q)) for q in (10, 50, 90)},
           "ratio_real_to_query_null_median": float(np.median(real / np.maximum(qn, 1e-9)))}
pairs_all = [p for r in records for p in r["pairs"]]
summary["pairs"] = len(pairs_all)
summary["pairs_real_above_query_p90"] = int(sum(1 for p in pairs_all if p["real_overlap"] > p["query_null_p90"]))
summary["pairs_real_above_query_p99"] = int(sum(1 for p in pairs_all if p["real_overlap"] > p["query_null_p99"]))
summary["pairs_median_fraction_of_null_draws_below_real"] = float(np.median([p["real_exceeds_null_fraction"] for p in pairs_all]))
summary["read_positions_per_context"] = "50, 75, 90 and 100 percent of the context"
import subprocess as _sp, datetime as _dt
provenance = {"file": "overlap_test.json", "generated": _dt.datetime.now().astimezone().isoformat(), "script": "overlap_test.py",
              "git_head": _sp.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "parameters": {"K": K, "GAP": GAP, "NDRAW": NDRAW, "seed": 0, "contexts": "EXP-001 jsweep cases 0 and 21 (seed 20260902, probe step 3)", "read_positions": "50, 75, 90, 100 percent",
                             "head_selection": "tau_head > 1000 tokens on both contexts, from alpha_calibration.json"},
              "model": "mlx-community/Qwen3.5-4B-MLX-4bit", "authorisation": "Director direct, standing lift as heard by the Chief; run by the Chief; no other model process alive"}
json.dump({"summary": summary, "provenance": provenance, "records": records}, open(REPO / "outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/overlap_test.json", "w"), indent=1)
print(json.dumps(summary, indent=1)); print("elapsed", round(time.time() - t0, 1), "s")
