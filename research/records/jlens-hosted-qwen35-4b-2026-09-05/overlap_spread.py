"""10g re-measurement: overlap of top-10 far-source sets and query cosine on the SAME read positions
spread across the context (25, 35, 50, 65, 80, 90, 100 percent), per head, against the
query-randomised null; per-head cross; the eighteen entries from the cos60 cross inspected by name."""
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

import json, sys, time, subprocess, os, datetime
from pathlib import Path
import numpy as np
REPO = Path("/Users/daniel.tipton/Desktop/An app"); sys.path.insert(0, str(REPO / "src"))
import mlx.core as mx
mx.set_cache_limit(2 * 1024 ** 3)
def _other_model_processes():
    pids = subprocess.run(["pgrep", "-f", "agent-v2-|run_hosted_lens|contrast_reactions|calibrate_alpha|overlap_test|query_cosine|overlap_spread|mlx_lm convert"], capture_output=True, text=True).stdout.split()
    mine = {str(os.getpid()), str(os.getppid())}; found = []
    for pid in pids:
        if pid in mine: continue
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout.strip().replace("\n", " ")
        if not cmd or "pgrep" in cmd or "ps -o" in cmd: continue
        if any(tag in cmd.split("<<")[0] for tag in ("agent-v2-", "run_hosted_lens", "contrast_reactions", "calibrate_alpha.py", "overlap_test.py", "query_cosine.py", "overlap_spread.py", "mlx_lm convert")): found.append(f"{pid}: {cmd[:120]}")
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
A = REPO / "outputs/probes/jlens-hosted-qwen35-4b-2026-09-05"
calib = json.load(open(A / "alpha_calibration.json"))
tau = {0: np.zeros((24, 32)), 1: np.zeros((24, 32))}
for hi in calib["head_info"]: tau[hi["context"]][hi["block"]] = np.array(hi["tau_head"])
long_heads = [(b, h) for b in range(24) for h in range(32) if tau[0][b, h] > 1000 and tau[1][b, h] > 1000]
prev = json.load(open(A / "overlap_by_cosine.json"))["rows"]
eighteen = {(r["context"], r["block"], r["head"]) for r in prev if r["ratio"] > 2 and r["cos60"] < 0.5}
print("long-gate heads:", len(long_heads), "; entries flagged on cos60 (ratio>2, cos60<0.5):", len(eighteen))
def alpha_rows_batch(Q, kk, g, beta, t):
    W = Q.copy(); out = np.zeros((t + 1, Q.shape[0]))
    for r in range(t, -1, -1):
        kr = kk[r]; d = W @ kr; out[r] = beta[r] * d; W = g[r] * (W - beta[r] * d[:, None] * kr[None, :])
    return out
rng = np.random.default_rng(0); GAP = 257; K = 10; NDRAW = 200
FRACS = (0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
records = []
for ci, ids in enumerate(contexts):
    captures.clear(); view.residuals(ids, [view.num_layers]); T = len(ids)
    positions = sorted(set(int(round(T * f)) - 1 for f in FRACS))
    for (b, h) in long_heads:
        cap = captures[b]
        q = np.array(cap["q"].astype(mx.float32))[0]; k = np.array(cap["k"].astype(mx.float32))[0]
        Hk = q.shape[1]; kk = k[:, h // (32 // Hk)]; qq = q[:, h // (32 // Hk)]
        beta = np.array(mx.sigmoid(cap["b"]).astype(mx.float32))[0][:, h]; g = np.array(gd.compute_g(cap["A_log"], cap["a"], cap["dt_bias"]).astype(mx.float32))[0][:, h]
        qnorm = np.linalg.norm(qq, axis=1).mean()
        real_sets, null_sets = {}, {}
        for t in positions:
            n_elig = t - GAP + 1
            if n_elig < 50: continue
            Ar_real = np.abs(alpha_rows_batch(qq[t][None, :], kk, g, beta, t))[: n_elig, 0]
            real_sets[t] = set(np.argsort(Ar_real)[-K:].tolist())
            Qr = rng.standard_normal((NDRAW, kk.shape[1])); Qr *= qnorm / np.linalg.norm(Qr, axis=1, keepdims=True)
            Ar = np.abs(alpha_rows_batch(Qr, kk, g, beta, t))[: n_elig]
            null_sets[t] = [set(np.argsort(Ar[:, d])[-K:].tolist()) for d in range(NDRAW)]
        ts = sorted(real_sets)
        if len(ts) < 2: continue
        reals, nulls_m = [], []
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                t1, t2 = ts[i], ts[j]; n_common = t1 - GAP + 1
                r1 = {s for s in real_sets[t1] if s < n_common}; r2 = {s for s in real_sets[t2] if s < n_common}
                reals.append(len(r1 & r2))
                nulls_m.append(np.mean([len({s for s in null_sets[t1][d] if s < n_common} & {s for s in null_sets[t2][d] if s < n_common}) for d in range(NDRAW)]))
        Qn = qq[ts] / (np.linalg.norm(qq[ts], axis=1, keepdims=True) + 1e-12); C = Qn @ Qn.T; iu = np.triu_indices(len(ts), 1)
        Kn = kk[ts] / (np.linalg.norm(kk[ts], axis=1, keepdims=True) + 1e-12); Ck = Kn @ Kn.T
        records.append({"context": ci, "block": b, "head": h, "key_head": h // (32 // Hk), "positions": ts, "pairs": len(reals),
                        "overlap": float(np.mean(reals)), "query_null": float(np.mean(nulls_m)), "ratio": float(np.mean(reals) / max(np.mean(nulls_m), 1e-9)),
                        "query_cosine_same_positions": float(np.median(C[iu])), "key_cosine_same_positions": float(np.median(Ck[iu])),
                        "flagged_on_cos60": (ci, b, h) in eighteen})
    print(f"context {ci} done: T={T}, positions={positions}, {time.time()-t0:.0f}s")
ov = np.array([r["overlap"] for r in records]); nul = np.array([r["query_null"] for r in records]); ratio = np.array([r["ratio"] for r in records]); cq = np.array([r["query_cosine_same_positions"] for r in records])
from scipy.stats import spearmanr
def quad(ct, rt): return {"retrieval-like": int(((ratio > rt) & (cq < ct)).sum()), "static": int(((ratio > rt) & (cq >= ct)).sum()), "null_lowcos": int(((ratio <= rt) & (cq < ct)).sum()), "null_highcos": int(((ratio <= rt) & (cq >= ct)).sum())}
flagged = [r for r in records if r["flagged_on_cos60"]]
summary = {"entries": len(records), "read_positions_fractions": FRACS, "K": K, "GAP": GAP, "NDRAW": NDRAW,
           "mean_overlap": float(ov.mean()), "mean_query_null": float(nul.mean()), "median_ratio": float(np.median(ratio)),
           "query_cosine_same_positions_percentiles": {p: float(np.percentile(cq, p)) for p in (10, 25, 50, 75, 90)},
           "spearman_overlap_vs_cosine": float(spearmanr(ov, cq).correlation),
           "cross": {f"ratio>{rt} & cos<{ct}": quad(ct, rt) for ct, rt in ((0.6, 1.5), (0.5, 1.5), (0.5, 2.0), (0.3, 1.5), (0.3, 2.0))},
           "flagged_eighteen": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "positions"} for r in flagged],
           "flagged_eighteen_summary": {"n": len(flagged), "ratio_median": float(np.median([r["ratio"] for r in flagged])) if flagged else None,
                                        "query_cosine_median": float(np.median([r["query_cosine_same_positions"] for r in flagged])) if flagged else None,
                                        "retrieval_like_at_ratio2_cos0.5": int(sum(1 for r in flagged if r["ratio"] > 2 and r["query_cosine_same_positions"] < 0.5))}}
provenance = {"file": "overlap_spread.json", "generated": datetime.datetime.now().astimezone().isoformat(), "script": "overlap_spread.py",
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "parameters": {"K": K, "GAP": GAP, "NDRAW": NDRAW, "seed": 0, "read_positions": "fractions " + ", ".join(str(f) for f in FRACS) + " of each context", "contexts": "EXP-001 jsweep cases 0 and 21", "head_selection": "tau > 1000 on both contexts"},
              "model": "mlx-community/Qwen3.5-4B-MLX-4bit", "authorisation": "Director direct, standing lift as heard by the Chief; run by the Chief; no other model process alive"}
json.dump({"summary": summary, "provenance": provenance, "records": records}, open(A / "overlap_spread.json", "w"), indent=1)
# also give overlap_by_cosine.json the summary and provenance it lacked
obc = json.load(open(A / "overlap_by_cosine.json"))
obc["summary"] = {"note": "cross of overlap_test.json (four read positions at 50/75/90/100 percent) against query_cosine.json; superseded for the F8 question by overlap_spread.json, which measures overlap and cosine on the same spread positions",
                  "entries": len(obc["rows"]), "spearman_overlap_vs_cos_read": obc.get("spearman_overlap_cosread")}
obc["provenance"] = dict(provenance, file="overlap_by_cosine.json", script="inline cross computed 2026-09-05 late evening from overlap_test.json and query_cosine.json", parameters={"inputs": ["overlap_test.json", "query_cosine.json"]})
json.dump(obc, open(A / "overlap_by_cosine.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "flagged_eighteen"}, indent=1))
print("flagged eighteen on the spread measure:")
for r in summary["flagged_eighteen"]: print("  ", r)
print("elapsed", round(time.time() - t0, 1), "s")
