"""Live calibration of the WP3 reconstruction gate on the cached Qwen3.5-4B.

Hooks gated_delta_update as the block calls it (post-conv, post-norm, post-scale q/k), recomputes
read weights alpha[t, s] by the backward scan for sampled positions t on real contexts, and
reports the reconstruction error of y_t = sum_s alpha[t, s] v_s against (a) the kernel's own
output and (b) a float32 recompute of the same block by the library's ops path. Also previews the
scale-free transport statistics of the WP3 design (mass share by gap bin against the uniform null,
retention ratio to bin 1-4) so the gate and the statistic are calibrated on the same tensors.
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

import json, math, sys, time, os, subprocess
from pathlib import Path
import numpy as np

REPO = Path("/Users/daniel.tipton/Desktop/An app")
sys.path.insert(0, str(REPO / "src"))
import mlx.core as mx
mx.set_cache_limit(2 * 1024 ** 3)

# concurrency rule of record: refuse if another model-loading process is alive
others = subprocess.run(["pgrep", "-fl", "agent-v2-|run_hosted_lens|contrast_reactions|mlx_lm convert"], capture_output=True, text=True).stdout.strip().splitlines()
others = [o for o in others if "calibrate_alpha" not in o and "pgrep" not in o]
if others:
    print("REFUSED: another model-loading process is alive:\n" + "\n".join(others)); sys.exit(2)

from mlx_lm.models import qwen3_5 as q35
from mlx_lm.models import gated_delta as gd
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.evaluate import load_policy
from local_llm_lab.pipeline.jlens import encode

captures = []  # one per block call, in forward order
_orig = q35.gated_delta_update
def hooked(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    out, st = _orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    mx.eval(out)
    captures.append({"q": q, "k": k, "v": v, "a": a, "b": b, "A_log": A_log, "dt_bias": dt_bias, "out": out, "mask": mask,
                     "dtypes": {n: str(t.dtype) for n, t in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b), ("out", out))}, "use_kernel": use_kernel})
    return out, st
q35.gated_delta_update = hooked

t0 = time.time()
spec = load_model_spec("qwen35-4b")
model, tokenizer, view, resolved = load_policy(spec, None)
print("model loaded", round(time.time() - t0, 1), "s; linear blocks:", sum(1 for i in range(view.num_layers) if view.layer_kind(i) == "linear_attention"))

from local_llm_lab.pipeline.tasks import make_jspace_tasks
from local_llm_lab.probes.jspace_sweep import select_cases
tasks = make_jspace_tasks("jsweep", 720, 20260902)
cases = select_cases(tasks, tokenizer, spec=spec, probe_step=3)
contexts = [encode(tokenizer, cases[i]["prompt"]) for i in (0, 21)]
print("contexts:", [len(c) for c in contexts])

def alpha_rows(q, k, g, beta, t, Hk):
    """alpha[t, s] for all s <= t, all value heads at once. q,k: (T,Hk,Dk); g,beta: (T,Hv)."""
    T, Hv = g.shape
    rep = Hv // Hk
    kk = np.repeat(k, rep, axis=1); qq = np.repeat(q, rep, axis=1)     # value head h uses key head h // rep
    w = qq[t].copy()                                                    # (Hv, Dk)
    out = np.zeros((t + 1, Hv))
    for r in range(t, -1, -1):
        kr = kk[r]
        out[r] = beta[r] * np.einsum("hd,hd->h", kr, w)
        w = g[r][:, None] * (w - beta[r][:, None] * kr * np.einsum("hd,hd->h", kr, w)[:, None])
    return out

BINS = [(1, 4), (5, 16), (17, 64), (65, 256), (257, 1024), (1025, 2688)]
results = {"contexts": [len(c) for c in contexts], "blocks": [], "dtypes": None, "use_kernel": None}
worst_abs_kernel = worst_rel_kernel = worst_abs_ops = 0.0
cells = 0
mass = {b: [] for b in BINS}; ret = {b: [] for b in BINS}; unif = {b: [] for b in BINS}; zero_cells = {b: [0, 0] for b in BINS}
gate_tau = []
head_info = []
cell_head = {b: [] for b in BINS}   # (context, block, head) per cell, aligned with mass/ret lists
conc = {b: [] for b in BINS}        # fraction of the bin's |alpha| mass in the top-10 source positions, per cell
conc_uniform = {b: [] for b in BINS}
for ci, ids in enumerate(contexts):
    captures.clear()
    view.residuals(ids, [view.num_layers])   # one forward, all blocks call the hooked update
    T = len(ids)
    positions = sorted(set([T - 1] + [int(round(T * f)) - 1 for f in (0.06, 0.12, 0.25, 0.5, 0.75, 0.9)]))
    print(f"context {ci}: T={T}, blocks captured={len(captures)}, positions={positions}")
    for bi, cap in enumerate(captures):
        q = np.array(cap["q"].astype(mx.float32))[0]; k = np.array(cap["k"].astype(mx.float32))[0]; v = np.array(cap["v"].astype(mx.float32))[0]
        beta = np.array(mx.sigmoid(cap["b"]).astype(mx.float32))[0]
        g = np.array(gd.compute_g(cap["A_log"], cap["a"], cap["dt_bias"]).astype(mx.float32))[0]
        out_k = np.array(cap["out"].astype(mx.float32))[0]
        # float32 ops-path recompute of the same block from the same inputs
        out_ops = np.array(gd.gated_delta_ops(cap["q"].astype(mx.float32), cap["k"].astype(mx.float32), cap["v"].astype(mx.float32),
                                              gd.compute_g(cap["A_log"], cap["a"], cap["dt_bias"]), mx.sigmoid(cap["b"]).astype(mx.float32),
                                              mx.zeros((1, v.shape[1], v.shape[2], k.shape[2]), dtype=mx.float32), None)[0])[0]
        Hk = q.shape[1]; Hv = v.shape[1]
        gate_tau.append(float(-1.0 / np.log(g).mean()))
        logg = np.log(np.clip(g, 1e-30, 1.0))
        tau_head = 1.0 / np.maximum(-logg.mean(axis=0), 1e-12)           # per value head, over all positions: tau = -1 / E[log g] > 0
        # longest stretch of near-open gate per head: max over t of the product of g over the trailing 1024 tokens
        cum = np.cumsum(logg, axis=0)
        span = 1024
        if T > span:
            prod1024 = np.exp((cum[span:] - cum[:-span]).max(axis=0))
        else:
            prod1024 = np.exp(cum[-1])
        head_info.append({"context": ci, "block": bi, "tau_head": tau_head.tolist(), "max_gate_product_over_1024": prod1024.tolist(),
                          "median_beta": np.median(beta, axis=0).tolist(), "beta_p90": np.percentile(beta, 90, axis=0).tolist(),
                          "beta_p99": np.percentile(beta, 99, axis=0).tolist(), "beta_max": beta.max(axis=0).tolist(),
                          "beta_frac_over_0.5": (beta > 0.5).mean(axis=0).tolist(), "beta_frac_over_0.2": (beta > 0.2).mean(axis=0).tolist()})
        blk = {"block": bi, "T": T, "positions": positions, "worst_abs_kernel": 0.0, "worst_rel_kernel": 0.0, "worst_abs_ops": 0.0,
               "per_position_head_abs_kernel": [], "per_position_head_abs_ops": [], "context": ci}
        for t in positions:
            A = alpha_rows(q, k, g, beta, t, Hk)          # (t+1, Hv)
            y = np.einsum("sh,shd->hd", A, v[: t + 1])      # (Hv, Dv)
            per_head_k = np.abs(y - out_k[t]).max(axis=1); per_head_o = np.abs(y - out_ops[t]).max(axis=1)
            blk["per_position_head_abs_kernel"].append([float(x) for x in per_head_k]); blk["per_position_head_abs_ops"].append([float(x) for x in per_head_o])
            err_k = per_head_k.max(); rel_k = err_k / (np.abs(out_k[t]).max() + 1e-12)
            err_o = per_head_o.max()
            blk["worst_abs_kernel"] = max(blk["worst_abs_kernel"], float(err_k)); blk["worst_rel_kernel"] = max(blk["worst_rel_kernel"], float(rel_k)); blk["worst_abs_ops"] = max(blk["worst_abs_ops"], float(err_o))
            worst_abs_kernel = max(worst_abs_kernel, float(err_k)); worst_rel_kernel = max(worst_rel_kernel, float(rel_k)); worst_abs_ops = max(worst_abs_ops, float(err_o)); cells += Hv
            # transport preview (WP3 statistics) at this position
            gaps = t - np.arange(t + 1)                       # gap of source s from t
            absA = np.abs(A)                                  # (t+1, Hv)
            tot = absA.sum(axis=0) + 1e-30
            near = absA[(gaps >= 1) & (gaps <= 4)].mean(axis=0) if t >= 1 else None
            for (lo, hi) in BINS:
                sel = (gaps >= lo) & (gaps <= hi)
                n = int(sel.sum())
                if n == 0: continue
                share = absA[sel].sum(axis=0) / tot
                mass[(lo, hi)].extend(share.tolist()); unif[(lo, hi)].append(n / (t + 1))
                cell_head[(lo, hi)].extend([(ci, bi, h) for h in range(Hv)])
                binA = absA[sel]                                            # (n, Hv)
                top10 = np.sort(binA, axis=0)[-10:].sum(axis=0) / (binA.sum(axis=0) + 1e-30)
                conc[(lo, hi)].extend(top10.tolist()); conc_uniform[(lo, hi)].append(min(10, n) / n)
                if near is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r = np.where(near > 0, absA[sel].mean(axis=0) / near, np.inf)
                    ret[(lo, hi)].extend(r.tolist())
                zero_cells[(lo, hi)][0] += int((absA[sel].max(axis=0) == 0).sum()); zero_cells[(lo, hi)][1] += Hv
        results["blocks"].append(blk)
        if results["dtypes"] is None: results["dtypes"] = cap["dtypes"]; results["use_kernel"] = cap["use_kernel"]
    print(f"  block errors (kernel abs / rel, ops abs): " + ", ".join(f"{b['worst_abs_kernel']:.1e}/{b['worst_rel_kernel']:.1e}/{b['worst_abs_ops']:.1e}" for b in results['blocks'][-len(captures):]))
results["worst_abs_vs_kernel"] = worst_abs_kernel; results["worst_rel_vs_kernel"] = worst_rel_kernel; results["worst_abs_vs_float32_ops"] = worst_abs_ops; results["cells_checked"] = cells
results["gate_tau_tokens_median_over_blocks"] = float(np.median(gate_tau)); results["gate_tau_tokens_max"] = float(np.max(gate_tau))
# per-cell arrays for both statistics (the Head's check: retention quantiles must be matched by mass-share quantiles)
tp = {}
for (lo, hi) in BINS:
    m = np.array(mass[(lo, hi)]); r = np.array(ret[(lo, hi)]); u = np.array(unif[(lo, hi)])
    if m.size == 0:
        continue
    null_med = float(np.median(u))
    # zero rule: a cell whose bin-1-4 mean is exactly zero has an undefined ratio; np division gave inf/nan; exclude and count
    r_def = r[np.isfinite(r)]
    joint = int(((r > 0.05) & np.isfinite(r) & (m > null_med)).sum()) if r.size == m.size else None
    tp[f"{lo}-{hi}"] = {
        "cells": int(m.size), "uniform_null_median": null_med,
        "mass_share": {"p50": float(np.percentile(m, 50)), "p90": float(np.percentile(m, 90)), "p99": float(np.percentile(m, 99)), "max": float(m.max()),
                        "cells_above_uniform_null": int((m > null_med).sum())},
        "retention_to_bin_1_4": {"defined_cells": int(r_def.size), "undefined_zero_denominator": int(r.size - r_def.size),
                                 "p50": float(np.percentile(r_def, 50)) if r_def.size else None, "p90": float(np.percentile(r_def, 90)) if r_def.size else None,
                                 "p99": float(np.percentile(r_def, 99)) if r_def.size else None, "cells_above_0.05": int((r_def > 0.05).sum())},
        "joint_named_exception_cells (retention>0.05 AND mass_share>uniform null)": joint,
        "heads_exactly_zero": zero_cells[(lo, hi)],
    }
results["transport_preview"] = tp
# relate joint cells to per-head gate constants
tau_lookup = {(hi_["context"], hi_["block"]): (np.array(hi_["tau_head"]), np.array(hi_["max_gate_product_over_1024"])) for hi_ in head_info}
expl = {}
for (lo, hi) in BINS[-2:]:
    m = np.array(mass[(lo, hi)]); r = np.array(ret[(lo, hi)]); u = float(np.median(unif[(lo, hi)]))
    heads = cell_head[(lo, hi)]
    joint = (r > 0.05) & np.isfinite(r) & (m > u)
    taus = np.array([tau_lookup[(c, b)][0][h] for (c, b, h) in heads]); prods = np.array([tau_lookup[(c, b)][1][h] for (c, b, h) in heads])
    distinct_heads = sorted({(b, h) for (c, b, h), j in zip(heads, joint) if j})
    expl[f"{lo}-{hi}"] = {
        "joint_cells": int(joint.sum()), "distinct_(block,head)_in_joint": len(distinct_heads),
        "tau_head_median_joint": float(np.median(taus[joint])) if joint.any() else None, "tau_head_median_other": float(np.median(taus[~joint])),
        "tau_head_p10_joint": float(np.percentile(taus[joint], 10)) if joint.any() else None,
        "max_gate_product_1024_median_joint": float(np.median(prods[joint])) if joint.any() else None, "max_gate_product_1024_median_other": float(np.median(prods[~joint])),
        "joint_cells_with_tau_over_1000": int(((taus > 1000) & joint).sum()), "all_cells_with_tau_over_1000": int((taus > 1000).sum()),
        "distinct_joint_heads_by_block": {str(b): sum(1 for (bb, h) in distinct_heads if bb == b) for b in range(24)},
        "joint_cells_by_block": {str(b): int(sum(1 for (c, bb, h), j in zip(heads, joint) if j and bb == b)) for b in range(24)},
    }
# mechanism: are the joint cells write-once memory heads (beta spikes, concentrated mass) or stale smears (flat low beta, flat mass)?
mech = {}
beta_lookup = {(hi_["context"], hi_["block"]): hi_ for hi_ in head_info}
gp_all = np.concatenate([np.array(hi_["max_gate_product_over_1024"]) for hi_ in head_info]); bm_all = np.concatenate([np.array(hi_["median_beta"]) for hi_ in head_info])
from scipy.stats import spearmanr
rho = spearmanr(gp_all, bm_all).correlation
for (lo, hi) in BINS[-2:]:
    m = np.array(mass[(lo, hi)]); r = np.array(ret[(lo, hi)]); u = float(np.median(unif[(lo, hi)])); c = np.array(conc[(lo, hi)]); cu = float(np.median(conc_uniform[(lo, hi)]))
    heads = cell_head[(lo, hi)]
    joint = (r > 0.05) & np.isfinite(r) & (m > u)
    b_med = np.array([beta_lookup[(cc, bb)]["median_beta"][h] for (cc, bb, h) in heads]); b_p99 = np.array([beta_lookup[(cc, bb)]["beta_p99"][h] for (cc, bb, h) in heads])
    b_max = np.array([beta_lookup[(cc, bb)]["beta_max"][h] for (cc, bb, h) in heads]); b_f05 = np.array([beta_lookup[(cc, bb)]["beta_frac_over_0.5"][h] for (cc, bb, h) in heads])
    spike = (b_p99 > 0.5)                                         # at least 1 percent of positions write strongly
    concentrated = c > 3 * cu                                     # top-10 sources carry > 3x the uniform share of the bin's mass
    mech[f"{lo}-{hi}"] = {
        "joint_cells": int(joint.sum()),
        "beta_median_joint": float(np.median(b_med[joint])) if joint.any() else None, "beta_median_other": float(np.median(b_med[~joint])),
        "beta_p99_median_joint": float(np.median(b_p99[joint])) if joint.any() else None, "beta_p99_median_other": float(np.median(b_p99[~joint])),
        "beta_max_median_joint": float(np.median(b_max[joint])) if joint.any() else None,
        "frac_positions_beta_over_0.5_median_joint": float(np.median(b_f05[joint])) if joint.any() else None, "frac_positions_beta_over_0.5_median_other": float(np.median(b_f05[~joint])),
        "top10_concentration_uniform": cu,
        "top10_concentration_median_joint": float(np.median(c[joint])) if joint.any() else None, "top10_concentration_p90_joint": float(np.percentile(c[joint], 90)) if joint.any() else None,
        "top10_concentration_median_other": float(np.median(c[~joint])),
        "joint_cells_with_beta_spike (p99>0.5)": int((joint & spike).sum()), "joint_cells_concentrated (top10 > 3x uniform)": int((joint & concentrated).sum()),
        "joint_cells_spike_AND_concentrated": int((joint & spike & concentrated).sum()),
        "distinct_heads_spike_AND_concentrated": len({(bb, h) for (cc, bb, h), j in zip(heads, joint & spike & concentrated) if j}),
    }
results["mechanism"] = {"spearman_gate_product_vs_median_beta": float(rho), "long_bins": mech}
all_tau = np.concatenate([np.array(hi_["tau_head"]) for hi_ in head_info])
per_block_long = {}
for hi_ in head_info:
    t_ = np.array(hi_["tau_head"]); per_block_long.setdefault(hi_["block"], []).append(int((t_ > 1000).sum()))
results["per_head_gate"] = {"tau_head_percentiles_over_all_(context,block,head)": {q: float(np.percentile(all_tau, q)) for q in (1, 10, 25, 50, 75, 90, 99)},
                            "tau_head_max": float(all_tau.max()), "heads_with_tau_over_1000": int((all_tau > 1000).sum()), "heads_with_tau_over_256": int((all_tau > 256).sum()), "n": int(all_tau.size),
                            "heads_with_tau_over_1000_by_block (max over the two contexts)": {str(b): max(v) for b, v in sorted(per_block_long.items())},
                            "explanation_of_long_bin_joint_cells": expl}
results["head_info"] = head_info
results["per_cell"] = {f"{lo}-{hi}": {"mass_share": [float(x) for x in mass[(lo, hi)]], "retention": [float(x) if np.isfinite(x) else None for x in ret[(lo, hi)]]} for (lo, hi) in BINS}
# where does the worst error fall, and does it grow with position?
E = []  # (context, block, position_index, head, err_kernel, err_ops)
for blk in results["blocks"]:
    for pi, (rowk, rowo) in enumerate(zip(blk["per_position_head_abs_kernel"], blk["per_position_head_abs_ops"])):
        for h, (ek, eo) in enumerate(zip(rowk, rowo)):
            E.append((blk["context"], blk["block"], blk["positions"][pi], h, ek, eo))
ek = np.array([e[4] for e in E]); eo = np.array([e[5] for e in E])
iw = int(np.argmax(ek)); io = int(np.argmax(eo))
by_block = {}; by_pos = {}
for c, b, t, h, a, o in E:
    by_block.setdefault(b, []).append(a); by_pos.setdefault(t, []).append(a)
results["error_location"] = {
    "worst_vs_kernel": {"context": E[iw][0], "block": E[iw][1], "position": E[iw][2], "head": E[iw][3], "abs": float(ek[iw])},
    "worst_vs_float32_ops": {"context": E[io][0], "block": E[io][1], "position": E[io][2], "head": E[io][3], "abs": float(eo[io])},
    "cells": len(E),
    "cells_above_1e-4_vs_kernel": int((ek > 1e-4).sum()), "cells_above_1e-3_vs_kernel": int((ek > 1e-3).sum()), "cells_above_1e-2_vs_kernel": int((ek > 1e-2).sum()),
    "cells_above_1e-4_vs_ops": int((eo > 1e-4).sum()), "cells_above_1e-3_vs_ops": int((eo > 1e-3).sum()),
    "median_abs_vs_kernel": float(np.median(ek)), "p99_abs_vs_kernel": float(np.percentile(ek, 99)),
    "median_abs_vs_ops": float(np.median(eo)), "p99_abs_vs_ops": float(np.percentile(eo, 99)),
    "by_block_median_vs_kernel": {str(b): float(np.median(v)) for b, v in sorted(by_block.items())},
    "by_block_max_vs_kernel": {str(b): float(np.max(v)) for b, v in sorted(by_block.items())},
    "by_position_median_vs_kernel": {str(t): float(np.median(v)) for t, v in sorted(by_pos.items())},
    "by_position_max_vs_kernel": {str(t): float(np.max(v)) for t, v in sorted(by_pos.items())},
}
out = REPO / "outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/alpha_calibration.json"
json.dump(results, open(out, "w"), indent=1)
print(json.dumps({k: results[k] for k in ("dtypes", "use_kernel", "worst_abs_vs_kernel", "worst_rel_vs_kernel", "worst_abs_vs_float32_ops", "cells_checked", "gate_tau_tokens_median_over_blocks", "gate_tau_tokens_max")}, indent=1))
print(json.dumps(results["transport_preview"], indent=1))
print(json.dumps(results["error_location"], indent=1))
print(json.dumps(results["mechanism"], indent=1))
print("elapsed", round(time.time() - t0, 1), "s")
