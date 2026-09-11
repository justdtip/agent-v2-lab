"""Saved-array diagnostics for the A2 shares (experiments 1, 2 and 4 of the 2026-09-11 review), on the 600
out-of-domain cells, one dictionary and one layer per call. No model is loaded; the residuals are the capture's.

(1) Is the vocabulary-wide share counting errors that barely affect the decision?  For each cell: the existing
    share ||Le||/||Lh|| over all V tokens; the same after subtracting each score vector's vocabulary mean; the
    change of the score gaps among the original top-10 tokens when h is replaced by its reconstruction
    h_hat = h - e; whether the argmax survives and the margin over the runner-up before and after; the
    target token's relative error |Le_v|/|Lh_v|. Absolute magnitudes and denominators are kept.
(2) Is the dictionary's error unusually damaging, or would any error of that size be amplified?  Eight random
    Gaussian directions of norm ||e||, and eight angle-matched controls (the same projection on h, the same
    norm), each read through the same L; the actual share's rank among them, per cell.
(4) Decoder or encoder?  With the decoder frozen: nonnegative least squares on the cell's own active set (same
    features, refitted coefficients), and a nonnegative greedy re-selection with the same number of active
    features over all atoms; the raw share after each, against the encoder's.

usage: a2_diagnostics.py CONFIG LAYER OUT_JSON [SUFFIX]   (4B paths as in diag_identity.py; SUFFIX default width_16k_l0_small)
"""
import json, sys, time
from pathlib import Path
import numpy as np


def nnls(A, b, maxiter=None):
    """Lawson-Hanson active-set nonnegative least squares (scipy is not installed on the card). A: (m, n)."""
    A = np.asarray(A, np.float64); b = np.asarray(b, np.float64); m, n = A.shape
    x = np.zeros(n); P = np.zeros(n, bool); w = A.T @ (b - A @ x); it = 0; maxiter = maxiter or 3 * n
    while (~P).any() and (w[~P] > 1e-10).any() and it < maxiter:
        j = int(np.argmax(np.where(P, -np.inf, w))); P[j] = True
        while True:
            s = np.zeros(n); s[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (s[P] > 0).all(): x = s; break
            neg = P & (s <= 0); alpha = np.min(x[neg] / (x[neg] - s[neg] + 1e-300)); x = x + alpha * (s - x); P &= x > 1e-12; x[~P] = 0
            it += 1
            if it >= maxiter: break
        w = A.T @ (b - A @ x); it += 1
    return x, float(np.linalg.norm(b - A @ x))
from local_llm_lab.probes import device_bridge as DB, sae_bridge as B
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens

conf = json.loads(Path(sys.argv[1]).read_text()); layer = int(sys.argv[2]); out = Path(sys.argv[3]); suffix = sys.argv[4] if len(sys.argv) > 4 else "width_16k_l0_small"
if out.exists(): sys.exit(f"refused: {out} exists")
S4 = Path("/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
folder = f"resid_post_all/layer_{layer - 1}_{suffix}"
D4 = Path("/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1") / folder
R4 = Path("/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it") / folder / "DIGEST.json"
conf = dict(conf, dictionary_folder=folder); t0 = time.time(); rng = np.random.default_rng(20260911)
lens, metadata = load_admitted_device_lens(Path("/workspace/chief/out/lens4b-f32/admitted-maps.npz"), checkpoint=S4, model_base=conf["model_base"], expected_sha256=conf["lens_sha256"])
dmeta, receipt = DB._dictionary(D4, R4, conf, lens.hidden_size)
assert B.hook_alignment(dmeta, lens, base=conf["model_base"], nu=metadata["nu"]) == layer
residuals, readings, prov = DB._capture(Path("/workspace/chief/captures/4b"), Path("/workspace/chief/captures/4b/positions-sample.json"), Path("/workspace/chief/all-splits.jsonl"), S4, metadata, lens, conf, dmeta, Path("/workspace/chief/captures/4b/pairings-sample.json"))
dictionary = B.load_dictionary(D4 / "params.safetensors", D4 / "config.json", source=f"{receipt['repo']}/{receipt['folder']}")
unemb = B.load_unembedding(S4); J = B.lens_map_for_layer(lens, layer)
Lmat = (unemb.gain[:, None] * (J if J is not None else np.eye(lens.hidden_size, dtype=np.float32))).astype(np.float32)  # (d, d): x -> (J x) * g  as x @ Lmat? see L()
def L(x):  # gain-only lens readout, rows of x -> (V,)
    u = x if J is None else x @ J.T
    return (u * unemb.gain) @ unemb.weight.T
def share(a, b): nb = np.linalg.norm(b); return float(np.linalg.norm(a) / nb) if nb > 0 else None
K = 8; rows = []
print(json.dumps({"loaded_s": round(time.time() - t0, 1), "cells": len(readings), "dictionary": folder}), flush=True)
for cell, h in zip(readings, residuals, strict=True):
    h = np.asarray(h, np.float32).reshape(-1); v = int(cell["token_id"])
    z = B.encode(dictionary, h); hhat = B.decode(dictionary, z); e = h - hhat; S = np.flatnonzero(z > 0)
    lh, le = L(h), L(e); lhat = lh - le
    # (1) decision-focused
    top = np.argsort(-lh)[:10]; t1, t2 = int(top[0]), int(top[1])
    gaps_before = lh[top][:, None] - lh[top][None, :]; gaps_after = lhat[top][:, None] - lhat[top][None, :]
    margin_before = float(lh[t1] - lh[t2]); tophat = int(np.argmax(lhat)); srt = np.sort(lhat)[::-1]; margin_after = float(srt[0] - srt[1])
    rec = {"row": cell["row"], "position": cell["position"], "token_id": v, "active": int(S.size),
           "share_vocab": share(le, lh), "share_centred": share(le - le.mean(), lh - lh.mean()),
           "norm_lh": float(np.linalg.norm(lh)), "norm_le": float(np.linalg.norm(le)), "norm_h": float(np.linalg.norm(h)), "norm_e": float(np.linalg.norm(e)),
           "target_rel_err": float(abs(le[v]) / max(abs(lh[v]), 1e-9)), "target_score": float(lh[v]), "target_err": float(le[v]),
           "argmax_survives": bool(tophat == t1), "margin_before": margin_before, "margin_after": margin_after,
           "top10_gap_change_max_abs": float(np.abs(gaps_after - gaps_before).max()), "top10_gap_change_max_rel_to_margin": float(np.abs(gaps_after - gaps_before).max() / max(abs(margin_before), 1e-9)),
           "top10_gap_sign_flips": int(((gaps_before * gaps_after) < 0).sum() // 2)}
    # (2) random and angle-matched controls of the same norm
    ne, hn = np.linalg.norm(e), h / np.linalg.norm(h); proj = float(e @ hn)
    rand_shares, angle_shares = [], []
    for _ in range(K):
        r = rng.standard_normal(h.size).astype(np.float32); r *= ne / np.linalg.norm(r); rand_shares.append(share(L(r), lh))
        n = rng.standard_normal(h.size).astype(np.float32); n -= (n @ hn) * hn; perp = np.sqrt(max(ne**2 - proj**2, 0.0)); n *= perp / np.linalg.norm(n); a = proj * hn + n; angle_shares.append(share(L(a), lh))
    rec.update({"share_random": rand_shares, "share_angle_matched": angle_shares, "cos_e_h": float(proj / ne) if ne > 0 else None,
                "actual_rank_among_random": int(sum(s < rec["share_vocab"] for s in rand_shares)), "actual_rank_among_angle_matched": int(sum(s < rec["share_vocab"] for s in angle_shares)),
                "gain_ratio_actual": (rec["share_vocab"] / (ne / np.linalg.norm(h))) if ne > 0 else None, "gain_ratio_random_median": float(np.median(rand_shares) / (ne / np.linalg.norm(h))) if ne > 0 else None})
    # (4) decoder frozen: refit on the active set; greedy nonnegative re-selection with the same k
    target = h - dictionary.b_dec; raw_share = share(e, h)
    if S.size:
        zS, _ = nnls(dictionary.w_dec[S].T.astype(np.float64), target.astype(np.float64), maxiter=50 * S.size)
        e_refit = target - zS @ dictionary.w_dec[S]; rec["raw_share_refit_same_set"] = share(e_refit, h)
        resid, chosen = target.astype(np.float64).copy(), []
        for _ in range(S.size):
            corr = dictionary.w_dec.astype(np.float64) @ resid; corr[chosen] = -np.inf; j = int(np.argmax(corr))
            if corr[j] <= 0: break
            chosen.append(j); zc, _ = nnls(dictionary.w_dec[chosen].T.astype(np.float64), target.astype(np.float64), maxiter=50 * len(chosen)); resid = target - zc @ dictionary.w_dec[chosen]
        rec["raw_share_reselected_same_k"] = share(resid, h); rec["reselected_overlap_with_active"] = float(len(set(chosen) & set(S.tolist())) / S.size)
    rec["raw_share"] = raw_share
    rec = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else ([float(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], (np.floating, float)) else v)) for k, v in rec.items()}
    rows.append(rec)
def q(xs): xs = np.asarray([x for x in xs if x is not None], float); return {"n": int(xs.size), "min": float(xs.min()), "median": float(np.median(xs)), "p90": float(np.quantile(xs, .9)), "max": float(xs.max())} if xs.size else {"n": 0}
summary = {}
for pos in sorted({r["position"] for r in rows}):
    c = [r for r in rows if r["position"] == pos]
    summary[pos] = {"cells": len(c), "share_vocab": q([r["share_vocab"] for r in c]), "share_centred": q([r["share_centred"] for r in c]), "target_rel_err": q([r["target_rel_err"] for r in c]),
                    "argmax_survives": sum(r["argmax_survives"] for r in c), "margin_before": q([r["margin_before"] for r in c]), "margin_after": q([r["margin_after"] for r in c]),
                    "top10_gap_change_max_rel_to_margin": q([r["top10_gap_change_max_rel_to_margin"] for r in c]), "top10_gap_sign_flips": q([r["top10_gap_sign_flips"] for r in c]),
                    "share_random_pooled": q([s for r in c for s in r["share_random"]]), "share_angle_matched_pooled": q([s for r in c for s in r["share_angle_matched"]]),
                    "actual_rank_among_random_of_8": q([r["actual_rank_among_random"] for r in c]), "actual_rank_among_angle_matched_of_8": q([r["actual_rank_among_angle_matched"] for r in c]),
                    "cos_e_h": q([r["cos_e_h"] for r in c]), "gain_ratio_actual": q([r["gain_ratio_actual"] for r in c]), "gain_ratio_random_median": q([r["gain_ratio_random_median"] for r in c]),
                    "raw_share": q([r["raw_share"] for r in c]), "raw_share_refit_same_set": q([r.get("raw_share_refit_same_set") for r in c]), "raw_share_reselected_same_k": q([r.get("raw_share_reselected_same_k") for r in c]),
                    "reselected_overlap_with_active": q([r.get("reselected_overlap_with_active") for r in c]), "active": q([r["active"] for r in c])}
out.write_text(json.dumps({"basis": __doc__, "layer": layer, "dictionary": folder, "K_controls": K, "summary": summary, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, indent=1) + "\n")
print(json.dumps({"layer": layer, "elapsed_s": round(time.time() - t0, 1), **{p: {k: (v["median"] if isinstance(v, dict) and "median" in v else v) for k, v in s.items() if k in ("share_vocab", "share_centred", "target_rel_err", "argmax_survives", "share_random_pooled", "share_angle_matched_pooled", "actual_rank_among_random_of_8", "raw_share", "raw_share_refit_same_set", "raw_share_reselected_same_k")} for p, s in summary.items()}}), flush=True)
