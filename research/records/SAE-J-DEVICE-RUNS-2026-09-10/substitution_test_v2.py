"""Substitution test, revision 2 (the three corrections of the second review, 2026-09-11, plus the strength ladder).

Intervention: delta = h_hat - h (the reconstruction minus the residual); the patched residual is h + s*delta for
s in {1/4, 1/2, 1}.  Controls at s = 1: random directions of norm ||delta||; angle-matched directions with the SAME
norm and the SAME signed projection on h as delta (asserted per draw); the same-state write-back.
Predictions of the top-10 logit-gap changes at s = 1, three of them, so that a normalisation mismatch can be
told from an inaccurate averaged map:
  (i)  exact derivative of the logits along delta (forward-mode through the whole remaining model);
  (ii) the lens's residual change J*delta passed through the exact derivative of the model's own final readout
       (RMSNorm + unembedding) at the original final residual -- the lens map with the normalisation responding;
  (iii) the lens's gain-only scores divided by the fixed RMS of the original final residual, as before.
Before the final norm: the exact derivative of the final pre-norm residual along delta, against J*delta (cosine,
relative error) -- the averaged map's accuracy on its own, with no readout involved.
The derivative's accuracy is reported at each strength (relative error of s*derivative against the actual change).
KL is computed in float64 from the logits; paired differences KL_rec - KL_control are reported beside ratios.
Two samples of action-position cells: the first 32 of the sampled decisions, and the 32 with the smallest
original top-1/top-2 margin among all 300 (one unpatched forward per cell selects them; the margin does not
depend on the layer).

usage: substitution_test_v2.py CONFIG LAYERS(comma) OUT_JSON
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/chief/lab-a2/src")
from local_llm_lab import hf_text
from local_llm_lab.probes import device_bridge as DB, sae_bridge as B
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens
from local_llm_lab.pipeline.lens_fitting.upstream import load_upstream, upstream_index_of_repo_layer
import jlens.hf as upstream_hf

conf = json.loads(Path(sys.argv[1]).read_text()); layers = [int(x) for x in sys.argv[2].split(",")]; out = Path(sys.argv[3])
if out.exists(): sys.exit(f"refused: {out} exists")
S4 = Path("/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
HUB = Path("/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1/resid_post_all"); REC = Path("/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it/resid_post_all")
t0 = time.time(); rng = np.random.default_rng(20260911); K = 4; dev = "cuda:0"; STRENGTHS = (0.25, 0.5, 1.0); IU = np.triu_indices(10, 1)
lens, metadata = load_admitted_device_lens(Path("/workspace/chief/out/lens4b-f32/admitted-maps.npz"), checkpoint=S4, model_base=conf["model_base"], expected_sha256=conf["lens_sha256"])
prepared, prov = DB._capture_cells(Path("/workspace/chief/captures/4b"), Path("/workspace/chief/captures/4b/positions-sample.json"), Path("/workspace/chief/all-splits.jsonl"), S4, metadata, lens, conf, layers)
act_cells = [p for p in prepared if p["cell"]["position"] == "P_act"]
model, report = hf_text.load_text_causal_lm(S4, dtype="bfloat16", attn_implementation="eager", device=dev); model.eval(); model.to(torch.float32)
for p in model.parameters(): p.requires_grad_(False)
load_upstream(); lm = upstream_hf.HFLensModel(model, tokenizer=None); blocks = lm.layers; n_blocks = len(blocks)
inner = model.model if hasattr(model, "model") else model.language_model
norm_mod, head_mod = inner.norm, model.lm_head
unemb = B.load_unembedding(S4)
print(json.dumps({"loaded_s": round(time.time() - t0, 1), "act_cells": len(act_cells), "blocks": n_blocks, "norm": type(norm_mod).__name__}), flush=True)

class Patch:
    def __init__(self): self.vec = None; self.P = None; self.captured = None; self.final = None
    def hook(self, mod, inp, outp):
        hs = outp[0] if isinstance(outp, tuple) else outp
        if self.vec is not None:
            hs = hs.clone(); hs[0, self.P] = self.vec
            return (hs,) + tuple(outp[1:]) if isinstance(outp, tuple) else hs
        self.captured = hs[0, self.P].detach().clone()
    def final_hook(self, mod, inp, outp):
        hs = outp[0] if isinstance(outp, tuple) else outp; self.final = hs[0, self.P]
patch = Patch()
def forward_logits(ids, u, P, vec=None, grad=False):
    patch.vec, patch.P = vec, P
    h1 = blocks[u].register_forward_hook(patch.hook); h2 = blocks[n_blocks - 1].register_forward_hook(patch.final_hook)
    try:
        with torch.set_grad_enabled(grad): logits = model(input_ids=ids).logits[0, P].float()
    finally: h1.remove(); h2.remove()
    return logits
def kl64(l0, l1):
    a, b = torch.log_softmax(l0.double(), -1), torch.log_softmax(l1.double(), -1); return float((a.exp() * (a - b)).sum())
def gaps(l, top): g = l[top][:, None] - l[top][None, :]; return g[IU]
def summarise(l0, l1, top, t1, t2, v):
    a1 = int(torch.argmax(l1)); s = torch.sort(l1, descending=True).values
    g0, g1 = gaps(l0, top), gaps(l1, top)
    return {"argmax_same": a1 == t1, "argmax": a1, "margin": float(s[0] - s[1]), "kl": kl64(l0, l1), "target_logit_change": float(l1[v] - l0[v]), "top1_top2_gap_change": float((l1[t1] - l1[t2]) - (l0[t1] - l0[t2])),
            "top10_gap_change_max_abs": float((g1 - g0).abs().max()), "top10_gap_sign_flips": int(((g0 * g1) < 0).sum().item()), "top10_gap_changes": (g1 - g0).detach().cpu().numpy().tolist()}
def corr(a, b): a, b = np.asarray(a, float), np.asarray(b, float); return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
def relerr(pred, act): pred, act = np.asarray(pred, float), np.asarray(act, float); return float(np.linalg.norm(act - pred) / max(np.linalg.norm(act), 1e-9))

# ---- sample selection: the first 32, and the 32 lowest-margin among all 300 (one unpatched forward each)
margins = []
for c in act_cells:
    ids = torch.tensor([c["input_ids"]], device=dev); l0 = forward_logits(ids, 0, c["cell"]["token_index"]); s = torch.sort(l0, descending=True).values; margins.append(float(s[0] - s[1]))
order = np.argsort(margins); first = list(range(32)); lowest = [int(i) for i in order[:32]]
samples = {"first_32": first, "lowest_margin_32": lowest}
print(json.dumps({"margins_all_300": {"min": float(min(margins)), "median": float(np.median(margins)), "max": float(max(margins))}, "lowest_32_max_margin": float(max(margins[i] for i in lowest))}), flush=True)
rows = []
for L in layers:
    folder = f"layer_{L - 1}_width_16k_l0_small"; receipt = json.loads((REC / folder / "DIGEST.json").read_text())
    dictionary = B.load_dictionary(HUB / folder / "params.safetensors", HUB / folder / "config.json", source=f"{receipt['repo']}/{receipt['folder']}")
    J = B.lens_map_for_layer(lens, L); u = upstream_index_of_repo_layer(L); Jt = torch.as_tensor(J, device=dev)
    def Lscore(x): return ((x @ J.T) * unemb.gain) @ unemb.weight.T
    for sample, idxs in samples.items():
        for i in idxs:
            c = act_cells[i]; ids = torch.tensor([c["input_ids"]], device=dev); P = c["cell"]["token_index"]; v = int(c["cell"]["token_id"]); h_cap = np.asarray(c["residuals"][L], np.float32)
            l0 = forward_logits(ids, u, P); h = patch.captured.cpu().numpy(); h_final = patch.final.detach().clone()
            repro = float(np.max(np.abs(h - h_cap))); rms_final = float(torch.sqrt((h_final ** 2).mean()))
            z = B.encode(dictionary, h); hhat = B.decode(dictionary, z); delta = hhat - h; nd = float(np.linalg.norm(delta)); hn = h / np.linalg.norm(h); proj = float(delta @ hn)
            top = torch.topk(l0, 10).indices; t1, t2 = int(top[0]), int(top[1]); topn = top.cpu().numpy()
            row = {"sample": sample, "index": i, "row": c["cell"]["row"], "token_index": P, "target": v, "context_tokens": c["cell"]["context_tokens"], "layer": L, "reproduces_capture_max_abs": repro, "rms_final": rms_final,
                   "norm_h": float(np.linalg.norm(h)), "norm_delta": nd, "raw_share": nd / float(np.linalg.norm(h)), "cos_delta_h": proj / nd, "active": int((z > 0).sum()),
                   "orig": {"argmax": t1, "margin": float(l0[t1] - l0[t2]), "target_logit": float(l0[v]), "target_is_argmax": t1 == v, "top10": topn.tolist()}}
            ht, dt = torch.as_tensor(h, device=dev), torch.as_tensor(delta, device=dev)
            l_same = forward_logits(ids, u, P, ht); row["same_state"] = summarise(l0, l_same, top, t1, t2, v); row["same_state"]["max_abs_logit_diff"] = float((l_same - l0).abs().max())
            # (i) exact derivative of the logits along delta
            def f_logits(hp): return forward_logits(ids, u, P, hp, grad=True)
            _, jv = torch.autograd.functional.jvp(f_logits, (ht,), (dt,)); jv = jv.detach(); g_deriv = gaps(jv, top).cpu().numpy()
            # exact derivative of the final pre-norm residual along delta, against J*delta
            def f_final(hp):
                forward_logits(ids, u, P, hp, grad=True); return patch.final
            _, jf = torch.autograd.functional.jvp(f_final, (ht,), (dt,)); jf = jf.detach(); jd = (dt @ Jt.T)
            pre = {"cos_exact_vs_lens": float(torch.nn.functional.cosine_similarity(jf, jd, dim=0)), "rel_err_lens_vs_exact": float(torch.linalg.norm(jd - jf) / max(float(torch.linalg.norm(jf)), 1e-9)), "norm_exact": float(torch.linalg.norm(jf)), "norm_lens": float(torch.linalg.norm(jd))}
            # (ii) lens residual change through the exact derivative of the readout head at the original final residual
            def head(x): return head_mod(norm_mod(x[None, None, :]))[0, 0].float()
            _, jh = torch.autograd.functional.jvp(head, (h_final,), (jd,)); g_lens_head = gaps(jh.detach(), top).cpu().numpy()
            _, jh_exact = torch.autograd.functional.jvp(head, (h_final,), (jf,)); g_exact_head = gaps(jh_exact.detach(), top).cpu().numpy()  # exact pre-norm change through the head: isolates the head's linearisation
            # (iii) the fixed-RMS lens prediction, as before
            g_lens_fixed = gaps(torch.as_tensor(Lscore(delta.astype(np.float32)), device=dev), top).cpu().numpy() / rms_final
            # strength ladder
            row["strength"] = {}
            for s_ in STRENGTHS:
                ls_ = forward_logits(ids, u, P, ht + s_ * dt); sm = summarise(l0, ls_, top, t1, t2, v); act = np.array(sm["top10_gap_changes"])
                sm.update({"rel_err_derivative": relerr(s_ * g_deriv, act), "corr_derivative": corr(s_ * g_deriv, act), "rel_err_lens_through_head": relerr(s_ * g_lens_head, act), "corr_lens_through_head": corr(s_ * g_lens_head, act),
                           "rel_err_lens_fixed_rms": relerr(s_ * g_lens_fixed, act), "corr_lens_fixed_rms": corr(s_ * g_lens_fixed, act), "rel_err_exact_pre_norm_through_head": relerr(s_ * g_exact_head, act)})
                if s_ != 1.0: sm.pop("top10_gap_changes")
                row["strength"][str(s_)] = sm
            row["reconstruction"] = row["strength"]["1.0"]; row["pre_norm"] = pre
            # controls at s = 1
            row["random"], row["angle_matched"] = [], []
            for _ in range(K):
                r = rng.standard_normal(h.size).astype(np.float32); r *= nd / np.linalg.norm(r); sr = summarise(l0, forward_logits(ids, u, P, ht + torch.as_tensor(r, device=dev)), top, t1, t2, v); sr.pop("top10_gap_changes"); row["random"].append(sr)
                n = rng.standard_normal(h.size).astype(np.float32); n -= (n @ hn) * hn; n *= np.sqrt(max(nd**2 - proj**2, 0.0)) / np.linalg.norm(n); a = (proj * hn + n).astype(np.float32)
                assert abs(float(np.linalg.norm(a)) - nd) < 1e-3 * nd and abs(float(a @ hn) - proj) < 1e-3 * max(abs(proj), 1e-6), "angle-matched control does not match delta's norm and signed projection"
                sa = summarise(l0, forward_logits(ids, u, P, ht + torch.as_tensor(a, device=dev)), top, t1, t2, v); sa.pop("top10_gap_changes"); sa["norm"] = float(np.linalg.norm(a)); sa["projection_on_h"] = float(a @ hn); row["angle_matched"].append(sa)
            row["kl_paired_diff_vs_random_median"] = row["reconstruction"]["kl"] - float(np.median([s["kl"] for s in row["random"]])); row["kl_paired_diff_vs_angle_median"] = row["reconstruction"]["kl"] - float(np.median([s["kl"] for s in row["angle_matched"]]))
            row["kl_ratio_vs_random_median"] = (row["reconstruction"]["kl"] / float(np.median([s["kl"] for s in row["random"]]))) if float(np.median([s["kl"] for s in row["random"]])) > 1e-9 else None
            rows.append(row); print(json.dumps({"L": L, "sample": sample, "row": row["row"], "margin": round(row["orig"]["margin"], 2), "rec_flip": not row["reconstruction"]["argmax_same"], "rec_kl": f"{row['reconstruction']['kl']:.2e}", "pre_norm_cos": round(pre["cos_exact_vs_lens"], 3), "corr_deriv": round(row["reconstruction"]["corr_derivative"], 2), "corr_lens_head": round(row["reconstruction"]["corr_lens_through_head"] or 0, 2), "corr_lens_fixed": round(row["reconstruction"]["corr_lens_fixed_rms"] or 0, 2)}), flush=True)
def q(xs): xs = np.asarray([x for x in xs if x is not None], float); return {"n": int(xs.size), "min": float(xs.min()), "median": float(np.median(xs)), "p90": float(np.quantile(xs, .9)), "max": float(xs.max())} if xs.size else {"n": 0}
summary = {}
for L in layers:
    for sample in samples:
        rs = [r for r in rows if r["layer"] == L and r["sample"] == sample]
        summary[f"L{L}/{sample}"] = {"cells": len(rs), "orig_margin": q([r["orig"]["margin"] for r in rs]), "orig_target_is_argmax": sum(r["orig"]["target_is_argmax"] for r in rs), "same_state_max_abs_logit_diff": q([r["same_state"]["max_abs_logit_diff"] for r in rs]),
            "reconstruction": {"argmax_flips": sum(not r["reconstruction"]["argmax_same"] for r in rs), "kl": q([r["reconstruction"]["kl"] for r in rs]), "kl_min": float(min(r["reconstruction"]["kl"] for r in rs)), "margin_after": q([r["reconstruction"]["margin"] for r in rs]), "top1_top2_gap_change": q([r["reconstruction"]["top1_top2_gap_change"] for r in rs]), "top10_gap_sign_flips": q([r["reconstruction"]["top10_gap_sign_flips"] for r in rs])},
            "random": {"argmax_flips_per_cell": q([sum(not s["argmax_same"] for s in r["random"]) for r in rs]), "kl": q([s["kl"] for r in rs for s in r["random"]]), "top10_gap_sign_flips": q([s["top10_gap_sign_flips"] for r in rs for s in r["random"]])},
            "angle_matched": {"argmax_flips_per_cell": q([sum(not s["argmax_same"] for s in r["angle_matched"]) for r in rs]), "kl": q([s["kl"] for r in rs for s in r["angle_matched"]]), "top10_gap_sign_flips": q([s["top10_gap_sign_flips"] for r in rs for s in r["angle_matched"]])},
            "kl_paired_diff_vs_random_median": q([r["kl_paired_diff_vs_random_median"] for r in rs]), "kl_paired_diff_vs_angle_median": q([r["kl_paired_diff_vs_angle_median"] for r in rs]), "kl_ratio_vs_random_median_where_defined": q([r["kl_ratio_vs_random_median"] for r in rs]),
            "pre_norm_cos_exact_vs_lens": q([r["pre_norm"]["cos_exact_vs_lens"] for r in rs]), "pre_norm_rel_err_lens_vs_exact": q([r["pre_norm"]["rel_err_lens_vs_exact"] for r in rs]),
            "predictions_at_full_strength": {k: q([r["reconstruction"][k] for r in rs]) for k in ("corr_derivative", "rel_err_derivative", "corr_lens_through_head", "rel_err_lens_through_head", "corr_lens_fixed_rms", "rel_err_lens_fixed_rms", "rel_err_exact_pre_norm_through_head")},
            "derivative_rel_err_by_strength": {s_: q([r["strength"][s_]["rel_err_derivative"] for r in rs]) for s_ in ("0.25", "0.5", "1.0")}, "lens_through_head_rel_err_by_strength": {s_: q([r["strength"][s_]["rel_err_lens_through_head"] for r in rs]) for s_ in ("0.25", "0.5", "1.0")},
            "argmax_flips_by_strength": {s_: sum(not r["strength"][s_]["argmax_same"] for r in rs) for s_ in ("0.25", "0.5", "1.0")}}
out.write_text(json.dumps({"basis": __doc__, "conventions": {"dtype": str(next(model.parameters()).dtype), "attn": report["attn_implementation"], "K": K, "strengths": STRENGTHS, "kl": "float64 from the logits"}, "samples": samples, "margins_all_300": margins, "summary": summary, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, indent=1) + "\n")
print(json.dumps({"event": "done", "elapsed_s": round(time.time() - t0, 1), "summary": {k: {"flips": s["reconstruction"]["argmax_flips"], "kl_med": s["reconstruction"]["kl"]["median"], "kl_min": s["reconstruction"]["kl_min"], "pre_norm_cos": s["pre_norm_cos_exact_vs_lens"]["median"], "corr_deriv": s["predictions_at_full_strength"]["corr_derivative"]["median"], "corr_lens_head": s["predictions_at_full_strength"]["corr_lens_through_head"]["median"], "corr_lens_fixed": s["predictions_at_full_strength"]["corr_lens_fixed_rms"]["median"], "deriv_relerr_by_s": {k2: v2["median"] for k2, v2 in s["derivative_rel_err_by_strength"].items()}} for k, s in summary.items()}}), flush=True)
