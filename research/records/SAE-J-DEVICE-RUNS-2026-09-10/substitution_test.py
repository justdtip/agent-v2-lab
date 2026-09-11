"""Experiment 3 of the 2026-09-11 review: does substituting the dictionary's reconstruction change the model's
decision, and does the lens predict that change?  At a fixed decision position (P_act), layer L, the rest of the
model is run with (a) the original residual, (b) the same residual written back through the patch (the inert
control), (c) the dictionary reconstruction h_hat = b + D z, (d) four random errors of the same norm, (e) four
angle-matched errors (same projection on h, same norm).  Measured at that position: the next-token argmax and
its margin over the runner-up, KL(p_orig || p_arm), the target token's logit, and the change of the logit gaps
among the original top-10 tokens.  The lens's prediction of the same gap changes (gain-only scores, divided by
the measured RMS of the final pre-norm residual so they are on the logit scale) is reported beside the model's,
and an exact forward-mode derivative of the logits along e (torch.autograd.functional.jvp) beside both, so the
linear prediction, the averaged lens and the actual change can be compared.  Teacher-forced prefix, width 1,
float32 eager, the capture's conventions; the unpatched forward must reproduce the capture's residual.

usage: substitution_test.py CONFIG LAYERS(comma) N_CELLS OUT_JSON
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

conf = json.loads(Path(sys.argv[1]).read_text()); layers = [int(x) for x in sys.argv[2].split(",")]; n_cells = int(sys.argv[3]); out = Path(sys.argv[4])
if out.exists(): sys.exit(f"refused: {out} exists")
S4 = Path("/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
HUB = Path("/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1/resid_post_all"); REC = Path("/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it/resid_post_all")
t0 = time.time(); rng = np.random.default_rng(20260911); K = 4; dev = "cuda:0"
lens, metadata = load_admitted_device_lens(Path("/workspace/chief/out/lens4b-f32/admitted-maps.npz"), checkpoint=S4, model_base=conf["model_base"], expected_sha256=conf["lens_sha256"])
folder0 = "layer_17_width_16k_l0_small"; dmeta0, _ = DB._dictionary(HUB / folder0, REC / folder0 / "DIGEST.json", dict(conf, dictionary_folder=f"resid_post_all/{folder0}"), lens.hidden_size)
prepared, prov = DB._capture_cells(Path("/workspace/chief/captures/4b"), Path("/workspace/chief/captures/4b/positions-sample.json"), Path("/workspace/chief/all-splits.jsonl"), S4, metadata, lens, conf, layers)
cells = [p for p in prepared if p["cell"]["position"] == "P_act"][:n_cells]
model, report = hf_text.load_text_causal_lm(S4, dtype="bfloat16", attn_implementation="eager", device=dev); model.eval(); model.to(torch.float32)
for p in model.parameters(): p.requires_grad_(False)
load_upstream(); lm = upstream_hf.HFLensModel(model, tokenizer=None); blocks = lm.layers; n_blocks = len(blocks)
unemb = B.load_unembedding(S4)
print(json.dumps({"loaded_s": round(time.time() - t0, 1), "cells": len(cells), "blocks": n_blocks}), flush=True)

class Patch:
    """Replace the residual at (block u, position P) for batch element 0; record the final block's output at P."""
    def __init__(self): self.vec = None; self.u = None; self.P = None; self.final = None; self.captured = None
    def hook(self, mod, inp, outp):
        hs = outp[0] if isinstance(outp, tuple) else outp
        if self.vec is not None:
            hs = hs.clone(); hs[0, self.P] = self.vec
            return (hs,) + tuple(outp[1:]) if isinstance(outp, tuple) else hs
        self.captured = hs[0, self.P].detach().clone()
    def final_hook(self, mod, inp, outp):
        hs = outp[0] if isinstance(outp, tuple) else outp; self.final = hs[0, self.P].detach().clone()
patch = Patch()
def run(ids, u, P, vec=None):
    patch.vec, patch.u, patch.P = (None if vec is None else torch.as_tensor(vec, device=dev, dtype=torch.float32)), u, P
    h1 = blocks[u].register_forward_hook(patch.hook); h2 = blocks[n_blocks - 1].register_forward_hook(patch.final_hook)
    try:
        with torch.no_grad(): logits = model(input_ids=ids).logits[0, P].float()
    finally: h1.remove(); h2.remove()
    return logits, patch.captured, patch.final
def summarise(l0, l1, top, t1, t2, v):
    p0, p1 = torch.log_softmax(l0, -1), torch.log_softmax(l1, -1); kl = float((p0.exp() * (p0 - p1)).sum())
    a1 = int(torch.argmax(l1)); s = torch.sort(l1, descending=True).values; margin = float(s[0] - s[1])
    g0 = l0[top][:, None] - l0[top][None, :]; g1 = l1[top][:, None] - l1[top][None, :]
    return {"argmax_same": a1 == t1, "argmax": a1, "margin": margin, "kl": kl, "target_logit_change": float(l1[v] - l0[v]), "top1_top2_gap_change": float((l1[t1] - l1[t2]) - (l0[t1] - l0[t2])),
            "top10_gap_change_max_abs": float((g1 - g0).abs().max()), "top10_gap_sign_flips": int(((g0 * g1) < 0).sum().item() // 2), "top10_gap_changes": (g1 - g0)[np.triu_indices(10, 1)].cpu().numpy().tolist()}
rows = []
for L in layers:
    folder = f"layer_{L - 1}_width_16k_l0_small"; receipt = json.loads((REC / folder / "DIGEST.json").read_text())
    dictionary = B.load_dictionary(HUB / folder / "params.safetensors", HUB / folder / "config.json", source=f"{receipt['repo']}/{receipt['folder']}")
    J = B.lens_map_for_layer(lens, L); u = upstream_index_of_repo_layer(L)
    def Lscore(x): xx = x if J is None else x @ J.T; return (xx * unemb.gain) @ unemb.weight.T
    for c in cells:
        ids = torch.tensor([c["input_ids"]], device=dev); P = c["cell"]["token_index"]; v = int(c["cell"]["token_id"]); h_cap = np.asarray(c["residuals"][L], np.float32)
        l0, h_fwd, fin = run(ids, u, P); h = h_fwd.cpu().numpy()
        repro = float(np.max(np.abs(h - h_cap))); rms_final = float(torch.sqrt((fin ** 2).mean()))
        z = B.encode(dictionary, h); hhat = B.decode(dictionary, z); e = h - hhat; ne = float(np.linalg.norm(e)); hn = h / np.linalg.norm(h); proj = float(e @ hn)
        top = torch.topk(l0, 10).indices; t1, t2 = int(top[0]), int(top[1]); topn = top.cpu().numpy()
        # exact local derivative along e (forward mode), and the lens's linear prediction, both as top-10 gap changes
        def f(hp):
            patch.vec, patch.u, patch.P = hp, u, P
            hk = blocks[u].register_forward_hook(patch.hook)
            try: return model(input_ids=ids).logits[0, P].float()
            finally: hk.remove()
        with torch.enable_grad():
            _, jv = torch.autograd.functional.jvp(f, (torch.as_tensor(h, device=dev),), (torch.as_tensor(-e, device=dev),))  # d logits along (h_hat - h)
        jv = jv.detach(); gj = (jv[top][:, None] - jv[top][None, :])[np.triu_indices(10, 1)].cpu().numpy()
        le = Lscore(-e); gl = ((le[topn][:, None] - le[topn][None, :])[np.triu_indices(10, 1)]) / rms_final  # lens prediction on the logit scale
        row = {"row": c["cell"]["row"], "token_index": P, "target": v, "context_tokens": c["cell"]["context_tokens"], "layer": L, "reproduces_capture_max_abs": repro, "rms_final": rms_final,
               "norm_h": float(np.linalg.norm(h)), "norm_e": ne, "raw_share": ne / float(np.linalg.norm(h)), "cos_e_h": proj / ne, "active": int((z > 0).sum()),
               "orig": {"argmax": t1, "margin": float(l0[t1] - l0[t2]), "target_logit": float(l0[v]), "target_is_argmax": t1 == v, "top10": topn.tolist()}}
        l_same, _, _ = run(ids, u, P, h); row["same_state"] = summarise(l0, l_same, top, t1, t2, v); row["same_state"]["max_abs_logit_diff"] = float((l_same - l0).abs().max())
        l_rec, _, _ = run(ids, u, P, hhat); row["reconstruction"] = summarise(l0, l_rec, top, t1, t2, v)
        row["reconstruction"]["derivative_top10_gap_changes"] = gj.tolist(); row["reconstruction"]["lens_top10_gap_changes_logit_scale"] = gl.tolist()
        act = np.array(row["reconstruction"]["top10_gap_changes"])
        def corr(a, b): a, b = np.asarray(a), np.asarray(b); return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
        row["reconstruction"]["corr_actual_vs_derivative"] = corr(act, gj); row["reconstruction"]["corr_actual_vs_lens"] = corr(act, gl); row["reconstruction"]["corr_derivative_vs_lens"] = corr(gj, gl)
        row["reconstruction"]["rel_err_derivative_vs_actual"] = float(np.linalg.norm(act - gj) / max(np.linalg.norm(act), 1e-9)); row["reconstruction"]["rel_err_lens_vs_actual"] = float(np.linalg.norm(act - gl) / max(np.linalg.norm(act), 1e-9))
        row["random"], row["angle_matched"] = [], []
        for _ in range(K):
            r = rng.standard_normal(h.size).astype(np.float32); r *= ne / np.linalg.norm(r); lr, _, _ = run(ids, u, P, h + r); row["random"].append(summarise(l0, lr, top, t1, t2, v))
            n = rng.standard_normal(h.size).astype(np.float32); n -= (n @ hn) * hn; n *= np.sqrt(max(ne**2 - proj**2, 0.0)) / np.linalg.norm(n); a = proj * hn + n; la, _, _ = run(ids, u, P, h + a); row["angle_matched"].append(summarise(l0, la, top, t1, t2, v))
        for arm in ("random", "angle_matched"):
            for s in row[arm]: s.pop("top10_gap_changes", None)
        rows.append(row); print(json.dumps({"layer": L, "row": row["row"], "repro": repro, "rec_kl": round(row["reconstruction"]["kl"], 4), "rec_argmax_same": row["reconstruction"]["argmax_same"], "rand_kl_med": round(float(np.median([s["kl"] for s in row["random"]])), 4), "corr_act_deriv": row["reconstruction"]["corr_actual_vs_derivative"], "corr_act_lens": row["reconstruction"]["corr_actual_vs_lens"]}), flush=True)
def q(xs): xs = np.asarray([x for x in xs if x is not None], float); return {"n": int(xs.size), "min": float(xs.min()), "median": float(np.median(xs)), "p90": float(np.quantile(xs, .9)), "max": float(xs.max())} if xs.size else {"n": 0}
summary = {}
for L in layers:
    rs = [r for r in rows if r["layer"] == L]
    summary[str(L)] = {"cells": len(rs), "reproduces_capture_max_abs": q([r["reproduces_capture_max_abs"] for r in rs]), "same_state_max_abs_logit_diff": q([r["same_state"]["max_abs_logit_diff"] for r in rs]),
        "orig_margin": q([r["orig"]["margin"] for r in rs]), "orig_target_is_argmax": sum(r["orig"]["target_is_argmax"] for r in rs),
        "reconstruction": {"argmax_flips": sum(not r["reconstruction"]["argmax_same"] for r in rs), "kl": q([r["reconstruction"]["kl"] for r in rs]), "margin_after": q([r["reconstruction"]["margin"] for r in rs]), "top1_top2_gap_change": q([r["reconstruction"]["top1_top2_gap_change"] for r in rs]), "top10_gap_sign_flips": q([r["reconstruction"]["top10_gap_sign_flips"] for r in rs]),
                           "corr_actual_vs_derivative": q([r["reconstruction"]["corr_actual_vs_derivative"] for r in rs]), "corr_actual_vs_lens": q([r["reconstruction"]["corr_actual_vs_lens"] for r in rs]), "rel_err_derivative_vs_actual": q([r["reconstruction"]["rel_err_derivative_vs_actual"] for r in rs]), "rel_err_lens_vs_actual": q([r["reconstruction"]["rel_err_lens_vs_actual"] for r in rs])},
        "random": {"argmax_flips_per_cell": q([sum(not s["argmax_same"] for s in r["random"]) for r in rs]), "kl": q([s["kl"] for r in rs for s in r["random"]]), "top10_gap_sign_flips": q([s["top10_gap_sign_flips"] for r in rs for s in r["random"]])},
        "angle_matched": {"argmax_flips_per_cell": q([sum(not s["argmax_same"] for s in r["angle_matched"]) for r in rs]), "kl": q([s["kl"] for r in rs for s in r["angle_matched"]]), "top10_gap_sign_flips": q([s["top10_gap_sign_flips"] for r in rs for s in r["angle_matched"]])},
        "kl_reconstruction_over_random_median": q([r["reconstruction"]["kl"] / max(float(np.median([s["kl"] for s in r["random"]])), 1e-12) for r in rs]),
        "raw_share": q([r["raw_share"] for r in rs]), "cos_e_h": q([r["cos_e_h"] for r in rs])}
out.write_text(json.dumps({"basis": __doc__, "conventions": {"dtype": str(next(model.parameters()).dtype), "attn": report["attn_implementation"], "K": K}, "summary": summary, "rows": rows, "elapsed_s": round(time.time() - t0, 1)}, indent=1) + "\n")
print(json.dumps({"event": "done", "elapsed_s": round(time.time() - t0, 1), "summary": {L: {"rec_flips": s["reconstruction"]["argmax_flips"], "rec_kl_med": s["reconstruction"]["kl"]["median"], "rand_kl_med": s["random"]["kl"]["median"], "angle_kl_med": s["angle_matched"]["kl"]["median"], "corr_act_deriv": s["reconstruction"]["corr_actual_vs_derivative"]["median"], "corr_act_lens": s["reconstruction"]["corr_actual_vs_lens"]["median"]} for L, s in summary.items()}}), flush=True)
