"""Quantisation gap: the hosted-lens readings on the bfloat16 checkpoint against the 4-bit one.

Reads two run_hosted_lens output directories and writes quant_gap.json and quant_gap.md. Every
comparison is paired: the same 42 cases, the same layers, the same corpus contexts, the same
sanity prompts, one model against the other.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

A_DIR, B_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
A_NAME, B_NAME = sys.argv[4], sys.argv[5]


def load(d, name):
    return json.load(open(d / name))


def pearson(xs, ys):
    n = len(xs); mx_ = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for rank, i in enumerate(order): r[i] = rank
        return r
    return pearson(ranks(xs), ranks(ys))


out = {"a": A_NAME, "b": B_NAME, "cases": {}, "corpus": {}, "cka": {}, "sanity": {}}
lines = [f"# Quantisation gap: {B_NAME} against {A_NAME}\n"]

# ---- cases: per layer, paired over the 42 cases
ca, cb = load(A_DIR, "cases.json"), load(B_DIR, "cases.json")
pa = {r["task_id"]: r for r in ca["per_case"]}; pb = {r["task_id"]: r for r in cb["per_case"]}
shared = [t for t in pa if t in pb]
lines.append(f"## Cases: {len(shared)} shared of {len(pa)} and {len(pb)}\n")
lines.append("| readout | corr log P(true), matched | Spearman | mean abs diff log10 P(true) | sign agreement of true-vs-false margin | wins A | wins B |")
lines.append("| --- | --- | --- | --- | --- | --- | --- |")
readouts = [k for k in pa[shared[0]]["matched"].keys() if k.startswith("hosted_") or k == "model_output"]
for ro in readouts:
    la = [math.log10(max(pa[t]["matched"][ro]["true"], 1e-12)) for t in shared]
    lb = [math.log10(max(pb[t]["matched"][ro]["true"], 1e-12)) for t in shared]
    ma = [pa[t]["matched"][ro]["true"] - pa[t]["matched"][ro]["false"] for t in shared]
    mb = [pb[t]["matched"][ro]["true"] - pb[t]["matched"][ro]["false"] for t in shared]
    agree = sum(1 for x, y in zip(ma, mb) if (x > 0) == (y > 0)) / len(shared)
    mad = sum(abs(x - y) for x, y in zip(la, lb)) / len(shared)
    wa = ca["table"].get(ro, {}).get("matched_wins"); wb = cb["table"].get(ro, {}).get("matched_wins")
    out["cases"][ro] = {"pearson_log_ptrue": pearson(la, lb), "spearman_log_ptrue": spearman(la, lb), "mean_abs_diff_log10": mad, "margin_sign_agreement": agree, "matched_wins_a": wa, "matched_wins_b": wb}
    lines.append(f"| {ro} | {pearson(la, lb):.3f} | {spearman(la, lb):.3f} | {mad:.3f} | {agree:.2f} | {wa} | {wb} |")

# ---- corpus: per layer summary statistics
oa, ob = load(A_DIR, "corpus.json"), load(B_DIR, "corpus.json")
keys = ["topk_hit", "topk_hit_logit_lens", "excess_kurtosis_median", "top1_autocorr_d1", "top1_autocorr_d1_null", "eff_dim_frac_50"]
lines.append("\n## Corpus statistics per layer (A | B)\n")
lines.append("| layer | " + " | ".join(keys) + " |")
lines.append("| --- | " + " | ".join("---" for _ in keys) + " |")
for L in oa["summary"]:
    if L not in ob["summary"]: continue
    sa, sb = oa["summary"][L], ob["summary"][L]
    out["corpus"][L] = {k: [sa.get(k), sb.get(k)] for k in keys}
    lines.append(f"| {L} | " + " | ".join(f"{sa.get(k, float('nan')):.3f} / {sb.get(k, float('nan')):.3f}" for k in keys) + " |")

# ---- CKA: matrix difference
ka, kb = oa["cka"], ob["cka"]
diffs = [abs(ka[i][j] - kb[i][j]) for i in range(len(ka)) for j in range(len(ka[i])) if i < len(kb) and j < len(kb[i])]
out["cka"] = {"max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs), "n": len(diffs)}
lines.append(f"\n## CKA across layers: max |diff| {max(diffs):.3f}, mean |diff| {sum(diffs)/len(diffs):.3f} over {len(diffs)} entries\n")

# ---- sanity prompts
sa_, sb_ = load(A_DIR, "sanity.json"), load(B_DIR, "sanity.json")
lines.append("## Sanity prompts: layer of first target hit and final-layer target probability (A | B)\n")
for name in sa_:
    if name not in sb_: continue
    ea, eb = sa_[name], sb_[name]
    out["sanity"][name] = {"a": ea.get("layers"), "b": eb.get("layers")}
    lines.append(f"- **{name}** `{ea.get('prompt')}` targets {ea.get('targets')}")
    lines.append(f"  - A layers: {json.dumps(ea.get('layers'))[:600]}")
    lines.append(f"  - B layers: {json.dumps(eb.get('layers'))[:600]}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "quant_gap.json").write_text(json.dumps(out, indent=2))
(OUT / "quant_gap.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:16]))
print("...\nwrote", OUT / "quant_gap.md")
