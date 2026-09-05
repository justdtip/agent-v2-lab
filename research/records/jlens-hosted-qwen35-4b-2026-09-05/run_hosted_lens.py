"""Apply the hosted (Neuronpedia / Anthropic-recipe) Jacobian lens for Qwen3.5-4B to a local
MLX checkpoint, through the repository's own capture path.

Stages (each timed, each skippable):
  cases     regenerate the 42 EXP-001 probe points from the recorded seed and score them
            under the hosted lens, the logit lens and the model's own output, with the
            paired matched/mismatched sign tests and the P(true)/P(false) decomposition.
  corpus    the paper's band signatures on the sweep corpus: top-k next-token accuracy,
            excess kurtosis, top-1 autocorrelation against a shuffled null, and the
            effective dimensionality of the J-lens vectors; plus layer-to-layer CKA.
  sanity    three of the paper's qualitative readouts on raw prompts.
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

import argparse, dataclasses, json, math, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/Users/daniel.tipton/Desktop/An app")
sys.path.insert(0, str(REPO / "src"))

import mlx.core as mx

# Keep MLX's buffer cache from retaining the peak of a long-context forward pass (R: 26 GB box).
try:
    mx.set_cache_limit(2 * 1024 ** 3)
except Exception:
    pass


def log(msg, **kw):
    print(msg, json.dumps(kw) if kw else "", flush=True)


def sign_test(wins, total):
    from math import comb
    if total == 0:
        return 1.0
    tail = [comb(total, i) for i in range(total + 1)]
    observed = tail[wins]
    return min(1.0, sum(t for t in tail if t <= observed) / 2 ** total)


def paired_sign(a, b):
    """Two-sided sign test on pairs (a_i, b_i): wins = #(a_i > b_i), ties dropped."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    return wins, losses, sign_test(wins, wins + losses)


class HostedLens:
    def __init__(self, npz_path: str, num_layers: int):
        z = np.load(npz_path)
        self.J = {}
        for k in z.files:
            l = int(k[1:])
            self.J[l + 1] = mx.array(z[k].astype(np.float32))  # repo layer L = file l + 1
        self.num_layers = num_layers
        self.layers = sorted(self.J)
        log("lens loaded", layers=f"{self.layers[0]}..{self.layers[-1]}", count=len(self.layers))

    def has(self, L):
        return L in self.J or L == self.num_layers

    def transport(self, h, L):
        """h: (..., d). Library: residual @ J.T ; final layer: identity."""
        if L == self.num_layers:
            return h
        return h @ self.J[L].T


def dense_unembedding(view):
    """Dense W_U (vocab, d) float32 from the tied quantised embedding."""
    emb = view.text_module.embed_tokens
    w = emb.weight
    if hasattr(emb, "scales"):
        extra = {"mode": emb.mode} if hasattr(emb, "mode") else {}
        w = mx.dequantize(emb.weight, emb.scales, emb.biases, group_size=emb.group_size, bits=emb.bits, **extra)
    return w.astype(mx.float32)


def logits_from(view, lens, H, L, use_lens=True):
    """H: (T, d) float32 residual at layer L -> logits (T, vocab) = W_U norm(J H)."""
    v = lens.transport(H, L) if use_lens else H
    return view.unembed(view.final_norm(v))


def excess_kurtosis(x):
    x = x - x.mean(axis=-1, keepdims=True)
    m2 = (x * x).mean(axis=-1)
    m4 = (x ** 4).mean(axis=-1)
    return m4 / (m2 * m2) - 3.0


def stage_cases(view, tokenizer, spec, lens, args, out):
    from local_llm_lab.pipeline.tasks import make_jspace_tasks
    from local_llm_lab.probes.jspace_sweep import select_cases
    from local_llm_lab.pipeline.jlens import encode
    t0 = time.time()
    tasks = make_jspace_tasks("jsweep", 720, 20260902)
    cases = select_cases(tasks, tokenizer, spec=spec, probe_step=3)
    log("cases regenerated", n=len(cases), seconds=round(time.time() - t0, 1))
    art = json.load(open(REPO / "outputs/probes/jspace-qwen35-4b-base-2026-09-05-rerun/sweep.json"))
    art_cases = art["per_case"]
    by_task = {c["task_id"]: c for c in cases}
    mismatch = {}
    agree = 0
    for a in art_cases:
        c = by_task.get(a["task_id"])
        if c and c["true"] == a["true"] and c["false"] == a["false"]:
            agree += 1
        mismatch[a["task_id"]] = a["mismatched_task_id"]
    log("artifact agreement", artifact_cases=len(art_cases), regenerated=len(cases), matching_pairs=agree)
    layers = sorted(set(args.layers))
    # forward pass per case, all layers at once, last position only
    last = {}
    tokens_per_case = {}
    t0 = time.time()
    for c in cases:
        ids = encode(tokenizer, c["prompt"])
        res = view.residuals(ids, layers)
        last[c["task_id"]] = {L: res[L][0, -1] for L in layers}
        tokens_per_case[c["task_id"]] = len(ids)
        mx.eval(*last[c["task_id"]].values())
    log("residuals captured", cases=len(cases), median_tokens=float(np.median(list(tokens_per_case.values()))),
        seconds=round(time.time() - t0, 1))
    # readouts
    def probs_at(task_id, L, use_lens):
        h = last[task_id][L][None, :]
        lg = logits_from(view, lens, h, L, use_lens)[0]
        return mx.softmax(lg, axis=-1)
    readouts = {}
    for L in layers:
        readouts[f"hosted_L{L}"] = (L, True)
        readouts[f"logit_lens_L{L}"] = (L, False)
    readouts["model_output"] = (view.num_layers, False)
    per_case = []
    topk_dump = {}
    t0 = time.time()
    for c in cases:
        row = {"task_id": c["task_id"], "true": c["true"], "false": c["false"],
               "mismatched_task_id": mismatch.get(c["task_id"]), "matched": {}, "mismatched": {}}
        for name, (L, use) in readouts.items():
            p = probs_at(c["task_id"], L, use)
            row["matched"][name] = {"true": float(p[c["true_token"]]), "false": float(p[c["false_token"]])}
            other = mismatch.get(c["task_id"])
            if other in last:
                q = probs_at(other, L, use)
                row["mismatched"][name] = {"true": float(q[c["true_token"]]), "false": float(q[c["false_token"]])}
            if use and args.topk:
                order = mx.argsort(-p)[: args.topk].tolist()
                topk_dump.setdefault(c["task_id"], {})[f"L{L}"] = [
                    (tokenizer.decode([int(i)]), round(float(p[int(i)]), 5)) for i in order]
        per_case.append(row)
    log("readouts scored", seconds=round(time.time() - t0, 1))
    # tables
    table = {}
    for name in readouts:
        m = [r["matched"][name] for r in per_case]
        mm = [r["mismatched"][name] for r in per_case if name in r["mismatched"]]
        wins_m = sum(1 for x in m if x["true"] > x["false"])
        wins_mm = sum(1 for x in mm if x["true"] > x["false"])
        # paired: matched vs mismatched ordering (the ratified test)
        pairs = [(r["matched"][name]["true"] > r["matched"][name]["false"],
                  r["mismatched"][name]["true"] > r["mismatched"][name]["false"])
                 for r in per_case if name in r["mismatched"]]
        disc_w = sum(1 for a, b in pairs if a and not b)
        disc_l = sum(1 for a, b in pairs if b and not a)
        # decomposition: does each candidate's own probability move with its context?
        pt = paired_sign([r["matched"][name]["true"] for r in per_case if name in r["mismatched"]],
                         [r["mismatched"][name]["true"] for r in per_case if name in r["mismatched"]])
        pf = paired_sign([r["matched"][name]["false"] for r in per_case if name in r["mismatched"]],
                         [r["mismatched"][name]["false"] for r in per_case if name in r["mismatched"]])
        table[name] = {
            "matched_wins": wins_m, "matched_n": len(m), "matched_p": sign_test(wins_m, len(m)),
            "mismatched_wins": wins_mm, "mismatched_n": len(mm),
            "paired_discordant": [disc_w, disc_l], "paired_p": sign_test(disc_w, disc_w + disc_l),
            "P_true_moves": {"wins": pt[0], "losses": pt[1], "p": pt[2]},
            "P_false_moves": {"wins": pf[0], "losses": pf[1], "p": pf[2]},
            "mean_log_ratio_true_over_false_matched": float(np.mean([math.log((x["true"] + 1e-30) / (x["false"] + 1e-30)) for x in m])),
        }
    # artifact rows for the same layers (the 'all' JVP variant), for side-by-side
    art_rows = {}
    for L in layers:
        for key in (f"jlens_L{L}_all", f"jlens_L{L}_self", f"jlens_L{L}_future", f"logit_lens_L{L}"):
            if key in art["results"]:
                r = art["results"][key]
                art_rows[key] = {"matched_wins": r["matched_wins"], "mismatched_wins": r["mismatched_wins"], "matched_p": r["matched_p"]}
    art_rows["model_output"] = {k: art["results"]["model_output"][k] for k in ("matched_wins", "mismatched_wins", "matched_p")}
    json.dump({"table": table, "artifact_rows": art_rows, "per_case": per_case, "tokens_per_case": tokens_per_case,
               "layers": layers, "regenerated_cases": len(cases), "matching_pairs": agree},
              open(out / "cases.json", "w"), indent=1)
    json.dump(topk_dump, open(out / "cases_topk.json", "w"), indent=1)
    # markdown
    lines = ["| readout | true>false matched | mismatched | matched p | paired discordant (w/l) | paired p | P(true) moves p | P(false) moves p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in table.items():
        lines.append(f"| {name} | {r['matched_wins']}/{r['matched_n']} | {r['mismatched_wins']}/{r['mismatched_n']} | {r['matched_p']:.3g} | {r['paired_discordant'][0]}/{r['paired_discordant'][1]} | {r['paired_p']:.3g} | {r['P_true_moves']['p']:.3g} | {r['P_false_moves']['p']:.3g} |")
    lines.append("")
    lines.append("Artifact rows (finite-difference JVP, 48 samples) for the same layers:")
    lines.append("| readout | matched wins | mismatched wins | matched p |")
    lines.append("| --- | --- | --- | --- |")
    for k, r in art_rows.items():
        lines.append(f"| {k} | {r['matched_wins']}/42 | {r['mismatched_wins']}/42 | {r['matched_p']:.3g} |")
    (out / "cases.md").write_text("\n".join(lines) + "\n")
    return table


def stage_corpus(view, tokenizer, lens, args, out):
    from local_llm_lab.pipeline.jlens import build_corpus
    t0 = time.time()
    contexts = build_corpus(tokenizer, size=args.corpus_size, length=args.corpus_length)
    layers = list(range(0, view.num_layers + 1))
    skip = 16
    stats = {L: {"topk_hit": [], "top1_hit": [], "kurt": [], "top1": [], "kurt_ll": [], "topk_hit_ll": [], "top1_ll": [],
                 "entropy": [], "entropy_ll": [], "top1_mass": [], "top1_mass_ll": []} for L in layers}
    for ci, ids in enumerate(contexts):
        res = view.residuals(ids, layers)
        T = len(ids)
        final = res[view.num_layers][0]
        model_logits = view.unembed(view.final_norm(final))
        model_top1 = mx.argmax(model_logits, axis=-1)  # model's own next-token at each position
        for L in layers:
            H = res[L][0]
            if not lens.has(L):
                continue
            lg = logits_from(view, lens, H, L, True)
            ll = logits_from(view, lens, H, L, False)
            for name, Lg in (("", lg), ("_ll", ll)):
                order = mx.argsort(-Lg, axis=-1)[:, : args.k]
                hit = (order == model_top1[:, None]).any(axis=-1)
                top1 = order[:, 0]
                k_ex = excess_kurtosis(Lg)
                logp = Lg - mx.logsumexp(Lg, axis=-1, keepdims=True)
                p = mx.exp(logp)
                ent = -(p * logp).sum(axis=-1) / math.log(2)   # bits
                p1 = p.max(axis=-1)
                mx.eval(hit, top1, k_ex, ent, p1)
                sel = slice(skip, T - 1)
                stats[L]["topk_hit" + name].extend(hit[sel].tolist())
                stats[L]["kurt" + name].extend(k_ex[sel].tolist())
                stats[L]["top1" + name].append(top1[sel].tolist())
                stats[L]["entropy" + name].extend(ent[sel].tolist())
                stats[L]["top1_mass" + name].extend(p1[sel].tolist())
        log("corpus context done", index=ci, tokens=T, seconds=round(time.time() - t0, 1))
    # autocorrelation of top-1 across positions, against a within-context shuffle null
    rng = np.random.default_rng(0)
    summary = {}
    for L in layers:
        if not lens.has(L):
            continue
        s = stats[L]
        def autocorr(seqs, delta):
            same = tot = 0
            for q in seqs:
                q = np.array(q)
                if len(q) > delta:
                    same += int((q[:-delta] == q[delta:]).sum()); tot += len(q) - delta
            return same / tot if tot else float("nan")
        def null(seqs, delta):
            vals = []
            for _ in range(5):
                vals.append(autocorr([rng.permutation(np.array(q)) for q in seqs], delta))
            return float(np.mean(vals))
        summary[L] = {
            "topk_hit": float(np.mean(s["topk_hit"])), "topk_hit_logit_lens": float(np.mean(s["topk_hit_ll"])),
            "excess_kurtosis_median": float(np.median(s["kurt"])), "excess_kurtosis_median_logit_lens": float(np.median(s["kurt_ll"])),
            "entropy_bits_median": float(np.median(s["entropy"])), "entropy_bits_median_logit_lens": float(np.median(s["entropy_ll"])),
            "top1_mass_median": float(np.median(s["top1_mass"])), "top1_mass_median_logit_lens": float(np.median(s["top1_mass_ll"])),
            "top1_autocorr_d1": autocorr(s["top1"], 1), "top1_autocorr_d1_null": null(s["top1"], 1),
            "top1_autocorr_d4": autocorr(s["top1"], 4), "top1_autocorr_d4_null": null(s["top1"], 4),
            "top1_autocorr_d1_logit_lens": autocorr(s["top1_ll"], 1), "top1_autocorr_d1_logit_lens_null": null(s["top1_ll"], 1),
        }
    # effective dimensionality of the J-lens vectors W_U J_L, and CKA between layers
    t1 = time.time()
    WU = dense_unembedding(view)  # (vocab, d)
    n = WU.shape[0]
    mu = WU.mean(axis=0, keepdims=True)
    G = ((WU.T @ WU) / n - mu.T @ mu)  # covariance of unembedding rows (d, d)
    mx.eval(G)
    rng2 = np.random.default_rng(1)
    sub = mx.array(np.sort(rng2.choice(n, size=min(4096, n), replace=False)))
    Wsub = WU[sub]
    grams = {}
    for L in layers:
        if not lens.has(L):
            continue
        J = mx.eye(view.hidden_size) if L == view.num_layers else lens.J[L]
        C = J.T @ G @ J
        ev = np.sort(np.linalg.eigvalsh(np.array(C, dtype=np.float64)))[::-1]
        ev = np.clip(ev, 0, None)
        cs = np.cumsum(ev) / ev.sum()
        summary[L]["eff_dim_frac_50"] = float(np.searchsorted(cs, 0.5) + 1) / len(ev)
        summary[L]["eff_dim_frac_90"] = float(np.searchsorted(cs, 0.9) + 1) / len(ev)
        summary[L]["eff_dim_frac_99"] = float(np.searchsorted(cs, 0.99) + 1) / len(ev)
        summary[L]["participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum()) / len(ev)
        V = Wsub @ J  # (4096, d): J-lens vectors for the subsample
        V = V - V.mean(axis=0, keepdims=True)
        K = V @ V.T
        mx.eval(K)
        grams[L] = np.array(K, dtype=np.float64)
    Ls = sorted(grams)
    def hsic(K1, K2):
        return float((K1 * K2).sum())
    cka = np.zeros((len(Ls), len(Ls)))
    for i, a in enumerate(Ls):
        for j, b in enumerate(Ls):
            cka[i, j] = hsic(grams[a], grams[b]) / math.sqrt(hsic(grams[a], grams[a]) * hsic(grams[b], grams[b]))
    log("spectrum and CKA done", seconds=round(time.time() - t1, 1))
    json.dump({"summary": {str(k): v for k, v in summary.items()}, "cka_layers": Ls, "cka": cka.tolist(),
               "corpus": {"contexts": len(contexts), "tokens": [len(c) for c in contexts], "skip_first": skip, "k": args.k}},
              open(out / "corpus.json", "w"), indent=1)
    lines = ["| L | kind | top-k hit J | top-k hit logit | entropy bits J | entropy bits logit | top1 mass J | autocorr d1 J (null) | autocorr d1 logit (null) | dims for 90% var | part. ratio |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for L in Ls:
        s = summary[L]
        kind = "emb" if L == 0 else view.layer_kind(L - 1)
        lines.append(f"| {L} | {kind} | {s['topk_hit']:.2f} | {s['topk_hit_logit_lens']:.2f} | {s['entropy_bits_median']:.1f} | {s['entropy_bits_median_logit_lens']:.1f} | {s['top1_mass_median']:.3f} | {s['top1_autocorr_d1']:.2f} ({s['top1_autocorr_d1_null']:.2f}) | {s['top1_autocorr_d1_logit_lens']:.2f} ({s['top1_autocorr_d1_logit_lens_null']:.2f}) | {s['eff_dim_frac_90']:.2f} | {s['participation_ratio']:.3f} |")
    lines.append("")
    lines.append("CKA between layers' J-lens vector Gram matrices (4096-token subsample), rows/cols = layers " + str(Ls))
    lines.append("```")
    for i, a in enumerate(Ls):
        lines.append(f"{a:2d} " + " ".join(f"{cka[i, j]:.2f}" for j in range(len(Ls))))
    lines.append("```")
    (out / "corpus.md").write_text("\n".join(lines) + "\n")
    return summary


def stage_sanity(view, tokenizer, lens, args, out):
    from local_llm_lab.pipeline.jlens import encode
    prompts = {
        # Qwen tokenises numbers digit by digit, so the paper's 21/42/49 are multi-token here;
        # use single-digit intermediates: (1+2)=3, 3*3=9, 9-2=7.
        "arith": ("calc: ( 1 + 2 ) * 3 - 2 =", ["3", " 3", "9", " 9", "7", " 7"]),
        "spider": ("The number of legs on the animal that spins webs is", [" spider", "spider", " 8", "8", " eight"]),
        "rhyme": ("The soldier marched into the night,\n", [" fight", " light", " night", " sight"]),
        "france": ("The capital of France is", [" Paris", "Paris"]),
    }
    layers = list(range(0, view.num_layers + 1))
    result = {}
    for name, (text, targets) in prompts.items():
        ids = encode(tokenizer, text)
        res = view.residuals(ids, layers)
        tids = {}
        for t in targets:
            e = encode(tokenizer, t)
            if e:
                tids[t] = e[0]
        rows = {}
        for L in layers:
            if not lens.has(L):
                continue
            h = res[L][0, -1][None, :]
            lg = logits_from(view, lens, h, L, True)[0]
            ll = logits_from(view, lens, h, L, False)[0]
            order = mx.argsort(-lg)
            order_ll = mx.argsort(-ll)
            rank = {t: int(mx.argmax(order == tid)) + 1 for t, tid in tids.items()}
            rank_ll = {t: int(mx.argmax(order_ll == tid)) + 1 for t, tid in tids.items()}
            top = [tokenizer.decode([int(i)]) for i in order[:12].tolist()]
            rows[L] = {"top12": top, "rank_hosted": rank, "rank_logit_lens": rank_ll}
        result[name] = {"prompt": text, "targets": tids, "layers": {str(k): v for k, v in rows.items()}}
        log("sanity prompt done", name=name)
    json.dump(result, open(out / "sanity.json", "w"), indent=1)
    lines = []
    for name, r in result.items():
        lines.append(f"## {name}: `{r['prompt']!r}`")
        lines.append("| L | top-12 hosted J-lens | ranks hosted | ranks logit lens |")
        lines.append("| --- | --- | --- | --- |")
        for L, row in r["layers"].items():
            lines.append(f"| {L} | {' · '.join(t.replace('|','/').replace(chr(10),'⏎') for t in row['top12'])} | {row['rank_hosted']} | {row['rank_logit_lens']} |")
        lines.append("")
    (out / "sanity.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-4b", help="registry name, or a local MLX model directory")
    ap.add_argument("--name", default=None, help="label for a local model directory")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stages", default="cases,corpus,sanity")
    ap.add_argument("--layers", type=int, nargs="*", default=[5, 11, 12, 16, 20, 21, 27, 28, 32])
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--corpus-size", type=int, default=16)
    ap.add_argument("--corpus-length", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy
    t0 = time.time()
    if os.path.isdir(args.model):
        spec = dataclasses.replace(load_model_spec("qwen35-4b"), name=args.name or Path(args.model).name, hf_id=args.model)
    else:
        spec = load_model_spec(args.model)
    model, tokenizer, view, resolved = load_policy(spec, None)
    log("model loaded", name=spec.name, hf_id=spec.hf_id, layers=view.num_layers, hidden=view.hidden_size,
        vocab=view.vocab_size, seconds=round(time.time() - t0, 1))
    lens = HostedLens(args.lens, view.num_layers)
    assert lens.J[lens.layers[0]].shape[0] == view.hidden_size
    timings = {}
    results = {"model": spec.name, "hf_id": spec.hf_id, "lens": args.lens, "layer_convention": "repo layer L = file index L-1 = output of block L-1; L=32 identity"}
    for stage in args.stages.split(","):
        t1 = time.time()
        fn = {"cases": stage_cases, "corpus": stage_corpus, "sanity": stage_sanity}[stage]
        if stage == "cases":
            results[stage] = fn(view, tokenizer, spec, lens, args, out)
        else:
            results[stage] = fn(view, tokenizer, lens, args, out)
        timings[stage] = round(time.time() - t1, 1)
        log("stage done", stage=stage, seconds=timings[stage])
    results["timings_s"] = timings
    json.dump(results, open(out / "summary.json", "w"), indent=1, default=str)
    log("all done", total_seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
