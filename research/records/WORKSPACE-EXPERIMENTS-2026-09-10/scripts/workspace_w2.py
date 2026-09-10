"""Workspace experiments W-2 (global availability) and W-4 primary (decodability at P_note vs P_act).

Rank-r linear readout of the taken tool (six classes) from residuals: PCA to r dimensions fitted on
the training fold, then ridge regression to one-hot targets, argmax prediction. Folds by episode.
Three designs per (position, layer, representation, rank): within-family (episode folds stratified
by family), leave-families-out (fit on half the families, evaluate on the others, both halves by
rule), and the permutation null (labels permuted within family on the training fold, same folds).
Representations: lens-transported residual (h @ J^T) and raw residual, per layer of a declared
ladder of layers (all repo layers at the six registry fractions plus the last).

usage: workspace_w2.py CAPTURE_DIR MAPS_NPZ OUT_DIR [--ranks 1,2,4,8,16] [--folds 5] [--seed 20260910]
"""
from __future__ import annotations
import argparse, json, statistics, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("capture"); ap.add_argument("maps"); ap.add_argument("out")
ap.add_argument("--ranks", default="1,2,4,8,16"); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=20260910)
ap.add_argument("--layer-fractions", default="0.167,0.333,0.5,0.667,0.833,1.0"); ap.add_argument("--ridge", type=float, default=1.0)
args = ap.parse_args(); CAP = Path(args.capture); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
man = json.loads((CAP / "manifest.json").read_text()); TOOLS = man["tools"]; L = man["repo_layers"]
idx = [json.loads(l) for l in (CAP / "index.jsonl").read_text().splitlines() if l.strip()]
rng = np.random.default_rng(args.seed)
ranks = [int(x) for x in args.ranks.split(",")]
fracs = [float(x) for x in args.layer_fractions.split(",")]
layers = sorted({L[min(len(L) - 1, max(0, round(f * len(L)) - 1))] for f in fracs} | {L[-1]})
z = np.load(args.maps); J = {r: z[f"J{r}"] for r in layers}
res = {"note": np.load(CAP / "residual_note.npy", mmap_mode="r"), "act": np.load(CAP / "residual_act.npy", mmap_mode="r")}
y = np.array([r["tool_idx"] for r in idx]); fam = np.array([r["family"] for r in idx]); ep = np.array([r["task_id"] for r in idx])
N = len(idx); K = len(TOOLS); prior = Counter(y.tolist()); prior_acc = max(prior.values()) / N
episodes = sorted(set(ep.tolist())); fam_of_ep = {e: fam[np.where(ep == e)[0][0]] for e in episodes}
families = sorted(set(fam.tolist()))
# folds by episode, stratified by family
fold_of_ep = {}
for f in families:
    eps = sorted(e for e in episodes if fam_of_ep[e] == f); rng.shuffle(eps)
    for k, e in enumerate(eps): fold_of_ep[e] = k % args.folds
fold = np.array([fold_of_ep[e] for e in ep])
# leave-families-out: two halves by rule (alternating in sorted family order), evaluated both ways
halfA = set(families[0::2]); halfB = set(families[1::2])

def fit_predict(Xtr, ytr, Xte, r, ridge):
    mu = Xtr.mean(0); Xc = Xtr - mu
    # PCA to r via SVD on the training fold
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P = Vt[:r].T  # [d, r]
    Ztr = Xc @ P; Zte = (Xte - mu) @ P
    Y = np.eye(K)[ytr]
    W = np.linalg.solve(Ztr.T @ Ztr + ridge * np.eye(r), Ztr.T @ Y)  # ridge to one-hot
    return (Zte @ W).argmax(1)

def transported(pos, layer):
    li = L.index(layer); X = np.asarray(res[pos][:, li, :], dtype=np.float32)
    return X @ J[layer].T.astype(np.float32)

def raw(pos, layer):
    li = L.index(layer); return np.asarray(res[pos][:, li, :], dtype=np.float32)

def evaluate(X, design, r):
    correct = defaultdict(list)  # per episode: list of hits
    if design in ("within", "null"):
        for k in range(args.folds):
            tr, te = fold != k, fold == k
            ytr = y[tr].copy()
            if design == "null":  # permute labels within family on the training fold
                for f in families:
                    m = fam[tr] == f; ytr[m] = rng.permutation(ytr[m])
            pred = fit_predict(X[tr], ytr, X[te], r, args.ridge)
            for e, hit in zip(ep[te], pred == y[te]): correct[e].append(bool(hit))
    else:  # leave-families-out, both directions; the transfer null permutes labels within family on the training families
        for train_f, test_f in ((halfA, halfB), (halfB, halfA)):
            tr = np.isin(fam, list(train_f)); te = np.isin(fam, list(test_f))
            ytr = y[tr].copy()
            if design == "transfer_null":  # per-family random bijection of the six labels on the training families: destroys the cross-family correspondence of label identities, keeps each family's label structure
                for f in families:
                    m = fam[tr] == f
                    if m.any():
                        bij = rng.permutation(K); ytr[m] = bij[ytr[m]]
            pred = fit_predict(X[tr], ytr, X[te], r, args.ridge)
            for e, hit in zip(ep[te], pred == y[te]): correct[e].append(bool(hit))
    ep_means = [statistics.mean(v) for v in correct.values()]
    return {"episodes": len(ep_means), "decisions": sum(len(v) for v in correct.values()),
            "accuracy_episode_mean": statistics.mean(ep_means), "accuracy_episode_p10": sorted(ep_means)[int(0.1 * len(ep_means))],
            "accuracy_episode_p90": sorted(ep_means)[int(0.9 * len(ep_means))]}

out = {"basis": f"computed from {CAP}; r is the dimension of a PCA projection fitted on the training fold (a representation constraint, not the rank of the six-class contrast, which is at most five), with a ridge ({args.ridge}) readout to one-hot on top, argmax; the label is the EXPERT action; the result is cross-family linear decodability of the information the specified pipeline retains, not functional availability, and a lens-vs-raw difference at rank r reflects which directions the fitted PCA keeps (an invertible lens changes that), never information the lens created; the transfer null is a per-family random bijection of the labels on the training families (the within-family permutation is uninformative for family-constant strata and is used only for the within design); folds by episode stratified by family ({args.folds}); leave-families-out halves by sorted order; unit = episode",
       "tool_prior_accuracy": prior_acc, "ranks": ranks, "layers": layers, "families_A": sorted(halfA), "families_B": sorted(halfB), "cells": []}
for pos in ("act", "note"):
    for layer in layers:
        for rep_name, X in (("lens", transported(pos, layer)), ("raw", raw(pos, layer))):
            for r in ranks:
                if r > X.shape[1]: continue
                cell = {"position": "P_" + pos, "layer": layer, "representation": rep_name, "rank": r}
                for design in ("within", "null", "leave_families_out", "transfer_null"):
                    cell[design] = evaluate(X, design, r)
                out["cells"].append(cell)
                print(json.dumps({k: (v if not isinstance(v, dict) else round(v["accuracy_episode_mean"], 3)) for k, v in cell.items()}), flush=True)
(OUT / "w2.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps({"event": "done", "cells": len(out["cells"])}))
