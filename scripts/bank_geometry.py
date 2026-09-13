"""What is actually in the concept bank, before anything is injected anywhere.

Two hundred and forty concepts at four layers have been extracted, injected, trained on and argued
about, and nobody has looked at the object itself. Three questions, all answerable from the file on
disk with no forward pass:

  How many directions is it really? If 240 concept vectors span a handful of dimensions, then
  "a concept direction" is mostly one shared thing and a small remainder, and every result that
  compares a concept against a null drawn from the same covariance is comparing two samples of
  nearly the same object.

  How much of a concept is the mean? This is the scalar that separates concepts from our nulls at
  0.79 AUC. Its size at each layer says how much of tonight's detection result it could account
  for on its own.

  Does the model's own geometry know the semantic families? The eight families were assigned by
  hand, for stratifying negatives. If the bank reproduces them, the vectors carry meaning and not
  just carrier structure. If it does not, that is worth knowing before any claim about content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from local_llm_lab.introspect.vocabulary import all_concepts  # noqa: E402


def participation_ratio(eigenvalues: torch.Tensor) -> float:
    """(sum l)^2 / sum l^2: the number of directions the variance is effectively spread over.

    Equal to the dimension when every direction carries the same variance, and to 1 when one
    direction carries all of it. Reported instead of a count above a threshold, which is a choice
    of threshold dressed as a measurement.
    """
    s = float(eigenvalues.sum())
    return (s * s) / float((eigenvalues ** 2).sum()) if s > 0 else 0.0


def cosine_matrix(rows: torch.Tensor) -> torch.Tensor:
    unit = rows / rows.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return unit @ unit.T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    saved = torch.load(args.data / "bank.pt", map_location="cpu")
    words, layers = saved["words"], saved["layers"]
    families = all_concepts()
    fam = [families.get(w) for w in words]
    report: dict = {"layers": {}, "words": len(words)}

    print(f"{len(words)} concepts, layers {layers}, hidden {saved['bank'][layers[0]].shape[-1]}\n")
    print(f"{'layer':>5} {'|v| mean':>9} {'|v| sd':>7} {'|mu|':>7} {'mu/|v|':>7} "
          f"{'part.ratio':>11} {'top1 var':>9} {'top10 var':>10} {'mean-dir var':>13}")
    per_layer = {}
    for layer in layers:
        bank = saved["bank"][layer].float()
        norms = bank.norm(dim=-1)
        mu = bank.mean(dim=0)
        centred = bank - mu
        # eigenvalues of the covariance, via the singular values of the centred rows
        sv = torch.linalg.svdvals(centred.double())
        ev = (sv ** 2) / (bank.shape[0] - 1)
        pr = participation_ratio(ev)
        total = float(ev.sum())
        # how much of the UNCENTRED variance lies along the mean direction: the scalar in question
        unit_mu = mu / mu.norm().clamp(min=1e-12)
        along = (bank @ unit_mu)
        uncentred_total = float((bank ** 2).sum(dim=-1).mean())
        mean_share = float((along ** 2).mean()) / uncentred_total if uncentred_total else 0.0
        per_layer[layer] = dict(bank=bank, mu=mu, ev=ev)
        print(f"{layer:>5} {float(norms.mean()):>9.2f} {float(norms.std()):>7.2f} "
              f"{float(mu.norm()):>7.2f} {float(mu.norm() / norms.mean()):>7.3f} "
              f"{pr:>11.1f} {float(ev[0]) / total:>8.1%} "
              f"{float(ev[:10].sum()) / total:>9.1%} {mean_share:>12.1%}")
        report["layers"][str(layer)] = dict(
            norm_mean=float(norms.mean()), norm_sd=float(norms.std()), mu_norm=float(mu.norm()),
            participation_ratio=pr, top1=float(ev[0] / total),
            top10=float(ev[:10].sum() / total), mean_direction_share=mean_share)

    print("\nSEPARABILITY BY THE MEAN ALONE, the scalar the null generator leaves free.")
    print("A concept's projection on the bank mean against a draw from the CENTRED covariance.")
    from local_llm_lab.introspect.vectors import spectrum_matched
    for layer in layers:
        bank = per_layer[layer]["bank"]
        unit_mu = per_layer[layer]["mu"] / per_layer[layer]["mu"].norm().clamp(min=1e-12)
        pos = (bank @ unit_mu).tolist()
        for add_mean in (False, True):
            g = torch.Generator().manual_seed(0)
            neg = [float(spectrum_matched(bank, generator=g, add_mean=add_mean) @ unit_mu)
                   for _ in range(len(pos))]
            wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            auc = wins / (len(pos) * len(neg))
            print(f"  L{layer:<3} add_mean={str(add_mean):<5} AUC {auc:.3f}")
            report["layers"][str(layer)][f"auc_add_mean_{add_mean}"] = auc

    print("\nDOES THE GEOMETRY KNOW THE FAMILIES? within-family cosine against between-family.")
    for layer in layers:
        bank = per_layer[layer]["bank"]
        cos = cosine_matrix(bank - per_layer[layer]["mu"])   # mean removed: it inflates everything
        within, between = [], []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if fam[i] is None or fam[j] is None:
                    continue
                (within if fam[i] == fam[j] else between).append(float(cos[i, j]))
        w = sum(within) / len(within)
        b = sum(between) / len(between)
        sd = (sum((x - b) ** 2 for x in between) / max(len(between) - 1, 1)) ** 0.5
        print(f"  L{layer:<3} within {w:+.4f} (n={len(within)})  between {b:+.4f} (n={len(between)})"
              f"  separation {(w - b) / sd:+.2f} sd")
        report["layers"][str(layer)].update(within_family=w, between_family=b,
                                            family_separation_sd=(w - b) / sd)

    print("\nDOES THE SIMILARITY STRUCTURE SURVIVE DEPTH? correlation of the 240x240 cosine")
    print("matrices between layers. High means the same concepts stay near the same concepts.")
    mats = {l: cosine_matrix(per_layer[l]["bank"] - per_layer[l]["mu"]) for l in layers}
    iu = torch.triu_indices(len(words), len(words), offset=1)
    flat = {l: mats[l][iu[0], iu[1]] for l in layers}
    for a in range(len(layers)):
        for b in range(a + 1, len(layers)):
            la, lb = layers[a], layers[b]
            x, y = flat[la], flat[lb]
            r = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
            print(f"  L{la} vs L{lb}: r = {r:+.3f}")
            report[f"structure_r_L{la}_L{lb}"] = r

    if args.out:
        json.dump(report, args.out.open("w"), indent=1)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
