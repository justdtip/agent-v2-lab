"""The cross-fitting folds, assigned deterministically and digested, so the seal fixes them.

P5 requires every held-out prediction to be held out **by episode**, through cross-fitting: K folds
by episode, stratified by family and variant, the seed in the seal, applied to the probe and to the
transport rule alike, so that every transition receives an out-of-fold prediction rather than only
the ones that happen to fall in a single held-out split. Describing that in prose would leave the
assignment to whoever runs it; computing it here makes the assignment part of the pre-registration.

The stratification is on `(split, family, variant)` and the order within a stratum is by `task_id`,
so the assignment is a function of the corpus and the seed and of nothing else — no shuffle, no
dictionary iteration order, no wall clock.

    python research/records/STATE-PLAN-PROGRESS-2026-09-09/folds.py --decisions FILE --out FILE

CPU only, no model, no card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

K = 5
SEED = 20260910


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=K)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    decisions = [json.loads(line) for line in args.decisions.read_text().splitlines() if line.strip()]
    episodes: dict[str, dict] = {}
    for d in decisions:
        episodes.setdefault(d["task_id"], d)

    # Round-robin within each stratum, in sorted task order, offset by a digest of the stratum and
    # the seed. Round-robin rather than a hash bucket because it makes the fold sizes equal by
    # construction within a stratum; the offset stops every stratum from starting at fold 0, which
    # would bias fold 0 toward the smallest strata.
    strata: dict[tuple, list[str]] = defaultdict(list)
    for task, d in episodes.items():
        strata[(d["split"], d["family"], d["variant"])].append(task)

    fold_of: dict[str, int] = {}
    for stratum, tasks in sorted(strata.items()):
        key = f"{args.seed}:{':'.join(stratum)}".encode()
        offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % args.folds
        for index, task in enumerate(sorted(tasks)):
            fold_of[task] = (index + offset) % args.folds

    per_fold = Counter(fold_of.values())
    per_fold_split: dict[int, Counter] = defaultdict(Counter)
    per_fold_variant: dict[int, Counter] = defaultdict(Counter)
    for task, fold in fold_of.items():
        per_fold_split[fold][episodes[task]["split"]] += 1
        per_fold_variant[fold][episodes[task]["variant"]] += 1

    payload = {
        "basis": "measured-here",
        "rule": "round-robin within each (split, family, variant) stratum, tasks in sorted order, "
                "the stratum's start offset by sha256(seed:stratum); a function of the corpus and "
                "the seed and of nothing else",
        "folds": args.folds,
        "seed": args.seed,
        "episodes": len(fold_of),
        "per_fold": dict(sorted(per_fold.items())),
        "per_fold_split": {str(k): dict(sorted(v.items())) for k, v in sorted(per_fold_split.items())},
        "per_fold_variant": {str(k): dict(sorted(v.items())) for k, v in sorted(per_fold_variant.items())},
        "assignment_sha256": hashlib.sha256(
            json.dumps(sorted(fold_of.items()), sort_keys=True).encode()
        ).hexdigest(),
        "fold_of": dict(sorted(fold_of.items())),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"{len(fold_of)} episodes into {args.folds} folds, seed {args.seed}")
    print("per fold:", dict(sorted(per_fold.items())))
    print(f"{'fold':>5} " + "".join(f"{v:>14s}" for v in
                                    ("clean", "transient", "wrong_path", "unknown_tool",
                                     "failed_edit", "stale_path")))
    for fold in sorted(per_fold_variant):
        counts = per_fold_variant[fold]
        print(f"{fold:>5} " + "".join(f"{counts.get(v, 0):>14d}" for v in
                                      ("clean", "transient", "wrong_path", "unknown_tool",
                                       "failed_edit", "stale_path")))
    print("assignment sha256:", payload["assignment_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
