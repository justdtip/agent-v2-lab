"""E2, read through the seal: update closure as a retrieval score (§4.2, §7, §7.1).

**The transport rule's functional form is not fixed by §4.2, and this is the literal reading of the
sentence that describes it.** The rule "predicts +1 on an ordinary transition and the recovery cost
at a corrective one", which is a statement about how many steps the state should advance, so the
transport advances the read state by that many step-units:

    T(x, m) = x + m · d,    d = the mean per-step displacement over the fitting folds' ordinary
                                transitions, all of which advance by exactly one step.

`m` is the transition's step delta, 1 on an ordinary transition and on a contiguous correction, 2
where the correction crosses a step whose row was dropped. That delta *is* the recovery cost. The
rule has no free parameter to set after looking, which is what §4.2 says the design is for, and it
applies **no normalisation**, so the metric is exactly the residual space as captured and §4.2's
warning about a rule normalising what the candidates did not cannot arise.

Everything §4.2 does fix is obeyed: the candidate set is all N decisions of the episode, the source
included, so chance is exactly 1/N; a tie is a miss; the distance is Euclidean on native bf16
promoted to float32 and not otherwise transformed; the transport distance is descriptive, aggregated
within an episode and then across episodes, and enters no bound.

The rule is fitted on **ordinary transitions of the train split only**, out of fold, using the sealed
fold assignment (§7.1). Every one of the 553 corrective transitions is therefore out-of-sample by
construction, which is the claim being tested: that the update rule generalises to the perturbation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from local_llm_lab.pipeline.state_programme.decodability import HEADLINE_FRACTION, layers_for  # noqa: E402
from local_llm_lab.pipeline.state_programme.read_gate import require_seal  # noqa: E402

sys.path.insert(0, str(HERE))
from read_e1 import load_cells, load_layers  # noqa: E402


class Refused(RuntimeError):
    """A reading that cannot be made honestly is not made."""


def transitions(cells: list[dict]) -> list[dict]:
    """Consecutive captured decisions within an episode, classified as §4.2 classifies them."""
    by_task: dict[str, list[int]] = defaultdict(list)
    for row, cell in enumerate(cells):
        by_task[cell["task_id"]].append(row)
    out = []
    for task, rows in by_task.items():
        rows.sort(key=lambda row: cells[row]["step"])
        for source, target in zip(rows, rows[1:]):
            delta = int(cells[target]["step"]) - int(cells[source]["step"])
            corrective = bool(cells[target].get("recovery"))
            out.append({
                "task_id": task, "source": source, "target": target, "m": delta,
                "kind": "corrective" if corrective else "ordinary",
                "subgroup": ("contiguous" if delta == 1 else "across a gap") if corrective else "ordinary",
                "split": cells[target]["split"], "candidates": rows,
            })
    return out


def score(features: np.ndarray, moves: list[dict], displacement: dict[str, np.ndarray]) -> list[dict]:
    """One row per transition: hit or miss, its chance level, and the transport distance."""
    rows = []
    for move in moves:
        d = displacement.get(move["task_id"])
        if d is None:
            continue  # no out-of-fold rule for this episode; it is dropped and counted by the caller
        transported = features[move["source"]] + move["m"] * d
        candidates = np.array(move["candidates"])
        distances = np.linalg.norm(features[candidates] - transported, axis=1)
        best = distances.min()
        winners = np.flatnonzero(distances == best)
        hit = bool(len(winners) == 1 and candidates[winners[0]] == move["target"])  # a tie is a miss
        rows.append({**move, "hit": float(hit), "chance": 1.0 / len(candidates),
                     "distance": float(np.linalg.norm(features[move["target"]] - transported))})
    return rows


def by_episode(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§7's unit: one bounded observation per episode, and its chance level, and its distance."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    episodes = sorted(grouped)
    hits = np.array([np.mean([r["hit"] for r in grouped[e]]) for e in episodes])
    chance = np.array([np.mean([r["chance"] for r in grouped[e]]) for e in episodes])
    distance = np.array([np.mean([r["distance"] for r in grouped[e]]) for e in episodes])
    return hits, chance, distance


def read_model(name: str, directory: Path, folds: dict[str, int]) -> dict:
    cells = load_cells(directory)
    depth = int(cells[0]["layers"])
    chosen = layers_for(depth)
    moves = transitions(cells)
    ordinary_train = [m for m in moves if m["kind"] == "ordinary" and m["split"] == "train"]
    corrective = [m for m in moves if m["kind"] == "corrective"]
    print(f"  {name}: {len(moves)} transitions, {len(ordinary_train)} ordinary in train, "
          f"{len(corrective)} corrective")

    unassigned = sorted({m["task_id"] for m in moves} - set(folds))
    if unassigned:
        raise Refused(f"{len(unassigned)} episode(s) have no sealed fold, first {unassigned[0]}")

    features_all = load_layers(directory, cells, sorted(chosen.values()))
    results = []
    for position, layer in enumerate(sorted(chosen.values())):
        fraction = next(f for f, l in chosen.items() if l == layer)
        features = features_all[:, position, :]

        # The rule, out of fold: the mean per-step displacement of the OTHER folds' ordinary
        # train transitions. Fitted per fold once, then looked up per episode.
        per_fold = {}
        for held_out in sorted(set(folds.values())):
            used = [m for m in ordinary_train if folds[m["task_id"]] != held_out]
            per_fold[held_out] = np.mean(
                [features[m["target"]] - features[m["source"]] for m in used], axis=0)
        displacement = {task: per_fold[fold] for task, fold in folds.items()}

        cell = {"layer": layer, "fraction": fraction, "headline": fraction == HEADLINE_FRACTION}
        for label, subset in (("ordinary_train", ordinary_train),
                              ("corrective", corrective),
                              ("corrective_contiguous", [m for m in corrective if m["subgroup"] == "contiguous"]),
                              ("corrective_across_a_gap", [m for m in corrective if m["subgroup"] == "across a gap"]),
                              ("ordinary_test", [m for m in moves if m["kind"] == "ordinary" and m["split"] == "test"])):
            hits, chance, distance = by_episode(score(features, subset, displacement))
            cell[label] = {
                "n_episodes": int(len(hits)), "n_transitions": len(subset),
                "hit_rate": float(hits.mean()), "chance": float(chance.mean()),
                "over_chance": float((hits - chance).mean()),
                "transport_distance": float(distance.mean()),
                "per_episode_over_chance": (hits - chance).tolist(),
            }
        results.append(cell)
    return {"model": name, "depth": depth, "layers": chosen, "cells": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=HERE)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-seal")
    args = parser.parse_args(argv)

    seal = require_seal(args.record, expected_digest=args.expect_seal)
    print("seal verified:", seal["verified"]["files"], "|", seal["verified"]["baseline"])
    assignment = json.loads((args.record / "folds.json").read_text())
    if assignment["assignment_sha256"] != seal["folds"]["assignment_sha256"]:
        raise Refused("folds.json is not the assignment the seal fixes")
    folds = {task: int(fold) for task, fold in assignment["fold_of"].items()}
    print(f"sealed folds: {len(folds)} episodes, K={assignment['folds']}, "
          f"{assignment['assignment_sha256'][:12]}…")

    results = []
    for name in sorted(p.name for p in args.captures.iterdir() if p.is_dir()):
        directory = args.captures / name
        if (directory / "manifest.jsonl").exists():
            results.append(read_model(name, directory, folds))

    args.out.write_text(json.dumps({
        "seal_sha256": seal["verified"]["seal_sha256"],
        "rule": "T(x, m) = x + m * d, d the mean per-step displacement of the fitting folds' "
                "ordinary train transitions; no normalisation, so the metric is the residual space "
                "as captured",
        "models": results}, indent=2) + "\n")

    print(f"\n{'model':<24} {'stratum':<24} {'hit':>7} {'chance':>7} {'over':>7} {'dist':>9} {'n':>5}")
    for model in results:
        for cell in model["cells"]:
            if not cell["headline"]:
                continue
            for label in ("ordinary_train", "corrective", "corrective_contiguous", "corrective_across_a_gap"):
                row = cell[label]
                print(f"{model['model']:<24} {label:<24} {row['hit_rate']:>7.3f} "
                      f"{row['chance']:>7.3f} {row['over_chance']:>7.3f} "
                      f"{row['transport_distance']:>9.2f} {row['n_episodes']:>5}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
