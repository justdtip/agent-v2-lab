"""E2, read through the seal: update closure as a retrieval score (§4.2, §7, §7.1).

**The transport rule is Amendment 1's**, sealed as addendum `def0b7e9…`: a rank-r supervised map to
the successor, fitted on the fitting folds' ordinary transitions and applied `m` times for a
transition of cost `m`. `m` is the step delta, 1 on an ordinary transition and on a contiguous
correction, 2 where the correction crosses a step whose row was dropped; that delta is the recovery
cost. The rule's input standardisation is internal and its output is in the untransformed residual
space, so no normalisation of the rule enters the metric.

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

from local_llm_lab.pipeline.state_programme.decodability import (  # noqa: E402
    HEADLINE_FRACTION, HEADLINE_RANK, LADDER, layers_for,
)
from local_llm_lab.pipeline.state_programme.read_gate import require_seal  # noqa: E402
from local_llm_lab.pipeline.state_programme.transport import fit, nearer_the_successor  # noqa: E402

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


def transport_rule(fitting_sources: np.ndarray, fitting_targets: np.ndarray, max_rank: int):
    """Amendment 1 §2's rule, in one place.

    Isolated deliberately. The addendum's builder pushes a fixture through **this** function, on
    which §2's rule and the withdrawn `x + m·d` give different answers, and refuses unless it returns
    §2's. A reader that drifts from the sealed rule is then caught by execution and not by reading.
    """
    return fit(fitting_sources, fitting_targets, max_rank)


def score(features: np.ndarray, moves: list[dict], rules: dict[str, object], rank: int) -> list[dict]:
    """One row per transition: hit or miss, its chance level, and the transport distance."""
    rows = []
    for move in moves:
        rule = rules.get(move["task_id"])
        if rule is None or rank > rule.max_rank:
            continue
        transported = rule.apply(features[move["source"]], rank=rank, times=move["m"])
        candidates = np.array(move["candidates"])
        distances = np.linalg.norm(features[candidates] - transported, axis=1)
        best = distances.min()
        winners = np.flatnonzero(distances == best)
        hit = bool(len(winners) == 1 and candidates[winners[0]] == move["target"])  # a tie is a miss
        rows.append({**move, "hit": float(hit), "chance": 1.0 / len(candidates),
                     "distance": float(np.linalg.norm(features[move["target"]] - transported))})
    return rows


def tails(values: np.ndarray) -> dict:
    """The distribution, not the centre: §7's unit is the episode and episodes differ."""
    if not len(values):
        return {}
    q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {"min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6]),
            "share_at_zero": float((values <= 0.0).mean()),
            "share_at_one": float((values >= 1.0).mean())}


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
    strata = (("ordinary_train", ordinary_train),
              ("corrective", corrective),
              ("corrective_contiguous", [m for m in corrective if m["subgroup"] == "contiguous"]),
              ("corrective_across_a_gap", [m for m in corrective if m["subgroup"] == "across a gap"]),
              ("ordinary_test", [m for m in moves if m["kind"] == "ordinary" and m["split"] == "test"]))

    results = []
    for position, layer in enumerate(sorted(chosen.values())):
        fraction = next(f for f, l in chosen.items() if l == layer)
        features = features_all[:, position, :]

        # Amendment 1's rule, out of fold: fitted on the OTHER folds' ordinary train transitions.
        per_fold = {}
        capability = {}
        for held_out in sorted(set(folds.values())):
            used = [m for m in ordinary_train if folds[m["task_id"]] != held_out]
            sources = np.array([features[m["source"]] for m in used])
            targets = np.array([features[m["target"]] for m in used])
            per_fold[held_out] = transport_rule(sources, targets, max(LADDER))
        rules = {task: per_fold[fold] for task, fold in folds.items()}

        # §5's capability reference, recomputed here on this depth's own held-out ordinary rows.
        for rank in LADDER:
            held = []
            for held_out in sorted(set(folds.values())):
                rows = [m for m in ordinary_train if folds[m["task_id"]] == held_out]
                if not rows or rank > per_fold[held_out].max_rank:
                    continue
                src = np.array([features[m["source"]] for m in rows])
                tgt = np.array([features[m["target"]] for m in rows])
                held.append(nearer_the_successor(
                    per_fold[held_out].apply(src, rank=rank, times=1), src, tgt))
            capability[rank] = float(np.concatenate(held).mean()) if held else None

        cell = {"layer": layer, "fraction": fraction, "headline": fraction == HEADLINE_FRACTION,
                "capability_reference": {str(r): capability[r] for r in LADDER},
                "components_per_fold": {str(f): per_fold[f].max_rank for f in sorted(per_fold)}}
        for label, subset in strata:
            cell[label] = {}
            for rank in LADDER:
                hits, chance, distance = by_episode(score(features, subset, rules, rank))
                if not len(hits):
                    cell[label][str(rank)] = None
                    continue
                over = hits - chance
                cell[label][str(rank)] = {
                    "n_episodes": int(len(hits)), "n_transitions": len(subset),
                    "hit_rate": float(hits.mean()), "chance": float(chance.mean()),
                    "over_chance": float(over.mean()),
                    "transport_distance": float(distance.mean()),
                    "tails_over_chance": tails(over), "tails_hit_rate": tails(hits),
                    "per_episode_over_chance": over.tolist(),
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
        "rule": "Amendment 1 (addendum def0b7e9): a rank-r supervised map to the successor, fitted "
                "on the fitting folds' ordinary transitions and applied m times for cost m",
        "headline": {"fraction": HEADLINE_FRACTION, "rank": HEADLINE_RANK},
        "models": results}, indent=2) + "\n")

    r = str(HEADLINE_RANK)
    print(f"\n=== HEADLINE: depth {HEADLINE_FRACTION}, rank {HEADLINE_RANK} ===")
    print(f"{'model':<22} {'stratum':<24} {'hit':>6} {'chance':>7} {'over':>7} {'p05':>7} "
          f"{'p50':>7} {'p95':>7} {'zero':>6} {'n':>5}")
    for model in results:
        for cell in model["cells"]:
            if not cell["headline"]:
                continue
            print(f"{'':<22} capability reference at r={HEADLINE_RANK}: "
                  f"{cell['capability_reference'][r]:.3f}  ({model['model']})")
            for label in ("ordinary_train", "corrective", "corrective_contiguous",
                          "corrective_across_a_gap", "ordinary_test"):
                row = cell[label][r]
                if row is None:
                    print(f"{model['model']:<22} {label:<24} {'--':>6}")
                    continue
                t = row["tails_over_chance"]
                print(f"{model['model']:<22} {label:<24} {row['hit_rate']:>6.3f} "
                      f"{row['chance']:>7.3f} {row['over_chance']:>7.3f} {t['p05']:>7.3f} "
                      f"{t['median']:>7.3f} {t['p95']:>7.3f} {t['share_at_zero']:>6.2f} "
                      f"{row['n_episodes']:>5}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
