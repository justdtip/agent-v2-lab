"""E2, read through the seal and the amendment's addendum: update closure as a retrieval score.

**Admission.** Nothing is read until the parent seal verifies *and* an active, non-superseded
addendum verifies against it and lists this reader's own current bytes and the rule module's. The
reader carries no addendum digest as a constant — it is itself one of the addendum's sealed files, so
the addendum is written after the reader is final and the reader asks only to find itself in it. The
verified addendum's identity is recorded in the output.

**The gate gates.** §5's headline capability is computed for **both** models, at the headline depth
and rank, **before any stratum of either model is scored**. If either model falls below 0.5 nothing
is scored and the run refuses. A capability figure printed after a result is not a gate.

**The rule** is Amendment 1 §2's: a rank-r supervised map to the successor, fitted out of fold on the
sealed assignment over ordinary train transitions, applied `m` times for a transition of cost `m`.
It lives in `transport_rule` so the addendum's builder can execute the function this reader runs.

**The arithmetic is §4.2's.** The transported point is cast to float32 before the distance and the
strict-nearest comparison, because the captured residuals are bf16 promoted to float32 *for the
arithmetic*, and float64 and float32 do not agree near a tie — where the tie rule decides a miss.

**Tolerances come from the seal**, not from constants here, and a stratum whose scored population is
smaller than its requested one never carries the complete stratum's tolerance.
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
sys.path.insert(0, str(HERE))

from local_llm_lab.pipeline.state_programme.decodability import (  # noqa: E402
    HEADLINE_FRACTION, HEADLINE_RANK, LADDER, layers_for,
)
from local_llm_lab.pipeline.state_programme.read_gate import (  # noqa: E402
    require_addendum, require_seal,
)
from local_llm_lab.pipeline.state_programme.tolerances import (  # noqa: E402
    ALPHA, RESAMPLES, paired_bootstrap_bounds, required_n,
)
from local_llm_lab.pipeline.state_programme.transport import fit, nearer_the_successor  # noqa: E402
from read_e1 import load_cells, load_layers  # noqa: E402

#: Amendment 1 §5's gate: the fraction the rule must clear at the headline depth and rank.
GATE = 0.5
#: Each stratum and the seal's tolerance row that governs it. `None` is descriptive, not an estimand.
STRATA = (("ordinary_train", "ε_ord"),
          ("corrective", "ε_sub"),
          ("corrective_contiguous", "contiguous"),
          ("corrective_across_a_gap", "across a gap"),
          ("ordinary_test", None))
RULE_MODULE = "src/local_llm_lab/pipeline/state_programme/transport.py"
READER = "research/records/STATE-PLAN-PROGRESS-2026-09-09/read_e2.py"


def declared_models(record: Path) -> list[str]:
    """The model pair the programme declares, from the sealed budget rather than from a constant."""
    budget = json.loads((record / "capture-budget.json").read_text())
    return sorted(budget["models"])


class Refused(RuntimeError):
    """A reading that cannot be made honestly is not made."""


def transport_rule(fitting_sources: np.ndarray, fitting_targets: np.ndarray, max_rank: int):
    """Amendment 1 §2's rule, in one place, so a conformity check executes what the reader runs."""
    return fit(fitting_sources, fitting_targets, max_rank)


def transitions(cells: list[dict], folds: dict[str, int]) -> list[dict]:
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
                "fold": folds[task],
                "kind": "corrective" if corrective else "ordinary",
                "subgroup": ("contiguous" if delta == 1 else "across a gap") if corrective else "ordinary",
                "split": cells[target]["split"], "candidates": rows,
            })
    return out


def score(features: np.ndarray, moves: list[dict], rules: dict, rank: int) -> tuple[list[dict], set]:
    """One row per scored transition, and the folds this rank could not reach (F4)."""
    rows, unreached = [], set()
    for move in moves:
        rule = rules.get(move["task_id"])
        if rule is None or rank > rule.max_rank:
            unreached.add(move["fold"])
            continue
        # §4.2's declared arithmetic: float32, because the tie rule decides a miss and float64 and
        # float32 disagree near a tie.
        transported = np.asarray(rule.apply(features[move["source"]], rank=rank, times=move["m"]),
                                 dtype=np.float32)
        candidates = np.array(move["candidates"])
        distances = np.linalg.norm(features[candidates] - transported, axis=1)
        best = distances.min()
        winners = np.flatnonzero(distances == best)
        hit = bool(len(winners) == 1 and candidates[winners[0]] == move["target"])  # a tie is a miss
        rows.append({**move, "hit": float(hit), "chance": 1.0 / len(candidates),
                     "distance": float(np.linalg.norm(features[move["target"]] - transported))})
    return rows, unreached


def tails(values: np.ndarray, *, kind: str) -> dict:
    """The distribution, not the centre. `kind` decides what the endpoint masses can honestly mean."""
    if not len(values):
        return {}
    q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    out = {"min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
           "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6])}
    if kind == "rate":  # bounded in [0, 1]; these are exact endpoint masses
        out["share_exactly_zero"] = float((values == 0.0).mean())
        out["share_exactly_one"] = float((values == 1.0).mean())
    else:  # a paired contrast spans [-1, 1]; "at zero" and "at or below chance" are different things
        out["share_at_or_below_chance"] = float((values <= 0.0).mean())
        out["share_exactly_at_chance"] = float((values == 0.0).mean())
    return out


def by_episode(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§7's unit: one bounded observation per episode, its chance level, and its distance."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    episodes = sorted(grouped)
    hits = np.array([np.mean([r["hit"] for r in grouped[e]]) for e in episodes])
    chance = np.array([np.mean([r["chance"] for r in grouped[e]]) for e in episodes])
    distance = np.array([np.mean([r["distance"] for r in grouped[e]]) for e in episodes])
    return hits, chance, distance


def fit_out_of_fold(features: np.ndarray, ordinary_train: list[dict], folds: dict[str, int]) -> dict:
    """Amendment 1's rule per fold, fitted on the OTHER folds' ordinary train transitions."""
    per_fold = {}
    for held_out in sorted(set(folds.values())):
        used = [m for m in ordinary_train if m["fold"] != held_out]
        sources = np.array([features[m["source"]] for m in used])
        targets = np.array([features[m["target"]] for m in used])
        per_fold[held_out] = transport_rule(sources, targets, max(LADDER))
    return per_fold


def capability(features: np.ndarray, ordinary_train: list[dict], per_fold: dict, rank: int) -> dict:
    """§5's two-point comparison on held-out ordinary transitions. Not the retrieval score."""
    held, requested, scored, unreached = [], 0, 0, set()
    for held_out, rule in sorted(per_fold.items()):
        rows = [m for m in ordinary_train if m["fold"] == held_out]
        requested += len(rows)
        if not rows or rank > rule.max_rank:
            unreached.add(held_out)
            continue
        src = np.array([features[m["source"]] for m in rows])
        tgt = np.array([features[m["target"]] for m in rows])
        moved = np.asarray(rule.apply(src, rank=rank, times=1), dtype=np.float32)
        held.append(nearer_the_successor(moved, src, tgt))
        scored += len(rows)
    return {"fraction": float(np.concatenate(held).mean()) if held else None,
            "requested": requested, "scored": scored,
            "folds_unavailable": sorted(unreached), "complete": not unreached}


def refitting_bootstrap(features: np.ndarray, subset: list[dict], ordinary_train: list[dict],
                        folds: dict[str, int], strata: dict[str, tuple], *, rank: int,
                        resamples: int, seed: int, alpha: float = ALPHA) -> dict:
    """§7 as sealed: episodes resampled **within `(split, family, variant)`**, and the rule
    **refitted out of fold inside every resample**.

    The cheap interval resamples fixed per-episode numbers and holds the fit still, so it describes
    the spread of a score computed once; this one carries the fit's own variability, which is where
    a transport rule's uncertainty actually lives. It is therefore the expensive one, and §7 declares
    it rather than the cheap one.
    """
    episodes = sorted({m["task_id"] for m in subset})
    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for task in episodes:
        by_stratum[strata[task]].append(task)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(resamples):
        drawn: list[str] = []
        for _key, members in sorted(by_stratum.items()):
            drawn += [members[i] for i in rng.integers(0, len(members), len(members))]
        multiplicity = defaultdict(int)
        for task in drawn:
            multiplicity[task] += 1
        per_fold = {}
        for held_out in sorted(set(folds.values())):
            rows = [m for m in ordinary_train
                    for _ in range(multiplicity.get(m["task_id"], 0)) if m["fold"] != held_out]
            if len(rows) <= rank:
                per_fold[held_out] = None
                continue
            per_fold[held_out] = transport_rule(
                np.array([features[m["source"]] for m in rows]),
                np.array([features[m["target"]] for m in rows]), rank)
        rules = {task: per_fold[fold] for task, fold in folds.items() if per_fold[fold] is not None}
        rows, _ = score(features, [m for m in subset if multiplicity.get(m["task_id"], 0)], rules, rank)
        if not rows:
            continue
        hits, chance, _ = by_episode(rows)
        means.append(float((hits - chance).mean()))
    if not means:
        return {"low": None, "high": None, "resamples": 0,
                "note": "no resample produced a scorable population"}
    values = np.array(means)
    return {"low": float(np.quantile(values, alpha / 2)),
            "high": float(np.quantile(values, 1 - alpha / 2)),
            "resamples": len(means), "stratified_by": "(split, family, variant)",
            "refitted": True}


def governing(cheap: dict, refit: dict | None) -> tuple[float, float, str]:
    """Both intervals are reported; the **wider** governs the reading."""
    if refit is None or refit.get("low") is None:
        return cheap["low"], cheap["high"], "paired, fit held still"
    if (refit["high"] - refit["low"]) >= (cheap["high"] - cheap["low"]):
        return refit["low"], refit["high"], "paired, refitted within each resample"
    return cheap["low"], cheap["high"], "paired, fit held still"


def verdict(low: float, high: float, epsilon: float) -> str:
    """Resolved only when the governing interval lies wholly beyond ε or wholly within it."""
    if min(abs(low), abs(high)) > epsilon and low * high > 0:
        return "resolved: the governing interval lies wholly beyond ε"
    if max(abs(low), abs(high)) <= epsilon:
        return "resolved: the governing interval lies wholly within ε"
    return "not resolvable at this n: the governing interval straddles ε"


def tolerance_for(seal: dict, key: str | None) -> dict | None:
    """The stratum's tolerance, from the seal rather than from a constant here."""
    if key is None:
        return None
    rows = [r for r in seal["tolerances"]["rows"] if key in r["name"]]
    if key == "ε_sub":  # the bare name also matches its two subgroups; take the unqualified row
        rows = [r for r in rows if "contiguous" not in r["name"] and "across a gap" not in r["name"]]
    if len(rows) != 1:
        raise Refused(f"{len(rows)} tolerance rows in the seal match {key!r}")
    row = rows[0]
    return {"name": row["name"], "epsilon": row["epsilon_declared"],
            "needs_n_at_least": row["needs_n_at_least"],
            "m": seal["tolerances"]["m"], "alpha": seal["tolerances"]["alpha"],
            "range_width": seal["tolerances"]["range_width"]}


def evaluate(subset: list[dict], features: np.ndarray, rules: dict, rank: int,
             tolerance: dict | None, seed: int, registered: bool,
             refit: dict | None = None) -> dict | None:
    """One (stratum, depth, rank) cell, with its coverage, its tolerance and its bound."""
    rows, unreached = score(features, subset, rules, rank)
    if not rows:
        return {"status": "unavailable", "reason": f"no transition reached rank {rank}",
                "requested_transitions": len(subset), "scored_transitions": 0,
                "folds_unavailable": sorted(unreached)}
    hits, chance, distance = by_episode(rows)
    over = hits - chance
    complete = not unreached and len(rows) == len(subset)
    cell = {
        "status": "scored" if complete else "reduced population",
        "registered": registered, "exploratory": not registered,
        "requested_transitions": len(subset), "scored_transitions": len(rows),
        "folds_unavailable": sorted(unreached),
        "n_episodes": int(len(hits)),
        "hit_rate": float(hits.mean()), "chance": float(chance.mean()),
        "over_chance": float(over.mean()),
        "transport_distance": float(distance.mean()),
        "tails_over_chance": tails(over, kind="contrast"),
        "tails_hit_rate": tails(hits, kind="rate"),
        "per_episode_over_chance": over.tolist(),
    }
    low, high = paired_bootstrap_bounds(hits, chance, seed=seed)
    cell["paired_bootstrap"] = {"low": low, "high": high, "refitted": False,
                                "note": "the fit is held still; §7's declared interval refits"}
    cell["refitting_bootstrap"] = refit
    governs = governing(cell["paired_bootstrap"], refit)
    cell["governing_interval"] = {"low": governs[0], "high": governs[1], "from": governs[2],
                                  "rule": "the wider of the two intervals governs the reading"}
    if tolerance is None:
        cell["tolerance"] = None
        cell["reading"] = "descriptive; this stratum carries no pre-registered tolerance"
        return cell
    if not complete:
        # F4: the complete stratum's tolerance does not describe a population it did not score.
        cell["tolerance"] = {**tolerance, "applied": False}
        cell["reading"] = ("the sealed tolerance is NOT applied: the scored population is smaller "
                           "than the registered one, and a tolerance derived for the whole stratum "
                           "does not govern a part of it")
        return cell
    needed = required_n(tolerance["m"], tolerance["epsilon"],
                        alpha=tolerance["alpha"], range_width=tolerance["range_width"])
    sufficient = len(hits) >= needed
    cell["tolerance"] = {**tolerance, "applied": True, "recomputed_needs_n": needed,
                         "episodes_sufficient": bool(sufficient)}
    if not sufficient:
        cell["reading"] = (f"insufficient: {len(hits)} episodes against {needed} required at "
                           f"ε = {tolerance['epsilon']}; no verdict is read at this resolution")
    else:
        cell["reading"] = verdict(governs[0], governs[1], tolerance["epsilon"])
        if "straddles" in cell["reading"]:
            cell["reading"] += "; this is not evidence of no effect"
    return cell


def load_for(name: str, directory: Path, folds: dict, fraction: float | None):
    """Cells, features, the depth map and the transitions. `fraction` None loads all six depths."""
    cells = load_cells(directory)
    depth = int(cells[0]["layers"])
    chosen = layers_for(depth)
    wanted = [chosen[fraction]] if fraction is not None else sorted(chosen.values())
    moves = transitions(cells, folds)
    features = load_layers(directory, cells, wanted)
    return cells, (features[:, 0, :] if fraction is not None else features), chosen, moves


def read_model(name: str, directory: Path, folds: dict, seal: dict, cached: dict,
               resamples: int) -> dict:
    cells, features_all, chosen, moves = load_for(name, directory, folds, None)
    strata = {c["task_id"]: (c["split"], c["family"], c["variant"]) for c in cells}
    ordinary_train = [m for m in moves if m["kind"] == "ordinary" and m["split"] == "train"]
    corrective = [m for m in moves if m["kind"] == "corrective"]
    subsets = {
        "ordinary_train": ordinary_train,
        "corrective": corrective,
        "corrective_contiguous": [m for m in corrective if m["subgroup"] == "contiguous"],
        "corrective_across_a_gap": [m for m in corrective if m["subgroup"] == "across a gap"],
        "ordinary_test": [m for m in moves if m["kind"] == "ordinary" and m["split"] == "test"],
    }
    print(f"  {name}: {len(moves)} transitions, {len(ordinary_train)} ordinary train, "
          f"{len(corrective)} corrective")

    seed = int(seal["seed"])
    results = []
    for position, layer in enumerate(sorted(chosen.values())):
        fraction = next(f for f, l in chosen.items() if l == layer)
        features = features_all[:, position, :]
        per_fold = cached.get(fraction) or fit_out_of_fold(features, ordinary_train, folds)
        rules = {task: per_fold[fold] for task, fold in folds.items()}
        cell = {"layer": layer, "fraction": fraction,
                "headline_depth": fraction == HEADLINE_FRACTION,
                "components_per_fold": {str(f): per_fold[f].max_rank for f in sorted(per_fold)},
                "capability_reference": {str(r): capability(features, ordinary_train, per_fold, r)
                                         for r in LADDER}}
        for label, key in STRATA:
            cell[label] = {}
            for rank in LADDER:
                # `ordinary_test` carries no registered tolerance, so it is descriptive even at the
                # headline coordinates (Codex F3): registered is about the estimand, not the cell.
                registered = (fraction == HEADLINE_FRACTION and rank == HEADLINE_RANK
                              and key is not None)
                refit = None
                if registered and resamples:
                    refit = refitting_bootstrap(features, subsets[label], ordinary_train, folds,
                                                strata, rank=rank, resamples=resamples, seed=seed)
                cell[label][str(rank)] = evaluate(subsets[label], features, rules, rank,
                                                  tolerance_for(seal, key), seed, registered, refit)
        results.append(cell)
    return {"model": name, "depth": int(cells[0]["layers"]), "layers": chosen, "cells": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=HERE)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-seal")
    parser.add_argument("--bootstrap-resamples", type=int, default=RESAMPLES,
                        help="§7 seals %(default)s; lower it only under an amendment, never to fit a night")
    parser.add_argument("--estimate-bootstrap", type=int, metavar="N",
                        help="time N refitting resamples, project the full cost, and read NOTHING")
    args = parser.parse_args(argv)

    root = args.record.resolve().parents[2]
    seal = require_seal(args.record, expected_digest=args.expect_seal)
    addendum = require_addendum(args.record, parent=seal, must_list={
        READER: Path(__file__).resolve(), RULE_MODULE: root / RULE_MODULE})
    print("seal verified:   ", seal["verified"]["files"], "|", seal["verified"]["baseline"])
    print("addendum:        ", addendum["verified"]["addendum"], addendum["verified"]["sha256"][:16] + "…",
          "| baseline", addendum["verified"]["baseline_commit"][:12] + "…")
    if addendum["verified"]["superseded_ignored"]:
        print("  superseded, ignored:", addendum["verified"]["superseded_ignored"])

    assignment = json.loads((args.record / "folds.json").read_text())
    if assignment["assignment_sha256"] != seal["folds"]["assignment_sha256"]:
        raise Refused("folds.json is not the assignment the seal fixes")
    folds = {task: int(fold) for task, fold in assignment["fold_of"].items()}

    # F2: the pair is declared by the sealed budget, not by whatever directories happen to exist.
    declared = declared_models(args.record)
    directories = {p.name: p for p in sorted(args.captures.iterdir())
                   if p.is_dir() and (p / "manifest.jsonl").exists()}
    missing = [m for m in declared if m not in directories]
    extra = [m for m in directories if m not in declared]
    if missing or extra:
        raise Refused(f"the declared pair is {declared}; missing {missing}, unexpected {extra}. "
                      "A reading of one model, or of none, is not this comparison.")

    # PHASE 1 — the gate, both models, before any stratum of either is scored (F2).
    print(f"\n=== GATE: §5, depth {HEADLINE_FRACTION}, rank {HEADLINE_RANK}, threshold {GATE} ===")
    gates, cached = {}, {}
    for name, directory in directories.items():
        cells, features, chosen, moves = load_for(name, directory, folds, HEADLINE_FRACTION)
        ordinary_train = [m for m in moves if m["kind"] == "ordinary" and m["split"] == "train"]
        per_fold = fit_out_of_fold(features, ordinary_train, folds)
        result = capability(features, ordinary_train, per_fold, HEADLINE_RANK)
        # F2: an incomplete gate population is not a conservative gate, it is a different gate.
        covered = result["complete"] and result["scored"] == result["requested"] > 0
        result.update({"layer": chosen[HEADLINE_FRACTION], "rank": HEADLINE_RANK,
                       "threshold": GATE, "coverage_complete": bool(covered),
                       "passes": bool(result["fraction"] is not None
                                      and result["fraction"] >= GATE and covered)})
        gates[name] = result
        cached[name] = {HEADLINE_FRACTION: per_fold}
        print(f"  {name:<24} {result['fraction']:.3f}  "
              f"{'PASS' if result['passes'] else 'FAIL'}  "
              f"({result['scored']}/{result['requested']} transitions"
              f"{'' if result['complete'] else ', folds unavailable ' + str(result['folds_unavailable'])})")
    failed = [n for n, g in gates.items() if not g["passes"]]
    if failed:
        reasons = "; ".join(
            f"{n}: fraction {gates[n]['fraction']}, "
            f"{gates[n]['scored']}/{gates[n]['requested']} transitions"
            f"{'' if gates[n]['coverage_complete'] else ', coverage incomplete'}" for n in failed)
        raise Refused(f"§5's gate fails on {failed} ({reasons}); no stratum is scored on either model")
    print("  gate passes on both models; scoring proceeds")

    if args.estimate_bootstrap:
        import time
        name, directory = next(iter(directories.items()))
        cells, features, chosen, moves = load_for(name, directory, folds, HEADLINE_FRACTION)
        strata = {c["task_id"]: (c["split"], c["family"], c["variant"]) for c in cells}
        ordinary_train = [m for m in moves if m["kind"] == "ordinary" and m["split"] == "train"]
        started = time.time()
        refitting_bootstrap(features, ordinary_train, ordinary_train, folds, strata,
                            rank=HEADLINE_RANK, resamples=args.estimate_bootstrap,
                            seed=int(seal["seed"]))
        each = (time.time() - started) / args.estimate_bootstrap
        print(f"\n{name}: {each:.1f}s per refitting resample at rank {HEADLINE_RANK}")
        print(f"  §7's {args.bootstrap_resamples} resamples on this model alone: "
              f"{each * args.bootstrap_resamples / 3600:.1f} h, serial")
        print("NOTHING WAS READ: this mode produces no bound and no stratum.")
        return 0

    # PHASE 2 — the strata, once.
    print("\n=== STRATA ===")
    results = [read_model(name, directory, folds, seal, cached[name], args.bootstrap_resamples)
               for name, directory in directories.items()]

    args.out.write_text(json.dumps({
        "seal_sha256": seal["verified"]["seal_sha256"],
        "addendum": addendum["verified"],
        "rule": "Amendment 1 §2: a rank-r supervised map to the successor, fitted out of fold on "
                "ordinary train transitions and applied m times for cost m",
        "arithmetic": "float32 at the retrieval boundary, per §4.2",
        "headline": {"fraction": HEADLINE_FRACTION, "rank": HEADLINE_RANK},
        "gate": gates, "models": results}, indent=2) + "\n")

    r = str(HEADLINE_RANK)
    print(f"\n=== HEADLINE (registered): depth {HEADLINE_FRACTION}, rank {HEADLINE_RANK} ===")
    print(f"{'model':<22} {'stratum':<24} {'ε':>5} {'hit':>6} {'chance':>7} {'over':>7} "
          f"{'p05':>7} {'p50':>7} {'p95':>7} {'≤chance':>8} {'n':>5}  reading")
    for model in results:
        for cell in model["cells"]:
            if not cell["headline_depth"]:
                continue
            for label, _key in STRATA:
                row = cell[label][r]
                if row is None or row.get("status") == "unavailable":
                    print(f"{model['model']:<22} {label:<24} {'--':>5}")
                    continue
                t = row["tails_over_chance"]
                eps = row["tolerance"]["epsilon"] if row["tolerance"] else None
                print(f"{model['model']:<22} {label:<24} "
                      f"{(f'{eps:.2f}' if eps else '--'):>5} "
                      f"{row['hit_rate']:>6.3f} {row['chance']:>7.3f} {row['over_chance']:>7.3f} "
                      f"{t['p05']:>7.3f} {t['median']:>7.3f} {t['p95']:>7.3f} "
                      f"{t['share_at_or_below_chance']:>8.2f} {row['n_episodes']:>5}  "
                      f"{row['reading'][:46]}")
    print(f"\nwritten: {args.out}")
    print("Everything outside the headline row is the exploratory depth/rank profile, "
          "not a registered quantity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
