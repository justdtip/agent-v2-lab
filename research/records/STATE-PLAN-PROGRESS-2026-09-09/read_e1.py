"""E1, read through the seal: decodability at matched rank on both models (§4.1, §5, §7).

Refuses without the seal. Refuses if any sealed task's horizon cannot be recovered from the task
definition, because §6 forbids inferring a horizon from a row count and a missing horizon is a
missing target, not a target to guess.

Fitting is on the **train** split only. The 240 clean test episodes are the evaluation set and enter
no fit — not the probe's, not either null's (§5). Every figure carries its rank and its layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from local_llm_lab.pipeline.state_programme.decodability import (  # noqa: E402
    HEADLINE_FRACTION, HEADLINE_RANK, LADDER, TARGETS,
    decompose, episode_scores, fit_at_rank, layers_for,
    mean_absolute_error, permute_within, r_squared,
)
from local_llm_lab.pipeline.state_programme.read_gate import require_seal  # noqa: E402
from local_llm_lab.pipeline.tasks import make_tasks  # noqa: E402

#: The seven generation groups the corpus was rendered from, with the parameters its manifest records.
GROUPS = (("train", 480, 0, True), ("train1", 240, 1, True), ("train2", 120, 2, True),
          ("valid", 24, 1, False), ("valid2", 24, 2, False),
          ("test", 180, 2, False), ("test3", 60, 3, False))
GENERATOR_SEED = 20260902


class Refused(RuntimeError):
    """A reading that cannot be made honestly is not made."""


def horizons() -> dict[str, int]:
    """Task.horizon for every task, from the task definition and never from a row count (§6)."""
    out = {}
    for split, count, difficulty, perturb in GROUPS:
        for task in make_tasks(split, count, GENERATOR_SEED, perturb=perturb, difficulty=difficulty):
            out[task.task_id] = task.horizon
    return out


def load_cells(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines() if line.strip()]


def load_layers(directory: Path, cells: list[dict], layers: list[int]) -> np.ndarray:
    """(cells, len(layers), d_model) in float32, gathered in one pass over the shards."""
    import torch

    by_shard: dict[int, list[int]] = {}
    for row, cell in enumerate(cells):
        by_shard.setdefault(int(cell["shard"]), []).append(row)
    indices = np.array([int(c["index_in_shard"]) for c in cells])
    picked = np.array([layer - 1 for layer in layers])  # §5's numbers are one-based

    out = np.empty((len(cells), len(layers), int(cells[0]["d_model"])), dtype=np.float32)
    for shard in sorted(by_shard):
        block = torch.load(directory / f"residuals-{shard:05d}.pt", map_location="cpu")
        rows = by_shard[shard]
        out[rows] = block[indices[rows]][:, picked, :].to(torch.float32).numpy()
        del block
    return out


def read_model(name: str, directory: Path, horizon: dict[str, int], seed: int) -> dict:
    cells = load_cells(directory)
    depth = int(cells[0]["layers"])
    chosen = layers_for(depth)
    print(f"  {name}: {len(cells)} cells, {depth} layers, reading {sorted(chosen.values())}")

    task_ids = np.array([c["task_id"] for c in cells])
    steps = np.array([int(c["step"]) for c in cells])
    families = np.array([str(c["family"]) for c in cells])
    splits = np.array([str(c["split"]) for c in cells])

    unknown = sorted({t for t in task_ids if t not in horizon})
    if unknown:
        raise Refused(f"{len(unknown)} task(s) have no horizon from the task definition, first {unknown[0]}")
    remaining = np.array([horizon[t] for t in task_ids]) - steps - 1
    if remaining.min() < 0:
        bad = task_ids[remaining.argmin()]
        raise Refused(f"steps remaining is negative at {bad}: the horizon disagrees with the step index")
    targets = {"step_index": steps, "steps_remaining": remaining}

    # The step-0 feature of a decision is the residual at the first decision of the same task.
    first_row: dict[str, int] = {}
    for row, task in enumerate(task_ids):
        best = first_row.get(task)
        if best is None or steps[row] < steps[best]:
            first_row[task] = row
    step0_row = np.array([first_row[t] for t in task_ids])
    # §6: 334 tasks have one decision with no rendered row. Where that decision is the first, the
    # "step-0" feature is the earliest decision that exists, which is not step 0. Say so rather
    # than let the null quietly change meaning for those tasks.
    late_start = sorted({t for t in first_row if steps[first_row[t]] != 0})
    if late_start:
        print(f"  {name}: {len(late_start)} task(s) have no step-0 decision captured; the step-0 null "
              f"uses their earliest captured decision instead, first {late_start[0]}")

    features = load_layers(directory, cells, sorted(chosen.values()))
    train = np.flatnonzero(splits == "train")
    evaluate = np.flatnonzero(splits == "test")
    print(f"  {name}: fitting on {len(train)} train decisions, evaluating on {len(evaluate)} test "
          f"decisions in {len(np.unique(task_ids[evaluate]))} episodes")

    rng = np.random.default_rng(seed)
    rows = []
    for position, layer in enumerate(sorted(chosen.values())):
        fraction = next(f for f, l in chosen.items() if l == layer)
        x = features[:, position, :]
        x0 = x[step0_row]
        for target in TARGETS:
            y = targets[target]
            permuted = permute_within(y[train], families[train], rng)
            books = {
                "probe": (decompose(x[train], y[train], max(LADDER)), x[evaluate]),
                "permutation_null": (decompose(x[train], permuted, max(LADDER)), x[evaluate]),
                "step0_null": (decompose(x0[train], y[train], max(LADDER)), x0[evaluate]),
            }
            per_rank = {}
            for rank in LADDER:
                cell = {}
                for arm, (components, evaluation_features) in books.items():
                    if rank > components.max_rank:
                        cell[arm] = None
                        continue
                    predicted = fit_at_rank(components, rank).predict(evaluation_features)
                    episodes, scores = episode_scores(predicted, y[evaluate], task_ids[evaluate])
                    cell[arm] = {"episode_mean": float(scores.mean()), "per_episode": scores.tolist(),
                                 "mae": mean_absolute_error(predicted, y[evaluate]),
                                 "r2": r_squared(predicted, y[evaluate])}
                per_rank[rank] = cell
                if cell["probe"] and cell["permutation_null"]:
                    over_perm = np.array(cell["probe"]["per_episode"]) - np.array(cell["permutation_null"]["per_episode"])
                    over_step0 = np.array(cell["probe"]["per_episode"]) - np.array(cell["step0_null"]["per_episode"])
                    rows.append({
                        "model": name, "target": target, "layer": layer, "fraction": fraction, "rank": rank,
                        "n_episodes": len(over_perm),
                        "probe": cell["probe"]["episode_mean"],
                        "permutation_null": cell["permutation_null"]["episode_mean"],
                        "step0_null": cell["step0_null"]["episode_mean"],
                        "over_permutation": float(over_perm.mean()),
                        "over_step0": float(over_step0.mean()),
                        "mae": cell["probe"]["mae"], "r2": cell["probe"]["r2"],
                        "headline": bool(rank == HEADLINE_RANK and fraction == HEADLINE_FRACTION),
                        "per_episode_over_permutation": over_perm.tolist(),
                        "per_episode_over_step0": over_step0.tolist(),
                    })
    return {"model": name, "depth": depth, "layers": chosen, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=HERE)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-seal")
    args = parser.parse_args(argv)

    seal = require_seal(args.record, expected_digest=args.expect_seal)
    print("seal verified:", seal["verified"]["files"], "|", seal["verified"]["baseline"])
    horizon = horizons()
    print(f"horizons recovered for {len(horizon)} tasks from the task definition (§6)")

    results = []
    for name in sorted(p.name for p in args.captures.iterdir() if p.is_dir()):
        directory = args.captures / name
        if (directory / "manifest.jsonl").exists():
            results.append(read_model(name, directory, horizon, int(seal["seed"])))

    payload = {"seal_sha256": seal["verified"]["seal_sha256"], "seed": seal["seed"],
               "fit_on": "train split only; the 240 clean test episodes enter no fit (§5)",
               "headline": {"rank": HEADLINE_RANK, "fraction": HEADLINE_FRACTION},
               "models": results}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n{'model':<24} {'target':<16} {'lyr':>4} {'r':>3} {'probe':>7} {'perm':>7} {'step0':>7} {'over':>7}")
    for model in results:
        for row in model["rows"]:
            if row["headline"]:
                print(f"{row['model']:<24} {row['target']:<16} {row['layer']:>4} {row['rank']:>3} "
                      f"{row['probe']:>7.3f} {row['permutation_null']:>7.3f} {row['step0_null']:>7.3f} "
                      f"{row['over_permutation']:>7.3f}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
