"""Amendment 1 §5's capability table, measured so that it can be replayed rather than trusted.

Out of fold on the sealed assignment, on **ordinary train transitions only**, at one depth, for every
rank on the ladder: the fraction of transitions where the transported point is strictly nearer the
successor than the source. That is the gate's quantity and the capability report's quantity — a
two-point comparison, not the retrieval score, and never computed on a corrective transition here.

It refuses without the seal, and refuses if `folds.json` is not the assignment the seal fixes, so the
table cannot be produced against a different fold set than the one E2 will use.

    python measure_gate.py --captures /workspace/captures --out gate-table.json \
        --expect-seal 998b3bca…

`--depth headline` (the default) reads §5's 0.5 fraction; a float reads that fraction instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

from local_llm_lab.pipeline.state_programme.decodability import (  # noqa: E402
    HEADLINE_FRACTION, LADDER, layers_for,
)
from local_llm_lab.pipeline.state_programme.read_gate import require_seal  # noqa: E402
from local_llm_lab.pipeline.state_programme.transport import fit, nearer_the_successor  # noqa: E402
from read_e1 import load_cells, load_layers  # noqa: E402
from read_e2 import transitions  # noqa: E402


class Refused(RuntimeError):
    """A table that cannot be produced honestly is not produced."""


def measure(directory: Path, folds: dict[str, int], fraction: float) -> dict:
    cells = load_cells(directory)
    depth = int(cells[0]["layers"])
    layer = layers_for(depth)[fraction]
    moves = [m for m in transitions(cells) if m["kind"] == "ordinary" and m["split"] == "train"]
    features = load_layers(directory, cells, [layer])[:, 0, :]

    held: dict[int, list[np.ndarray]] = {rank: [] for rank in LADDER}
    short: list[str] = []
    for fold in sorted(set(folds.values())):
        fitting = [m for m in moves if folds[m["task_id"]] != fold]
        evaluating = [m for m in moves if folds[m["task_id"]] == fold]
        if not evaluating:
            continue
        sources = np.array([features[m["source"]] for m in fitting])
        targets = np.array([features[m["target"]] for m in fitting])
        rule = fit(sources, targets, max(LADDER))
        if rule.max_rank < max(LADDER):
            short.append(f"fold {fold} yielded {rule.max_rank} components, not {max(LADDER)}")
        held_sources = np.array([features[m["source"]] for m in evaluating])
        held_targets = np.array([features[m["target"]] for m in evaluating])
        for rank in LADDER:
            if rank <= rule.max_rank:
                moved = rule.apply(held_sources, rank=rank, times=1)
                held[rank].append(nearer_the_successor(moved, held_sources, held_targets))
    return {
        "layer": layer, "fraction": fraction, "depth": depth,
        "n_transitions": len(moves),
        "by_rank": {str(rank): (float(np.concatenate(held[rank]).mean()) if held[rank] else None)
                    for rank in LADDER},
        "short_folds": short,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=HERE)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--depth", default="headline")
    parser.add_argument("--expect-seal")
    args = parser.parse_args(argv)

    seal = require_seal(args.record, expected_digest=args.expect_seal)
    print("seal verified:", seal["verified"]["files"], "|", seal["verified"]["baseline"])
    assignment = json.loads((args.record / "folds.json").read_text())
    if assignment["assignment_sha256"] != seal["folds"]["assignment_sha256"]:
        raise Refused("folds.json is not the assignment the seal fixes")
    folds = {task: int(fold) for task, fold in assignment["fold_of"].items()}
    fraction = HEADLINE_FRACTION if args.depth == "headline" else float(args.depth)

    rows = {}
    for name in sorted(p.name for p in args.captures.iterdir() if p.is_dir()):
        directory = args.captures / name
        if (directory / "manifest.jsonl").exists():
            rows[name] = measure(directory, folds, fraction)
            print(f"  {name}: layer {rows[name]['layer']} of {rows[name]['depth']}, "
                  f"{rows[name]['n_transitions']} ordinary train transitions")
            for note in rows[name]["short_folds"]:
                print(f"    short: {note}")

    print(f"\n{'model':<24} " + "".join(f"{'r=' + str(r):>9}" for r in LADDER))
    for name, row in rows.items():
        print(f"{name:<24} " + "".join(
            f"{row['by_rank'][str(r)]:>9.3f}" if row["by_rank"][str(r)] is not None else f"{'--':>9}"
            for r in LADDER))
    print("\nOrdinary train transitions only, out of fold. No corrective transition is scored here.")

    if args.out:
        args.out.write_text(json.dumps(
            {"seal_sha256": seal["verified"]["seal_sha256"],
             "assignment_sha256": assignment["assignment_sha256"],
             "population": "ordinary transitions of the train split, out of fold",
             "quantity": "fraction where the transported point is strictly nearer the successor "
                         "than the source, one application of the rule",
             "models": rows}, indent=2) + "\n")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
