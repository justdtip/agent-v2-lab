"""E1's reading against §7's tolerances, in the three states §10 allows.

Nothing here re-fits. It reads `read_e1.py`'s output and answers one question per comparison: is the
observed contrast larger than the resolution this corpus supports? A contrast smaller than its ε is
reported as **not resolvable at this n**, never as "no difference" and never as a small difference,
because the pre-registration's opening says E1 supports a gap above about 23 accuracy points and
nothing smaller.

The paired bootstrap interval is reported beside each contrast as a description of the sampling
spread. It is not a coverage guarantee: §7.1 states the assumptions it rests on and which of them
the cross-fitting makes false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from local_llm_lab.pipeline.state_programme.tolerances import paired_bootstrap_bounds  # noqa: E402

EPSILON_MAIN = 0.23  # §7's operative table, E1 on the test split, 240 episodes


def verdict(observed: float, epsilon: float) -> str:
    return "resolved" if abs(observed) > epsilon else "not resolvable at this n"


def headline_rows(payload: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for model in payload["models"]:
        for row in model["rows"]:
            if row["headline"]:
                out[(row["model"], row["target"])] = row
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.readings.read_text())
    seed = int(payload["seed"])
    rows = headline_rows(payload)
    models = sorted({model for model, _ in rows})
    targets = sorted({target for _, target in rows})

    findings = []
    print(f"E1 at the pre-registered headline: rank {payload['headline']['rank']}, "
          f"fraction {payload['headline']['fraction']}. ε_main = {EPSILON_MAIN}\n")
    print(f"{'model':<24} {'target':<16} {'probe':>7} {'over perm':>10} {'95% lo':>8} {'95% hi':>8}  verdict")
    for model in models:
        for target in targets:
            row = rows[(model, target)]
            probe = np.array(row["per_episode_over_permutation"])
            low, high = paired_bootstrap_bounds(probe, np.zeros_like(probe), seed=seed)
            state = verdict(row["over_permutation"], EPSILON_MAIN)
            print(f"{model:<24} {target:<16} {row['probe']:>7.3f} {row['over_permutation']:>10.3f} "
                  f"{low:>8.3f} {high:>8.3f}  {state}")
            findings.append({"comparison": "probe over permutation null", "model": model,
                             "target": target, "observed": row["over_permutation"],
                             "bootstrap": [low, high], "epsilon": EPSILON_MAIN, "verdict": state})

    if len(models) == 2:
        print(f"\n{'comparison':<24} {'target':<16} {'12B':>7} {'4B':>7} {'diff':>8} {'95% lo':>8} {'95% hi':>8}  verdict")
        big = [m for m in models if "12b" in m.lower()][0]
        small = [m for m in models if m != big][0]
        for target in targets:
            a = np.array(rows[(big, target)]["per_episode_over_permutation"])
            b = np.array(rows[(small, target)]["per_episode_over_permutation"])
            difference = float((a - b).mean())
            low, high = paired_bootstrap_bounds(a, b, seed=seed)
            state = verdict(difference, EPSILON_MAIN)
            print(f"{'12B minus 4B':<24} {target:<16} {rows[(big, target)]['probe']:>7.3f} "
                  f"{rows[(small, target)]['probe']:>7.3f} {difference:>8.3f} {low:>8.3f} {high:>8.3f}  {state}")
            findings.append({"comparison": "12B minus 4B, paired by episode", "target": target,
                             "observed": difference, "bootstrap": [low, high],
                             "epsilon": EPSILON_MAIN, "verdict": state})

    print("\nthe ladder, over the permutation null, to show the headline is not a lucky rank:")
    print(f"{'model':<24} {'target':<16} " + "".join(f"{'r=' + str(r):>8}" for r in (1, 2, 4, 8, 16, 32)))
    for model in payload["models"]:
        for target in targets:
            by_rank = {row["rank"]: row["over_permutation"] for row in model["rows"]
                       if row["target"] == target and row["fraction"] == payload["headline"]["fraction"]}
            print(f"{model['model']:<24} {target:<16} " +
                  "".join(f"{by_rank.get(r, float('nan')):>8.3f}" for r in (1, 2, 4, 8, 16, 32)))

    if args.out:
        args.out.write_text(json.dumps({"epsilon_main": EPSILON_MAIN, "seed": seed,
                                        "seal_sha256": payload["seal_sha256"],
                                        "findings": findings}, indent=2) + "\n")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
