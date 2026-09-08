"""Issue 88: the arm 1 checkpoints across the whole 180-task test split, against the same base.

The 19-task reader (`paired_tally.py`) answers "which checkpoint", on the set the divergence
record used. This one answers "by how much", on every task, and it is a separate file rather than
a flag because the two ask different questions and the 19-task tallies must stay exactly as they
were reported.

The base is the same 180 greedy trajectories throughout, read from disk and never rerun. Wall
clock is deliberately not reported: a Director-authorized lens diagnostic shared the box for part
of the 800 run, so elapsed times are not comparable across the two runs or against the 19-task
rate. The verdicts are, because decoding is greedy.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "outputs/agent-v2/evals/base-test.json"
ARM = "outputs/agent-v2e-qwen35-4b-top8/evals"


def read(path: Path) -> dict[str, dict]:
    return {t["task_id"]: t for t in json.loads(path.read_text())["trajectories"]}


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = passes / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denominator, (centre + half) / denominator)


def exact_paired(wins: int, losses: int) -> float:
    """Two-sided exact binomial on the discordant pairs -- McNemar's exact form.

    Discordant pairs only, because a task both policies pass or both fail carries no evidence
    about which is better; with 5 discordant pairs no split can reach 0.05, which is why the
    19-task margin was reported as plausible rather than established.
    """
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**n)


def tally(name: str, adapter: dict[str, dict], base: dict[str, dict]) -> dict:
    shared = [task for task in base if task in adapter]
    passes = sum(adapter[task]["verdict"]["success"] for task in shared)
    base_passes = sum(base[task]["verdict"]["success"] for task in shared)
    won = [task for task in shared if adapter[task]["verdict"]["success"]]
    base_won = [task for task in shared if base[task]["verdict"]["success"]]
    gains = sorted(set(won) - set(base_won))
    losses = sorted(set(base_won) - set(won))
    exhausted = [t for t in shared if adapter[t].get("exhausted")]
    loops = [t for t in shared if adapter[t].get("loop_detected")]
    parse = [t for t in shared if adapter[t].get("parse_error")]
    violations = sum(len(adapter[t]["integrity"]["violations"]) for t in shared)
    low, high = wilson(passes, len(shared))
    base_low, base_high = wilson(base_passes, len(shared))

    families: dict[str, list[int]] = {}
    for task in shared:
        row = families.setdefault(adapter[task]["family"], [0, 0, 0])
        row[0] += 1
        row[1] += adapter[task]["verdict"]["success"]
        row[2] += base[task]["verdict"]["success"]

    print(f"\n=== {name} on {len(shared)} tasks")
    print(f"  passes            {passes}  ({passes / len(shared):.1%}, "
          f"Wilson {low:.3f}-{high:.3f})")
    print(f"  base passes       {base_passes}  ({base_passes / len(shared):.1%}, "
          f"Wilson {base_low:.3f}-{base_high:.3f})")
    print(f"  adapter wins      {len(gains)}   base wins {len(losses)}   "
          f"exact paired p = {exact_paired(len(gains), len(losses)):.2e}")
    print(f"  exhausted {len(exhausted)}  loops {len(loops)}  parse errors {len(parse)}  "
          f"integrity violations {violations}")
    print("  by family (adapter / base / n):")
    ordered = sorted(families.items(), key=lambda item: item[1][2] - item[1][1])
    for family, (n, adapted, based) in ordered:
        delta = "" if adapted == based else f"   {adapted - based:+d}"
        print(f"    {family:20s} {adapted:2d} / {based:2d} / {n:2d}{delta}")
    return {
        "name": name, "n": len(shared), "passes": passes, "base_passes": base_passes,
        "adapter_wins": len(gains), "base_wins": len(losses),
        "exact_paired_p": exact_paired(len(gains), len(losses)),
        "wilson": [low, high], "base_wilson": [base_low, base_high],
        "exhausted": len(exhausted), "loops": len(loops), "parse_errors": len(parse),
        "integrity_violations": violations,
        "by_family": {
            family: {"n": n, "adapter": a, "base": b}
            for family, (n, a, b) in sorted(families.items())
        },
        "adapter_wins_tasks": sorted(gains), "base_wins_tasks": sorted(losses),
    }


def main() -> int:
    base = read(ROOT / BASE)
    out = []
    for stem, label in (("ckpt-0000800-full", "arm 1 @ 800 rows"),
                        ("ckpt-0001200-full", "arm 1 @ 1,200 rows")):
        path = ROOT / ARM / f"{stem}-test.json"
        if not path.exists():
            print(f"\n=== {label}: not yet written ({path.name})")
            continue
        out.append(tally(label, read(path), base))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
