"""Whether a run loops, and where, read from the trajectories two runs actually took.

The question the diagnostic re-run exists to answer is not "did the pass rate move" — three
episodes cannot say that — but "is the failure the same failure". So this reads the shape: how
often a run repeats a call it has already made, how long its longest repeated cycle is, and
whether it ever uses a path an observation gave it rather than one it derived itself.

No model, no box. Two record directories in, one table out.

    python loop_shape.py <control-dir> <run-dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def episodes(directory: str) -> dict[str, dict]:
    """Label to manifest entry, for the agentic episodes a run completed."""
    path = os.path.join(directory, "manifest.json")
    with open(path) as handle:
        manifest = json.load(handle)
    return {
        entry["label"]: entry
        for entry in manifest.get("episodes", [])
        if entry.get("kind") == "agentic"
    }


def call_signatures(entry: dict) -> list[str]:
    """One stable string per step: the tool and its arguments, as the model issued them."""
    signatures = []
    for step in entry.get("steps", []):
        action = step.get("action")
        signatures.append(json.dumps(action, sort_keys=True) if action is not None else "<none>")
    return signatures


def longest_repeated_cycle(signatures: list[str]) -> int:
    """The length of the longest block that repeats immediately, or 0.

    A cycle of length 3 repeated four times and a single call repeated twelve times are both
    loops and are not the same loop, so the length is reported rather than a boolean.
    """
    best = 0
    for size in range(1, len(signatures) // 2 + 1):
        for start in range(len(signatures) - 2 * size + 1):
            first = signatures[start : start + size]
            if first == signatures[start + size : start + 2 * size]:
                best = max(best, size)
    return best


def summarise(entry: dict) -> dict:
    signatures = call_signatures(entry)
    counts = collections.Counter(signatures)
    return {
        "steps": len(signatures),
        "distinct_calls": len(counts),
        "most_repeated_call": max(counts.values()) if counts else 0,
        "longest_repeated_cycle": longest_repeated_cycle(signatures),
        "success": bool(entry.get("success")),
        "loop_detected": bool(entry.get("loop_detected")),
        "exhausted": bool(entry.get("exhausted")),
        "generated_tokens": entry.get("generated_tokens"),
        "seconds": entry.get("seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control")
    parser.add_argument("run")
    parser.add_argument("--json")
    args = parser.parse_args()

    control, run = episodes(args.control), episodes(args.run)
    shared = [label for label in run if label in control]
    if not shared:
        raise SystemExit("the two runs share no agentic episode; there is nothing to compare")

    rows = {}
    header = f"{'episode':38} {'steps':>11} {'distinct':>10} {'max repeat':>12} {'cycle':>9} {'pass':>10}"
    print(header)
    for label in sorted(shared):
        a, b = summarise(control[label]), summarise(run[label])
        rows[label] = {"control": a, "run": b}
        print(
            f"{label[:38]:38} {a['steps']:>5}->{b['steps']:<5} "
            f"{a['distinct_calls']:>4}->{b['distinct_calls']:<4} "
            f"{a['most_repeated_call']:>5}->{b['most_repeated_call']:<5} "
            f"{a['longest_repeated_cycle']:>4}->{b['longest_repeated_cycle']:<4} "
            f"{str(a['success']):>5}->{str(b['success']):<5}"
        )
    payload = {"control": os.path.abspath(args.control), "run": os.path.abspath(args.run), "episodes": rows}
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
