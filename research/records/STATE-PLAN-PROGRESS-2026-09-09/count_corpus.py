"""Re-run the Chief's corpus count independently, as the order requires, and say where it differs.

`STATE-PLAN-PROGRESS-ORDER-2026-09-10` §1 records three facts measured on
`data/agent_v2e-gemma3-4b/{train,valid,test}.jsonl` and instructs the D-CRO to re-run the count and
put the script in the record. This is that script. It measures each claim rather than restating it,
prints a verdict per claim, and exits nonzero if any disagrees, so a disagreement cannot be read as
agreement by someone skimming.

    python research/records/STATE-PLAN-PROGRESS-2026-09-09/count_corpus.py [--data DIR] [--json OUT]

CPU only, no model, no card.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: What §1 of the order says, as a table this script tries to falsify.
CLAIMED = {
    "rendered_rows": 8907,
    "tasks": 1128,
    "tasks_train": 840,
    "tasks_valid": 48,
    "tasks_test": 240,
    "consecutive_pairs": 6501,
    "non_prefix_pairs": 4509,
    "rows_without_family_train": 240,
    "rows_without_family_test": 60,
    "rows_without_family_valid": 48,
    "decisions_per_episode_max": 17,
}


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def rendered_prompt(row: dict) -> str:
    """What the model sees at this row, as one string: the system and the turns before its own."""
    return "\n\x00".join(f"{m['role']}\x01{m['content']}" for m in row.get("messages", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/agent_v2e-gemma3-4b"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    per_split: dict[str, list[dict]] = {}
    for split in ("train", "valid", "test"):
        path = args.data / f"{split}.jsonl"
        if not path.exists():
            print(f"MISSING {path}", file=sys.stderr)
            return 2
        per_split[split] = list(rows(path))

    measured: dict[str, object] = {}
    measured["rendered_rows"] = sum(len(v) for v in per_split.values())

    # Tasks, and the rows that carry no family. `task_id` is the episode; a row without one is a
    # chat replay row and is not an agentic episode at all.
    tasks: set[str] = set()
    no_family: Counter[str] = Counter()
    tasks_per_split: dict[str, set[str]] = defaultdict(set)
    steps_per_task: dict[str, list[int]] = defaultdict(list)
    horizons: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for split, items in per_split.items():
        for row in items:
            meta = row.get("metadata", {})
            task_id = meta.get("task_id")
            if meta.get("family") is None:
                no_family[split] += 1
            if task_id is None:
                continue
            tasks.add(task_id)
            tasks_per_split[split].add(task_id)
            steps_per_task[task_id].append(int(meta.get("step", -1)))
    measured["tasks"] = len(tasks)
    for split in ("train", "valid", "test"):
        measured[f"tasks_{split}"] = len(tasks_per_split[split])
        measured[f"rows_without_family_{split}"] = no_family[split]

    counts = [len(v) for v in steps_per_task.values()]
    measured["decisions_per_episode_mean"] = round(statistics.mean(counts), 2) if counts else None
    measured["decisions_per_episode_max"] = max(counts) if counts else None
    measured["decisions_per_episode_median"] = statistics.median(counts) if counts else None

    # Fact 1: the step index is the turn count. Under teacher forcing each row is one decision, so
    # a task's recorded steps should be 0..n-1 with no gaps and no repeats.
    contiguous = sum(1 for v in steps_per_task.values() if sorted(v) == list(range(len(v))))
    measured["tasks_with_contiguous_steps"] = contiguous
    measured["tasks_total_for_step_check"] = len(steps_per_task)

    # Fact 2: horizon is a function of (family, difficulty, variant), except pointer_chain.
    for items in per_split.values():
        for row in items:
            meta = row.get("metadata", {})
            family = meta.get("family")
            if family is None:
                continue
            key = (family, int(meta.get("difficulty", -1)), str(meta.get("variant")))
            horizons[key].add(len(steps_per_task.get(meta.get("task_id"), [])))
    ambiguous = sorted({key[0] for key, values in horizons.items() if len(values) > 1})
    measured["families_with_ambiguous_horizon"] = ambiguous
    measured["horizon_keys"] = len(horizons)

    # Fact 3: consecutive rows of one task are mostly NOT prefixes of each other.
    by_task: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for items in per_split.values():
        for row in items:
            meta = row.get("metadata", {})
            if meta.get("task_id") is not None:
                by_task[meta["task_id"]].append((int(meta.get("step", -1)), rendered_prompt(row)))
    pairs = non_prefix = 0
    for entries in by_task.values():
        entries.sort()
        for (_, earlier), (_, later) in zip(entries, entries[1:], strict=False):
            pairs += 1
            non_prefix += not later.startswith(earlier)
    measured["consecutive_pairs"] = pairs
    measured["non_prefix_pairs"] = non_prefix
    measured["non_prefix_fraction"] = round(non_prefix / pairs, 4) if pairs else None

    # `(task_id, step)` is **not** unique, and the reason is `recovery_repeats` in `data.py`: the
    # training split repeats each corrective row two or six times, byte for byte, to weight it in
    # the fine-tuning mix. So a repeated key is one decision written several times, not several
    # decisions, and the deduplicated basis is the one that counts decisions. Both are reported,
    # because a count whose basis is unstated is a count nobody can reproduce.
    # (This comment first said the repeats were a transient retry. They are not; see the record's
    # correction and `prereg_inputs.py`, which refuses if a repeated key is not an identical row.)
    deduped = {task: sorted({step for step, _ in entries}) for task, entries in
               ((task, [(s, p) for s, p in entries]) for task, entries in by_task.items())}
    dedup_counts = [len(v) for v in deduped.values()]
    measured["rows_with_task_id"] = sum(len(v) for v in by_task.values())
    measured["rows_deduplicated"] = sum(dedup_counts)
    measured["duplicate_step_rows"] = measured["rows_with_task_id"] - measured["rows_deduplicated"]
    measured["consecutive_pairs_deduplicated"] = sum(len(v) - 1 for v in deduped.values())
    measured["decisions_per_episode_max_deduplicated"] = max(dedup_counts) if dedup_counts else None
    measured["decisions_per_episode_mean_deduplicated"] = (
        round(statistics.mean(dedup_counts), 2) if dedup_counts else None
    )

    print(f"{'claim':34s} {'order':>10s} {'measured':>10s}  verdict")
    disagreements = []
    for key, claimed in CLAIMED.items():
        got = measured.get(key)
        ok = got == claimed
        disagreements += [] if ok else [(key, claimed, got)]
        print(f"{key:34s} {claimed:>10} {str(got):>10}  {'agrees' if ok else 'DIFFERS'}")
    print()
    for key in ("decisions_per_episode_mean", "decisions_per_episode_median",
                "tasks_with_contiguous_steps", "tasks_total_for_step_check",
                "families_with_ambiguous_horizon", "horizon_keys", "non_prefix_fraction",
                "rows_with_task_id", "rows_deduplicated", "duplicate_step_rows",
                "consecutive_pairs_deduplicated", "decisions_per_episode_max_deduplicated",
                "decisions_per_episode_mean_deduplicated"):
        print(f"{key:34s} {measured.get(key)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"claimed": CLAIMED, "measured": measured,
             "disagreements": [{"key": k, "order": c, "measured": g} for k, c, g in disagreements],
             "basis": "measured-here", "data": str(args.data)},
            indent=2, sort_keys=True, default=str) + "\n")
    if disagreements:
        print(f"\n{len(disagreements)} claim(s) differ; the count does not confirm §1 as written.")
        return 1
    print("\nevery claim in §1 reproduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
