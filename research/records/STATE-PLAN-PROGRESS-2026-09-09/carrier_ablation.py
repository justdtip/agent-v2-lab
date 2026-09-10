"""A **screening diagnostic** for candidate removal rules: does the step survive as a distinct string?

This is a screen and not the gate. It answers one narrow question — after stripping a candidate
clause, are the remaining notes still distinct strings within an episode? — and a rule that fails it
has certainly not removed the carrier. **A rule that passes it has not been shown to have removed the
carrier either**: distinctness is a property of strings, and a step can be recoverable from text that
repeats, or unrecoverable from text that differs in an irrelevant token. The pre-registration's
two-part acceptance gate (drive the share to the floor **and** leave the tool call and next-action
clause byte-identical) is **not implemented here**; the second half is not checked at all, and no
result from this script should be read as that gate having been applied.

The test is mechanical and needs no model. Within one episode, strip a candidate clause from every
note and count how many distinct strings remain.

    python research/records/STATE-PLAN-PROGRESS-2026-09-09/carrier_ablation.py --data DIR --out FILE

CPU only, no model, no card.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

#: The explicit progress statement, per the shapes the renderer produces. Each is anchored at the
#: start of the note, because that is where `pipeline/tasks.py` puts the counter.
COUNTER = re.compile(
    r"^(?:"
    r"(?:Inspected|Applied|Read|Visited|Checked|Updated|Loaded)\s+\d+\s+of\s+\d+\.\s*"
    r"|Visited\s+\d+\s+node\(s\)[^.]*\.\s*"
    r"|approved:[^.]*\.\s*"
    r"|loads so far:[^.]*\.\s*"
    r")"
)
#: The clause that names the next action. Removing it removes the action, not the state.
NEXT = re.compile(r"Next:[^.]*\.\s*")
#: The queue and the pending list, both of which shrink as the episode advances.
QUEUE = re.compile(r"(?:queue|pending|remaining):[^.]*\.\s*")


def note_of(row: dict) -> str:
    """The model's own note at this decision: the completion's prose, before the fenced call."""
    return row["completion"].split("```")[0].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    episodes: dict[str, dict[int, str]] = defaultdict(dict)
    family_of: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        with (args.data / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = row.get("metadata", {})
                if meta.get("family") is None:
                    continue
                episodes[meta["task_id"]][int(meta["step"])] = note_of(row)
                family_of[meta["task_id"]] = str(meta["family"])

    variants = {
        "whole note": lambda text: text,
        "counter removed": lambda text: COUNTER.sub("", text, count=1),
        "counter and queue removed": lambda text: QUEUE.sub("", COUNTER.sub("", text, count=1)),
        "counter, queue and next removed": lambda text: NEXT.sub(
            "", QUEUE.sub("", COUNTER.sub("", text, count=1))
        ),
    }

    per_family: dict[str, dict[str, dict]] = defaultdict(dict)
    for name, rule in variants.items():
        tally: dict[str, list[bool]] = defaultdict(list)
        for task, notes in episodes.items():
            if len(notes) < 2:
                continue
            ablated = [rule(notes[step]) for step in sorted(notes)]
            # How much of the carrier survives, as a gradient rather than a boolean: the number of
            # distinct remaining texts over the number of steps. 1.0 means every step is still
            # told apart by what is left, so the ablation removed a label; 1/n means the notes have
            # become indistinguishable and the step is no longer readable from them.
            tally[family_of[task]].append(len(set(ablated)) / len(ablated))
        for family, shares in tally.items():
            per_family[family][name] = {
                "episodes": len(shares),
                "distinguishable_share": round(sum(shares) / len(shares), 4),
                # Named for what it counts, not for what one might hope it means: episodes
                # where at most half the steps remain distinct strings. It is not a count of
                # episodes in which the carrier was removed, and it was called that.
                "episodes_at_or_below_half_distinct": sum(1 for s in shares if s <= 0.5),
            }

    payload = {
        "basis": "measured-here",
        "what_this_is": "a screening diagnostic, not the acceptance gate: the second half of "
                        "the gate — that the tool call and next-action clause are byte-identical "
                        "after the removal — is not implemented here",
        "question": "after removing a candidate clause from every note in an episode, how many of "
                    "the remaining texts are still distinct strings?",
        "reading": "distinguishable_share is the mean over episodes of (distinct remaining notes / "
                   "steps). 1.0 means every step is still told apart by what is left, so the rule "
                   "removed a label and not the carrier. The arm is constructible for a family "
                   "only where a rule drives this near its floor of 1/n.",
        "rules": {"counter": COUNTER.pattern, "queue": QUEUE.pattern, "next": NEXT.pattern},
        "by_family": {k: per_family[k] for k in sorted(per_family)},
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    order = list(variants)
    print(f"{'family':20s} " + "".join(f"{n[:22]:>24s}" for n in order))
    for family in sorted(per_family):
        cells = "".join(
            f"{per_family[family][n]['distinguishable_share']:>24.3f}"
            if n in per_family[family] else f"{'-':>24}"
            for n in order
        )
        print(f"{family:20s} {cells}")
    print("\nmean share of steps whose note remains a distinct string after the removal; 1.0 "
          "means the rule certainly did not remove the carrier. A low share is a screen passed, "
          "not the acceptance gate met: the byte-identity half of the gate is not implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
