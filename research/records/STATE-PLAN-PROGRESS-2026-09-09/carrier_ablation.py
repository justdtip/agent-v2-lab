"""Can the plan's progress be removed from the note without removing the next action?

The carrier-ablation arm re-renders the transcript with the state-bearing fact removed from the
note, so that a surviving effect must be carried by the model rather than re-read from the text. It
only works if such a removal exists. This measures whether it does, per family, before the
pre-registration promises one.

The test is mechanical and needs no model. Within one episode, strip a candidate clause from every
note and ask whether the remaining text still differs from step to step. If it does, the step index
is still recoverable from what is left and the ablation has removed a label rather than the carrier.

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
                "fully_ablated_episodes": sum(1 for s in shares if s <= 0.5),
            }

    payload = {
        "basis": "measured-here",
        "question": "after removing a candidate clause from every note in an episode, is the step "
                    "index still exactly recoverable from the remaining text?",
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
    print("\nmean share of steps still told apart by what the rule leaves behind; 1.0 = the rule "
          "removed a label, not the carrier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
