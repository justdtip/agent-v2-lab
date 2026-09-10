"""Compute every corpus-derived fact the plan-progress pre-registration must carry, before sealing.

The pre-registration names the decisions, the transitions and the gaps it will read. Naming them by
hand would make the document a claim about the corpus; computing them here makes it a description of
one. Everything is deterministic from the rendered rows, so re-running this on the device copy must
reproduce it exactly — which is the point, and why the file digests go into the same output.

The script's spine is the **distinct decision**, not the rendered row. `data.py` oversamples
corrective decisions in the training split only (`recovery_repeats`, applied at
`expert_rows = [row for row in expert_rows for _ in range(_repeats(...))]`), so a recovery decision
appears two or six times as a byte-identical row. That is a training-mix weight, not two decisions,
and a capture keyed on rows would forward the same tokens repeatedly and weight the fit by the
training recipe. The script checks the byte-identity rather than assuming it, and refuses if a
group's members differ.

    python research/records/STATE-PLAN-PROGRESS-2026-09-09/prereg_inputs.py --data DIR --out FILE

CPU only, no model, no card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

#: R6's exploratory stratum: one episode per family, test split, difficulty 2, clean variant. Chosen
#: by the first `task_id` in sorted order within each family, so the choice is a rule and not a pick.
EXPLORATORY = {"split": "test", "difficulty": 2, "variant": "clean", "per_family": 1}

SPLITS = ("train", "valid", "test")


def read(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, help="write the capture set, one decision per line")
    parser.add_argument("--corpus-sha256", default=None, help="the shipped archive's digest")
    args = parser.parse_args(argv)

    rows_per_split: Counter[str] = Counter()
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    replay: list[dict] = []
    for split in SPLITS:
        for ordinal, row in enumerate(read(args.data / f"{split}.jsonl")):
            rows_per_split[split] += 1
            meta = row.get("metadata", {})
            if meta.get("family") is None:
                replay.append({"split": split, "ordinal": ordinal, **meta})
                continue
            groups[(str(meta["task_id"]), int(meta["step"]))].append(
                {
                    "split": split,
                    "ordinal": ordinal,
                    "meta": meta,
                    # The **messages** the model is given at this decision, and the whole row
                    # beside it. This is canonical JSON over the semantic record and not the bytes
                    # the model reads, so it is named for what it is: two rows with one message
                    # record and different rendered prompts share it. The capture binds the
                    # rendered prompt's byte digest and the ids consumed separately, at the seam.
                    "messages_sha256": sha(row.get("messages", [])[:-1]),
                    "row_sha256": sha(row),
                }
            )

    # Every repeated key must be the same row, or the oversampling reading is wrong and the count
    # basis has to be settled before anything downstream is written.
    inconsistent = {
        f"{task}:{step}": sorted({item["row_sha256"][:12] for item in items})
        for (task, step), items in groups.items()
        if len({item["row_sha256"] for item in items}) > 1
    }

    decisions = []
    for (task, step), items in sorted(groups.items()):
        first = items[0]
        meta = first["meta"]
        decisions.append(
            {
                "task_id": task,
                "step": step,
                "split": first["split"],
                "family": str(meta.get("family")),
                "variant": str(meta.get("variant")),
                "difficulty": int(meta.get("difficulty", -1)),
                "recovery": bool(meta.get("recovery")),
                "rendered_rows": len(items),
                "messages_sha256": first["messages_sha256"],
                "row_ordinals": sorted(item["ordinal"] for item in items),
            }
        )

    steps: dict[str, list[int]] = defaultdict(list)
    for decision in decisions:
        steps[decision["task_id"]].append(decision["step"])
    split_of = {d["task_id"]: d["split"] for d in decisions}
    meta_of = {d["task_id"]: d for d in decisions}

    # A2: which decisions of each task have no rendered row. The renderer drops a row whose
    # observation begins ERROR unless that step is an injected fault, so the gap is real and the
    # pre-registration must name it rather than let a reader infer a horizon from a row count.
    missing = {
        task: sorted(set(range(max(values) + 1)) - set(values))
        for task, values in steps.items()
        if set(range(max(values) + 1)) - set(values)
    }

    # The E2 census, on the distinct decisions in step order. A transition is a consecutive pair;
    # the update rule is judged on what the step index does across it.
    # Gap and recovery are independent, and collapsing them would hide the constraint that matters:
    #   ordinary                     the step advances by one into an ordinary decision
    #   into_recovery                the step advances by one into the corrective decision
    #   into_recovery_across_a_gap   the corrective decision is entered over a step that has no
    #                                rendered row — the failing turn itself, which the renderer drops
    #   gap                          the step advances by more than one into an ordinary decision
    # The pairs marked with a gap cannot be scored as ordinary transitions, and the third kind is
    # E2's informative stratum for four of the five recovery variants, so it is counted separately.
    by_task: dict[str, list[dict]] = defaultdict(list)
    for decision in decisions:
        by_task[decision["task_id"]].append(decision)
    transitions: Counter[str] = Counter()
    by_variant_transitions: dict[str, Counter[str]] = defaultdict(Counter)
    by_split_transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for task, items in by_task.items():
        items.sort(key=lambda d: d["step"])
        variant, split = items[0]["variant"], items[0]["split"]
        for before, after in zip(items, items[1:], strict=False):
            move = after["step"] - before["step"]
            gapped, corrective = move > 1, after["recovery"]
            if move < 1:
                kind = "other"
            elif corrective:
                kind = "into_recovery_across_a_gap" if gapped else "into_recovery"
            else:
                kind = "gap" if gapped else "ordinary"
            transitions[kind] += 1
            by_variant_transitions[variant][kind] += 1
            by_split_transitions[split][kind] += 1

    # R6's twelve, by rule.
    candidates: dict[str, list[str]] = defaultdict(list)
    for task, decision in meta_of.items():
        if (
            decision["split"] == EXPLORATORY["split"]
            and decision["difficulty"] == EXPLORATORY["difficulty"]
            and decision["variant"] == EXPLORATORY["variant"]
        ):
            candidates[decision["family"]].append(task)
    exploratory = {family: sorted(ids)[0] for family, ids in sorted(candidates.items())}

    oversampled = Counter(d["rendered_rows"] for d in decisions)
    by_variant = Counter(meta_of[t]["variant"] for t in steps)

    # The archive digest names a tarball; these name the bytes inside it. A file digest is what the
    # device copy can be checked against directly, without rebuilding an archive whose digest
    # depends on mtimes and member order.
    digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.data.iterdir())
        if path.is_file()
    }

    payload = {
        "basis": "measured-here",
        "generated_from": str(args.data),
        "corpus_archive_sha256": args.corpus_sha256,
        "file_sha256": digests,
        "rows": {
            "total": sum(rows_per_split.values()),
            "per_split": dict(rows_per_split),
            "agentic": sum(len(v) for v in groups.values()),
            "chat_replay": len(replay),
        },
        "decisions": {
            "note": "the capture unit: one distinct (task_id, step). The extra rendered rows are "
                    "byte-identical copies made by `recovery_repeats` in the training split only "
                    "(data.py), which is a training-mix weight and not a second decision",
            "total": len(decisions),
            "per_split": dict(Counter(d["split"] for d in decisions)),
            "recovery": sum(1 for d in decisions if d["recovery"]),
            "rendered_rows_per_decision": dict(sorted(oversampled.items())),
            "extra_rows_from_oversampling": sum(len(v) - 1 for v in groups.values()),
            "repeated_keys_that_are_not_identical_rows": inconsistent,
            "capture_set_sha256": sha([[d["task_id"], d["step"], d["messages_sha256"]] for d in decisions]),
        },
        "tasks": {
            "total": len(steps),
            "per_split": dict(Counter(split_of.values())),
            "per_variant": dict(sorted(by_variant.items())),
        },
        "rows_without_a_family": {
            "what_they_are": "chat replay rows, not agentic episodes: no task, no step, no plan "
                             "progress, so no ground truth for this state variable exists for them",
            "total": len(replay),
            "per_split": dict(Counter(r["split"] for r in replay)),
            "per_category": dict(sorted(Counter(str(r.get("category")) for r in replay).items())),
            "per_source": dict(sorted(Counter(str(r.get("source")) for r in replay).items())),
            "disposition": "excluded from the capture",
        },
        "a2_decisions_without_a_rendered_row": {
            "rule": "the renderer drops a row whose observation begins ERROR unless that step is an "
                    "injected fault; the dropped turn still enters the context, so the turn count "
                    "is still the step index",
            "tasks_affected": len(missing),
            "missing_steps_total": sum(len(v) for v in missing.values()),
            "by_task": missing,
        },
        "e2_transition_census": {
            "rule": "a transition is a consecutive pair of distinct decisions in step order; the "
                    "update rule predicts +1 ordinarily and the recovery cost at a corrective "
                    "decision. A transition that spans a step with no rendered row is named as "
                    "such and is never scored as an ordinary +1.",
            "total": sum(transitions.values()),
            "by_kind": dict(sorted(transitions.items())),
            "by_variant": {k: dict(sorted(v.items())) for k, v in sorted(by_variant_transitions.items())},
            "by_split": {k: dict(sorted(v.items())) for k, v in sorted(by_split_transitions.items())},
        },
        "r6_exploratory_episodes": {
            "rule": EXPLORATORY | {"selection": "first task_id in sorted order within each family"},
            "count": len(exploratory),
            "by_family": exploratory,
            "detail": {
                task: {
                    "family": family,
                    "decisions": len(steps[task]),
                    "highest_step": max(steps[task]),
                    "decisions_without_a_row": missing.get(task, []),
                }
                for family, task in exploratory.items()
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.decisions:
        with args.decisions.open("w", encoding="utf-8") as stream:
            for decision in decisions:
                stream.write(json.dumps(decision, sort_keys=True) + "\n")

    print("rows      ", payload["rows"]["total"], payload["rows"]["per_split"],
          f"| agentic {payload['rows']['agentic']} | chat replay {payload['rows']['chat_replay']}")
    print("decisions ", len(decisions), payload["decisions"]["per_split"],
          f"| recovery {payload['decisions']['recovery']}")
    print("rendered rows per decision:", payload["decisions"]["rendered_rows_per_decision"],
          "| extra rows:", payload["decisions"]["extra_rows_from_oversampling"])
    print("repeated keys that are NOT identical rows:", len(inconsistent))
    print("tasks     ", len(steps), payload["tasks"]["per_split"], dict(sorted(by_variant.items())))
    print("A2 tasks with a dropped decision:", len(missing),
          "| missing steps:", sum(len(v) for v in missing.values()))
    print("E2 transitions:", sum(transitions.values()), dict(sorted(transitions.items())))
    for variant, kinds in sorted(by_variant_transitions.items()):
        print(f"   {variant:14s} {dict(sorted(kinds.items()))}")
    print("capture set sha256:", payload["decisions"]["capture_set_sha256"])
    print("R6 exploratory episodes:", len(exploratory))
    for family, task in exploratory.items():
        detail = payload["r6_exploratory_episodes"]["detail"][task]
        print(f"   {family:20s} {task:34s} decisions={detail['decisions']:3d} "
              f"top_step={detail['highest_step']:3d} dropped={detail['decisions_without_a_row']}")
    print("file digests")
    for name, digest in digests.items():
        print(f"   {name:20s} {digest}")

    if inconsistent:
        print(f"\nREFUSED: {len(inconsistent)} repeated keys are not byte-identical rows; the "
              "oversampling reading does not hold and the count basis is unsettled")
        return 1
    if len(exploratory) != 12:
        print(f"\nWARNING: the rule selected {len(exploratory)} families, not 12")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
