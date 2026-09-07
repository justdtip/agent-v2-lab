"""Issue 88 arm 1: the paired tally per checkpoint, against arm A's six defect classes.

Two readers, because arm A and arm 1 are recorded in different forms. Arm A's 19 adapter
trajectories survive only as markdown transcripts; arm 1's are eval JSON. Both are reduced to the
same shape -- a list of (call, observation) pairs -- so one rule runs over both and arm A's
recorded 12 locked-in failures is a check on the rule rather than a number taken on trust.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the checkout, so the record re-runs anywhere
CALL = re.compile(r"^`(\w+) (\{.*\})`$")
ROOTED = ("workspace/", "lab/")


def from_transcripts(directory: Path) -> dict[str, list[tuple[str, str]]]:
    """Arm A: one markdown file a task, calls on backticked lines, observations in fences."""
    runs: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(directory.glob("*.md")):
        text = path.read_text()
        title = re.search(r"^# .*:: (\S+)", text, re.M)
        if not title:
            continue
        steps: list[tuple[str, str]] = []
        for block in text.split("**Step ")[1:]:
            matches = (CALL.match(line.strip()) for line in block.splitlines())
            call = next((match for match in matches if match), None)
            fence = re.search(r"```\n(.*?)\n```", block, re.S)
            if call:
                arguments = json.dumps(json.loads(call.group(2)), sort_keys=True)
            steps.append((f"{call.group(1)} {arguments}", fence.group(1) if fence else ""))
        runs[title.group(1)] = steps
    return runs


def from_eval(path: Path) -> dict[str, dict]:
    return {t["task_id"]: t for t in json.loads(path.read_text())["trajectories"]}


def calls_of(trajectory: dict) -> list[tuple[str, str]]:
    out = []
    for step in trajectory["steps"]:
        action = step.get("action") or {}   # a step that failed to parse has no action
        name = action.get("name", "?")
        args = json.dumps(action.get("arguments", {}), sort_keys=True)
        out.append((f"{name} {args}", step.get("observation") or ""))
    return out


def locked_in(steps: list[tuple[str, str]]) -> int | None:
    """The record's rule: an identical repeat of the previous call. Returns the first such step."""
    for index in range(1, len(steps)):
        if steps[index][0] == steps[index - 1][0]:
            return index
    return None


def defect(steps: list[tuple[str, str]], meta: dict | None = None) -> tuple[str, str]:
    """Arm A's six, taken at the earliest step any of them fires.

    Earliest-step rather than class-priority, because that is how the divergence record read the
    same trajectories: the defect that caused the failure is the first one, and a later malformed
    expression is usually the consequence of an earlier wrong turn. Three of the six are
    mechanical -- an unrooted path, an expression that is not arithmetic, an identical repeat of a
    call that had just succeeded. The other three (wrong hop, wrong file, state-note comparison
    error) need the task's expected answer, so they are left unclassified here and read by hand.
    """
    for index, (call, observation) in enumerate(steps):
        name, raw = call.split(" ", 1)
        arguments = json.loads(raw)
        for key in ("path", "directory"):
            value = arguments.get(key)
            if isinstance(value, str) and value and not value.startswith(ROOTED):
                return "path abbreviation", f"step {index}: {key}={value!r} is not rooted"
        expression = arguments.get("expression")
        if name == "calculate" and isinstance(expression, str) and (
            not re.fullmatch(r"[\d\s+\-*/().]+", expression)
            or expression.count("(") != expression.count(")")
        ):
            return "malformed argument", f"step {index}: expression={expression!r}"
        complaint = observation.lower()
        if observation.startswith("ERROR") and ("syntax" in complaint or "invalid" in complaint):
            return "malformed argument", f"step {index}: {observation[:60]}"
        if index and call == steps[index - 1][0] and not steps[index - 1][1].startswith("ERROR"):
            return ("re-issued call after success",
                    f"step {index} repeats step {index - 1} after a good result")
    # Two classes arm A never showed, so they are named rather than folded into its six.
    meta = meta or {}
    if meta.get("parse_error") or any(call.startswith("? ") for call, _ in steps):
        why = meta.get("parse_error") or "no action emitted"
        return "truncated tool call", f"step {len(steps) - 1}: {why}"
    unused = [r for r in meta.get("reasons", []) if r.startswith("required tools unused")]
    if unused:
        return "required tool unused", f"{unused[0]} after {len(steps)} steps"
    return "unclassified", f"{len(steps)} steps, first identical repeat at {locked_in(steps)}"


def report(name: str, adapter: dict[str, list[tuple[str, str]]], successes: dict[str, bool],
           base: dict[str, bool], meta: dict[str, dict] | None = None) -> dict:
    failures = [task for task in adapter if not successes[task]]
    pairs = [task for task in failures if base.get(task)]
    gains = [task for task in adapter if successes[task] and not base.get(task)]
    locks = [task for task in failures if locked_in(adapter[task]) is not None]
    classes: dict[str, int] = {}
    notes: list[str] = []
    for task in failures:
        label, why = defect(adapter[task], (meta or {}).get(task))
        classes[label] = classes.get(label, 0) + 1
        notes.append(f"    {task:34s} {label:28s} {why}")
    passes = len(adapter) - len(failures)
    print(f"\n{name}: {passes}/{len(adapter)} pass · {len(pairs)} base-pass/adapter-fail pairs · "
          f"{len(gains)} adapter-pass/base-fail · {len(locks)}/{len(failures)} failures locked in")
    if gains:
        print(f"    recovers what the base missed: {', '.join(sorted(gains))}")
    for line in notes:
        print(line)
    return {"name": name, "n": len(adapter), "passes": passes, "failures": len(failures),
            "pairs": len(pairs), "gains": len(gains), "locked_in": len(locks), "defects": classes}


def main() -> int:
    base_eval = from_eval(ROOT / "outputs/agent-v2/evals/base-test.json")
    order = list(base_eval)[:19]
    base = {task: base_eval[task]["verdict"]["success"] for task in order}
    print(f"base (recorded, greedy, reused not rerun): {sum(base.values())}/19 pass")

    summaries = []
    arm_a = from_transcripts(ROOT / "outputs/agent-v2e-qwen35-4b/transcripts/best-adapter-test")
    arm_a = {task: steps for task, steps in arm_a.items() if task in set(order)}
    divergence = ROOT / "research/records/ARM-A-DIVERGENCE-2026-09-07/descriptive.json"
    record = json.loads(divergence.read_text())
    arm_a_success = {t["task_id"]: t["adapter_success"] for t in record["tasks"]}
    summaries.append(
        report("arm A @ 800 rows (the divergence record, re-read)", arm_a, arm_a_success, base)
    )
    print(f"    record says: {record['adapter_passes']} pass, "
          f"{record['pairs_base_pass_adapter_fail']} pairs, "
          f"{record['adapter_failures_locked_in']} locked in, defects {record['defects']}")

    for step in ("0000800", "0001200", "0000400"):
        path = ROOT / f"outputs/agent-v2e-qwen35-4b-top8/evals/ckpt-{step}-test.json"
        if not path.exists():  # the record's own copy, for a reader who has no outputs tree
            path = Path(__file__).with_name("evals") / f"ckpt-{step}-test.json"
        if not path.exists():
            print(f"\nckpt-{step}: not yet written")
            continue
        trajectories = from_eval(path)
        steps_by_task = {task: calls_of(t) for task, t in trajectories.items()}
        success = {task: t["verdict"]["success"] for task, t in trajectories.items()}
        meta = {task: {"parse_error": t.get("parse_error"), "reasons": t["verdict"]["reasons"]}
                for task, t in trajectories.items()}
        summaries.append(report(f"arm 1 ckpt-{step}", steps_by_task, success, base, meta))

    (Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/dev/null")).write_text(
        json.dumps(summaries, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
