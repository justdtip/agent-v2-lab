from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.tasks import Task, task_from_id

__all__ = [
    "Fact",
    "IntegrityReport",
    "Violation",
    "check_trajectory",
    "completion_patterns",
    "main",
    "required_carry",
]


@dataclass(frozen=True)
class Fact:
    kind: str
    value: str
    observed_at: int


@dataclass(frozen=True)
class Violation:
    step: int
    kind: str
    detail: str


@dataclass(frozen=True)
class IntegrityReport:
    violations: tuple[Violation, ...]
    first_violation: Violation | None
    counts: dict[str, int]
    clean: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "violations": [asdict(violation) for violation in self.violations],
            "first_violation": (
                asdict(self.first_violation) if self.first_violation is not None else None
            ),
            "counts": dict(sorted(self.counts.items())),
            "clean": self.clean,
        }


def completion_patterns() -> tuple[re.Pattern[str], ...]:
    flags = re.IGNORECASE
    return (
        re.compile(r"\(\s*full\s*\)", flags),
        re.compile(r"\(\s*final\s*\)", flags),
        re.compile(r"\bcomplete\b", flags),
        re.compile(r"\ball\s+\d+\b.*?\b(?:read|inspected|applied|verified)\b", flags),
        re.compile(r"\bpending\s*:\s*none\b", flags),
        re.compile(r"\btask\s+complete\b", flags),
    )


def _normalise(value: Any) -> str:
    return " ".join(str(value).split())


def _action_text(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        return ""
    return _normalise(" ".join(str(value) for value in arguments.values()))


def _extract_facts(observation: str, observed_at: int) -> set[Fact]:
    facts: set[Fact] = set()
    if "status=approved" in observation:
        match = re.search(r"\bamount=(\d+)\b", observation)
        if match:
            facts.add(Fact("amount", match.group(1), observed_at))
    metric = re.search(r"\bmetric=\d+\b.*?\bvalue=(\d+)\b", observation, re.DOTALL)
    if metric:
        facts.add(Fact("metric", metric.group(1), observed_at))
    service = re.search(r"\bname=(service-\d+)\b.*?\bload=(\d+)\b", observation, re.DOTALL)
    if service:
        facts.add(Fact("load", f"{service.group(1)}={service.group(2)}", observed_at))
    next_key = re.search(r"\bNext-Key:\s*([^\s]+)", observation, re.IGNORECASE)
    if next_key:
        facts.add(Fact("next_key", next_key.group(1), observed_at))
    total = re.search(r"\bRESULT:\s*([^\s]+)", observation, re.IGNORECASE)
    if total:
        facts.add(Fact("total", total.group(1), observed_at))
    for listing in re.finditer(r"(?:^|\n)(?:FILES|MATCHES):\s*([^\n]*)", observation):
        for path in listing.group(1).split(","):
            basename = path.strip().rsplit("/", 1)[-1]
            if basename:
                facts.add(Fact("path", basename, observed_at))
    for line in observation.splitlines():
        manifest = re.fullmatch(r"\s*([^|]+)\|mode=([^\s]+)->mode=([^\s]+)\s*", line)
        if manifest:
            basename = manifest.group(1).strip().rsplit("/", 1)[-1]
            facts.add(
                Fact(
                    "worker",
                    f"{basename} mode={manifest.group(2)} -> mode={manifest.group(3)}",
                    observed_at,
                )
            )
    return facts


def _fact_is_referenced(fact: Fact, text: str) -> bool:
    text = _normalise(text)
    if fact.kind in {"amount", "metric", "total"}:
        return re.search(rf"(?<!\d){re.escape(fact.value)}(?!\d)", text) is not None
    return _normalise(fact.value).casefold() in text.casefold()


def required_carry(task: Task, *, keep_last: int) -> dict[int, frozenset[Fact]]:
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")
    simulator = Simulator.for_task(task)
    observations: list[tuple[int, str]] = []
    result: dict[int, frozenset[Fact]] = {}
    for index, step in enumerate(task.steps):
        prior = observations[:-keep_last] if keep_last else observations
        reference = f"{step.thought} {_action_text({'arguments': step.action.arguments})}"
        facts = {
            fact
            for observed_at, observation in prior
            for fact in _extract_facts(observation, observed_at)
            if _fact_is_referenced(fact, reference)
        }
        result[index] = frozenset(facts)
        observation = simulator.execute(step.action)
        if step.action.name != "finish":
            observations.append((index, observation))
    return result


def check_trajectory(
    task: Task,
    steps: list[dict[str, Any]],
    *,
    keep_last: int,
) -> IntegrityReport:
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")
    carry = required_carry(task, keep_last=keep_last)
    canonical_notes = [_normalise(step.thought) for step in task.steps]
    violations: list[tuple[int, int, Violation]] = []
    actual_observations: list[str] = []
    previous_note = ""

    for position, record in enumerate(steps):
        if not isinstance(record, dict):
            record = {}
        saved_index = record.get("index")
        canonical_index = (
            saved_index
            if isinstance(saved_index, int) and 0 <= saved_index < len(task.steps)
            else position
        )
        violation_step = canonical_index
        note = _normalise(record.get("thought") or "")
        canonical = (
            canonical_notes[canonical_index]
            if 0 <= canonical_index < len(canonical_notes)
            else ""
        )

        def add(
            order: int, kind: str, detail: str, *, step: int = violation_step
        ) -> None:
            violations.append(
                (step, order, Violation(step, kind, detail))
            )

        if note and previous_note and note == previous_note:
            add(0, "verbatim_copy", "note repeats the previous step verbatim")

        stale_labels = _contradictory_fields(note, canonical)
        visible = actual_observations[-keep_last:] if keep_last else []
        missing: list[Fact] = []
        if canonical_index in carry:
            reference = note
            for fact in sorted(carry[canonical_index], key=lambda item: (item.kind, item.value)):
                if fact.kind in {"path", "worker"}:
                    continue
                if _fact_is_referenced(fact, reference):
                    continue
                if any(
                    any(
                        visible_fact.kind == fact.kind and visible_fact.value == fact.value
                        for visible_fact in _extract_facts(observation, -1)
                    )
                    for observation in visible
                ):
                    continue
                if _fact_covered_by_stale_field(fact, stale_labels, canonical):
                    continue
                missing.append(fact)
        if missing:
            values = ", ".join(dict.fromkeys(fact.value for fact in missing))
            add(1, "value_drop", f"missing required values: {values}")

        if _has_premature_completion(
            note,
            canonical,
            at_or_after_horizon=canonical_index >= len(task.steps) - 1,
        ):
            add(2, "premature_completion", "completion claim is not licensed at this step")

        actual_counts = re.findall(r"\b(\d+)\s+of\s+(\d+)\b", note, re.IGNORECASE)
        canonical_counts = re.findall(r"\b(\d+)\s+of\s+(\d+)\b", canonical, re.IGNORECASE)
        if actual_counts and actual_counts != canonical_counts:
            add(
                3,
                "count_mismatch",
                f"count {actual_counts!r} contradicts canonical {canonical_counts!r}",
            )

        if stale_labels:
            labels = ", ".join(sorted(stale_labels))
            add(4, "stale_fact", f"contradictory structured fields: {labels}")

        omitted = _omitted_queue_entries(note, canonical)
        if omitted:
            add(5, "queue_loss", f"omitted remaining entries: {', '.join(sorted(omitted))}")

        observation = record.get("observation")
        if isinstance(observation, str) and observation != "FINISHED":
            actual_observations.append(observation)
        previous_note = note

    ordered = tuple(item[2] for item in sorted(violations, key=lambda item: item[:2]))
    counts = dict(sorted(Counter(violation.kind for violation in ordered).items()))
    return IntegrityReport(ordered, ordered[0] if ordered else None, counts, not ordered)


_FIELD_PATTERN = re.compile(
    r"\b(approved|first half|second half|highest so far)\s*:\s*"
    r"(.*?)(?=;|\.\s+(?:Reading|Computing)|,\s+above threshold|$)",
    re.IGNORECASE,
)


def _structured_fields(note: str) -> dict[str, str]:
    return {
        match.group(1).casefold(): _normalise(match.group(2)).casefold()
        for match in _FIELD_PATTERN.finditer(note)
    }


def _field_tokens(value: str) -> set[str]:
    return set(re.findall(r"service-\d+=\d+|(?<!\d)\d+(?!\d)|\bnone\b", value))


def _contradictory_fields(note: str, canonical: str) -> set[str]:
    actual_fields = _structured_fields(note)
    canonical_fields = _structured_fields(canonical)
    contradictory: set[str] = set()
    for label, actual in actual_fields.items():
        expected = canonical_fields.get(label)
        if expected is None or actual == expected:
            continue
        actual_tokens = _field_tokens(actual)
        expected_tokens = _field_tokens(expected)
        if not actual_tokens.issubset(expected_tokens):
            contradictory.add(label)
    return contradictory


def _fact_covered_by_stale_field(fact: Fact, labels: set[str], canonical: str) -> bool:
    fields = _structured_fields(canonical)
    return any(
        _fact_is_referenced(fact, fields[label])
        for label in labels
        if label in fields
    )


def _queue_entries(note: str) -> dict[str, set[str]]:
    entries: dict[str, set[str]] = {}
    patterns = {
        "pending": r"\bpending\s*:\s*(.*?)(?:\.(?:\s|$)|$)",
        "remaining": r"\bRemaining after this\s*:\s*(.*?)(?:\.(?:\s|$)|$)",
    }
    for label, pattern in patterns.items():
        match = re.search(pattern, note, re.IGNORECASE)
        if match:
            entries[label] = set(
                re.findall(r"[A-Za-z0-9_-]+\.[A-Za-z0-9]+", match.group(1))
            )
    return entries


def _omitted_queue_entries(note: str, canonical: str) -> set[str]:
    actual = _queue_entries(note)
    expected = _queue_entries(canonical)
    omitted: set[str] = set()
    for label, canonical_entries in expected.items():
        if label in actual and actual[label] < canonical_entries:
            omitted.update(canonical_entries - actual[label])
    return omitted


def _has_premature_completion(
    note: str, canonical: str, *, at_or_after_horizon: bool
) -> bool:
    if at_or_after_horizon:
        return False
    patterns = completion_patterns()
    actual = [bool(pattern.search(note)) for pattern in patterns]
    expected = [bool(pattern.search(canonical)) for pattern in patterns]
    if actual[5] and not (expected[5] or at_or_after_horizon):
        return True
    if actual[4] and not expected[4]:
        return True
    if actual[1] and not expected[1]:
        return True
    if actual[0] and not expected[0]:
        return True
    if actual[3] and not expected[3]:
        return True
    phase_licensed = expected[0] or expected[2] or expected[3] or expected[5]
    return actual[2] and not (phase_licensed or at_or_after_horizon)


def _analyse_evaluation(path: Path, fallback_seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("trajectories", [])
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError(f"{path}: expected summary object and trajectories list")
    seed = summary.get("data_seed", fallback_seed)
    keep_last = summary.get("keep_last")
    if not isinstance(seed, int) or not isinstance(keep_last, int) or keep_last < 0:
        raise ValueError(f"{path}: data_seed and non-negative keep_last are required")
    analysed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("task_id"), str):
            raise ValueError(f"{path}: trajectory is missing task_id")
        task_id = record["task_id"]
        if task_id in analysed:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        difficulty = record.get("difficulty")
        if type(difficulty) is int and difficulty == -1:
            difficulty = None
        elif difficulty is not None and (
            type(difficulty) is not int or difficulty < 0
        ):
            raise ValueError(f"{path}: {task_id} has invalid difficulty")
        task = task_from_id(task_id, seed, difficulty=difficulty)
        steps = record.get("steps", [])
        integrity = check_trajectory(
            task, steps if isinstance(steps, list) else [], keep_last=keep_last
        )
        verdict = record.get("verdict", {})
        success = bool(verdict.get("success")) if isinstance(verdict, dict) else False
        analysed[task_id] = {
            "family": task.family,
            "success": success,
            "integrity": integrity,
        }
    return {
        "path": path,
        "label": str(summary.get("label") or path.stem),
        "seed": seed,
        "keep_last": keep_last,
        "records": analysed,
    }


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    rendered = [[str(cell) for cell in row] for row in rows]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rendered),
    ]


def _render_evaluations(paths: list[Path], seed: int) -> str:
    evaluations = [_analyse_evaluation(path, seed) for path in paths]
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    task_ids = set(evaluations[0]["records"])
    for evaluation in evaluations[1:]:
        other_ids = set(evaluation["records"])
        if other_ids != task_ids:
            missing = sorted(task_ids - other_ids)
            extra = sorted(other_ids - task_ids)
            raise ValueError(
                "evaluation task_id sets differ: "
                f"{evaluation['path']} missing={missing[:3]} extra={extra[:3]}"
            )

    lines = ["# Offline note-integrity comparison", "", "## Sources and configuration", ""]
    lines.extend(
        _table(
            ["run", "source", "data seed", "keep-last", "tasks"],
            [
                [
                    evaluation["label"],
                    evaluation["path"],
                    evaluation["seed"],
                    evaluation["keep_last"],
                    len(evaluation["records"]),
                ]
                for evaluation in evaluations
            ],
        )
    )

    lines.extend(["", "## Overall", ""])
    overall_rows = []
    for evaluation in evaluations:
        records = list(evaluation["records"].values())
        successes = sum(record["success"] for record in records)
        clean = sum(record["integrity"].clean for record in records)
        failed = len(records) - successes
        failed_affected = sum(
            not record["success"] and not record["integrity"].clean for record in records
        )
        overall_rows.append(
            [
                evaluation["label"],
                _fraction(successes, len(records)),
                _fraction(clean, len(records)),
                len(records) - clean,
                _fraction(failed_affected, failed),
            ]
        )
    lines.extend(
        _table(
            [
                "run",
                "outcome success",
                "integrity clean",
                "affected trajectories",
                "failed with violation",
            ],
            overall_rows,
        )
    )

    lines.extend(["", "## Per-family", ""])
    family_rows = []
    for evaluation in evaluations:
        records = evaluation["records"]
        for family in sorted({record["family"] for record in records.values()}):
            group = [record for record in records.values() if record["family"] == family]
            successes = sum(record["success"] for record in group)
            clean = sum(record["integrity"].clean for record in group)
            family_rows.append(
                [
                    evaluation["label"],
                    family,
                    _fraction(successes, len(group)),
                    len(group) - successes,
                    _fraction(clean, len(group)),
                ]
            )
    lines.extend(
        _table(
            ["run", "family", "outcome success", "failures", "integrity clean"],
            family_rows,
        )
    )

    lines.extend(["", "## Violation kinds (affected trajectories)", ""])
    violation_rows = []
    for evaluation in evaluations:
        affected: Counter[tuple[str, str]] = Counter()
        for record in evaluation["records"].values():
            for kind in record["integrity"].counts:
                affected[(record["family"], kind)] += 1
        for (family, kind), count in sorted(affected.items()):
            violation_rows.append([evaluation["label"], family, kind, count])
    lines.extend(
        _table(
            ["run", "family", "violation", "affected trajectories"],
            violation_rows,
        )
    )

    if len(evaluations) >= 2:
        first, second = evaluations[:2]
        pairs = [
            (first["records"][task_id], second["records"][task_id])
            for task_id in sorted(task_ids)
        ]
        integrity_flips = Counter(
            (left["integrity"].clean, right["integrity"].clean) for left, right in pairs
        )
        outcome_flips = Counter((left["success"], right["success"]) for left, right in pairs)
        labels = {
            (True, True): "clean → clean",
            (True, False): "clean → affected",
            (False, True): "affected → clean",
            (False, False): "affected → affected",
        }
        outcome_labels = {
            (True, True): "success → success",
            (True, False): "success → failure",
            (False, True): "failure → success",
            (False, False): "failure → failure",
        }
        lines.extend(["", "## Integrity-clean paired flips", ""])
        lines.extend(
            _table(
                ["transition", "tasks"],
                [[labels[key], integrity_flips[key]] for key in labels],
            )
        )
        lines.extend(["", "## Outcome paired flips", ""])
        lines.extend(
            _table(
                ["transition", "tasks"],
                [[outcome_labels[key], outcome_flips[key]] for key in outcome_labels],
            )
        )

    acceptance = {
        ("cross_reference", "verbatim_copy"): 15,
        ("batch_update", "premature_completion"): 14,
        ("ledger_reconcile", "value_drop"): 6,
    }
    last = evaluations[-1]
    observed: Counter[tuple[str, str]] = Counter()
    for record in last["records"].values():
        for kind in record["integrity"].counts:
            observed[(record["family"], kind)] += 1
    lines.extend(["", "## Memo acceptance checks", ""])
    lines.extend(
        _table(
            ["family", "violation", "expected", "observed", "status"],
            [
                [family, kind, expected, observed[(family, kind)], "PASS" if observed[(family, kind)] == expected else "FAIL"]
                for (family, kind), expected in acceptance.items()
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved evaluations for note integrity")
    parser.add_argument("--eval", dest="evaluations", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=20260902, type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _render_evaluations(args.evaluations, args.seed), encoding="utf-8"
    )
