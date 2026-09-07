from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import SubprocessError
from typing import Any

from local_llm_lab import spawn
from local_llm_lab.pipeline.env import Simulator
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, Task, replay_task_from_id
from local_llm_lab.runlog import RunLog, git_commit

__all__ = [
    "Fact",
    "IntegrityReport",
    "Violation",
    "check_trajectory",
    "completion_patterns",
    "git_tree_dirty",
    "main",
    "required_carry",
]

_GIT_TIMEOUT = 10.0


def _git_argv(cwd: Path | None, *arguments: str) -> list[str]:
    """``git`` with the working directory expressed as ``-C``, not as ``subprocess``'s ``cwd``.

    Same effect, different mechanism, and the mechanism is the point: CPython takes
    ``posix_spawn`` only when ``cwd`` is ``None``, so passing a directory to ``subprocess``
    forks -- which aborts an interpreter that has initialised Metal (R45, issue 84 item 3).
    ``git -C`` moves the directory into argv, where it costs nothing.

    Failure behaviour is unchanged: ``git -C`` on a path that does not exist exits non-zero,
    which both callers already read as "not determined".
    """
    return ["git", *(("-C", str(cwd)) if cwd is not None else ()), *arguments]


def git_tree_dirty(cwd: Path | None = None) -> bool | None:
    """Whether the working tree carries uncommitted changes, for R26(e) identity blocks.

    Lives here because this is the only module the three CLIs of the run-log lane
    (``evaluate``, ``prefer``, ``integrity``) can all import without pulling in a model
    dependency, and ``runlog.py`` -- which owns its sibling :func:`runlog.git_commit` --
    belongs to another lane this cycle.

    Mirrors ``git_commit``'s contract: provenance is a nice-to-have and must never be the
    thing that kills a run, so any error (no git, not a repository, timeout) is reported as
    ``None`` -- "not determined" -- rather than raised or silently read as clean.
    """
    try:
        completed = spawn.run(
            _git_argv(cwd, "status", "--porcelain"),
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, ValueError, spawn.UnsafeSpawnError, SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


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


def _matched_fact(
    observation: str,
    pattern: str,
    kind: str,
    observed_at: int,
    *,
    flags: int = 0,
) -> Fact | None:
    match = re.search(pattern, observation, flags)
    return Fact(kind, match.group(1), observed_at) if match else None


def _approved_amount_fact(observation: str, observed_at: int) -> Fact | None:
    if "status=approved" not in observation:
        return None
    return _matched_fact(observation, r"\bamount=(\d+)\b", "amount", observed_at)


def _load_fact(observation: str, observed_at: int) -> Fact | None:
    match = re.search(
        r"\bname=(service-\d+)\b.*?\bload=(\d+)\b", observation, re.DOTALL
    )
    if not match:
        return None
    return Fact("load", f"{match.group(1)}={match.group(2)}", observed_at)


def _listing_facts(observation: str, observed_at: int) -> set[Fact]:
    facts: set[Fact] = set()
    for listing in re.finditer(r"(?:^|\n)(?:FILES|MATCHES):\s*([^\n]*)", observation):
        for path in listing.group(1).split(","):
            basename = path.strip().rsplit("/", 1)[-1]
            if basename:
                facts.add(Fact("path", basename, observed_at))
    return facts


def _worker_facts(observation: str, observed_at: int) -> set[Fact]:
    facts: set[Fact] = set()
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


def _extract_facts(observation: str, observed_at: int) -> set[Fact]:
    scalar_facts = (
        _approved_amount_fact(observation, observed_at),
        _matched_fact(
            observation,
            r"\bmetric=\d+\b.*?\bvalue=(\d+)\b",
            "metric",
            observed_at,
            flags=re.DOTALL,
        ),
        _load_fact(observation, observed_at),
        _matched_fact(
            observation,
            r"\bNext-Key:\s*([^\s]+)",
            "next_key",
            observed_at,
            flags=re.IGNORECASE,
        ),
        _matched_fact(
            observation,
            r"\bRESULT:\s*([^\s]+)",
            "total",
            observed_at,
            flags=re.IGNORECASE,
        ),
    )
    facts = {fact for fact in scalar_facts if fact is not None}
    facts.update(_listing_facts(observation, observed_at))
    facts.update(_worker_facts(observation, observed_at))
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


def _policy_index(record: dict[str, Any], position: int, horizon: int) -> int:
    saved_index = record.get("index")
    if isinstance(saved_index, int) and 0 <= saved_index < horizon:
        return saved_index
    return position


def _canonical_note(notes: list[str], index: int) -> str:
    return notes[index] if 0 <= index < len(notes) else ""


def _verbatim_copy_violation(step: int, note: str, previous_note: str) -> Violation | None:
    if note and previous_note and note == previous_note:
        return Violation(step, "verbatim_copy", "note repeats the previous step verbatim")
    return None


def _fact_is_visible(fact: Fact, observations: list[str]) -> bool:
    return any(
        any(
            visible_fact.kind == fact.kind and visible_fact.value == fact.value
            for visible_fact in _extract_facts(observation, -1)
        )
        for observation in observations
    )


def _fact_is_missing(
    fact: Fact,
    note: str,
    visible: list[str],
    stale_labels: set[str],
    canonical: str,
) -> bool:
    return (
        fact.kind not in {"path", "worker"}
        and not _fact_is_referenced(fact, note)
        and not _fact_is_visible(fact, visible)
        and not _fact_covered_by_stale_field(fact, stale_labels, canonical)
    )


def _value_drop_violation(
    step: int,
    facts: frozenset[Fact],
    note: str,
    visible: list[str],
    stale_labels: set[str],
    canonical: str,
) -> Violation | None:
    missing = [
        fact
        for fact in sorted(facts, key=lambda item: (item.kind, item.value))
        if _fact_is_missing(fact, note, visible, stale_labels, canonical)
    ]
    if not missing:
        return None
    values = ", ".join(dict.fromkeys(fact.value for fact in missing))
    return Violation(step, "value_drop", f"missing required values: {values}")


def _completion_violation(
    step: int, note: str, canonical: str, *, at_or_after_horizon: bool
) -> Violation | None:
    if not _has_premature_completion(
        note, canonical, at_or_after_horizon=at_or_after_horizon
    ):
        return None
    return Violation(step, "premature_completion", "completion claim is not licensed at this step")


def _count_violation(step: int, note: str, canonical: str) -> Violation | None:
    actual = re.findall(r"\b(\d+)\s+of\s+(\d+)\b", note, re.IGNORECASE)
    expected = re.findall(r"\b(\d+)\s+of\s+(\d+)\b", canonical, re.IGNORECASE)
    if not actual or actual == expected:
        return None
    return Violation(step, "count_mismatch", f"count {actual!r} contradicts canonical {expected!r}")


def _stale_violation(step: int, stale_labels: set[str]) -> Violation | None:
    if not stale_labels:
        return None
    labels = ", ".join(sorted(stale_labels))
    return Violation(step, "stale_fact", f"contradictory structured fields: {labels}")


def _queue_violation(step: int, note: str, canonical: str) -> Violation | None:
    omitted = _omitted_queue_entries(note, canonical)
    if not omitted:
        return None
    return Violation(
        step, "queue_loss", f"omitted remaining entries: {', '.join(sorted(omitted))}"
    )


def _step_violations(
    *,
    step: int,
    note: str,
    previous_note: str,
    canonical: str,
    facts: frozenset[Fact],
    visible: list[str],
    at_or_after_horizon: bool,
) -> tuple[Violation | None, ...]:
    stale_labels = _contradictory_fields(note, canonical)
    return (
        _verbatim_copy_violation(step, note, previous_note),
        _value_drop_violation(step, facts, note, visible, stale_labels, canonical),
        _completion_violation(
            step, note, canonical, at_or_after_horizon=at_or_after_horizon
        ),
        _count_violation(step, note, canonical),
        _stale_violation(step, stale_labels),
        _queue_violation(step, note, canonical),
    )


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
    for position, raw_record in enumerate(steps):
        record = raw_record if isinstance(raw_record, dict) else {}
        step = _policy_index(record, position, len(task.steps))
        note = _normalise(record.get("thought") or "")
        canonical = _canonical_note(canonical_notes, step)
        visible = actual_observations[-keep_last:] if keep_last else []
        candidates = _step_violations(
            step=step,
            note=note,
            previous_note=previous_note,
            canonical=canonical,
            facts=carry.get(step, frozenset()),
            visible=visible,
            at_or_after_horizon=step >= len(task.steps) - 1,
        )
        violations.extend(
            (step, order, violation)
            for order, violation in enumerate(candidates)
            if violation is not None
        )
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


def _artifact_generator_version(recorded: object, explicit: int | None, *, source: Path) -> int:
    for label, value in (("recorded", recorded), ("explicit", explicit)):
        if value is not None and (type(value) is not int or not 1 <= value <= GENERATOR_VERSION):
            raise ValueError(f"{source}: invalid {label} generator_version {value!r}")
    if recorded is None and explicit is None:
        raise ValueError(f"{source}: artifact has no generator_version; bind one explicitly")
    if recorded is not None and explicit is not None and recorded != explicit:
        raise ValueError(
            f"{source}: recorded generator_version {recorded} conflicts with explicit {explicit}"
        )
    return int(recorded if recorded is not None else explicit)


def _generator_version_basis(recorded: object, explicit: int | None) -> str:
    """R23: why the version in force was chosen, recorded beside the version itself.

    Called only after :func:`_artifact_generator_version` has accepted the pair, so an
    unbound or conflicting artifact never reaches a basis string: the fail-closed R12 error
    stands, and a refusal is never dressed up as a recorded rationale.  Shape follows
    ``probes/patch.py:526``.
    """
    if recorded is not None and explicit is not None:
        return "recorded in the evaluation artifact, confirmed by the explicit binding"
    if recorded is not None:
        return "recorded in the evaluation artifact"
    if explicit == 1:
        return (
            "R23 pre-versioning binding: explicit --generator-version 1; R5 defines v1 as "
            "the C-reproducing generator and no version field could exist before versioning"
        )
    return "explicit --generator-version binding; the artifact records no generator_version"


SEED_FROM_ARTIFACT = "recorded in the evaluation artifact"
SEED_FROM_CALLER = "caller's --seed; the artifact records no data_seed"
"""Why the seed the replay used is the seed it used.

The generator version fails closed when the artifact records none; the data seed does not,
because refusing an artifact without one would refuse every evaluation currently on disk.
So the fallback stays and stops being silent: a replay under the caller's seed rebuilds a
*different task* from the one the trajectory was run against, and every integrity verdict
below it is then computed against that other task.  The basis is recorded per artifact and
warned about in the run log, so nobody reads the comparison without seeing which it was.
"""


def _artifact_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    """R26(e): which policy produced this saved evaluation, and on what split.

    ``run_evaluation`` writes ``model`` as the resolved spec record, so the registry name
    and hf_id sit under ``model.spec``; an older artifact that recorded a bare string is
    read as the model name with no hf_id rather than rejected.
    """
    model = summary.get("model")
    spec = model.get("spec") if isinstance(model, dict) else None
    legacy_name = model if isinstance(model, str) else None
    name = spec.get("name") if isinstance(spec, dict) else legacy_name
    return {
        "model": name,
        "hf_id": spec.get("hf_id") if isinstance(spec, dict) else None,
        "policy": summary.get("label"),
        "adapter": summary.get("adapter"),
        "split": summary.get("split"),
    }


def _analyse_evaluation(
    path: Path, fallback_seed: int, *, generator_version: int | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    records = payload.get("trajectories", [])
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError(f"{path}: expected summary object and trajectories list")
    recorded_version = summary.get("generator_version", payload.get("generator_version"))
    version = _artifact_generator_version(recorded_version, generator_version, source=path)
    basis = _generator_version_basis(recorded_version, generator_version)
    # Key presence, not value: a summary carrying an explicit null ``data_seed`` is malformed
    # and must reach the type check below, not be quietly swapped for the caller's seed.
    seed_recorded = "data_seed" in summary
    seed = summary["data_seed"] if seed_recorded else fallback_seed
    seed_source = SEED_FROM_ARTIFACT if seed_recorded else SEED_FROM_CALLER
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
        task = replay_task_from_id(task_id, seed, version, difficulty=difficulty)
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
        # The seed alone cannot say whether it came from the artifact or from the caller, and
        # those two replay different tasks. Recorded here so the report and events.jsonl carry
        # the distinction instead of leaving a reader to assume the artifact supplied it.
        "seed_source": seed_source,
        "keep_last": keep_last,
        "generator_version": version,
        # R23: the version alone does not say why it is in force; a reader of the report or
        # of events.jsonl must not have to infer a v1 binding from the absence of a field.
        "generator_version_basis": basis,
        "identity": _artifact_identity(summary),
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


def _validated_task_ids(evaluations: list[dict[str, Any]]) -> set[str]:
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
    return task_ids


def _section(title: str, headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["", f"## {title}", "", *_table(headers, rows)]


def _source_rows(evaluations: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            evaluation["label"],
            evaluation["path"],
            evaluation["seed"],
            evaluation["seed_source"],
            evaluation["keep_last"],
            evaluation["generator_version"],
            evaluation["generator_version_basis"],
            len(evaluation["records"]),
        ]
        for evaluation in evaluations
    ]


def _overall_rows(evaluations: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for evaluation in evaluations:
        records = list(evaluation["records"].values())
        successes = sum(record["success"] for record in records)
        clean = sum(record["integrity"].clean for record in records)
        failed = len(records) - successes
        failed_affected = sum(
            not record["success"] and not record["integrity"].clean for record in records
        )
        rows.append(
            [
                evaluation["label"],
                _fraction(successes, len(records)),
                _fraction(clean, len(records)),
                len(records) - clean,
                _fraction(failed_affected, failed),
            ]
        )
    return rows


def _family_rows(evaluations: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for evaluation in evaluations:
        records = evaluation["records"]
        families = sorted({record["family"] for record in records.values()})
        for family in families:
            group = [record for record in records.values() if record["family"] == family]
            successes = sum(record["success"] for record in group)
            clean = sum(record["integrity"].clean for record in group)
            rows.append(
                [
                    evaluation["label"],
                    family,
                    _fraction(successes, len(group)),
                    len(group) - successes,
                    _fraction(clean, len(group)),
                ]
            )
    return rows


def _affected_counts(evaluation: dict[str, Any]) -> Counter[tuple[str, str]]:
    affected: Counter[tuple[str, str]] = Counter()
    for record in evaluation["records"].values():
        for kind in record["integrity"].counts:
            affected[(record["family"], kind)] += 1
    return affected


def _violation_rows(evaluations: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for evaluation in evaluations:
        for (family, kind), count in sorted(_affected_counts(evaluation).items()):
            rows.append([evaluation["label"], family, kind, count])
    return rows


def _paired_sections(
    evaluations: list[dict[str, Any]], task_ids: set[str]
) -> list[str]:
    if len(evaluations) < 2:
        return []
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
    return [
        *_section(
            "Integrity-clean paired flips",
            ["transition", "tasks"],
            [[labels[key], integrity_flips[key]] for key in labels],
        ),
        *_section(
            "Outcome paired flips",
            ["transition", "tasks"],
            [[outcome_labels[key], outcome_flips[key]] for key in outcome_labels],
        ),
    ]


def _acceptance_rows(evaluation: dict[str, Any]) -> list[list[Any]]:
    acceptance = {
        ("cross_reference", "verbatim_copy"): 15,
        ("batch_update", "premature_completion"): 14,
        ("ledger_reconcile", "value_drop"): 6,
    }
    observed = _affected_counts(evaluation)
    return [
        [
            family,
            kind,
            expected,
            observed[(family, kind)],
            "PASS" if observed[(family, kind)] == expected else "FAIL",
        ]
        for (family, kind), expected in acceptance.items()
    ]


def _render_evaluations(
    paths: list[Path], seed: int, *, generator_version: int | None = None, log: RunLog | None = None
) -> str:
    evaluations = []
    for number, path in enumerate(paths, 1):
        evaluation = _analyse_evaluation(path, seed, generator_version=generator_version)
        evaluations.append(evaluation)
        if log is not None:
            # R26(g): one progress event per checked artifact, carrying that artifact's R26(e)
            # identity -- the run itself spans several policies, so there is no single one to
            # put in the start line.
            log.progress(
                number,
                len(paths),
                "evaluation",
                source=str(path),
                **evaluation["identity"],
                data_seed=evaluation["seed"],
                data_seed_source=evaluation["seed_source"],
                keep_last=evaluation["keep_last"],
                generator_version=evaluation["generator_version"],
                generator_version_basis=evaluation["generator_version_basis"],
                tasks=len(evaluation["records"]),
            )
            if evaluation["seed_source"] == SEED_FROM_CALLER:
                # The one silent substitution left in this stage, so it is said out loud: the
                # tasks replayed below are the caller's-seed tasks, not necessarily the ones
                # this artifact was produced against, and no error will be raised about it.
                log.warn(
                    "replaying under the caller's seed",
                    source=str(path),
                    data_seed=evaluation["seed"],
                    reason=SEED_FROM_CALLER,
                )
    task_ids = _validated_task_ids(evaluations)

    lines = ["# Offline note-integrity comparison", "", "## Sources and configuration", ""]
    lines.extend(
        _table(
            [
                "run",
                "source",
                "data seed",
                "data seed source",
                "keep-last",
                "generator version",
                "generator version basis",
                "tasks",
            ],
            _source_rows(evaluations),
        )
    )
    lines.extend(
        _section(
            "Overall",
            [
                "run",
                "outcome success",
                "integrity clean",
                "affected trajectories",
                "failed with violation",
            ],
            _overall_rows(evaluations),
        )
    )
    lines.extend(
        _section(
            "Per-family",
            ["run", "family", "outcome success", "failures", "integrity clean"],
            _family_rows(evaluations),
        )
    )
    lines.extend(
        _section(
            "Violation kinds (affected trajectories)",
            ["run", "family", "violation", "affected trajectories"],
            _violation_rows(evaluations),
        )
    )
    lines.extend(_paired_sections(evaluations, task_ids))
    lines.extend(
        _section(
            "Memo acceptance checks",
            ["family", "violation", "expected", "observed", "status"],
            _acceptance_rows(evaluations[-1]),
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved evaluations for note integrity")
    parser.add_argument("--eval", dest="evaluations", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=20260902, type=int)
    parser.add_argument("--generator-version", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # R26(a): the log opens before any artifact is read, so a run that dies on a malformed or
    # unbound evaluation still leaves run.log and events.jsonl behind. The identity block
    # carries what is known before the artifacts are opened; the per-artifact model, policy,
    # adapter and split arrive with each progress event.
    log = RunLog.open(
        args.output.parent,
        name="integrity",
        command=list(sys.argv),
        identity={
            "evaluations": [str(path) for path in args.evaluations],
            "output": str(args.output),
            "data_seed": args.seed,
            "generator_version": args.generator_version,
            "git_commit": git_commit(),
            "git_dirty": git_tree_dirty(),
        },
    )
    completed = False
    try:
        rendered = _render_evaluations(
            args.evaluations, args.seed, generator_version=args.generator_version, log=log
        )
        args.output.write_text(rendered, encoding="utf-8")
        completed = True
    finally:
        # R26(c)'s incomplete_run rule on a non-training stage: a run that ended before it
        # scored every artifact produced no comparison anything downstream may read.
        log.close(
            status="ok" if completed else "error",
            incomplete_run=not completed,
            evaluations=len(args.evaluations),
        )
