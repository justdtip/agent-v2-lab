"""Gated, fake-testable causal patching at the dropped-value decision."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.evaluate import load_policy, wilson
from local_llm_lab.pipeline.integrity import (
    _canonical_note,
    _extract_facts,
    _fact_is_visible,
    _normalise,
    _policy_index,
    check_trajectory,
)
from local_llm_lab.pipeline.preflight import require_preflight
from local_llm_lab.pipeline.protocol import (
    SYSTEM_PROMPT,
    assistant_message,
    build_prompt,
    parse_turn,
    strip_thinking,
    tool_message,
    turn_is_complete,
    window_messages,
)
from local_llm_lab.pipeline.tasks import (
    GENERATOR_VERSION,
    Task,
    difficulty,
    render_expert_note,
    replay_task_from_id,
    task_from_id,
)
from local_llm_lab.probes.capture import InjectionHook, capture_residuals
from local_llm_lab.probes.policies import resolve_layers, resolve_policy, validate_layer_syntax
from local_llm_lab.probes.state_probe import preflight_precision_block, spec_capture_dtype
from local_llm_lab.provenance import write_provenance

PROMPT_GROUPS = (
    "system_prompt",
    "task_prompt",
    "previous_notes",
    "note_value_tokens",
    "last_two_observations",
    "final_token",
)
"""Groups a tokenizer can resolve on ONE rendered prompt (``position_groups``)."""

POSITION_GROUPS = (
    "system_prompt",
    "task_prompt",
    "previous_notes",
    "shared_value_tokens",
    "dropped_value_slot",
    "last_two_observations",
    "final_token",
)
"""The seven P6 cells (R25d).

``note_value_tokens`` is not a cell: R25(b) splits it into ``shared_value_tokens`` (values the
two notes have in common) and ``dropped_value_slot`` (the separator slot where a value the
target note dropped should have appeared).  Both are pairwise quantities, so they exist only
after ``align_groups`` has seen both prompts.
"""

CONTROLS = ("unrelated_task", "random_positions", "content_swap")
"""R27(5) adds ``content_swap``: the treatment rows with the dropped value's own rows
replaced by an unrelated case's, which separates "this position gates the decision" from
"this value's representation is what the injection carries"."""

OUTCOMES = ("flip", "corrupted", "unchanged", "empty", "parse_error")
"""R27(1): the strict per-generation outcome. A wrong number is never a flip.

R30(5) splits ``empty`` out of ``parse_error``: a note that parses but restates no value is
not a parse failure — the model wrote a turn, it simply did not carry the list — and folding
the two together hid which of the two a cell's non-flip mass was made of.
"""

ARTIFACT_SCHEMA = "p6-patch-r30"
"""Names the artifact shape; the payload carried no schema/version field before R25."""
_FAMILIES = frozenset({"aggregate_report", "ledger_reconcile"})

MINIMUM_SCORED_CASES = 5
"""R30a(3): the fewest scored cases a secondary section may be read from — the primary's own
count, and the only floor in this file that is a judgement rather than a structural need. It
is not a target: the section scores every placeable case it has and reports the count."""

PRIMARY_CONDITION = "primary"
SECONDARY_CONDITIONS = ("aggregate_report",)
"""R27(5): the bound secondary condition (issue #26) — ``aggregate_report`` failures scored
against the designed-correct generator note, kept in their own section of the artifact."""

_CONDITION_LABELS = {
    PRIMARY_CONDITION: "empirically_passing_preferred",
    "aggregate_report_secondary": "designed_correct",
}

BOUND_VS_HEAD_BASIS = (
    "bound generator version vs HEAD (R24): the counterfactual is the passing run's own "
    "note, so the judgement that selected the case and the judgement flip scoring uses must "
    "name the same decision step and dropped-value set"
)
HEAD_ALONE_BASIS = (
    "HEAD alone (R30): the counterfactual is a HEAD generator note, so there is no "
    "version-bound side to compare with and no v1 replay is performed"
)
_CONDITION_ELIGIBILITY_BASES = {
    PRIMARY_CONDITION: BOUND_VS_HEAD_BASIS,
    "aggregate_report_secondary": HEAD_ALONE_BASIS,
}

_RAW_HEAD_CHARS = 200
"""How much of an unparseable generation the artifact keeps, so a parse error is auditable."""


@dataclass(frozen=True)
class DropJudgement:
    """One integrity judgement of a failing trajectory (R24).

    ``decision_step`` is the earliest value-drop step, ``dropped_values`` the values that
    step is missing (``None`` when the judging source records no values, which can never
    count as identical to anything), and ``judged_under`` names the generator version or
    saved source the judgement came from.
    """

    decision_step: int | None
    dropped_values: tuple[str, ...] | None
    judged_under: str


def scoring_version_stable(bound: DropJudgement, head: DropJudgement) -> bool:
    """R24: HEAD flip scoring is admissible for a case only when the bound-version and
    HEAD judgements name the same decision step and the same dropped-value set."""
    return (
        bound.decision_step is not None
        and bound.decision_step == head.decision_step
        and bound.dropped_values is not None
        and head.dropped_values is not None
        and set(bound.dropped_values) == set(head.dropped_values)
    )


@dataclass(frozen=True)
class PatchCase:
    task: Task
    decision_step: int
    failing_steps: tuple[dict[str, Any], ...]
    passing_steps: tuple[dict[str, Any], ...] | None = None
    bound_judgement: DropJudgement | None = None
    head_judgement: DropJudgement | None = None
    condition: str = PRIMARY_CONDITION

    @property
    def counterfactual_label(self) -> str:
        """What kind of note the counterfactual is (R27(5)).

        The primary condition prefers the passing run's own note (R22b); the secondary
        condition has no passing run to draw on, so its note is the generator's
        designed-correct one and must never be read as an empirically passing note.
        """
        return _CONDITION_LABELS[self.condition]

    @property
    def eligibility_basis(self) -> str:
        """Under what the case's eligibility was judged (R30(2)), recorded per case.

        The primary condition keeps R24's bound-vs-HEAD comparison.  The secondary condition
        is judged under HEAD alone: its counterfactual is already a HEAD generator note, so
        a version-bound judgement has nothing to bind to, and comparing one anyway excluded
        cases for a disagreement that cannot reach the score.
        """
        return _CONDITION_ELIGIBILITY_BASES[self.condition]

    @property
    def scoring_version_stable(self) -> bool:
        """Whether HEAD flip scoring is admissible for this case (R24); fails closed when a
        case carries no judgements rather than guessing."""
        if self.bound_judgement is None or self.head_judgement is None:
            raise ValueError(
                f"{self.task.task_id}: case carries no bound/HEAD value-drop judgements"
            )
        return scoring_version_stable(self.bound_judgement, self.head_judgement)

    def scoring_record(self) -> dict[str, Any]:
        """The artifact's per-case R24 fields: the flag beside both judgements, so a reader
        sees why a case is unstable instead of trusting the flag."""
        stable = self.scoring_version_stable
        bound, head = self.bound_judgement, self.head_judgement
        assert bound is not None and head is not None
        return {
            "scoring_version_stable": stable,
            "decision_step_bound": bound.decision_step,
            "decision_step_head": head.decision_step,
            "dropped_values_bound": _listed(bound.dropped_values),
            "dropped_values_head": _listed(head.dropped_values),
            "judged_under_bound": bound.judged_under,
            "judged_under_head": head.judged_under,
            "eligibility_basis": self.eligibility_basis,
        }


def _listed(values: tuple[str, ...] | None) -> list[str] | None:
    return list(values) if values is not None else None


def _records(
    payload: dict[str, Any], name: str, *, data_seed: int | None = None
) -> tuple[list[dict[str, Any]], int, str]:
    """Return trajectories plus the resolved data seed and its source (R22a).

    The evaluation's own ``data_seed`` field wins when present; the ``data_seed`` override is
    used only when the field is absent; a present field that disagrees with the override is an
    error naming both values, and absence of both is an error.

    ``data_seed`` is looked for under ``summary`` first and at the top level second, the same
    two levels :func:`_generator_version_binding` reads. It was read at the top level alone,
    which is not where ``evaluate.write_report`` puts it: the writer frames the payload as
    ``{"summary": ..., "trajectories": ...}`` and ``evaluation_metadata`` puts the seed inside
    the summary. So the recorded seed was invisible, the override was always "used only when
    the field is absent", and an override disagreeing with the artifact was accepted in
    silence — the one outcome the R22a conflict check exists to make impossible.
    """
    if isinstance(data_seed, bool) or (data_seed is not None and not isinstance(data_seed, int)):
        raise ValueError("--data-seed override must be an integer")
    if not isinstance(payload, dict):
        raise ValueError(f"{name} evaluation must be a mapping")
    records = payload.get("trajectories")
    if not isinstance(records, list):
        raise ValueError(f"{name} evaluation must contain trajectories")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{name} trajectories must be mappings")
    summary = payload.get("summary")
    levels = [summary, payload] if isinstance(summary, dict) else [payload]
    recorded = next((level for level in levels if "data_seed" in level), None)
    if recorded is None:
        if data_seed is None:
            raise ValueError(
                f"{name} evaluation lacks data_seed and no --data-seed override was given"
            )
        return records, data_seed, "flag"
    seed = recorded["data_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{name} evaluation must contain integer data_seed")
    if data_seed is not None and data_seed != seed:
        raise ValueError(
            f"{name} evaluation data_seed {seed} conflicts with --data-seed {data_seed}"
        )
    return records, seed, "evaluation"


def _derived_difficulty(task_id: str) -> int:
    """Derive the pipeline difficulty from the task id when the evaluation omits the field.

    The tool adapts, the evidence is never edited: ``difficulty(split, index)`` is the
    pipeline's source of truth for the split/index encoded in the id.
    """
    try:
        split, _family, index_text, _variant = task_id.rsplit("-", 3)
        return difficulty(split, int(index_text))
    except ValueError as error:
        raise ValueError(f"{task_id}: cannot derive difficulty from task id") from error


def _generator_version_binding(
    payload: dict[str, Any], name: str, override: int | None
) -> tuple[int | None, str]:
    """Resolve the generator version an integrity recomputation must replay under (R12).

    The evaluation's recorded version wins (``summary.generator_version``, else the
    top-level field); an explicit ``--generator-version`` binding is used only when the
    evaluation records none, and a conflict is an error naming both values. Returns
    ``(None, "")`` when neither exists — the caller fails closed if a recomputation is
    actually needed, never defaulting to HEAD.
    """
    if isinstance(override, bool) or (override is not None and not isinstance(override, int)):
        raise ValueError("--generator-version binding must be an integer")
    if override is not None and not 1 <= override <= GENERATOR_VERSION:
        raise ValueError(f"--generator-version {override} must be in [1, {GENERATOR_VERSION}]")
    summary = payload.get("summary")
    recorded = summary.get("generator_version") if isinstance(summary, dict) else None
    if recorded is None:
        recorded = payload.get("generator_version")
    if recorded is None:
        return (override, "flag") if override is not None else (None, "")
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, int)
        or not 1 <= recorded <= GENERATOR_VERSION
    ):
        raise ValueError(f"{name} evaluation records invalid generator_version {recorded!r}")
    if override is not None and override != recorded:
        raise ValueError(
            f"{name} evaluation generator_version {recorded} conflicts with "
            f"--generator-version {override}"
        )
    return recorded, "evaluation"


_VALUE_DROP_DETAIL = "missing required values: "


def _dropped_values(detail: Any) -> tuple[str, ...] | None:
    """Parse the value list ``integrity`` writes into a value-drop violation's detail;
    ``None`` when the detail is absent or not in that form."""
    if not isinstance(detail, str) or not detail.startswith(_VALUE_DROP_DETAIL):
        return None
    values = tuple(value for value in detail[len(_VALUE_DROP_DETAIL) :].split(", ") if value)
    return values or None


def _violation_fields(violation: Any) -> tuple[Any, Any, Any]:
    if isinstance(violation, dict):
        return violation.get("kind"), violation.get("step"), violation.get("detail")
    return (
        getattr(violation, "kind", None),
        getattr(violation, "step", None),
        getattr(violation, "detail", None),
    )


def _judgement(violations: Iterable[Any], judged_under: str) -> DropJudgement:
    """The earliest value-drop step and its dropped values, from ``Violation`` objects or
    the saved dicts an evaluation's ``integrity`` block carries."""
    drops: dict[int, Any] = {}
    for violation in violations:
        kind, step, detail = _violation_fields(violation)
        if (
            kind == "value_drop"
            and isinstance(step, int)
            and not isinstance(step, bool)
            and step >= 0
        ):
            drops.setdefault(step, detail)
    if not drops:
        return DropJudgement(None, None, judged_under)
    step = min(drops)
    return DropJudgement(step, _dropped_values(drops[step]), judged_under)


def _recomputed_judgement(
    task: Task, steps: list[dict[str, Any]], *, keep_last: int, judged_under: str
) -> DropJudgement:
    """Recompute the value-drop judgement from the saved steps.

    For eligibility ``task`` must be the version-bound replay of the trajectory's
    generation (R12/R22): the judgement compares old notes against that version's
    canonical notes, never HEAD's, so a template-style drift cannot manufacture a value
    drop. The same call over the HEAD task is exactly what ``_is_flip`` later scores
    against, which is what R24 compares it with.
    """
    report = check_trajectory(task, steps, keep_last=keep_last)
    return _judgement(report.violations, judged_under)


def _saved_judgement(record: dict[str, Any]) -> DropJudgement:
    integrity = record.get("integrity")
    violations = integrity.get("violations") if isinstance(integrity, dict) else None
    judged_under = "saved evaluation integrity"
    if not isinstance(violations, list):
        return DropJudgement(None, None, judged_under)
    return _judgement(violations, judged_under)


def select_patch_cases(
    passing_payload: dict[str, Any],
    failing_payload: dict[str, Any],
    *,
    keep_last: int,
    data_seed: int | None = None,
    generator_version: int | None = None,
    secondary_condition: str | None = None,
) -> tuple[list[PatchCase], dict[str, Any]]:
    """Select the SPEC-004 §5 universe — task ids that pass under B and fail under C —
    restricted to value-drop decisions, and reconstruct C's task declaration.

    ``secondary_condition`` selects the bound secondary universe instead (R27(5), issue
    #26): the named family's failures under C, WITHOUT the passing-run intersection,
    because that family fails under B and C alike and supplies no empirically passing
    note. Those cases carry no ``passing_steps``, so ``counterfactual_note`` falls back to
    ``render_expert_note`` at HEAD and the case is labelled ``designed_correct``. The two
    universes are disjoint populations and are never merged into one headline.

    Returns the cases plus provenance: per-input data-seed sources (R22a) under
    ``data_seeds`` and per-input eligibility sources under ``eligibility``. A failing
    trajectory that carries ``difficulty``/``integrity`` uses the saved values untouched;
    when a field is absent the tool adapts — difficulty is derived from the task id, and
    integrity is recomputed from the saved steps via ``check_trajectory`` over the
    trajectory's generation version replayed through ``replay_task_from_id`` (R12; the
    binding comes from the evaluation or the explicit ``generator_version``, and its
    absence fails closed rather than defaulting to HEAD) — for selection only, with the
    source recorded so a reader can weight the judgement. Each case carries the passing
    run's saved trajectory steps when they are well formed, so the counterfactual note
    can prefer the note that empirically produced a pass (R22b), and both value-drop
    judgements — the bound one that selected it and the HEAD one flip scoring will use —
    so ``scoring_version_stable`` is a recorded fact per case (R24).
    """
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")
    if secondary_condition is not None and secondary_condition not in SECONDARY_CONDITIONS:
        raise ValueError(
            f"unknown secondary condition {secondary_condition!r}; "
            f"expected one of {', '.join(SECONDARY_CONDITIONS)}"
        )
    families = _FAMILIES if secondary_condition is None else frozenset({secondary_condition})
    condition = (
        PRIMARY_CONDITION if secondary_condition is None else f"{secondary_condition}_secondary"
    )
    passing, passing_seed, passing_source = _records(passing_payload, "passing", data_seed=data_seed)
    failing, failing_seed, failing_source = _records(failing_payload, "failing", data_seed=data_seed)
    replay_version, version_source = _generator_version_binding(
        failing_payload, "failing", generator_version
    )
    seeds = {
        "passing": {"data_seed": passing_seed, "data_seed_source": passing_source},
        "failing": {"data_seed": failing_seed, "data_seed_source": failing_source},
    }
    successful: dict[str, dict[str, Any]] = {
        record["task_id"]: record
        for record in passing
        if isinstance(record.get("task_id"), str)
        and isinstance(record.get("verdict"), dict)
        and record["verdict"].get("success") is True
    }
    selected: list[PatchCase] = []
    integrity_counts = {"evaluation": 0, "recomputed": 0, "head_alone": 0}
    difficulty_counts = {"evaluation": 0, "recomputed": 0}
    for record in failing:
        task_id = record.get("task_id")
        steps = record.get("steps")
        verdict = record.get("verdict")
        if (
            not isinstance(task_id, str)
            # The secondary universe deliberately drops the passing-run intersection
            # (R27(5)): its family has no pass/fail pair to intersect with.
            or (secondary_condition is None and task_id not in successful)
            or not isinstance(steps, list)
            # SPEC-004 §5 universe: the task must FAIL under C, not merely drop a value.
            or not isinstance(verdict, dict)
            or verdict.get("success") is not False
        ):
            continue
        if "difficulty" in record:
            level = record["difficulty"]
            if isinstance(level, bool) or not isinstance(level, int):
                continue
            difficulty_source = "evaluation"
        else:
            level = _derived_difficulty(task_id)
            difficulty_source = "recomputed"
        task = task_from_id(task_id, failing_seed, level)
        if task.family not in families:
            continue
        if not all(isinstance(step, dict) for step in steps):
            raise ValueError(f"{task_id}: malformed failing steps")
        difficulty_counts[difficulty_source] += 1
        head_judged_under = f"generator_v{GENERATOR_VERSION} HEAD"
        if secondary_condition is not None:
            # R30(2): the secondary condition is judged under HEAD ALONE. Its counterfactual
            # is already a HEAD generator note, so no version-bound judgement applies and no
            # v1 replay is performed; the basis is recorded per case and in the provenance.
            # The two sides are the SAME judgement, not two that happen to agree.
            bound = head = _recomputed_judgement(
                task, steps, keep_last=keep_last, judged_under=head_judged_under
            )
            integrity_counts["head_alone"] += 1
        elif "integrity" in record:
            bound = _saved_judgement(record)
            integrity_counts["evaluation"] += 1
        else:
            if replay_version is None:
                raise ValueError(
                    "failing evaluation records no generator_version and no "
                    "--generator-version binding was given; integrity recomputation "
                    "must replay the trajectory's generation version (R12), never HEAD"
                )
            replayed = replay_task_from_id(task_id, failing_seed, replay_version, level)
            bound = _recomputed_judgement(
                replayed,
                steps,
                keep_last=keep_last,
                judged_under=f"generator_v{replay_version} replay",
            )
            integrity_counts["recomputed"] += 1
        dropped = bound.decision_step
        if dropped is None:
            continue
        if dropped >= len(steps):
            raise ValueError(f"{task_id}: malformed failing steps")
        if secondary_condition is None:
            # R24: flip scoring judges the HEAD task these steps were built for
            # (``_is_flip``); judge it here too, so the artifact records whether that seam is
            # admissible.  The secondary condition already holds this judgement (R30(2)).
            head = _recomputed_judgement(
                task, steps, keep_last=keep_last, judged_under=head_judged_under
            )
        saved = successful[task_id].get("steps") if task_id in successful else None
        passing_steps = (
            tuple(dict(step) for step in saved)
            if secondary_condition is None
            and isinstance(saved, list)
            and all(isinstance(step, dict) for step in saved)
            else None
        )
        selected.append(
            PatchCase(
                task,
                dropped,
                tuple(dict(step) for step in steps),
                passing_steps,
                bound,
                head,
                condition,
            )
        )
    recomputed_total = integrity_counts["recomputed"] + difficulty_counts["recomputed"]
    saved_total = integrity_counts["evaluation"] + difficulty_counts["evaluation"]
    failing_eligibility: dict[str, Any] = {
        "eligibility_source": (
            "evaluation"
            if recomputed_total == 0
            else "recomputed"
            if saved_total == 0
            else "mixed"
        ),
        "integrity": integrity_counts,
        "difficulty": difficulty_counts,
        # R30(2): what the eligibility judgement was made under, recorded beside the counts
        # so a reader never has to infer it from the condition name.
        "eligibility_basis": _CONDITION_ELIGIBILITY_BASES[condition],
        # What the FAILING EVALUATION declares — a fact about the input file, never the
        # version this section's eligibility was judged under. The two are different, and on
        # the secondary condition they are deliberately unrelated: R30(2) judges it under HEAD
        # alone, so the name has to say "evaluation" or the number would read as a bound
        # version the section explicitly does not have. Written unconditionally, because the
        # conditional block below fires only when a case was recomputed, and a run against an
        # artifact this repository's writer produced recomputes nothing — every trajectory
        # carries saved integrity — so the input's version was recorded exactly never.
        "evaluation_generator_version": replay_version if version_source == "evaluation" else None,
        "evaluation_generator_version_source": version_source or "unbound",
    }
    if integrity_counts["recomputed"]:
        failing_eligibility["recomputed_generator_version"] = replay_version
        failing_eligibility["generator_version_source"] = version_source
        failing_eligibility["generator_version_basis"] = (
            "recorded in the failing evaluation"
            if version_source == "evaluation"
            else "explicit --generator-version binding; the evaluation records no generator_version"
        )
    provenance = {
        "data_seeds": seeds,
        "eligibility": {
            "passing": {"eligibility_source": "evaluation"},
            "failing": failing_eligibility,
        },
        "condition": condition,
        "counterfactual_label": _CONDITION_LABELS[condition],
    }
    return selected, provenance


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(value) for value in tokenizer.encode(text)]


def _char_span_once(text: str, needle: str, *, start: int, label: str) -> tuple[int, int]:
    """Return the unique ``[start, end)`` character span of ``needle`` at or after ``start``."""
    if not needle:
        raise ValueError(f"missing token span for {label}")
    first = text.find(needle, start)
    if first < 0:
        raise ValueError(f"missing token span for {label}")
    if text.find(needle, first + 1) >= 0:
        raise ValueError(f"ambiguous token span for {label}")
    return first, first + len(needle)


def _value_pattern(value: str) -> str:
    r"""A numeric value matched on its own boundaries, never inside a longer number.

    ``32`` occurring inside ``132`` is a different value, not a second occurrence of this one
    (issue #29): matching by plain substring aborted such a note as ambiguous.  The lookarounds
    exclude a neighbouring digit, decimal point, or sign, so a genuine duplicate still matches
    twice and is still refused.

    The trailing lookahead is ``(?![0-9])(?!\.[0-9])`` rather than ``(?![0-9.])``.  The older form
    rejected a full stop of any kind, so a value ending a clause -- ``Approved total = 359.`` --
    matched nowhere and the note was refused for having *too few* occurrences rather than too
    many.  That was 420 of 5,226 v1 notes and 167 of 5,226 v4 notes across the test split, and
    fifteen of the seventy value-level failures behind the unscored ``aggregate_report``
    secondary.  Splitting the lookahead keeps ``72`` out of ``72.5``, which is what the decimal
    exclusion was for, while admitting a sentence-final value, which it never intended to reject.
    """
    return rf"(?<![0-9.\-]){re.escape(value)}(?![0-9])(?!\.[0-9])"


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _token_span(
    tokenizer: Any,
    prompt_text: str,
    ids: Sequence[int],
    span: tuple[int, int],
    *,
    label: str,
) -> tuple[tuple[int, ...], int]:
    """Map a character span of the rendered prompt to the token positions covering it.

    The prompt prefix up to each character boundary is tokenised and its length taken as the
    boundary's token index.  Tokenisation is context dependent: the prefix's last token can
    differ from the full prompt's token at that index when the boundary falls inside a merge
    (trap 6; ``capture.response_mean_activations`` repairs the same class).  The boundary is
    then widened to the merged token by comparing against the full prompt's ids, and the
    repair is counted so the artifact records how many spans needed it.
    """
    char_start, char_end = span
    repairs = 0
    head = _encode(tokenizer, prompt_text[:char_start])
    start = len(head)
    if list(ids[:start]) != head:
        start = _common_prefix_length(ids, head)
        repairs += 1
    body = _encode(tokenizer, prompt_text[:char_end])
    end = len(body)
    if list(ids[:end]) != body:
        end = min(_common_prefix_length(ids, body) + 1, len(ids))
        repairs += 1
    # A merge moves a boundary by at most one token on a prefix-stable tokenizer.  A larger
    # move means the tokenizer is not prefix-stable or the span is mislocated; refuse loudly
    # rather than widen silently backwards (Chief's condition on issue #29).
    if abs(start - len(head)) > 1 or abs(end - len(body)) > 1:
        raise ValueError(
            f"token span for {label} moved more than one token at a boundary "
            f"(start {len(head)}->{start}, end {len(body)}->{end})"
        )
    if end <= start or start < 0 or end > len(ids):
        raise ValueError(f"missing token span for {label}")
    return tuple(range(start, end)), repairs


def position_groups(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    prompt_text: str,
    system_text: str,
    task_text: str,
    previous_notes: Sequence[str],
    note_values: Sequence[str],
    observations: Sequence[str],
    stats: dict[str, int] | None = None,
    value_spans: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Locate the six per-prompt token groups, refusing missing or ambiguous content spans.

    Every text is located by its unique character span in the rendered ``prompt_text`` and
    mapped to tokens through ``_token_span``; ``observations`` must be the observations that
    are present verbatim in that prompt (the windowed view), and the last two of them form
    ``last_two_observations``.  ``stats`` receives the per-group ``boundary_repairs`` counts,
    and ``value_spans`` receives each note value's own token positions in note order, which
    is what R25's pairwise value alignment needs.
    """
    ids = [int(value) for value in prompt_ids]
    if not ids:
        raise ValueError("prompt token ids must not be empty")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt text must not be empty")
    repairs = {name: 0 for name in PROMPT_GROUPS}

    def locate(text: str, *, start: int, label: str, group: str) -> tuple[tuple[int, ...], int]:
        span = _char_span_once(prompt_text, text, start=start, label=label)
        tokens, repaired = _token_span(tokenizer, prompt_text, ids, span, label=label)
        repairs[group] += repaired
        return tokens, span[1]

    system, cursor = locate(system_text, start=0, label="system_prompt", group="system_prompt")
    task, cursor = locate(task_text, start=cursor, label="task_prompt", group="task_prompt")
    after_task = cursor
    previous: list[int] = []
    note_span: tuple[int, int] | None = None
    for note in previous_notes:
        note_span = _char_span_once(prompt_text, note, start=cursor, label="previous_note")
        tokens, repaired = _token_span(tokenizer, prompt_text, ids, note_span, label="previous_note")
        repairs["previous_notes"] += repaired
        previous.extend(tokens)
        cursor = note_span[1]
    values: list[int] = []
    spans: dict[str, tuple[int, ...]] = {}
    region = "" if note_span is None else prompt_text[note_span[0] : note_span[1]]
    # R30a(1): the note's own list fields, each value carrying the span it was read from, in
    # the order ``_note_values`` yields them.  ``aggregate_report`` repeats a value on
    # purpose -- the running list carries it twice and the subtotal arithmetic names it
    # again -- so a search of the whole note for a unique match refuses the data instead of
    # locating it.  Consuming the field slots in step with ``note_values`` gives the k-th
    # number under a label one position by construction, and never lets the arithmetic
    # outside a list field stand in as a candidate.
    slots = _field_value_slots(region)
    slot_index = 0
    for value in note_values:
        if note_span is None:
            raise ValueError("note_value token span is missing or ambiguous in the substituted note")
        if slot_index < len(slots) and slots[slot_index][0] == value:
            start, _end = slots[slot_index][1]
            slot_index += 1
            offset = note_span[0] + start
        else:
            # The residue: a value no list field carries -- a subtotal, a running maximum --
            # keeps the pre-R30a rule, one match over the note or none. It is the only path
            # that can still refuse a note, and R30a(2) makes that refusal a skipped case.
            found = list(re.finditer(_value_pattern(value), region))
            if len(found) != 1:
                raise ValueError(
                    "note_value token span is missing or ambiguous in the substituted note"
                )
            offset = note_span[0] + found[0].start()
        tokens, repaired = _token_span(
            tokenizer, prompt_text, ids, (offset, offset + len(value)), label="note_value_tokens"
        )
        repairs["note_value_tokens"] += repaired
        # R25 pairs values across the two prompts by string identity, so a repeated value
        # needs exactly one canonical span: the first, which is the one the two notes share
        # when only one of them repeats it. Every occurrence still contributes its own
        # tokens to the group.
        spans.setdefault(value, tokens)
        values.extend(tokens)
    observation_spans: list[tuple[int, ...]] = []
    cursor = after_task
    for observation in observations:
        tokens, cursor = locate(observation, start=cursor, label="observation", group="last_two_observations")
        observation_spans.append(tokens)
    last_observations = tuple(position for span in observation_spans[-2:] for position in span)
    if value_spans is not None:
        value_spans.clear()
        value_spans.update(spans)
    if stats is not None:
        stats.clear()
        stats.update(repairs)
    return {
        "system_prompt": system,
        "task_prompt": task,
        "previous_notes": tuple(previous),
        "note_value_tokens": tuple(values),
        "last_two_observations": last_observations,
        "final_token": (len(ids) - 1,),
    }


def random_control_positions(
    valid_positions: Sequence[int],
    treatment_positions: Sequence[int],
    *,
    seed: int,
    label: str,
) -> tuple[int, ...]:
    """Return deterministic non-treatment positions with matching cardinality."""
    treatment = tuple(dict.fromkeys(int(position) for position in treatment_positions))
    alternatives = sorted(set(map(int, valid_positions)) - set(treatment))
    if len(alternatives) < len(treatment):
        raise ValueError("random control cannot satisfy treatment cardinality")
    return tuple(sorted(random.Random(f"{seed}:{label}").sample(alternatives, len(treatment))))


def greedy_generate(
    view: ArchitectureView,
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    max_tokens: int,
) -> str:
    """Greedily run the public view seams, preserving prompt/cache absolute positions."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    import mlx.core as mx

    cache = view.make_cache()
    generated: list[int] = []
    forward_ids = list(token_ids)
    for _ in range(max_tokens):
        h = view.embed(mx.array(forward_ids, dtype=mx.int32)[None, :])
        masks = view.masks(h, cache)
        for layer in range(view.num_layers):
            h = view.run_block(layer, h, masks, cache[layer])
        logits = view.unembed(view.final_norm(h))
        token = int(mx.argmax(logits[0, -1]).item())
        generated.append(token)
        text = tokenizer.decode(generated)
        if turn_is_complete(text):
            return text
        forward_ids = [token]
    return tokenizer.decode(generated)


def aggregate_task_flips(task_flips: dict[str, Sequence[bool]]) -> dict[str, Any]:
    """Aggregate a cell by task id, never by repeated generations from one task."""
    values = {task_id: any(outcomes) for task_id, outcomes in task_flips.items()}
    numerator = sum(values.values())
    denominator = len(values)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
        "wilson_95": list(wilson(numerator, denominator)),
    }


def aggregate_task_outcomes(task_outcomes: dict[str, Sequence[str]]) -> dict[str, Any]:
    """R27(2): the flip rate over cases, beside the per-outcome generation counts.

    ``numerator``/``denominator``/``rate``/``wilson_95`` are unchanged — one case counts
    once and flips when ANY of its generations is a ``flip`` — so the headline is directly
    comparable with the pre-R27 artifact. ``outcomes`` counts GENERATIONS, not cases, so a
    reader can see how much of a cell's non-flip mass is corruption rather than an
    unchanged note; that distinction is the whole point of the ruling.
    """
    counts = dict.fromkeys(OUTCOMES, 0)
    for outcomes in task_outcomes.values():
        for outcome in outcomes:
            if outcome not in counts:
                raise ValueError(f"unknown R27 outcome {outcome!r}")
            counts[outcome] += 1
    summary = aggregate_task_flips(
        {
            task_id: [outcome == "flip" for outcome in outcomes]
            for task_id, outcomes in task_outcomes.items()
        }
    )
    return {**summary, "outcomes": counts, "generations": sum(counts.values())}


def _action(record: dict[str, Any]) -> Any:
    raw = record.get("action")
    if isinstance(raw, dict) and isinstance(raw.get("name"), str):
        from local_llm_lab.agent_protocol import Action

        arguments = raw.get("arguments", {})
        return Action(raw["name"], arguments if isinstance(arguments, dict) else {})
    raise ValueError("failing step is missing an action")


def counterfactual_note(case: PatchCase) -> tuple[str | None, dict[str, Any]]:
    """Choose the note substituted before the decision step (R22b).

    The primary source is the passing run's saved note at ``decision_step - 1``, used when the
    passing trajectory's actions (name and arguments) equal the failing run's at every step
    before the decision — the criterion under which the saved note is the one that empirically
    led to a pass over the same prefix. Otherwise ``render_expert_note`` at HEAD is the
    fallback, and the returned provenance records the reason so a reader can weight the case.
    """
    if case.decision_step < 1:
        return None, {
            "counterfactual_source": None,
            "counterfactual_basis": "decision step 0 has no prior note to substitute",
        }
    position = case.decision_step - 1

    def fallback(reason: str) -> tuple[str, dict[str, Any]]:
        return render_expert_note(case.task, position), {
            "counterfactual_source": f"generator_v{GENERATOR_VERSION}",
            "counterfactual_basis": reason,
        }

    if case.passing_steps is None:
        return fallback("passing trajectory steps unavailable")
    if len(case.passing_steps) < case.decision_step:
        return fallback(
            f"passing trajectory has {len(case.passing_steps)} steps, shorter than the "
            f"{case.decision_step}-step decision prefix"
        )
    for index in range(case.decision_step):
        try:
            passing_action = _action(case.passing_steps[index])
        except ValueError:
            return fallback(f"passing step {index} is missing an action")
        if passing_action != _action(case.failing_steps[index]):
            return fallback(f"passing and failing actions diverge at step {index}")
    note = case.passing_steps[position].get("thought")
    if not isinstance(note, str) or not note:
        return fallback(f"passing step {position} has no saved note text")
    return note, {
        "counterfactual_source": "passing_transcript",
        "counterfactual_basis": (
            f"passing actions equal failing actions at steps 0..{position} (name and arguments)"
        ),
    }


def replay_counterfactual(
    case: PatchCase,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replay C before the decision, replacing only its immediately prior note (R22b).

    Returns the failing and counterfactual message lists plus the substituted note's
    provenance (``counterfactual_source`` and ``counterfactual_basis``).
    """
    note, provenance = counterfactual_note(case)
    failing = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": case.task.prompt}]
    counterfactual = [dict(message) for message in failing]
    for position, record in enumerate(case.failing_steps[: case.decision_step]):
        action = _action(record)
        thought = record.get("thought")
        if not isinstance(thought, str):
            raise ValueError("failing step is missing thought")
        replacement = (
            note if position == case.decision_step - 1 and note is not None else thought
        )
        failing.append(assistant_message(thought, action))
        counterfactual.append(assistant_message(replacement, action))
        observation = record.get("observation")
        if isinstance(observation, str):
            message = tool_message(action.name, observation)
            failing.append(message)
            counterfactual.append(dict(message))
    return failing, counterfactual, provenance


def _is_flip(task: Task, steps: list[dict[str, Any]], decision_step: int, *, keep_last: int) -> bool:
    """The PRE-R27 rule: no ``value_drop`` remains at the decision step.

    Retained as a recorded diagnostic only (``ScoredGeneration.value_drop_cleared``), never
    as an outcome.  It is not a flip test: a regenerated note carrying a WRONG number makes
    ``integrity._contradictory_fields`` mark the field contradictory, and
    ``_fact_covered_by_stale_field`` then treats every canonical value — the dropped one
    included — as covered, so the drop disappears and a corruption scores as a flip
    (``integrity.py:374-395``).  R27 replaces it with ``classify_generation``; keeping it
    beside the strict outcome lets the rerun be compared with the 2026-09-04 artifact
    without spending a second run.
    """
    report = check_trajectory(task, steps, keep_last=keep_last)
    return not any(
        violation.kind == "value_drop" and violation.step == decision_step
        for violation in report.violations
    )


def _message_contents(
    messages: Sequence[dict[str, Any]], *, keep_last: int
) -> tuple[str, str, list[str], list[str]]:
    """Extract canonical group inputs from the same windowed view ``build_prompt`` renders.

    ``window_messages`` replaces all but the last ``keep_last`` tool observations with hidden
    stubs, so only the observations still present verbatim are returned; stubs are not
    observations.
    """
    windowed = window_messages([dict(message) for message in messages], keep_last)
    system = next(
        (message["content"] for message in messages if message.get("role") == "system"),
        "",
    )
    task = next(
        (message["content"] for message in messages if message.get("role") == "user"),
        "",
    )
    notes = [
        message["content"]
        for message in messages
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
    ]
    observations = [
        message["content"]
        for message, original in zip(windowed, messages)
        if message.get("role") == "tool"
        and isinstance(message.get("content"), str)
        and message["content"] == original.get("content")
    ]
    if not isinstance(system, str) or not isinstance(task, str):
        raise ValueError("replayed messages must contain system and task text")
    return system, task, notes, observations


_NUMBER = r"-?\d+(?:\.\d+)?"

_VALUE_FIELD = re.compile(
    r"\b(approved|values so far|(?:first|second)\s+half(?:\s+\w+)*)\s*:\s*([^;.\n]+)",
    re.IGNORECASE,
)
r"""The note's LIST fields: a label, a colon, and the values carried under it.

R30(1) allows words between ``half`` and the colon — run C's ``aggregate_report`` notes read
``first half complete: 67 + 16; second half complete: 71 + 26.``, which the pre-R30 form
(colon immediately after ``half``) missed entirely, leaving F empty for those cases.  The
allowance is ``\w`` only, so it cannot cross a ``;``, a ``.`` or a colon and therefore stays
inside one clause.  A subtotal or ``highest so far`` expression is a value but is NOT a list
field, which is the distinction R30(3)'s field scoping rests on.
"""


def _note_value_fields(note: str) -> dict[str, tuple[str, ...]]:
    """Map each list-field label in the note to the values carried under it (R30(1)/(3)).

    Labels are normalised (whitespace collapsed, case-folded) so ``First half complete`` and
    ``first  half complete`` name the same field on both sides of a comparison.
    """
    thought = note.split("\n```json", maxsplit=1)[0]
    fields: dict[str, tuple[str, ...]] = {}
    for match in _VALUE_FIELD.finditer(thought):
        label = " ".join(match.group(1).split()).casefold()
        fields[label] = fields.get(label, ()) + tuple(re.findall(_NUMBER, match.group(2)))
    return fields


def _field_value_slots(note: str) -> list[tuple[str, tuple[int, int]]]:
    """Every list-field value with the character span it was read from, in note order.

    R30a(1): this is the locator's whole basis. The span is measured on the note itself, so
    the k-th number under a label needs no search to be placed, and a value the arithmetic
    repeats outside the field is not a candidate for it. ``_note_values`` opens with exactly
    this sequence, which is what lets ``position_groups`` walk the two in step; defining the
    values in terms of the slots rather than beside them is what keeps the two from drifting
    apart, and a scorer reading the note differently from the writer is how this path came
    to abandon a section in the first place.
    """
    thought = note.split("\n```json", maxsplit=1)[0]
    slots: list[tuple[str, tuple[int, int]]] = []
    for match in _VALUE_FIELD.finditer(thought):
        base = match.start(2)
        for number in re.finditer(_NUMBER, match.group(2)):
            slots.append((number.group(0), (base + number.start(), base + number.end())))
    return slots


def _note_values(note: str) -> list[str]:
    """Return numeric fact values, excluding labels, ordinals, and action expressions."""
    thought = note.split("\n```json", maxsplit=1)[0]
    number = _NUMBER
    values: list[str] = [value for value, _span in _field_value_slots(note)]
    values.extend(
        match.group(1)
        for match in re.finditer(
            rf"\b(?:first|second|grand|approved)\s+(?:subtotal|total)\s*=\s*({number})",
            thought,
            flags=re.IGNORECASE,
        )
    )
    values.extend(
        match.group(1)
        for match in re.finditer(
            rf"\bhighest so far\s*:\s*(?:[^;=]*=\s*)?({number})",
            thought,
            flags=re.IGNORECASE,
        )
    )
    return values


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class CaseScoring:
    """The value sets R27(1) scores one case's generations against.

    ``failing`` (F) is the failing note's own value set, ``dropped`` (D) the values the
    checker says that note is missing at the decision step, and ``canonical`` the value set
    of the ground-truth note at the same step.  All three are read with ``_note_values``,
    the same field-scoped, digit-bounded extraction R25's value alignment uses, so the
    scorer and the injection agree on what a "value" is.

    The expected set E is ``canonical | dropped``: D is a ``required_carry`` fact and is
    expected at that step by construction, so a case whose canonical note happens to carry
    the value outside a value field still admits a flip instead of failing closed.

    R30(3): ``canonical`` is FIELD-SCOPED — the canonical note's values under the same list
    field(s) the failing note's values live in, not flattened across every field.  Flattening
    admitted a computed subtotal from elsewhere in the canonical note as a legitimate value,
    so a regenerated list that replaced a carried value with that subtotal escaped the
    corruption test.  ``fields`` records the scope; empty means the failing note carried no
    list field and the scope fell back to the canonical note's own list fields.
    """

    failing: tuple[str, ...]
    dropped: tuple[str, ...]
    canonical: tuple[str, ...]
    fields: tuple[str, ...] = ()

    @property
    def expected(self) -> frozenset[str]:
        """E: every number the regenerated note may carry."""
        return frozenset(self.canonical) | frozenset(self.dropped)

    @property
    def required(self) -> frozenset[str]:
        """F ∪ D: every number the regenerated note must carry to count as a flip."""
        return frozenset(self.failing) | frozenset(self.dropped)

    @property
    def flip_satisfiable(self) -> bool:
        """R30(4): whether ANY generation could score ``flip`` for this case.

        ``required ⊄ expected`` means the flip condition is unsatisfiable by construction —
        typically a failing note whose only number is a computed subtotal the canonical value
        list does not contain.  Such a case can only ever score ``corrupted``, so scoring it
        would report a false 0.0; it is excluded from the headline with the reason recorded.
        """
        return self.required <= self.expected

    def record(self) -> dict[str, Any]:
        return {
            "failing_values": list(self.failing),
            "dropped_values_scored": list(self.dropped),
            "canonical_values": list(self.canonical),
            "expected_values": sorted(self.expected),
            "expected_field_scope": list(self.fields),
            "flip_satisfiable": self.flip_satisfiable,
            # A selected case's failing note is a subset of the canonical field by
            # construction — a foreign number there is scored ``stale_fact``, not
            # ``value_drop``, so the case would not have been selected.  A non-empty list
            # therefore marks an anomaly a reader must weigh, never a silent adjustment.
            "failing_values_outside_canonical": sorted(
                frozenset(self.failing) - self.expected
            ),
        }


def case_scoring(case: PatchCase) -> CaseScoring:
    """Measure F, D and the canonical value set once per case (R27(1)).

    The canonical note is located exactly as ``check_trajectory`` locates it — the saved
    ``index`` when it is in range, else the position, then the task's own step note,
    normalised — so the scorer judges against the same ground truth the checker does.
    """
    judgement = case.head_judgement
    if judgement is None or judgement.dropped_values is None:
        raise ValueError(
            f"{case.task.task_id}: strict R27 scoring needs the HEAD value-drop judgement"
        )
    record = case.failing_steps[case.decision_step]
    note = record.get("thought")
    step = _policy_index(record, case.decision_step, len(case.task.steps))
    canonical = _canonical_note([_normalise(item.thought) for item in case.task.steps], step)
    # R30(3): scope the expected set to the list field(s) the failing note's values live in.
    # A failing note carrying no list field at all has no scope to take, so it falls back to
    # every list field of the canonical note — still field-scoped, so a computed subtotal
    # elsewhere in that note is still not expected.
    canonical_fields = _note_value_fields(canonical)
    scope = tuple(label for label in _note_value_fields(note or "") if label in canonical_fields)
    labels = scope or tuple(canonical_fields)
    return CaseScoring(
        failing=_unique(_note_values(note) if isinstance(note, str) else ()),
        dropped=_unique(judgement.dropped_values),
        canonical=_unique(value for label in labels for value in canonical_fields[label]),
        fields=scope,
    )


def classify_generation(values: Sequence[str], scoring: CaseScoring) -> str:
    """R27(1)'s strict outcome for one parsed value set S.

    ``corrupted`` when S carries any number outside E — a wrong number is never a flip,
    which is the defect this ruling closes: a corrupted ``approved:`` list is marked
    contradictory by ``integrity._contradictory_fields`` and its canonical values, the
    dropped one included, are then treated as covered, so the pre-R27 rule scored it as a
    flip.  ``flip`` when S carries every value the failing note already had plus every
    dropped value.  ``unchanged`` when S parses and stays inside E but does not restore D.
    ``empty`` (R30(5)) when the note parses but restates no value at all: a real outcome, not
    a parse failure, and one a reader must be able to tell apart from an unchanged list.
    """
    parsed = frozenset(values)
    if not parsed:
        return "empty"
    if not parsed <= scoring.expected:
        return "corrupted"
    if scoring.required <= parsed:
        return "flip"
    return "unchanged"


@dataclass(frozen=True)
class ScoredGeneration:
    """One generation's R27 outcome beside the evidence it was scored from.

    ``value_drop_cleared`` is the PRE-R27 rule (``_is_flip``: no ``value_drop`` remains at
    the decision step), recorded as a diagnostic only so the rerun can be compared with the
    2026-09-04 artifact without a second run.  It never decides ``outcome``.
    """

    outcome: str
    note: str
    values: tuple[str, ...]
    raw_head: str = ""
    value_drop_cleared: bool | None = None

    @property
    def is_flip(self) -> bool:
        return self.outcome == "flip"

    def record(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "note": self.note,
            "values": list(self.values),
            "raw_head": self.raw_head,
            "value_drop_cleared": self.value_drop_cleared,
        }


def score_generation(raw: str, scoring: CaseScoring) -> ScoredGeneration:
    """Parse one generated turn and classify it (R27(1)).

    A turn that does not parse is ``parse_error``, and the first 200 characters of the raw
    text are kept so the failure is auditable offline instead of vanishing into a rate.  A
    turn that DOES parse but restates no value is ``empty`` (R30(5)), not a parse failure:
    nothing is unaccounted for, so no raw head is kept and the legacy ``value_drop_cleared``
    diagnostic is still measured for it.
    """
    head = raw[:_RAW_HEAD_CHARS]
    try:
        _thinking, cleaned = strip_thinking(raw)
        note = parse_turn(cleaned).thought
    except Exception:
        return ScoredGeneration("parse_error", "", (), head)
    if not isinstance(note, str):
        return ScoredGeneration("parse_error", "", (), head)
    values = _unique(_note_values(note))
    outcome = classify_generation(values, scoring)
    return ScoredGeneration(
        outcome, note, values, head if outcome == "parse_error" else ""
    )


def dropped_value_visibility(case: PatchCase, *, keep_last: int) -> dict[str, Any]:
    """R27(3): is the dropped value visible in the retained observations?

    Measured with the integrity module's own extraction, never a substring grep (the
    correction on issue #26): the failing trajectory's observations before the decision
    step are collected exactly as ``check_trajectory`` collects them — skipping
    ``FINISHED`` — the last ``keep_last`` are the visible window, and a dropped value
    counts as visible only when ``_fact_is_visible`` finds a fact of the same kind and
    value there.  A value that appears in no earlier observation cannot appear in the
    retained ones either, so it is correctly not visible.

    True excludes the case from the headline: the checker never scores a visible fact as a
    drop, so such a case would be measuring something other than restoration.
    """
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")
    judgement = case.head_judgement
    dropped = frozenset(judgement.dropped_values or ()) if judgement is not None else frozenset()
    observations = [
        record["observation"]
        for record in case.failing_steps[: case.decision_step]
        if isinstance(record.get("observation"), str) and record["observation"] != "FINISHED"
    ]
    retained = observations[-keep_last:] if keep_last else []
    facts = {
        fact
        for observed_at, observation in enumerate(observations)
        for fact in _extract_facts(observation, observed_at)
        if fact.value in dropped
    }
    visible = sorted({fact.value for fact in facts if _fact_is_visible(fact, retained)})
    return {
        "dropped_value_visible_in_retained_observations": bool(visible),
        "visible_dropped_values": visible,
        "retained_observations": len(retained),
    }


def _groups_for(
    tokenizer: Any,
    token_ids: Sequence[int],
    messages: Sequence[dict[str, Any]],
    *,
    prompt_text: str,
    keep_last: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, int], dict[str, tuple[int, ...]]]:
    """Resolve the six per-prompt groups; return them with boundary repairs and value spans.

    The third element maps each of the substituted note's values to its own token positions,
    in note order; R25's ``shared_value_tokens``/``dropped_value_slot`` alignment pairs those
    across the two prompts by string identity.
    """
    system, task, notes, observations = _message_contents(messages, keep_last=keep_last)
    values = _note_values(notes[-1]) if notes else []
    repairs: dict[str, int] = {}
    spans: dict[str, tuple[int, ...]] = {}
    groups = position_groups(
        tokenizer,
        token_ids,
        prompt_text=prompt_text,
        system_text=system,
        task_text=task,
        previous_notes=notes,
        note_values=values,
        observations=observations,
        stats=repairs,
        value_spans=spans,
    )
    if tuple(groups) != PROMPT_GROUPS or any(not groups[name] for name in PROMPT_GROUPS):
        raise ValueError("each P6 position group must resolve to at least one token")
    return groups, {name: int(repairs.get(name, 0)) for name in PROMPT_GROUPS}, spans


@dataclass(frozen=True)
class GroupAlignment:
    """How one P6 cell's source rows are matched to its target positions (R25).

    ``source_positions`` are taken one-for-one; ``pooled_sources`` instead names, per target
    position, the source rows mean-pooled into the single row written there (the
    ``dropped_value_slot`` cell).  Exactly one of the two is populated.
    """

    group: str
    rule: str
    source_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    source_cardinality: int
    target_cardinality: int
    unpatched_source_tokens: tuple[int, ...] = ()
    pooled_sources: tuple[tuple[int, ...], ...] = ()
    slots: tuple[dict[str, Any], ...] = ()
    shared_values: tuple[str, ...] = ()
    dropped_values: tuple[str, ...] = ()
    residue: dict[str, Any] | None = None

    @property
    def treatment_source_positions(self) -> tuple[int, ...]:
        """Every source position the treatment reads, pooled or not — the control's exclusion
        set, so a random control never redraws a treated position."""
        if self.pooled_sources:
            return tuple(
                dict.fromkeys(position for rows in self.pooled_sources for position in rows)
            )
        return self.source_positions

    def record(self) -> dict[str, Any]:
        """The JSON-safe artifact entry R25(d) requires for every cell."""
        entry: dict[str, Any] = {
            "group": self.group,
            "rule": self.rule,
            "source_cardinality": self.source_cardinality,
            "target_cardinality": self.target_cardinality,
            "patched_positions": len(self.target_positions),
            "target_positions": list(self.target_positions),
            "source_positions": list(self.source_positions),
            "unpatched_source_tokens": self.residue
            or {"count": 0, "positions": [], "token_ids": [], "text": ""},
        }
        if self.slots:
            entry["slots"] = [dict(slot) for slot in self.slots]
        if self.shared_values or self.dropped_values:
            entry["shared_values"] = list(self.shared_values)
            entry["dropped_values"] = list(self.dropped_values)
        return entry


def _decoded(tokenizer: Any, ids: Sequence[int]) -> str:
    """Token text for the artifact; empty when the tokenizer cannot decode (fakes)."""
    if not ids:
        return ""
    try:
        return str(tokenizer.decode(list(ids)))
    except Exception:  # pragma: no cover - defensive: the artifact records ids regardless
        return ""


def _residue_record(
    tokenizer: Any, ids: Sequence[int], positions: Sequence[int]
) -> dict[str, Any]:
    token_ids = [int(ids[position]) for position in positions]
    return {
        "count": len(token_ids),
        "positions": [int(position) for position in positions],
        "token_ids": token_ids,
        "text": _decoded(tokenizer, token_ids),
    }


def _tail_alignment(
    tokenizer: Any,
    group: str,
    source: Sequence[int],
    target: Sequence[int],
    source_ids: Sequence[int],
) -> GroupAlignment:
    """R25(a): equal cardinalities pass through; a longer source patches from its tail."""
    source = tuple(int(position) for position in source)
    target = tuple(int(position) for position in target)
    if len(source) < len(target):
        raise ValueError(
            f"{group}: treatment source group has {len(source)} positions, fewer than the "
            f"target's {len(target)}; tail alignment cannot match cardinality"
        )
    split = len(source) - len(target)
    residue = source[:split]
    return GroupAlignment(
        group=group,
        rule="identity" if split == 0 else "tail",
        source_positions=source[split:],
        target_positions=target,
        source_cardinality=len(source),
        target_cardinality=len(target),
        unpatched_source_tokens=residue,
        residue=_residue_record(tokenizer, source_ids, residue),
    )


def _value_alignment(
    tokenizer: Any,
    *,
    source_values: dict[str, tuple[int, ...]],
    target_values: dict[str, tuple[int, ...]],
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    source_cardinality: int,
    target_cardinality: int,
) -> tuple[GroupAlignment, GroupAlignment]:
    """R25(b): split the value cell into the shared values and the dropped-value slots.

    Values are paired by string identity, not by position, so a source note that lists the
    same values in a different order still pairs each with its twin.  ``position_groups``
    refuses a note whose value string occurs twice, so each string names one span per note and
    the pairing is a plain intersection.
    """
    shared = tuple(value for value in source_values if value in target_values)
    dropped = tuple(value for value in source_values if value not in target_values)
    if not shared:
        raise ValueError(
            "shared_value_tokens: the counterfactual and failing notes share no value, so "
            "no dropped-value slot can be placed relative to a shared one"
        )
    for value in shared:
        if len(source_values[value]) != len(target_values[value]):
            raise ValueError(
                f"shared_value_tokens: value {value!r} spans {len(source_values[value])} "
                f"source tokens and {len(target_values[value])} target tokens; identical "
                "strings must share cardinality by construction"
            )
    shared_cell = GroupAlignment(
        group="shared_value_tokens",
        rule="shared_value_identity",
        source_positions=tuple(p for value in shared for p in source_values[value]),
        target_positions=tuple(p for value in shared for p in target_values[value]),
        source_cardinality=source_cardinality,
        target_cardinality=target_cardinality,
        residue=_residue_record(tokenizer, source_ids, ()),
        shared_values=shared,
        dropped_values=dropped,
    )
    if not dropped:
        raise ValueError(
            "dropped_value_slot: the failing note drops no value the counterfactual note "
            "carries, so the cell has no slot to patch"
        )
    order = list(source_values)
    slots: list[dict[str, Any]] = []
    pooled: list[tuple[int, ...]] = []
    positions: list[int] = []
    for value in dropped:
        preceding = next(
            (
                earlier
                for earlier in reversed(order[: order.index(value)])
                if earlier in target_values
            ),
            None,
        )
        if preceding is None:
            # The value would have been first, so the slot is the separator that precedes the
            # first shared value instead of one that follows a shared value.
            slot = target_values[shared[0]][0] - 1
            rule = "before_first_shared_value"
        else:
            slot = target_values[preceding][-1] + 1
            rule = "after_preceding_shared_value"
        if not 0 <= slot < len(target_ids):
            raise ValueError(
                f"dropped_value_slot: slot {slot} for value {value!r} is outside the "
                f"{len(target_ids)}-token failing prompt"
            )
        if slot in positions:
            raise ValueError(
                f"dropped_value_slot: value {value!r} resolves to slot {slot}, already "
                "claimed by an earlier dropped value; one injection cannot write two rows "
                "to one position"
            )
        pooled_ids = [int(source_ids[position]) for position in source_values[value]]
        slots.append(
            {
                "dropped_value": value,
                "slot_rule": rule,
                "preceding_shared_value": preceding,
                "slot_position": int(slot),
                "overwritten_token_id": int(target_ids[slot]),
                "overwritten_token_text": _decoded(tokenizer, [target_ids[slot]]),
                "pooled_source_positions": [int(p) for p in source_values[value]],
                "pooled_source_token_ids": pooled_ids,
                "pooled_source_text": _decoded(tokenizer, pooled_ids),
            }
        )
        pooled.append(source_values[value])
        positions.append(int(slot))
    slot_cell = GroupAlignment(
        group="dropped_value_slot",
        rule="dropped_value_slot",
        source_positions=(),
        target_positions=tuple(positions),
        source_cardinality=sum(len(rows) for rows in pooled),
        target_cardinality=len(positions),
        pooled_sources=tuple(pooled),
        slots=tuple(slots),
        residue=_residue_record(tokenizer, source_ids, ()),
        shared_values=shared,
        dropped_values=dropped,
    )
    return shared_cell, slot_cell


def align_groups(
    tokenizer: Any,
    *,
    source_groups: dict[str, tuple[int, ...]],
    source_values: dict[str, tuple[int, ...]],
    source_ids: Sequence[int],
    target_groups: dict[str, tuple[int, ...]],
    target_values: dict[str, tuple[int, ...]],
    target_ids: Sequence[int],
) -> dict[str, GroupAlignment]:
    """Match the counterfactual (source) prompt's groups onto the failing (target) prompt's.

    R25, grounded in ``capture.InjectionHook``: ``replace=True`` maps source row *i* to
    ``at_positions[i]`` and refuses unequal counts, and the failing prompt has no position for
    a dropped value.  Unequal groups therefore align on their tail, and the value cell splits
    into an identity-aligned shared cell and a pooled-into-one-slot dropped cell.
    """
    aligned: dict[str, GroupAlignment] = {}
    for group in POSITION_GROUPS:
        if group in ("shared_value_tokens", "dropped_value_slot"):
            continue
        aligned[group] = _tail_alignment(
            tokenizer, group, source_groups[group], target_groups[group], source_ids
        )
    shared_cell, slot_cell = _value_alignment(
        tokenizer,
        source_values=source_values,
        target_values=target_values,
        source_ids=source_ids,
        target_ids=target_ids,
        source_cardinality=len(source_groups["note_value_tokens"]),
        target_cardinality=len(target_groups["note_value_tokens"]),
    )
    aligned["shared_value_tokens"] = shared_cell
    aligned["dropped_value_slot"] = slot_cell
    if any(not aligned[group].target_positions for group in POSITION_GROUPS):
        raise ValueError("each P6 position group must resolve to at least one token")
    return {group: aligned[group] for group in POSITION_GROUPS}


def _take_rows(rows: Any, positions: Sequence[int]) -> Any:
    import mlx.core as mx

    if not positions:
        raise ValueError("P6 position group must not be empty")
    return mx.take(rows, mx.array(tuple(positions), dtype=mx.int32), axis=0)


def _alignment_rows(rows: Any, alignment: GroupAlignment) -> Any:
    """The source rows one injection writes, in target-position order.

    A pooled cell (``dropped_value_slot``) contributes one row per slot: the mean of that
    dropped value's source rows, taken in float32 (briefing rule 1.5) and shaped like a single
    residual row, because the failing prompt has only the one separator position to write to.
    """
    import mlx.core as mx

    if not alignment.pooled_sources:
        return _take_rows(rows, alignment.source_positions)
    pooled = [
        mx.mean(_take_rows(rows, positions).astype(mx.float32), axis=0)
        for positions in alignment.pooled_sources
    ]
    return mx.stack(pooled, axis=0)


def _match_rows(rows: Any, count: int) -> Any:
    """Deterministically truncate or cycle source rows to an injection target."""
    import mlx.core as mx

    if count <= 0 or rows.shape[0] <= 0:
        raise ValueError("P6 source and target groups must not be empty")
    return mx.take(rows, mx.array([index % rows.shape[0] for index in range(count)], dtype=mx.int32), axis=0)


def dropped_value_source_positions(alignment: dict[str, GroupAlignment]) -> tuple[int, ...]:
    """The counterfactual-prompt positions holding the dropped value's own tokens.

    R25(b) already located them: ``dropped_value_slot`` records each dropped value's source
    rows in ``pooled_sources`` (``pooled_source_positions`` in the artifact).  The content
    control (R27(5)) is defined against exactly those positions, so it reads them from the
    alignment rather than re-deriving a second, possibly disagreeing, notion of "the value".
    """
    return tuple(
        dict.fromkeys(
            int(position)
            for rows in alignment["dropped_value_slot"].pooled_sources
            for position in rows
        )
    )


def content_swap_indices(
    alignment: GroupAlignment, dropped_positions: Sequence[int]
) -> tuple[int, ...]:
    """Which of a cell's rows, in target order, come from a dropped-value position.

    Empty means the cell's treatment reads none of those positions, so R27(5)'s content
    control is ``not_applicable`` there and is recorded as such rather than faked.
    """
    dropped = {int(position) for position in dropped_positions}
    if alignment.pooled_sources:
        return tuple(
            index
            for index, rows in enumerate(alignment.pooled_sources)
            if any(int(position) in dropped for position in rows)
        )
    return tuple(
        index
        for index, position in enumerate(alignment.source_positions)
        if int(position) in dropped
    )


def _swap_rows(rows: Any, indices: Sequence[int], replacement: Any) -> Any:
    """Return ``rows`` with the named target-order rows replaced, order preserved."""
    import mlx.core as mx

    if replacement.shape[0] != len(indices):
        raise ValueError("content control replacement must match the swapped row count")
    order = {int(index): position for position, index in enumerate(indices)}
    return mx.concatenate(
        [
            (replacement[order[row]] if row in order else rows[row])[None, :]
            for row in range(rows.shape[0])
        ],
        axis=0,
    )


def _random_control_pair(
    *,
    source_length: int,
    target_length: int,
    source_treatment: Sequence[int],
    target_treatment: Sequence[int],
    cardinality: int,
    seed: int,
    task_id: str,
    layer: int,
    group: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Draw independent non-treatment source/target groups from one exact P6 seed."""
    source_candidates = sorted(set(range(source_length)) - set(source_treatment))
    target_candidates = sorted(set(range(target_length)) - set(target_treatment))
    if len(source_candidates) < cardinality or len(target_candidates) < cardinality:
        raise ValueError("random control candidate pool cannot satisfy treatment cardinality")
    rng = random.Random(f"{seed}:{task_id}:{layer}:{group}")
    return (
        tuple(sorted(rng.sample(source_candidates, cardinality))),
        tuple(sorted(rng.sample(target_candidates, cardinality))),
    )


def _score_patch(
    view: ArchitectureView,
    tokenizer: Any,
    case: PatchCase,
    *,
    scoring: CaseScoring,
    layer: int,
    source_rows: Any,
    target_positions: Sequence[int],
    failing_ids: Sequence[int],
    keep_last: int,
    max_tokens: int,
) -> ScoredGeneration:
    """Generate the decision turn under one injection and score it strictly (R27(1))."""
    with InjectionHook(
        view,
        layer - 1,
        source_rows,
        at_positions=target_positions,
        replace=True,
    ):
        raw = greedy_generate(view, tokenizer, failing_ids, max_tokens=max_tokens)
    scored = score_generation(raw, scoring)
    if scored.outcome == "parse_error":
        return scored
    steps = [dict(step) for step in case.failing_steps]
    steps[case.decision_step]["thought"] = scored.note
    return replace(
        scored,
        value_drop_cleared=_is_flip(case.task, steps, case.decision_step, keep_last=keep_last),
    )


_CONTENT_SWAP_NOT_APPLICABLE = (
    "the cell's treatment source rows include none of the dropped value's token positions, "
    "so there is nothing to swap (R27)"
)


def _control_record(
    control: str,
    task_outcomes: dict[str, Sequence[str]],
    swapped: dict[str, int],
    *,
    cases: int,
) -> dict[str, Any]:
    """One control's cell entry; ``content_swap`` may be ``not_applicable`` (R27(5)).

    R30(6): a control's rate covers only the cases it could actually be run on, so the count
    it was taken over (``applicable_cases``) and the cell's full population (``cases``) are
    both recorded.  A rate of 0.4 over 5 cases and 0.4 over 2 are different claims, and the
    reader must not have to infer which one a cell is making.
    """
    scope = {"applicable_cases": len(task_outcomes), "cases": cases}
    if control != "content_swap":
        return {**aggregate_task_outcomes(task_outcomes), **scope}
    if not task_outcomes:
        return {"applicable": False, "reason": _CONTENT_SWAP_NOT_APPLICABLE, **scope}
    return {
        **aggregate_task_outcomes(task_outcomes),
        **scope,
        "applicable": True,
        "swapped_rows": dict(sorted(swapped.items())),
    }


def condition_of(cases: Sequence[PatchCase]) -> str:
    """The one condition a run's cases belong to; mixing them is refused (R27(5))."""
    conditions = {case.condition for case in cases}
    if len(conditions) != 1:
        raise ValueError(
            "a P6 run scores one condition at a time; "
            f"got {', '.join(sorted(conditions)) or 'none'}"
        )
    return conditions.pop()


def run_patch_probe(
    model: Any,
    tokenizer: Any,
    cases: Sequence[PatchCase],
    *,
    spec: ModelSpec,
    resolved: ResolvedSpec,
    layers: Sequence[int],
    policy: str,
    keep_last: int,
    max_tokens: int,
    seed: int,
    command: Sequence[str],
    minimum_scored: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the P6 cells and return per-generation, JSON-safe records.

    The headline cells run over scoring-version-stable cases (R24) whose dropped value is
    NOT visible in the retained observations (R27(3)); every other case is listed under
    ``excluded_cases`` with its reason and both judgements, and never merged into a cell.
    R30a(2) puts a case whose note the locator cannot place in the same container, for the
    same reason: it is a case that was not measured, and one of those must not decide the
    fate of the ones behind it, which is what happened to the 2026-09-05 secondary run.
    ``minimum_scored`` is the caller's floor on how many cases must survive that filter for
    the section to be worth reading; zero leaves only the structural minimum of two that
    the unrelated-task control needs.
    Aggregate-only artifacts are not permitted for P6 (R27(2)), so every generation's note
    text, parsed value set and outcome is recorded in the top-level ``generations`` list —
    one flat row per (case, layer, cell, condition), which keeps ``cells`` comparable with
    the pre-R27 artifact and lets a re-scoring run offline.  This seam is deliberately
    composed from fakes in tests; the CLI is the only real-model entry point and remains
    gated by its GPU guard.

    ``progress`` is an optional ``(step, total, label)`` callback reporting completed work:
    one call per prepared case and one per (layer, group) cell once its cases are scored
    (R26(g)). It is reporting only — the payload is identical with and without it — so the
    function stays free of I/O and composes from fakes.
    """
    view = ArchitectureView.from_model(model)
    if not cases:
        raise ValueError("no eligible patch cases")
    if not layers or any(layer < 1 or layer > view.num_layers for layer in layers):
        raise ValueError("layers must be residual indices in [1, num_layers]")
    # R18b: the registry decides the capture precision. Every consumer here already casts to
    # float32 -- the pooled slot mean in ``_alignment_rows`` and ``InjectionHook`` itself --
    # so ``native`` halves the captured bytes without moving a single injected number.
    capture_dtype = spec_capture_dtype(spec)
    stable = [case for case in cases if case.scoring_version_stable]
    unstable = [case for case in cases if not case.scoring_version_stable]
    visibility = {
        case.task.task_id: dropped_value_visibility(case, keep_last=keep_last)
        for case in stable
    }
    def _visible(case: PatchCase) -> bool:
        return bool(
            visibility[case.task.task_id]["dropped_value_visible_in_retained_observations"]
        )

    visible_cases = [case for case in stable if _visible(case)]
    measurable = [case for case in stable if not _visible(case)]
    # R30(4): a case whose required set is not inside its expected set can never score a
    # flip, so scoring it would report a false 0.0.  Mark it and keep it out of the headline.
    scorings = {case.task.task_id: case_scoring(case) for case in measurable}
    unsatisfiable = [
        case for case in measurable if not scorings[case.task.task_id].flip_satisfiable
    ]
    hidden = [case for case in measurable if scorings[case.task.task_id].flip_satisfiable]
    if len(hidden) < 2:
        raise ValueError(
            "P6 unrelated-task control requires at least two scoring-version-stable patch "
            f"cases (R24); {len(stable)} stable, {len(unstable)} unstable, "
            f"{len(visible_cases)} excluded for a dropped value visible in the retained "
            f"observations (R27), {len(unsatisfiable)} excluded as flip_unsatisfiable (R30)"
        )

    def case_record(case: PatchCase, provenance: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": case.task.task_id,
            "decision_step": case.decision_step,
            "condition": case.condition,
            "counterfactual_label": case.counterfactual_label,
            **provenance,
            **case.scoring_record(),
            # Measured once for every stable case above; an unstable case is measured here
            # so its record carries the same fields, never a blank the reader must guess at.
            **(
                visibility[case.task.task_id]
                if case.task.task_id in visibility
                else dropped_value_visibility(case, keep_last=keep_last)
            ),
        }

    excluded: list[dict[str, Any]] = []
    for case, reason in (
        [(case, "scoring_version_unstable") for case in unstable]
        + [(case, "dropped_value_visible_in_retained_observations") for case in visible_cases]
        + [(case, "flip_unsatisfiable") for case in unsatisfiable]
    ):
        _failing, _counterfactual, provenance = replay_counterfactual(case)
        record = {**case_record(case, provenance), "excluded_reason": reason}
        if case.task.task_id in scorings:
            # R30(4): the excluded case shows the sets it was judged on, so "unsatisfiable"
            # is a reader-checkable claim rather than a verdict.
            record["value_sets"] = scorings[case.task.task_id].record()
        excluded.append(record)
    prepared: list[dict[str, Any]] = []
    case_provenance: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}

    def _record_skip(
        case: PatchCase, provenance: dict[str, Any], reason: str, error: ValueError
    ) -> None:
        """File a case that could not be prepared beside the cases excluded before it.

        Same container, same fields, one more reason: a reader counting the fifteen finds
        every one of them in ``excluded_cases`` or in the headline, and never has to infer a
        case's fate from its absence.
        """
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        record = {**case_record(case, provenance), "excluded_reason": reason}
        record["excluded_detail"] = str(error)
        if case.task.task_id in scorings:
            record["value_sets"] = scorings[case.task.task_id].record()
        excluded.append(record)

    for case_number, case in enumerate(hidden, start=1):
        failing, counterfactual, provenance = replay_counterfactual(case)
        failing_prompt = build_prompt(tokenizer, failing, spec=spec, keep_last=keep_last)
        counter_prompt = build_prompt(tokenizer, counterfactual, spec=spec, keep_last=keep_last)
        failing_ids = list(tokenizer.encode(failing_prompt, add_special_tokens=False))
        counter_ids = list(tokenizer.encode(counter_prompt, add_special_tokens=False))
        # R30a(2): preparing one case must not decide the fate of the ones behind it. The two
        # refusals reachable here name the two things that can still leave a case with no
        # positions to patch -- a note value the locator cannot place, and a pair of dropped
        # values R25's slot rule cannot give distinct separators -- and each is recorded
        # against its own case, with the refusal's own words, instead of raised.
        try:
            failing_groups, failing_repairs, failing_values = _groups_for(
                tokenizer, failing_ids, failing, prompt_text=failing_prompt, keep_last=keep_last
            )
            counter_groups, counter_repairs, counter_values = _groups_for(
                tokenizer,
                counter_ids,
                counterfactual,
                prompt_text=counter_prompt,
                keep_last=keep_last,
            )
        except ValueError as error:
            _record_skip(case, provenance, "note_values_unlocatable", error)
            if progress is not None:
                progress(case_number, len(hidden), f"skip {case.task.task_id}")
            continue
        try:
            alignment = align_groups(
                tokenizer,
                source_groups=counter_groups,
                source_values=counter_values,
                source_ids=counter_ids,
                target_groups=failing_groups,
                target_values=failing_values,
                target_ids=failing_ids,
            )
        except ValueError as error:
            _record_skip(case, provenance, "alignment_unplaceable", error)
            if progress is not None:
                progress(case_number, len(hidden), f"skip {case.task.task_id}")
            continue
        # Counted only for a case that survived preparation, so the source tally describes
        # the cases the cells were actually taken over.
        source = provenance.get("counterfactual_source") or "none"
        source_counts[source] = source_counts.get(source, 0) + 1
        alignment_record = {group: cell.record() for group, cell in alignment.items()}
        scoring = scorings[case.task.task_id]
        case_provenance.append(
            {
                **case_record(case, provenance),
                "value_sets": scoring.record(),
                "boundary_repairs": {"failing": failing_repairs, "counterfactual": counter_repairs},
                "alignment": alignment_record,
            }
        )
        prepared.append(
            {
                "case": case,
                "scoring": scoring,
                "dropped_source_positions": dropped_value_source_positions(alignment),
                "failing_ids": failing_ids,
                "counter_ids": counter_ids,
                "failing_groups": failing_groups,
                "counter_groups": counter_groups,
                "alignment": alignment,
                "alignment_record": alignment_record,
                "failing_residuals": capture_residuals(
                    view, failing_ids, layers, positions="all", dtype=capture_dtype
                ),
                "counter_residuals": capture_residuals(
                    view, counter_ids, layers, positions="all", dtype=capture_dtype
                ),
            }
        )
        if progress is not None:
            progress(case_number, len(hidden), f"capture {case.task.task_id}")
    # R30a(2)/(3): the section refuses on the count that survived preparation, not on the
    # first case that did not. Two is structural -- the unrelated-task control patches from
    # another case's context and has nowhere to read one from below it -- and the caller's
    # floor sits above it.
    if len(prepared) < max(2, minimum_scored):
        raise ValueError(
            f"P6 section scored {len(prepared)} of {len(cases)} selected case(s), below the "
            f"minimum of {max(2, minimum_scored)}: "
            + (
                ", ".join(f"{count} {reason}" for reason, count in sorted(skipped_reasons.items()))
                or "no case was skipped in preparation"
            )
        )
    cells: dict[str, dict[str, Any]] = {}
    generations: list[dict[str, Any]] = []
    cell_total = len(layers) * len(POSITION_GROUPS)
    cell_number = 0

    def score(
        item: dict[str, Any],
        *,
        condition: str,
        layer: int,
        group: str,
        rows: Any,
        positions: Sequence[int],
        sink: dict[str, dict[str, list[str]]],
    ) -> None:
        """Score one generation and record it under both the cell and ``generations``."""
        case = item["case"]
        scored = _score_patch(
            view,
            tokenizer,
            case,
            scoring=item["scoring"],
            layer=layer,
            source_rows=rows,
            target_positions=positions,
            failing_ids=item["failing_ids"],
            keep_last=keep_last,
            max_tokens=max_tokens,
        )
        sink[condition][case.task.task_id] = [scored.outcome]
        generations.append(
            {
                "task_id": case.task.task_id,
                "layer": layer,
                "group": group,
                "condition": condition,
                **scored.record(),
            }
        )

    for layer in layers:
        for group in POSITION_GROUPS:
            outcomes: dict[str, dict[str, list[str]]] = {
                name: {} for name in ("treatment", *CONTROLS)
            }
            swapped: dict[str, int] = {}
            for index, item in enumerate(prepared):
                case = item["case"]
                alignment = item["alignment"][group]
                target = alignment.target_positions
                treatment_rows = _alignment_rows(item["counter_residuals"][layer], alignment)
                if treatment_rows.shape[0] != len(target):
                    raise ValueError("treatment source and target group cardinality must match")
                score(
                    item, condition="treatment", layer=layer, group=group,
                    rows=treatment_rows, positions=target, sink=outcomes,
                )
                unrelated = prepared[(index + 1) % len(prepared)]
                unrelated_rows = _match_rows(
                    _alignment_rows(unrelated["counter_residuals"][layer], unrelated["alignment"][group]),
                    len(target),
                )
                score(
                    item, condition="unrelated_task", layer=layer, group=group,
                    rows=unrelated_rows, positions=target, sink=outcomes,
                )
                random_source, random_target = _random_control_pair(
                    source_length=len(item["counter_ids"]),
                    target_length=len(item["failing_ids"]),
                    source_treatment=alignment.treatment_source_positions,
                    target_treatment=target,
                    cardinality=len(target),
                    seed=seed,
                    task_id=case.task.task_id,
                    layer=layer,
                    group=group,
                )
                random_rows = _take_rows(item["counter_residuals"][layer], random_source)
                score(
                    item, condition="random_positions", layer=layer, group=group,
                    rows=random_rows, positions=random_target, sink=outcomes,
                )
                # R27(5) content control: the treatment rows with the dropped value's own
                # rows replaced by an unrelated case's dropped-value rows.  A cell whose
                # source rows never touch those positions has nothing to swap, so it is
                # recorded ``not_applicable`` rather than given a fabricated number.
                indices = content_swap_indices(alignment, item["dropped_source_positions"])
                if not indices:
                    continue
                # The foreign value's rows are pooled exactly as R25(b) pools the treatment's
                # own: ``_alignment_rows`` over the unrelated case's slot cell yields one
                # float32 mean row per dropped value.  Taking that value's first token row
                # unpooled would make the control a different manipulation from the
                # treatment it exists to isolate, and the comparison would not be a
                # content swap at all.
                foreign_cell = unrelated["alignment"]["dropped_value_slot"]
                if not foreign_cell.pooled_sources:
                    continue
                replacement = _match_rows(
                    _alignment_rows(unrelated["counter_residuals"][layer], foreign_cell),
                    len(indices),
                )
                swapped[case.task.task_id] = len(indices)
                score(
                    item, condition="content_swap", layer=layer, group=group,
                    rows=_swap_rows(treatment_rows, indices, replacement),
                    positions=target, sink=outcomes,
                )
            cells[f"{layer}:{group}"] = {
                "layer": layer,
                "group": group,
                # R25(d): every cell records the alignment rule, both cardinalities, and the
                # residue or slot detail, per case, so the heat map is never read without it.
                "alignment": {
                    item["case"].task.task_id: item["alignment_record"][group]
                    for item in prepared
                },
                "treatment": {
                    **aggregate_task_outcomes(outcomes["treatment"]),
                    "applicable_cases": len(outcomes["treatment"]),
                    "cases": len(prepared),
                },
                "controls": {
                    control: _control_record(
                        control, outcomes[control], swapped, cases=len(prepared)
                    )
                    for control in CONTROLS
                },
            }
            cell_number += 1
            if progress is not None:
                progress(cell_number, cell_total, f"layer {layer} {group}")
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "model": resolved.as_dict(),
        "policy": policy,
        "layers": list(layers),
        # R18b: the precision the residuals were captured at; R18a: the preflight's measured
        # float32-vs-native deviation, copied verbatim so the cells are read beside it.
        "capture_dtype": capture_dtype,
        "fp32_manual_vs_native": preflight_precision_block(spec),
        "seed": seed,
        "command": list(command),
        "condition": condition_of(cases),
        "groups": list(POSITION_GROUPS),
        "controls": list(CONTROLS),
        "outcomes": list(OUTCOMES),
        "selected_task_ids": [case.task.task_id for case in cases],
        "scoring_generator_version": GENERATOR_VERSION,
        "stable_cases": len(stable),
        "unstable_cases": len(unstable),
        "headline_cases": len(prepared),
        "visibility_excluded_cases": len(visible_cases),
        "flip_unsatisfiable_cases": len(unsatisfiable),
        # R30a(2)/(3): scored and skipped out of the selection, so a rate is never read as
        # covering cases nothing was measured on. ``skipped_cases`` counts every selected
        # case that reached no cell, whichever filter took it.
        "selected_cases": len(cases),
        "scored_cases": len(prepared),
        "skipped_cases": len(cases) - len(prepared),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "headline_task_ids": [item["case"].task.task_id for item in prepared],
        "cases": case_provenance,
        "generations": generations,
        "excluded_cases": excluded,
        "counterfactual_sources": source_counts,
        "cells": cells,
    }


def _applicable_n(summary: dict[str, Any]) -> str:
    """``n=<applicable>/<cases>`` for a cell entry (R30(6)); ``n=?`` when a payload predates it."""
    applicable = summary.get("applicable_cases")
    cases = summary.get("cases")
    if applicable is None or cases is None:
        return "n=?"
    return f"n={applicable}/{cases}"


def _outcome_counts_table(payload: dict[str, Any]) -> list[str]:
    """R27(2)/(7): every cell's per-outcome generation counts, treatment and controls.

    A rate alone cannot be read after R27 — 0.0 from four corruptions and 0.0 from four
    unchanged notes mean opposite things — so the counts sit beside the heat map.
    """
    names = payload.get("outcomes") or list(OUTCOMES)
    lines = [
        "| layer/group | condition | n | " + " | ".join(names) + " |",
        "|---|---|---|" + "|".join("---:" for _ in names) + "|",
    ]
    for key, cell in payload["cells"].items():
        for condition in ("treatment", *payload["controls"]):
            summary = cell["treatment"] if condition == "treatment" else cell["controls"][condition]
            scope = _applicable_n(summary)
            if not summary.get("applicable", True):
                lines.append(
                    f"| {key} | {condition} | {scope} | "
                    + " | ".join("n/a" for _ in names)
                    + " |"
                )
                continue
            counts = summary.get("outcomes", {})
            lines.append(
                f"| {key} | {condition} | {scope} | "
                + " | ".join(str(counts.get(name, 0)) for name in names)
                + " |"
            )
    return lines


def _run_health(payload: dict[str, Any]) -> dict[str, Any]:
    """R30a(4): the run's own verdict, named by the sections that scored nothing.

    A section is unscored when it carries an ``error`` where its cells belong; that is the
    only shape a refusal takes here, and it is the shape the aborted run wrote. The verdict
    follows R26(c)'s vocabulary for a non-training stage: a run that did not do the thing it
    was asked to do is not healthy, whatever its exit code used to say.
    """
    unscored = [
        name
        for name in ("secondary",)
        if isinstance(payload.get(name), dict) and payload[name].get("error")
    ]
    return {
        "verdict": "unscored_section" if unscored else "scored",
        "status": "error" if unscored else "ok",
        "unscored_sections": unscored,
        "section_errors": {name: str(payload[name]["error"]) for name in unscored},
    }


def _coverage_lines(payload: dict[str, Any]) -> list[str]:
    """R30a(2): how many of the selected cases were scored, how many skipped, and why.

    Placed above the rates rather than below them, because a rate over a third of the
    selection is a different number from the same rate over all of it, and the reader meets
    the rate first.  A payload written before R30a carries no counts and gets no line.
    """
    if "scored_cases" not in payload:
        return []
    reasons = payload.get("skipped_reasons") or {}
    detail = ", ".join(f"{count} {reason}" for reason, count in sorted(reasons.items()))
    return [
        f"{payload['scored_cases']} of {payload['selected_cases']} selected case(s) scored; "
        f"{payload['skipped_cases']} skipped"
        + (f" ({detail})." if detail else "."),
        "",
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact layer×group treatment table and named-control tables.

    The heat map covers headline cases only — scoring-version-stable (R24) and with the
    dropped value hidden from the retained observations (R27(3)); every excluded case gets
    its own table so it is never read as part of the headline.  The strict outcome counts
    (R27(1)) and the content control (R27(5)) each get their own section, and the secondary
    condition, when a run carries one, is rendered separately and never merged.
    """
    groups = payload["groups"]
    lines = ["# P6 causal patching", ""]
    health = payload.get("health")
    if health and health.get("unscored_sections"):
        # R30a(4): before the condition, before the rates. A reader who stops after the
        # first screen must not stop believing the run measured what it set out to.
        lines.extend(
            [
                "**Unscored section(s): "
                + ", ".join(
                    f"{name} — {health['section_errors'][name]}"
                    for name in health["unscored_sections"]
                )
                + "**",
                "",
            ]
        )
    if payload.get("condition"):
        lines.extend([f"Condition: `{payload['condition']}`.", ""])
    lines.extend(_coverage_lines(payload))
    lines.extend(["## Treatment flip rate", ""])
    if "stable_cases" in payload:
        lines.extend(
            [
                f"Headline over {payload.get('headline_cases', payload['stable_cases'])} "
                f"case(s) of {payload['stable_cases']} scoring-version-stable; "
                f"{payload['unstable_cases']} unstable case(s) excluded (R24) and "
                f"{payload.get('visibility_excluded_cases', 0)} excluded because the dropped "
                "value is visible in the retained observations (R27).",
                "",
                "A flip is strict (R27): the note parses, carries every value the failing "
                "note had and every dropped value, and no number outside the canonical set. "
                "Read the rates with the outcome counts below.",
                "",
            ]
        )
    lines.extend(["| layer | " + " | ".join(groups) + " |", "|---|" + "|".join("---" for _ in groups) + "|"])
    for layer in payload["layers"]:
        values = []
        for group in groups:
            summary = payload["cells"][f"{layer}:{group}"]["treatment"]
            low, high = summary["wilson_95"]
            values.append(f"{summary['rate']:.3f} [{low:.3f}, {high:.3f}]")
        lines.append(f"| {layer} | " + " | ".join(values) + " |")
    for control in payload["controls"]:
        # R30(6): every control row prints the n its rate was taken over, beside the cell's
        # full population, so a rate is never read as covering more cases than it does.
        lines.extend(
            [
                "",
                f"## Control: {control}",
                "",
                "| layer/group | n | rate | 95% Wilson |",
                "|---|---|---:|---|",
            ]
        )
        for key, cell in payload["cells"].items():
            summary = cell["controls"][control]
            scope = _applicable_n(summary)
            if not summary.get("applicable", True):
                lines.append(
                    f"| {key} | {scope} | not_applicable | {summary.get('reason', '')} |"
                )
                continue
            low, high = summary["wilson_95"]
            lines.append(
                f"| {key} | {scope} | {summary['rate']:.3f} | [{low:.3f}, {high:.3f}] |"
            )
    if payload.get("cells"):
        lines.extend(["", "## Outcome counts per cell (R27)", ""])
        lines.extend(_outcome_counts_table(payload))
    sources = payload.get("counterfactual_sources")
    if sources:
        lines.extend(["", "## Counterfactual note sources", ""])
        lines.extend(f"- {name}: {sources[name]}" for name in sorted(sources))
    alignment_lines = _alignment_table(payload.get("cases", []))
    if alignment_lines:
        lines.extend(["", "## Alignment (R25)", ""])
        lines.extend(alignment_lines)
    if "stable_cases" in payload:
        lines.extend(["", "## Scoring version stability and value visibility (R24, R27)", ""])
        lines.extend(_stability_table(payload.get("cases", []), "headline"))
        lines.extend(["", "### Excluded cases (never in the headline)", ""])
        if payload.get("excluded_cases"):
            lines.extend(_stability_table(payload["excluded_cases"], "excluded"))
        else:
            lines.append("None.")
    secondary = payload.get("secondary")
    if secondary:
        lines.extend(["", "# Secondary condition (R27, issue #26)", ""])
        lines.extend(_secondary_section(secondary))
    return "\n".join(lines)


def _secondary_section(secondary: dict[str, Any]) -> list[str]:
    """Render the secondary condition on its own, never merged into the ledger headline."""
    lines = [
        f"Condition `{secondary.get('condition', 'unknown')}`, counterfactual note labelled "
        f"`{secondary.get('counterfactual_label', 'designed_correct')}`. This population is "
        "disjoint from the headline above and its rates are never pooled with it.",
        "",
    ]
    if secondary.get("error"):
        lines.extend(
            [
                f"Not scored: {secondary['error']}",
                "",
                "Selected task ids: "
                + (", ".join(secondary.get("selected_task_ids", [])) or "none"),
            ]
        )
        return lines
    lines.append(render_markdown(secondary))
    return lines


def _alignment_table(records: Sequence[dict[str, Any]]) -> list[str]:
    """One row per (case, cell): the rule, both cardinalities, and the residue or slot."""
    rows = [
        (record["task_id"], group, entry)
        for record in records
        if isinstance(record.get("alignment"), dict)
        for group, entry in record["alignment"].items()
    ]
    if not rows:
        return []
    lines = [
        "| task | cell | rule | source | target | unpatched | detail |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for task_id, group, entry in rows:
        slots = entry.get("slots") or []
        detail = (
            "; ".join(
                f"{slot['dropped_value']} -> {slot['slot_position']} "
                f"({slot['slot_rule']}, overwrote {slot['overwritten_token_text']!r})"
                for slot in slots
            )
            if slots
            else entry["unpatched_source_tokens"]["text"] or "-"
        )
        lines.append(
            f"| {task_id} | {group} | {entry['rule']} | {entry['source_cardinality']} | "
            f"{entry['target_cardinality']} | "
            f"{entry['unpatched_source_tokens']['count']} | {detail} |"
        )
    return lines


def _stability_table(records: Sequence[dict[str, Any]], population: str) -> list[str]:
    def values(listed: Any) -> str:
        return ", ".join(listed) if isinstance(listed, list) else "unknown"

    lines = [
        f"| task ({population}) | stable | decision step bound / HEAD | "
        "dropped values bound / HEAD | value visible in retained obs (R27) | excluded because |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        visible = record.get("dropped_value_visible_in_retained_observations")
        lines.append(
            f"| {record['task_id']} | {'yes' if record['scoring_version_stable'] else 'no'} | "
            f"{record['decision_step_bound']} / {record['decision_step_head']} | "
            f"{values(record['dropped_values_bound'])} / {values(record['dropped_values_head'])} | "
            f"{'unknown' if visible is None else ('yes' if visible else 'no')} | "
            f"{record.get('excluded_reason', '-')} |"
        )
    return lines


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"evaluation {path} must be a JSON object")
    return value


def main() -> None:
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu
    from local_llm_lab.runlog import RunLog, git_commit, sha256_of

    parser = argparse.ArgumentParser(description="P6 causal residual patching at value drops.")
    parser.add_argument("--passing-eval", type=Path, required=True)
    parser.add_argument("--failing-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen25-coder-3b")
    parser.add_argument("--policy", default="base")
    parser.add_argument("--layers")
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="data seed override, used only when an evaluation lacks the data_seed field (R22a)",
    )
    parser.add_argument(
        "--generator-version",
        type=int,
        default=None,
        help=(
            "generator version the integrity recomputation replays under, used only when "
            "the failing evaluation records no generator_version (R12/R22)"
        ),
    )
    parser.add_argument(
        "--secondary-condition",
        choices=SECONDARY_CONDITIONS,
        default=None,
        help=(
            "also score the bound secondary condition (R27(5), issue #26): the named "
            "family's failures under C, with the designed-correct generator note as the "
            "counterfactual. Written to a separate section of the artifact; never merged "
            "into the headline"
        ),
    )
    parser.add_argument(
        "--skip-preflight-check",
        action="store_true",
        help="run without the model's preflight evidence (SPEC-001 §10 override)",
    )
    add_gpu_arguments(parser)
    args = parser.parse_args()
    if not args.passing_eval.is_file() or not args.failing_eval.is_file():
        parser.error("--passing-eval and --failing-eval must name existing files")
    if args.keep_last < 0 or args.max_tokens <= 0:
        parser.error("--keep-last must be non-negative and --max-tokens must be positive")
    spec = load_model_spec(args.model)
    try:
        validate_layer_syntax(args.layers)
    except ValueError as error:
        parser.error(str(error))
    passing_payload = _load_payload(args.passing_eval)
    failing_payload = _load_payload(args.failing_eval)
    secondary_cases: list[PatchCase] = []
    secondary_provenance: dict[str, Any] = {}
    try:
        cases, selection_provenance = select_patch_cases(
            passing_payload,
            failing_payload,
            keep_last=args.keep_last,
            data_seed=args.data_seed,
            generator_version=args.generator_version,
        )
        if args.secondary_condition is not None:
            secondary_cases, secondary_provenance = select_patch_cases(
                passing_payload,
                failing_payload,
                keep_last=args.keep_last,
                data_seed=args.data_seed,
                generator_version=args.generator_version,
                secondary_condition=args.secondary_condition,
            )
    except ValueError as error:
        parser.error(str(error))
    if not cases:
        parser.error("no eligible patch cases")
    try:
        adapter = resolve_policy(args.policy, spec)
    except ValueError as error:
        parser.error(str(error))
    # R26(e), issue #35: the log opens once the adapter is known but before the GPU guard
    # and the model load, so run.log alone says what ran and covers the whole run.
    with RunLog.open(
        args.output,
        name="p6-patch",
        command=sys.argv,
        identity={
            "model": spec.name,
            "hf_id": spec.hf_id,
            "policy": args.policy,
            "adapter": str(adapter) if adapter else None,
            "passing_eval_sha256": sha256_of(args.passing_eval),
            "failing_eval_sha256": sha256_of(args.failing_eval),
            "layers": args.layers,
            "keep_last": args.keep_last,
            "data_seed": args.data_seed,
            "generator_version": args.generator_version,
            "git_commit": git_commit(),
        },
    ) as log:
        stable = sum(1 for case in cases if case.scoring_version_stable)
        log.info(
            "selected cases",
            cases=len(cases),
            stable=stable,
            unstable=len(cases) - stable,
            decision_steps=[case.decision_step for case in cases],
        )
        # SPEC-001 §10: the preflight gate sits before the GPU guard and the load, so a model
        # without evidence is refused by name and no weights are touched.
        require_preflight(spec, skip=args.skip_preflight_check)
        require_idle_gpu(parser, args, "running P6 causal patching")
        log.info("loading policy", model=args.model, policy=args.policy)
        model, tokenizer, view, resolved = load_policy(spec, adapter)
        try:
            selection = resolve_layers(args.layers, spec, view.num_layers)
        except ValueError as error:
            parser.error(str(error))
        log.info("layers", selected=list(selection.indices))
        del view
        payload = run_patch_probe(
            model,
            tokenizer,
            cases,
            spec=spec,
            resolved=resolved,
            layers=selection.indices,
            policy=args.policy,
            keep_last=args.keep_last, max_tokens=args.max_tokens, seed=args.seed, command=sys.argv,
            progress=log.progress,
        )
        if not isinstance(payload, dict) or "cells" not in payload:
            parser.error("run_patch_probe returned an invalid payload")
        payload["layer_selection"] = selection.as_dict()
        payload["data_seeds"] = selection_provenance["data_seeds"]
        payload["eligibility"] = selection_provenance["eligibility"]
        if args.secondary_condition is not None:
            # R27(5): a separate population with its own headline. It runs after the
            # primary so a refusal here (too few eligible cases, say) costs the primary
            # nothing; the reason is recorded in place of the section's cells.
            log.info(
                "secondary condition",
                condition=args.secondary_condition,
                cases=len(secondary_cases),
            )
            section: dict[str, Any] = {
                "condition": secondary_provenance.get("condition", args.secondary_condition),
                "counterfactual_label": secondary_provenance.get(
                    "counterfactual_label", "designed_correct"
                ),
                "selected_task_ids": [case.task.task_id for case in secondary_cases],
                "eligibility": secondary_provenance.get("eligibility", {}),
            }
            try:
                section = {
                    **section,
                    **run_patch_probe(
                        model,
                        tokenizer,
                        secondary_cases,
                        spec=spec,
                        resolved=resolved,
                        layers=selection.indices,
                        policy=args.policy,
                        keep_last=args.keep_last,
                        max_tokens=args.max_tokens,
                        seed=args.seed,
                        command=sys.argv,
                        # R30a(3): the floor the ruling names, applied to the section it was
                        # ruled on. The primary keeps only the structural minimum, so a
                        # count that has always been read stays readable.
                        minimum_scored=MINIMUM_SCORED_CASES,
                        progress=log.progress,
                    ),
                }
            except ValueError as error:
                section["error"] = str(error)
                log.info("secondary condition not scored", reason=str(error))
            payload["secondary"] = section
        # R30a(4): a section that scored nothing is the run's state, not a footnote in one
        # of its blocks. It decides the health record, the end event, the first line of the
        # markdown and the exit status together, because the 2026-09-05 run had an honest
        # ``error`` field and a clean everything-else, and the everything-else is what a
        # reader sees first.
        health = _run_health(payload)
        payload["health"] = health
        markdown = render_markdown(payload)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "patch.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output / "patch.md").write_text(markdown + "\n", encoding="utf-8")
        log.info(
            "wrote",
            json=str(args.output / "patch.json"),
            md=str(args.output / "patch.md"),
        )
        write_provenance(
            args.output,
            resolved=resolved,
            spec=spec,
            extra={
                "stage": "p6-patch",
                "policy": args.policy,
                "artifacts": [
                    str(args.output / "patch.json"),
                    str(args.output / "patch.md"),
                ],
            },
        )
        # Closed here rather than left to the context manager, which has only the exception
        # to go on and would call this run ``ok``.
        log.close(
            status=health["status"],
            verdict=health["verdict"],
            unscored_sections=health["unscored_sections"],
        )
    if health["unscored_sections"]:
        raise SystemExit(
            "P6 run incomplete: "
            + "; ".join(
                f"{name} section not scored: {health['section_errors'][name]}"
                for name in health["unscored_sections"]
            )
            + f"; see {args.output / 'patch.json'}"
        )
