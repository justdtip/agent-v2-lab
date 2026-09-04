"""Gated, fake-testable causal patching at the dropped-value decision."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import ModelSpec, ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.evaluate import load_policy, wilson
from local_llm_lab.pipeline.integrity import check_trajectory
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

CONTROLS = ("unrelated_task", "random_positions")
ARTIFACT_SCHEMA = "p6-patch-r25"
"""Names the artifact shape; the payload carried no schema/version field before R25."""
_FAMILIES = frozenset({"aggregate_report", "ledger_reconcile"})


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
    if "data_seed" not in payload:
        if data_seed is None:
            raise ValueError(
                f"{name} evaluation lacks data_seed and no --data-seed override was given"
            )
        return records, data_seed, "flag"
    seed = payload.get("data_seed")
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
) -> tuple[list[PatchCase], dict[str, Any]]:
    """Select the SPEC-004 §5 universe — task ids that pass under B and fail under C —
    restricted to value-drop decisions, and reconstruct C's task declaration.

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
    integrity_counts = {"evaluation": 0, "recomputed": 0}
    difficulty_counts = {"evaluation": 0, "recomputed": 0}
    for record in failing:
        task_id = record.get("task_id")
        steps = record.get("steps")
        verdict = record.get("verdict")
        if (
            not isinstance(task_id, str)
            or task_id not in successful
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
        if task.family not in _FAMILIES:
            continue
        if not all(isinstance(step, dict) for step in steps):
            raise ValueError(f"{task_id}: malformed failing steps")
        difficulty_counts[difficulty_source] += 1
        if "integrity" in record:
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
        # R24: flip scoring judges the HEAD task these steps were built for (``_is_flip``);
        # judge it here too, so the artifact records whether that seam is admissible.
        head = _recomputed_judgement(
            task, steps, keep_last=keep_last, judged_under=f"generator_v{GENERATOR_VERSION} HEAD"
        )
        saved = successful[task_id].get("steps")
        passing_steps = (
            tuple(dict(step) for step in saved)
            if isinstance(saved, list) and all(isinstance(step, dict) for step in saved)
            else None
        )
        selected.append(
            PatchCase(
                task, dropped, tuple(dict(step) for step in steps), passing_steps, bound, head
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
    """A numeric value matched on its own boundaries, never inside a longer number.

    ``32`` occurring inside ``132`` is a different value, not a second occurrence of this one
    (issue #29): matching by plain substring aborted such a note as ambiguous.  The lookarounds
    exclude a neighbouring digit, decimal point, or sign, so a genuine duplicate still matches
    twice and is still refused.
    """
    return rf"(?<![0-9.\-]){re.escape(value)}(?![0-9.])"


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
    for value in note_values:
        if note_span is None:
            raise ValueError("note_value token span is missing or ambiguous in the substituted note")
        region = prompt_text[note_span[0] : note_span[1]]
        found = list(re.finditer(_value_pattern(value), region))
        if len(found) != 1:
            raise ValueError("note_value token span is missing or ambiguous in the substituted note")
        offset = note_span[0] + found[0].start()
        tokens, repaired = _token_span(
            tokenizer, prompt_text, ids, (offset, offset + len(value)), label="note_value_tokens"
        )
        repairs["note_value_tokens"] += repaired
        spans[value] = tokens
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


def _note_values(note: str) -> list[str]:
    """Return numeric fact values, excluding labels, ordinals, and action expressions."""
    thought = note.split("\n```json", maxsplit=1)[0]
    number = r"-?\d+(?:\.\d+)?"
    values: list[str] = []
    for match in re.finditer(
        r"\b(?:approved|values so far|first half|second half)\s*:\s*([^;.\n]+)",
        thought,
        flags=re.IGNORECASE,
    ):
        values.extend(re.findall(number, match.group(1)))
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
    layer: int,
    source_rows: Any,
    target_positions: Sequence[int],
    failing_ids: Sequence[int],
    keep_last: int,
    max_tokens: int,
) -> bool:
    with InjectionHook(
        view,
        layer - 1,
        source_rows,
        at_positions=target_positions,
        replace=True,
    ):
        raw = greedy_generate(view, tokenizer, failing_ids, max_tokens=max_tokens)
    try:
        _thinking, cleaned = strip_thinking(raw)
        turn = parse_turn(cleaned)
    except Exception:
        return False
    scored = [dict(step) for step in case.failing_steps]
    scored[case.decision_step]["thought"] = turn.thought
    return _is_flip(case.task, scored, case.decision_step, keep_last=keep_last)


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
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the P6 cells and return aggregate-only, JSON-safe records.

    The headline cells run over scoring-version-stable cases only (R24); unstable cases
    are listed under ``excluded_cases`` with both judgements and never merged into a cell.
    This seam is deliberately composed from fakes in tests; the CLI is the only real-model
    entry point and remains gated by its GPU guard.

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
    stable = [case for case in cases if case.scoring_version_stable]
    unstable = [case for case in cases if not case.scoring_version_stable]
    if len(stable) < 2:
        raise ValueError(
            "P6 unrelated-task control requires at least two scoring-version-stable patch "
            f"cases (R24); {len(stable)} stable, {len(unstable)} unstable"
        )

    def case_record(case: PatchCase, provenance: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": case.task.task_id,
            "decision_step": case.decision_step,
            **provenance,
            **case.scoring_record(),
        }

    excluded: list[dict[str, Any]] = []
    for case in unstable:
        _failing, _counterfactual, provenance = replay_counterfactual(case)
        excluded.append(
            {**case_record(case, provenance), "excluded_reason": "scoring_version_unstable"}
        )
    prepared: list[dict[str, Any]] = []
    case_provenance: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for case_number, case in enumerate(stable, start=1):
        failing, counterfactual, provenance = replay_counterfactual(case)
        source = provenance.get("counterfactual_source") or "none"
        source_counts[source] = source_counts.get(source, 0) + 1
        failing_prompt = build_prompt(tokenizer, failing, spec=spec, keep_last=keep_last)
        counter_prompt = build_prompt(tokenizer, counterfactual, spec=spec, keep_last=keep_last)
        failing_ids = list(tokenizer.encode(failing_prompt, add_special_tokens=False))
        counter_ids = list(tokenizer.encode(counter_prompt, add_special_tokens=False))
        failing_groups, failing_repairs, failing_values = _groups_for(
            tokenizer, failing_ids, failing, prompt_text=failing_prompt, keep_last=keep_last
        )
        counter_groups, counter_repairs, counter_values = _groups_for(
            tokenizer, counter_ids, counterfactual, prompt_text=counter_prompt, keep_last=keep_last
        )
        alignment = align_groups(
            tokenizer,
            source_groups=counter_groups,
            source_values=counter_values,
            source_ids=counter_ids,
            target_groups=failing_groups,
            target_values=failing_values,
            target_ids=failing_ids,
        )
        alignment_record = {group: cell.record() for group, cell in alignment.items()}
        case_provenance.append(
            {
                **case_record(case, provenance),
                "boundary_repairs": {"failing": failing_repairs, "counterfactual": counter_repairs},
                "alignment": alignment_record,
            }
        )
        prepared.append(
            {
                "case": case,
                "failing_ids": failing_ids,
                "counter_ids": counter_ids,
                "failing_groups": failing_groups,
                "counter_groups": counter_groups,
                "alignment": alignment,
                "alignment_record": alignment_record,
                "failing_residuals": capture_residuals(view, failing_ids, layers, positions="all"),
                "counter_residuals": capture_residuals(view, counter_ids, layers, positions="all"),
            }
        )
        if progress is not None:
            progress(case_number, len(stable), f"capture {case.task.task_id}")
    cells: dict[str, dict[str, Any]] = {}
    cell_total = len(layers) * len(POSITION_GROUPS)
    cell_number = 0
    for layer in layers:
        for group in POSITION_GROUPS:
            outcomes = {name: {} for name in ("treatment", *CONTROLS)}
            for index, item in enumerate(prepared):
                case = item["case"]
                alignment = item["alignment"][group]
                target = alignment.target_positions
                treatment_rows = _alignment_rows(item["counter_residuals"][layer], alignment)
                if treatment_rows.shape[0] != len(target):
                    raise ValueError("treatment source and target group cardinality must match")
                outcomes["treatment"][case.task.task_id] = [_score_patch(
                    view, tokenizer, case, layer=layer, source_rows=treatment_rows,
                    target_positions=target, failing_ids=item["failing_ids"], keep_last=keep_last,
                    max_tokens=max_tokens,
                )]
                unrelated = prepared[(index + 1) % len(prepared)]
                unrelated_rows = _match_rows(
                    _alignment_rows(unrelated["counter_residuals"][layer], unrelated["alignment"][group]),
                    len(target),
                )
                outcomes["unrelated_task"][case.task.task_id] = [_score_patch(
                    view, tokenizer, case, layer=layer, source_rows=unrelated_rows,
                    target_positions=target, failing_ids=item["failing_ids"], keep_last=keep_last,
                    max_tokens=max_tokens,
                )]
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
                outcomes["random_positions"][case.task.task_id] = [_score_patch(
                    view, tokenizer, case, layer=layer, source_rows=random_rows,
                    target_positions=random_target, failing_ids=item["failing_ids"], keep_last=keep_last,
                    max_tokens=max_tokens,
                )]
            cells[f"{layer}:{group}"] = {
                "layer": layer,
                "group": group,
                # R25(d): every cell records the alignment rule, both cardinalities, and the
                # residue or slot detail, per case, so the heat map is never read without it.
                "alignment": {
                    item["case"].task.task_id: item["alignment_record"][group]
                    for item in prepared
                },
                "treatment": aggregate_task_flips(outcomes["treatment"]),
                "controls": {
                    control: aggregate_task_flips(outcomes[control]) for control in CONTROLS
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
        "seed": seed,
        "command": list(command),
        "groups": list(POSITION_GROUPS),
        "controls": list(CONTROLS),
        "selected_task_ids": [case.task.task_id for case in cases],
        "scoring_generator_version": GENERATOR_VERSION,
        "stable_cases": len(stable),
        "unstable_cases": len(unstable),
        "headline_task_ids": [case.task.task_id for case in stable],
        "cases": case_provenance,
        "excluded_cases": excluded,
        "counterfactual_sources": source_counts,
        "cells": cells,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact layer×group treatment table and named-control tables.

    The heat map covers scoring-version-stable cases only; unstable cases get their own
    table so they are never read as part of the headline (R24).
    """
    groups = payload["groups"]
    lines = ["# P6 causal patching", "", "## Treatment flip rate", ""]
    if "stable_cases" in payload:
        lines.extend(
            [
                f"Headline over {payload['stable_cases']} scoring-version-stable case(s); "
                f"{payload['unstable_cases']} unstable case(s) excluded (R24).",
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
        lines.extend(["", f"## Control: {control}", "", "| layer/group | rate | 95% Wilson |", "|---|---:|---|"])
        for key, cell in payload["cells"].items():
            summary = cell["controls"][control]
            low, high = summary["wilson_95"]
            lines.append(f"| {key} | {summary['rate']:.3f} | [{low:.3f}, {high:.3f}] |")
    sources = payload.get("counterfactual_sources")
    if sources:
        lines.extend(["", "## Counterfactual note sources", ""])
        lines.extend(f"- {name}: {sources[name]}" for name in sorted(sources))
    alignment_lines = _alignment_table(payload.get("cases", []))
    if alignment_lines:
        lines.extend(["", "## Alignment (R25)", ""])
        lines.extend(alignment_lines)
    if "stable_cases" in payload:
        lines.extend(["", "## Scoring version stability (R24)", ""])
        lines.extend(_stability_table(payload.get("cases", []), "headline"))
        lines.extend(["", "### Unstable cases (excluded from the headline)", ""])
        if payload.get("excluded_cases"):
            lines.extend(_stability_table(payload["excluded_cases"], "excluded"))
        else:
            lines.append("None.")
    return "\n".join(lines)


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
        f"| task ({population}) | stable | decision step bound / HEAD | dropped values bound / HEAD |",
        "|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['task_id']} | {'yes' if record['scoring_version_stable'] else 'no'} | "
            f"{record['decision_step_bound']} / {record['decision_step_head']} | "
            f"{values(record['dropped_values_bound'])} / {values(record['dropped_values_head'])} |"
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
    try:
        cases, selection_provenance = select_patch_cases(
            _load_payload(args.passing_eval),
            _load_payload(args.failing_eval),
            keep_last=args.keep_last,
            data_seed=args.data_seed,
            generator_version=args.generator_version,
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
        markdown = render_markdown(payload)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "patch.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output / "patch.md").write_text(markdown + "\n", encoding="utf-8")
        log.info(
            "wrote",
            json=str(args.output / "patch.json"),
            md=str(args.output / "patch.md"),
        )
