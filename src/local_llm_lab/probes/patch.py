"""Gated, fake-testable causal patching at the dropped-value decision."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections.abc import Sequence
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
)
from local_llm_lab.pipeline.tasks import Task, render_expert_note, task_from_id
from local_llm_lab.probes.capture import InjectionHook, capture_residuals
from local_llm_lab.probes.policies import resolve_policy

POSITION_GROUPS = (
    "system_prompt",
    "task_prompt",
    "previous_notes",
    "note_value_tokens",
    "last_two_observations",
    "final_token",
)
CONTROLS = ("unrelated_task", "random_positions")
_FAMILIES = frozenset({"aggregate_report", "ledger_reconcile"})


@dataclass(frozen=True)
class PatchCase:
    task: Task
    decision_step: int
    failing_steps: tuple[dict[str, Any], ...]


def _records(payload: dict[str, Any], name: str) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} evaluation must be a mapping")
    records = payload.get("trajectories")
    seed = payload.get("data_seed")
    if not isinstance(records, list):
        raise ValueError(f"{name} evaluation must contain trajectories")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{name} evaluation must contain integer data_seed")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{name} trajectories must be mappings")
    return records, seed


def _value_drop_step(record: dict[str, Any]) -> int | None:
    integrity = record.get("integrity")
    violations = integrity.get("violations") if isinstance(integrity, dict) else None
    if not isinstance(violations, list):
        return None
    found: list[int] = []
    for violation in violations:
        if isinstance(violation, dict) and violation.get("kind") == "value_drop":
            step = violation.get("step")
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                found.append(step)
    return min(found) if found else None


def select_patch_cases(
    passing_payload: dict[str, Any],
    failing_payload: dict[str, Any],
    *,
    keep_last: int,
) -> list[PatchCase]:
    """Select eligible B-pass/C-value-drop task ids and reconstruct C's task declaration."""
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")
    passing, _passing_seed = _records(passing_payload, "passing")
    failing, failing_seed = _records(failing_payload, "failing")
    successful = {
        record.get("task_id")
        for record in passing
        if isinstance(record.get("task_id"), str)
        and isinstance(record.get("verdict"), dict)
        and record["verdict"].get("success") is True
    }
    selected: list[PatchCase] = []
    for record in failing:
        task_id = record.get("task_id")
        difficulty = record.get("difficulty")
        steps = record.get("steps")
        dropped = _value_drop_step(record)
        if (
            not isinstance(task_id, str)
            or task_id not in successful
            or isinstance(difficulty, bool)
            or not isinstance(difficulty, int)
            or not isinstance(steps, list)
            or dropped is None
        ):
            continue
        task = task_from_id(task_id, failing_seed, difficulty)
        if task.family not in _FAMILIES:
            continue
        if dropped >= len(steps) or not all(isinstance(step, dict) for step in steps):
            raise ValueError(f"{task_id}: malformed failing steps")
        selected.append(PatchCase(task, dropped, tuple(dict(step) for step in steps)))
    return selected


def _find_once(ids: Sequence[int], needle: Sequence[int], *, start: int, label: str) -> tuple[int, ...]:
    if not needle:
        return ()
    matches = [
        index
        for index in range(start, len(ids) - len(needle) + 1)
        if list(ids[index : index + len(needle)]) == list(needle)
    ]
    if not matches:
        raise ValueError(f"missing token span for {label}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous token span for {label}")
    return tuple(range(matches[0], matches[0] + len(needle)))


def position_groups(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    system_text: str,
    task_text: str,
    previous_notes: Sequence[str],
    note_values: Sequence[str],
    observations: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    """Locate the six P6 token groups, refusing missing or ambiguous content spans."""
    ids = [int(value) for value in prompt_ids]

    def encoded(text: str) -> list[int]:
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))

    system = _find_once(ids, encoded(system_text), start=0, label="system_prompt")
    task = _find_once(ids, encoded(task_text), start=system[-1] + 1 if system else 0, label="task_prompt")
    note_spans = [
        _find_once(ids, encoded(note), start=task[-1] + 1 if task else 0, label="previous_note")
        for note in previous_notes
    ]
    previous = tuple(position for span in note_spans for position in span)
    note_region = note_spans[-1] if note_spans else ()
    values: list[int] = []
    for value in note_values:
        needle = encoded(value)
        starts = range(note_region[0], note_region[-1] - len(needle) + 2) if note_region else ()
        matches = [start for start in starts if ids[start : start + len(needle)] == needle]
        if len(matches) != 1:
            raise ValueError("note_value token span is missing or ambiguous in the substituted note")
        span = tuple(range(matches[0], matches[0] + len(needle)))
        values.extend(span)
    observation_spans = [
        _find_once(ids, encoded(observation), start=task[-1] + 1 if task else 0, label="observation")
        for observation in observations
    ]
    last_observations = tuple(position for span in observation_spans[-2:] for position in span)
    if not ids:
        raise ValueError("prompt token ids must not be empty")
    return {
        "system_prompt": system,
        "task_prompt": task,
        "previous_notes": previous,
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


def replay_counterfactual(case: PatchCase) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay C before the decision, replacing only its immediately prior note."""
    failing = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": case.task.prompt}]
    counterfactual = [dict(message) for message in failing]
    for position, record in enumerate(case.failing_steps[: case.decision_step]):
        action = _action(record)
        thought = record.get("thought")
        if not isinstance(thought, str):
            raise ValueError("failing step is missing thought")
        replacement = (
            render_expert_note(case.task, position)
            if position == case.decision_step - 1
            else thought
        )
        failing.append(assistant_message(thought, action))
        counterfactual.append(assistant_message(replacement, action))
        observation = record.get("observation")
        if isinstance(observation, str):
            message = tool_message(action.name, observation)
            failing.append(message)
            counterfactual.append(dict(message))
    return failing, counterfactual


def _is_flip(task: Task, steps: list[dict[str, Any]], decision_step: int, *, keep_last: int) -> bool:
    report = check_trajectory(task, steps, keep_last=keep_last)
    return not any(
        violation.kind == "value_drop" and violation.step == decision_step
        for violation in report.violations
    )


def _message_contents(messages: Sequence[dict[str, Any]]) -> tuple[str, str, list[str], list[str]]:
    """Extract canonical group inputs without altering the rendered prompt."""
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
        for message in messages
        if message.get("role") == "tool" and isinstance(message.get("content"), str)
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


def _groups_for(tokenizer: Any, token_ids: Sequence[int], messages: Sequence[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    system, task, notes, observations = _message_contents(messages)
    values = _note_values(notes[-1]) if notes else []
    groups = position_groups(
        tokenizer,
        token_ids,
        system_text=system,
        task_text=task,
        previous_notes=notes,
        note_values=values,
        observations=observations,
    )
    if tuple(groups) != POSITION_GROUPS or any(not groups[name] for name in POSITION_GROUPS):
        raise ValueError("each P6 position group must resolve to at least one token")
    return groups


def _take_rows(rows: Any, positions: Sequence[int]) -> Any:
    import mlx.core as mx

    if not positions:
        raise ValueError("P6 position group must not be empty")
    return mx.take(rows, mx.array(tuple(positions), dtype=mx.int32), axis=0)


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
) -> dict[str, Any]:
    """Run the P6 cells and return aggregate-only, JSON-safe records.

    This seam is deliberately composed from fakes in tests; the CLI is the only real-model
    entry point and remains gated by its GPU guard.
    """
    view = ArchitectureView.from_model(model)
    if not cases:
        raise ValueError("no eligible patch cases")
    if not layers or any(layer < 1 or layer > view.num_layers for layer in layers):
        raise ValueError("layers must be residual indices in [1, num_layers]")
    if len(cases) < 2:
        raise ValueError("P6 unrelated-task control requires at least two patch cases")
    prepared: list[dict[str, Any]] = []
    for case in cases:
        failing, counterfactual = replay_counterfactual(case)
        failing_prompt = build_prompt(tokenizer, failing, spec=spec)
        counter_prompt = build_prompt(tokenizer, counterfactual, spec=spec)
        failing_ids = list(tokenizer.encode(failing_prompt, add_special_tokens=False))
        counter_ids = list(tokenizer.encode(counter_prompt, add_special_tokens=False))
        prepared.append(
            {
                "case": case,
                "failing_ids": failing_ids,
                "counter_ids": counter_ids,
                "failing_groups": _groups_for(tokenizer, failing_ids, failing),
                "counter_groups": _groups_for(tokenizer, counter_ids, counterfactual),
                "failing_residuals": capture_residuals(view, failing_ids, layers, positions="all"),
                "counter_residuals": capture_residuals(view, counter_ids, layers, positions="all"),
            }
        )
    cells: dict[str, dict[str, Any]] = {}
    for layer in layers:
        for group in POSITION_GROUPS:
            outcomes = {name: {} for name in ("treatment", *CONTROLS)}
            for index, item in enumerate(prepared):
                case = item["case"]
                target = item["failing_groups"][group]
                source = item["counter_groups"][group]
                if len(source) != len(target):
                    raise ValueError("treatment source and target group cardinality must match")
                treatment_rows = _take_rows(item["counter_residuals"][layer], source)
                outcomes["treatment"][case.task.task_id] = [_score_patch(
                    view, tokenizer, case, layer=layer, source_rows=treatment_rows,
                    target_positions=target, failing_ids=item["failing_ids"], keep_last=keep_last,
                    max_tokens=max_tokens,
                )]
                unrelated = prepared[(index + 1) % len(prepared)]
                unrelated_rows = _match_rows(
                    _take_rows(unrelated["counter_residuals"][layer], unrelated["counter_groups"][group]),
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
                    source_treatment=source,
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
                "treatment": aggregate_task_flips(outcomes["treatment"]),
                "controls": {
                    control: aggregate_task_flips(outcomes[control]) for control in CONTROLS
                },
            }
    return {
        "model": resolved.as_dict(),
        "policy": policy,
        "layers": list(layers),
        "seed": seed,
        "command": list(command),
        "groups": list(POSITION_GROUPS),
        "controls": list(CONTROLS),
        "selected_task_ids": [case.task.task_id for case in cases],
        "cells": cells,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact layer×group treatment table and named-control tables."""
    groups = payload["groups"]
    lines = ["# P6 causal patching", "", "## Treatment flip rate", "", "| layer | " + " | ".join(groups) + " |", "|---|" + "|".join("---" for _ in groups) + "|"]
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
    return "\n".join(lines)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"evaluation {path} must be a JSON object")
    return value


def _parse_layers(raw: str | None, resolved: ResolvedSpec) -> tuple[int, ...]:
    if raw is None:
        return resolved.probe_layers
    values: list[int] = []
    for part in raw.split(","):
        try:
            number = float(part.strip())
        except ValueError as error:
            raise ValueError("--layers must be comma-separated indices or fractions") from error
        layer = round(number * resolved.num_layers) if 0 < number <= 1 and number != int(number) else int(number)
        if not 1 <= layer <= resolved.num_layers:
            raise ValueError(f"layer {part!r} is outside [1, {resolved.num_layers}]")
        values.append(layer)
    if not values:
        raise ValueError("--layers must not be empty")
    return tuple(dict.fromkeys(values))


def _validate_layer_syntax(raw: str | None) -> None:
    """Reject malformed layer text before any GPU or model-loading action."""
    if raw is None:
        return
    parts = raw.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("--layers must be comma-separated indices or fractions")
    for part in parts:
        try:
            number = float(part.strip())
        except ValueError as error:
            raise ValueError("--layers must be comma-separated indices or fractions") from error
        if not math.isfinite(number) or number <= 0 or (number > 1 and not number.is_integer()):
            raise ValueError("--layers must be positive indices or fractions in (0, 1]")


def main() -> None:
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu

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
    add_gpu_arguments(parser)
    args = parser.parse_args()
    if not args.passing_eval.is_file() or not args.failing_eval.is_file():
        parser.error("--passing-eval and --failing-eval must name existing files")
    if args.keep_last < 0 or args.max_tokens <= 0:
        parser.error("--keep-last must be non-negative and --max-tokens must be positive")
    try:
        _validate_layer_syntax(args.layers)
    except ValueError as error:
        parser.error(str(error))
    try:
        cases = select_patch_cases(
            _load_payload(args.passing_eval), _load_payload(args.failing_eval), keep_last=args.keep_last
        )
    except ValueError as error:
        parser.error(str(error))
    if not cases:
        parser.error("no eligible patch cases")
    spec = load_model_spec(args.model)
    require_idle_gpu(parser, args, "running P6 causal patching")
    adapter = resolve_policy(args.policy)
    model, tokenizer = load_policy(spec.hf_id, adapter)
    view = ArchitectureView.from_model(model)
    resolved = spec.resolve(model, tokenizer)
    try:
        layers = _parse_layers(args.layers, resolved)
    except ValueError as error:
        parser.error(str(error))
    del view
    payload = run_patch_probe(
        model, tokenizer, cases, spec=spec, resolved=resolved, layers=layers, policy=args.policy,
        keep_last=args.keep_last, max_tokens=args.max_tokens, seed=args.seed, command=sys.argv,
    )
    if not isinstance(payload, dict) or "cells" not in payload:
        parser.error("run_patch_probe returned an invalid payload")
    markdown = render_markdown(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "patch.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "patch.md").write_text(markdown + "\n", encoding="utf-8")
