from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.evaluate import mcnemar


def _outcomes(trajectories: object, task_count: object) -> dict[tuple[str, int], bool]:
    """Return outcomes only when the complete reported cohort is trustworthy."""
    if (
        not isinstance(task_count, int)
        or isinstance(task_count, bool)
        or task_count < 0
        or not isinstance(trajectories, list)
    ):
        return {}
    outcomes = {}
    for record in trajectories:
        if not isinstance(record, dict):
            return {}
        task_id = record.get("task_id")
        difficulty = record.get("difficulty")
        verdict = record.get("verdict")
        success = verdict.get("success") if isinstance(verdict, dict) else None
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(difficulty, int)
            or isinstance(difficulty, bool)
            or difficulty < 0
            or not isinstance(success, bool)
        ):
            return {}
        identity = (task_id, difficulty)
        if identity in outcomes:
            return {}
        outcomes[identity] = success
    return outcomes if len(outcomes) == task_count else {}


def _skip_reason(payload: object) -> str | None:
    """Why this file contributes no row, or ``None`` when it contributes one.

    Every rejection here used to be a bare ``continue``, so a directory of eight evaluations
    could render as five rows and say nothing about the other three. The table is read as the
    run's result; a row missing from it is indistinguishable from a run that never happened.
    """
    if not isinstance(payload, dict):
        return "top level is not a JSON object"
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return "no summary object"
    if "success_rate" not in summary:
        return "summary records no success_rate"
    return None


def load_summaries(evals: Path) -> list[dict[str, Any]]:
    """Every renderable summary in ``evals``, with each skipped file named on stderr.

    stderr, not the table: the skipped file has no row to carry the note, and the report is
    printed to stdout by the ``report`` stage, so a redirected run keeps the two apart.
    """
    summaries = []
    for path in sorted(evals.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        reason = _skip_reason(payload)
        if reason is not None:
            print(f"report: skipped {path.name}: {reason}", file=sys.stderr)
            continue
        summary = dict(payload["summary"])
        summary["_file"] = path.name
        summary["_outcomes"] = _outcomes(payload.get("trajectories"), summary.get("tasks"))
        summaries.append(summary)
    return summaries


def _rate(summary: dict[str, Any], key: str) -> str:
    """Percentage cell; ``-`` for summaries written before the metric existed."""
    value = summary.get(key)
    return "-" if value is None else f"{value:.0%}"


def _interval(summary: dict[str, Any], key: str) -> str:
    value = summary.get("wilson_95", {}).get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return ""
    return f" [{float(value[0]):.0%}-{float(value[1]):.0%}]"


def _paired_rows(summaries: list[dict[str, Any]]) -> list[str]:
    rows = []
    for left, right in combinations(summaries, 2):
        left_outcomes = left.get("_outcomes", {})
        right_outcomes = right.get("_outcomes", {})
        if (
            not left_outcomes
            or left.get("split") != right.get("split")
            or left.get("data_seed") is None
            or left.get("data_seed") != right.get("data_seed")
            or left_outcomes.keys() != right_outcomes.keys()
        ):
            continue
        result = mcnemar(left_outcomes, right_outcomes)
        left_label = left.get("label") or left.get("_file", "?")
        right_label = right.get("label") or right.get("_file", "?")
        rows.append(
            f"{left_label} vs {right_label}: pairs {result['tasks']}, "
            f"a-only {result['a_only']}, b-only {result['b_only']}, "
            f"discordant {result['discordant']}, p={result['p_value']:g}"
        )
    return rows


def render(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "No evaluation summaries found."
    families = sorted({family for s in summaries for family in s.get("by_family", {})})
    has_integrity = any(isinstance(summary.get("integrity"), dict) for summary in summaries)
    head = ["run", "split", "success", "clean", "valid", "schema", "exec", "errors", "steps"]
    if has_integrity:
        head.append("integrity-clean")
    head.extend(family[:10] for family in families)
    rows = [head]
    for s in summaries:
        label = s.get("label") or s.get("_file", "?")
        split = f"{s.get('split', '?')}{'+stress' if s.get('stress') else ''}"
        row = [
            label,
            split,
            f"{s['successes']}/{s['tasks']} ({s['success_rate']:.0%}){_interval(s, 'success')}",
            f"{s['clean_rate']:.0%}{_interval(s, 'clean')}",
            f"{s['valid_action_rate']:.0%}{_interval(s, 'valid_actions')}",
            f"{_rate(s, 'schema_validity_rate')}{_interval(s, 'schema_validity')}",
            f"{_rate(s, 'executable_call_rate')}{_interval(s, 'executable_calls')}",
            str(s["tool_errors"]),
            f"{s['mean_steps']:.1f}",
        ]
        if has_integrity:
            row.append(
                f"{_rate(s.get('integrity', {}), 'clean_rate')}{_interval(s, 'integrity_clean')}"
            )
        for family in families:
            stats = s.get("by_family", {}).get(family)
            row.append(
                "-"
                if stats is None
                else f"{stats['successes']}/{stats['tasks']}{_interval(stats, 'success')}"
            )
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    paired = _paired_rows(summaries)
    if paired:
        lines.extend(["", "Paired McNemar", *paired])
    return "\n".join(lines)
