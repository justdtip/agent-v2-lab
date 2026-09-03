from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.evaluate import mcnemar


def load_summaries(evals: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(evals.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary")
        if isinstance(summary, dict) and "success_rate" in summary:
            summary = dict(summary)
            summary["_file"] = path.name
            trajectories = payload.get("trajectories", [])
            summary["_outcomes"] = {
                str(record["task_id"]): bool(record.get("verdict", {}).get("success"))
                for record in trajectories
                if isinstance(record, dict) and isinstance(record.get("task_id"), str)
            }
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
            or left.get("difficulty") != right.get("difficulty")
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
            row.append(_rate(s.get("integrity", {}), "clean_rate"))
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
