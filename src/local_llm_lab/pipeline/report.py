from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summaries(evals: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(evals.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary")
        if isinstance(summary, dict) and "success_rate" in summary:
            summary = dict(summary)
            summary["_file"] = path.name
            summaries.append(summary)
    return summaries


def _rate(summary: dict[str, Any], key: str) -> str:
    """Percentage cell; ``-`` for summaries written before the metric existed."""
    value = summary.get(key)
    return "-" if value is None else f"{value:.0%}"


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
        label = s.get("label", s["_file"])
        split = f"{s.get('split', '?')}{'+stress' if s.get('stress') else ''}"
        row = [
            label,
            split,
            f"{s['successes']}/{s['tasks']} ({s['success_rate']:.0%})",
            f"{s['clean_rate']:.0%}",
            f"{s['valid_action_rate']:.0%}",
            _rate(s, "schema_validity_rate"),
            _rate(s, "executable_call_rate"),
            str(s["tool_errors"]),
            f"{s['mean_steps']:.1f}",
        ]
        if has_integrity:
            row.append(_rate(s.get("integrity", {}), "clean_rate"))
        for family in families:
            stats = s.get("by_family", {}).get(family)
            row.append("-" if stats is None else f"{stats['successes']}/{stats['tasks']}")
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
