from __future__ import annotations

import json
from pathlib import Path

from local_llm_lab.pipeline.report import load_summaries, render


def _summary(label: str, *, split: str = "valid", difficulty: int = 1) -> dict[str, object]:
    return {
        "label": label,
        "split": split,
        "difficulty": difficulty,
        "tasks": 2,
        "successes": 1,
        "success_rate": 0.5,
        "clean_rate": 0.5,
        "valid_action_rate": 1.0,
        "schema_validity_rate": 1.0,
        "executable_call_rate": 1.0,
        "tool_errors": 0,
        "mean_steps": 2.0,
        "by_family": {"read": {"successes": 1, "tasks": 2}},
        "wilson_95": {
            "success": [0.1, 0.9],
            "clean": [0.1, 0.9],
            "valid_actions": [0.5, 1.0],
            "schema_validity": [0.5, 1.0],
            "executable_calls": [0.5, 1.0],
            "integrity_clean": [0.1, 0.9],
        },
    }


def _write_eval(path: Path, summary: dict[str, object], outcomes: dict[str, bool]) -> None:
    payload = {
        "summary": summary,
        "trajectories": [
            {
                "task_id": task_id,
                "difficulty": summary["difficulty"],
                "verdict": {"success": success},
            }
            for task_id, success in outcomes.items()
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_render_shows_intervals_and_exact_paired_mcnemar_for_matching_evaluations(
    tmp_path: Path,
) -> None:
    """Catch interval-free rich summaries and a paired row with missing flip accounting."""
    _write_eval(
        tmp_path / "a.json",
        _summary("a"),
        {"valid-read-0000-clean": True, "valid-read-0012-clean": False},
    )
    _write_eval(
        tmp_path / "b.json",
        _summary("b"),
        {"valid-read-0000-clean": False, "valid-read-0012-clean": True},
    )

    rendered = render(load_summaries(tmp_path))

    assert "1/2 (50%) [10%-90%]" in rendered
    assert "Paired McNemar" in rendered
    assert "a vs b: pairs 2, a-only 1, b-only 1, discordant 2, p=1" in rendered


def test_render_refuses_pairs_for_different_ids_split_or_difficulty(tmp_path: Path) -> None:
    """Catch fabricated paired comparisons across non-equivalent evaluation cohorts."""
    _write_eval(tmp_path / "ids.json", _summary("ids"), {"valid-read-0000-clean": True})
    _write_eval(
        tmp_path / "split.json",
        _summary("split", split="test"),
        {"valid-read-0000-clean": True},
    )
    _write_eval(
        tmp_path / "difficulty.json",
        _summary("difficulty", difficulty=2),
        {"valid-read-0000-clean": True},
    )

    assert "Paired McNemar" not in render(load_summaries(tmp_path))


def test_render_preserves_legacy_only_and_mixed_integrity_layouts() -> None:
    """Catch a richer renderer that changes the historical legacy-only columns."""
    legacy = {
        "_file": "legacy.json",
        "label": "legacy",
        "split": "test",
        "tasks": 1,
        "successes": 1,
        "success_rate": 1.0,
        "clean_rate": 1.0,
        "valid_action_rate": 1.0,
        "schema_validity_rate": 1.0,
        "executable_call_rate": 1.0,
        "tool_errors": 0,
        "mean_steps": 1.0,
        "by_family": {},
    }
    assert "95% CI" not in render([legacy])
    mixed = render([legacy, _summary("new")])
    assert "integrity-clean" not in mixed
    assert "1/2 (50%) [10%-90%]" in mixed
