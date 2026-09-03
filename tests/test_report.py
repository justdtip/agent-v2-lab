from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_llm_lab.pipeline.report import load_summaries, render


def _summary(
    label: str, *, split: str = "valid", difficulty: int = 1, data_seed: int = 17
) -> dict[str, object]:
    return {
        "label": label,
        "split": split,
        "difficulty": difficulty,
        "data_seed": data_seed,
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
    left = _summary("a")
    left.update({"model": "model-a", "adapter": "adapter-a"})
    right = _summary("b")
    right.update({"model": "model-b", "adapter": "adapter-b"})
    _write_eval(
        tmp_path / "a.json",
        left,
        {"valid-read-0000-clean": True, "valid-read-0012-clean": False},
    )
    _write_eval(
        tmp_path / "b.json",
        right,
        {"valid-read-0000-clean": False, "valid-read-0012-clean": True},
    )

    rendered = render(load_summaries(tmp_path))

    assert "1/2 (50%) [10%-90%]" in rendered
    assert "Paired McNemar" in rendered
    assert "a vs b: pairs 2, a-only 1, b-only 1, discordant 2, p=1" in rendered


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("id-less", {"difficulty": 1, "verdict": {"success": True}}),
        ("non-dict", None),
        (
            "invalid-difficulty",
            {"task_id": "invalid", "difficulty": True, "verdict": {"success": True}},
        ),
        (
            "duplicate",
            {
                "task_id": "valid-read-0000-clean",
                "difficulty": 1,
                "verdict": {"success": True},
            },
        ),
        (
            "missing-outcome",
            {"task_id": "missing", "difficulty": 1, "verdict": {}},
        ),
        (
            "non-boolean-outcome",
            {"task_id": "invalid", "difficulty": 1, "verdict": {"success": "yes"}},
        ),
    ],
)
def test_render_refuses_entire_cohort_for_malformed_trajectory(
    tmp_path: Path, name: str, extra: object
) -> None:
    """A malformed extra trajectory must not be silently skipped or coerced into a pair."""
    outcomes = {"valid-read-0000-clean": True, "valid-read-0012-clean": False}
    _write_eval(tmp_path / "a.json", _summary("a"), outcomes)
    payload = {
        "summary": _summary("b"),
        "trajectories": [
            {
                "task_id": task_id,
                "difficulty": 1,
                "verdict": {"success": success},
            }
            for task_id, success in outcomes.items()
        ]
        + [extra],
    }
    (tmp_path / f"b-{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    summaries = load_summaries(tmp_path)

    assert summaries[1]["_outcomes"] == {}
    assert "Paired McNemar" not in render(summaries)


@pytest.mark.parametrize("tasks", [True, -1, "2", 1, 3])
def test_render_refuses_pair_when_summary_task_count_is_invalid_or_mismatched(
    tmp_path: Path, tasks: object
) -> None:
    """Pairing requires a truthful, exact non-boolean task count for every cohort."""
    outcomes = {"valid-read-0000-clean": True, "valid-read-0012-clean": False}
    _write_eval(tmp_path / "a.json", _summary("a"), outcomes)
    invalid = _summary("b")
    invalid["tasks"] = tasks
    _write_eval(tmp_path / "b.json", invalid, outcomes)

    summaries = load_summaries(tmp_path)

    assert summaries[1]["_outcomes"] == {}
    assert "Paired McNemar" not in render(summaries)


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


def test_render_refuses_pairs_for_same_ids_with_different_seed_or_task_difficulty(
    tmp_path: Path,
) -> None:
    """Catch pairing that treats a seed-shifted or difficulty-shifted task string as identical."""
    task_id = "valid-read-0000-clean"
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    _write_eval(seed_dir / "a.json", _summary("seed-a", data_seed=17), {task_id: True})
    _write_eval(seed_dir / "b.json", _summary("seed-b", data_seed=18), {task_id: False})
    level_dir = tmp_path / "level"
    level_dir.mkdir()
    _write_eval(level_dir / "a.json", _summary("level-a"), {task_id: True})
    payload = {
        "summary": _summary("level-b"),
        "trajectories": [
            {
                "task_id": task_id,
                "difficulty": 2,
                "prompt": "different seed would produce different files",
                "verdict": {"success": False},
            }
        ],
    }
    (level_dir / "b.json").write_text(json.dumps(payload), encoding="utf-8")

    assert "Paired McNemar" not in render(load_summaries(seed_dir))
    assert "Paired McNemar" not in render(load_summaries(level_dir))


def test_render_adds_top_level_integrity_interval_to_nested_clean_rate() -> None:
    """Catch an integrity-clean column that discards its summary Wilson interval."""
    rich = _summary("rich")
    rich["clean_rate"] = 1.0
    rich["integrity"] = {"clean_rate": 0.5}
    rich["wilson_95"] = {**rich["wilson_95"], "integrity_clean": [0.2, 0.8]}

    assert "50% [20%-80%]" in render([rich])


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
