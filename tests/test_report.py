from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from local_llm_lab.models import ResolvedSpec, load_model_spec
from local_llm_lab.pipeline.evaluate import evaluation_metadata, summarize, write_report
from local_llm_lab.pipeline.report import load_summaries, render
from local_llm_lab.pipeline.runner import Trajectory

_PAIR = {"valid-read-0000-clean": True, "valid-read-0012-clean": False}


def _resolved() -> ResolvedSpec:
    """The record ``evaluation_metadata`` puts under ``summary.model``, as the real class.

    ``ModelSpec.resolve`` needs loaded weights; this suite has none, so the dataclass is built
    directly over a real registry declaration.  ``report.py`` never reads it -- it is here so
    the summary this suite writes is the summary a run writes, field for field.
    """
    return ResolvedSpec(
        spec=load_model_spec("qwen35-4b"),
        num_layers=4,
        hidden_size=8,
        vocab_size=32,
        tie_word_embeddings=True,
        layer_types=("full_attention",) * 4,
        lora_keys=("self_attn.q_proj",),
        trainable_parameters=64,
        probe_layers=(1, 2, 3),
        cache_strategy="none",
        cache_strategy_reason="explicit:none",
        snapshot_revision=None,
        jvp_method="untested",
    )


def _cohort(outcomes: dict[str, bool], difficulty: int) -> list[Trajectory]:
    """One real ``Trajectory`` per outcome, as the runner hands them to ``write_report``.

    ``_outcomes`` reads ``task_id``, ``difficulty`` and ``verdict.success`` off the serialised
    record, and the serialisation is ``Trajectory.as_dict`` plus the two fields
    ``write_report`` overlays, so the objects have to be the real dataclass for the record to
    have the real shape.
    """
    return [
        Trajectory(
            task_id=task_id,
            family="read",
            variant="clean",
            label="x",
            prompt="p",
            difficulty=difficulty,
            turns=2,
            valid_turns=2,
            verdict={
                "success": success,
                "clean": success,
                "errors": 0,
                "recovered_errors": 0,
                "reasons": [],
                "calls": 1,
                "schema_failures": 0,
                "executable_calls": 1,
            },
        )
        for task_id, success in outcomes.items()
    ]


def _summary(
    label: str,
    *,
    split: str = "valid",
    difficulty: int = 1,
    data_seed: int = 17,
    outcomes: dict[str, bool] | None = None,
) -> dict[str, object]:
    """The summary a real evaluation records for this cohort (R38).

    ``summarize`` produces the rates and intervals ``render`` formats; ``evaluation_metadata``
    produces the identity keys ``load_summaries`` and ``_paired_rows`` select on --
    ``label``, ``split``, ``data_seed``, ``difficulty``.  Neither is restated here, so a
    renamed or relevelled key takes this suite down instead of quietly rendering ``-``.
    """
    trajectories = _cohort(_PAIR if outcomes is None else outcomes, difficulty)
    summary = summarize(trajectories)
    summary.update(
        evaluation_metadata(
            label=label,
            resolved=_resolved(),
            adapter=None,
            split=split,
            difficulties=[difficulty],
            stress=False,
            temperature=0.0,
            keep_last=2,
            seed=data_seed,
            use_cache=True,
            elapsed_seconds=0.0,
        )
    )
    return summary


def _write_eval(path: Path, summary: dict[str, object], outcomes: dict[str, bool]) -> None:
    """Write the artifact through ``write_report``, the function that writes the real ones.

    ``summary`` stays a parameter because several tests below pair a summary against a cohort
    that does not match it -- the exact disagreement ``_outcomes`` exists to refuse.
    """
    write_report(path, summary, _cohort(outcomes, int(summary.get("difficulty") or 0)))


def _single_task_summary(
    label: str, *, split: str = "valid", difficulty: int = 1, data_seed: int = 17
) -> dict[str, object]:
    return _summary(
        label,
        split=split,
        difficulty=difficulty,
        data_seed=data_seed,
        outcomes={"valid-read-0000-clean": True},
    )


def _write_records(
    path: Path, summary: dict[str, object], trajectories: list[object]
) -> None:
    """A real artifact whose trajectory list is then replaced by records no writer emits.

    The malformed records below -- a missing ``task_id``, a bare ``None``, a boolean
    difficulty -- cannot come from ``write_report``, which is the point: they are what a
    corrupted or truncated file looks like, and ``_outcomes`` must refuse the whole cohort.
    So the file is written for real and only the list is swapped, leaving the summary the
    reader selects on exactly as a run recorded it.
    """
    _write_eval(path, summary, {})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trajectories"] = trajectories
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cells(line: str) -> list[str]:
    """Split a report row on its two-space column separator (cells may contain one space)."""
    return re.split(r" {2,}", line.strip())


def test_report_table_lists_runs_and_families(tmp_path: Path) -> None:
    _write_eval(tmp_path / "x.json", _summary("x", outcomes={"a": True}), {"a": True})

    table = render(load_summaries(tmp_path))

    assert "x" in table and "read" in table and "1/1 (100%) [21%-100%]" in table
    header = _cells(table.splitlines()[0])
    assert header[:9] == [
        "run",
        "split",
        "success",
        "clean",
        "valid",
        "schema",
        "exec",
        "errors",
        "steps",
    ]
    cells = _cells(table.splitlines()[2])
    assert cells[2] == "1/1 (100%) [21%-100%]"
    assert cells[5] == "100% [21%-100%]"
    assert cells[6] == "100% [21%-100%]"


def test_render_shows_intervals_and_exact_paired_mcnemar_for_matching_evaluations(
    tmp_path: Path,
) -> None:
    """Catch interval-free rich summaries and a paired row with missing flip accounting."""
    left = _summary("a")
    right = _summary("b")
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

    # The real Wilson interval for 1 of 2, not the round [10%-90%] the hand-made summary
    # asserted: that number was invented to match itself and could not have caught a
    # renderer reading the wrong ``wilson_95`` entry.
    assert "1/2 (50%) [9%-91%]" in rendered
    assert "Paired McNemar" in rendered
    assert "a vs b: pairs 2, a-only 1, b-only 1, discordant 2, p=1" in rendered


@pytest.mark.parametrize(
    ("name", "left_records", "right_records", "empty_outcomes"),
    [
        (
            "id-less",
            [{"task_id": "valid", "difficulty": 1, "verdict": {"success": True}}],
            [
                {"task_id": "valid", "difficulty": 1, "verdict": {"success": True}},
                {"difficulty": 1, "verdict": {"success": True}},
            ],
            [False, True],
        ),
        (
            "non-dict",
            [{"task_id": "valid", "difficulty": 1, "verdict": {"success": True}}],
            [
                {"task_id": "valid", "difficulty": 1, "verdict": {"success": True}},
                None,
            ],
            [False, True],
        ),
        (
            "invalid-difficulty",
            [{"task_id": "valid", "difficulty": True, "verdict": {"success": True}}],
            [{"task_id": "valid", "difficulty": True, "verdict": {"success": False}}],
            [True, True],
        ),
        (
            "duplicate",
            [
                {"task_id": "valid", "difficulty": 1, "verdict": {"success": True}},
                {"task_id": "valid", "difficulty": 1, "verdict": {"success": False}},
            ],
            [
                {"task_id": "valid", "difficulty": 1, "verdict": {"success": False}},
                {"task_id": "valid", "difficulty": 1, "verdict": {"success": True}},
            ],
            [True, True],
        ),
        (
            "missing-outcome",
            [{"task_id": "valid", "difficulty": 1, "verdict": {}}],
            [{"task_id": "valid", "difficulty": 1, "verdict": {}}],
            [True, True],
        ),
        (
            "non-boolean-outcome",
            [{"task_id": "valid", "difficulty": 1, "verdict": {"success": "yes"}}],
            [{"task_id": "valid", "difficulty": 1, "verdict": {"success": "yes"}}],
            [True, True],
        ),
    ],
)
def test_render_refuses_entire_cohort_for_malformed_trajectory(
    tmp_path: Path,
    name: str,
    left_records: list[object],
    right_records: list[object],
    empty_outcomes: list[bool],
) -> None:
    """Each malformed identity/outcome alone must fail closed for an otherwise valid pair."""
    _write_records(tmp_path / f"a-{name}.json", _single_task_summary("a"), left_records)
    _write_records(tmp_path / f"b-{name}.json", _single_task_summary("b"), right_records)

    summaries = load_summaries(tmp_path)

    assert [summary["_outcomes"] == {} for summary in summaries] == empty_outcomes
    assert "Paired McNemar" not in render(summaries)


def test_render_refuses_pair_when_summary_task_count_is_boolean(tmp_path: Path) -> None:
    """A bool must not masquerade as the exact count of one valid trajectory."""
    task_id = "valid-read-0000-clean"
    left = _single_task_summary("a")
    right = _single_task_summary("b")
    left["tasks"] = True
    right["tasks"] = True
    _write_eval(tmp_path / "a.json", left, {task_id: True})
    _write_eval(tmp_path / "b.json", right, {task_id: False})

    summaries = load_summaries(tmp_path)

    assert all(summary["_outcomes"] == {} for summary in summaries)
    assert "Paired McNemar" not in render(summaries)


@pytest.mark.parametrize("tasks", [-1, "2", 1, 3])
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
    ids_dir = tmp_path / "ids"
    ids_dir.mkdir()
    _write_eval(ids_dir / "a.json", _single_task_summary("a"), {"valid-read-0000-clean": True})
    _write_eval(ids_dir / "b.json", _single_task_summary("b"), {"valid-read-0001-clean": False})
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    _write_eval(split_dir / "a.json", _single_task_summary("a"), {"valid-read-0000-clean": True})
    _write_eval(
        split_dir / "b.json",
        _single_task_summary("b", split="test"),
        {"valid-read-0000-clean": True},
    )
    difficulty_dir = tmp_path / "difficulty"
    difficulty_dir.mkdir()
    _write_eval(
        difficulty_dir / "a.json", _single_task_summary("a"), {"valid-read-0000-clean": True}
    )
    _write_eval(
        difficulty_dir / "b.json",
        _single_task_summary("b", difficulty=2),
        {"valid-read-0000-clean": True},
    )

    assert "Paired McNemar" not in render(load_summaries(ids_dir))
    assert "Paired McNemar" not in render(load_summaries(split_dir))
    assert "Paired McNemar" not in render(load_summaries(difficulty_dir))


def test_render_refuses_pairs_for_same_ids_with_different_seed_or_task_difficulty(
    tmp_path: Path,
) -> None:
    """Catch pairing that treats a seed-shifted or difficulty-shifted task string as identical."""
    task_id = "valid-read-0000-clean"
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    _write_eval(seed_dir / "a.json", _single_task_summary("seed-a", data_seed=17), {task_id: True})
    _write_eval(seed_dir / "b.json", _single_task_summary("seed-b", data_seed=18), {task_id: False})
    level_dir = tmp_path / "level"
    level_dir.mkdir()
    _write_eval(level_dir / "a.json", _single_task_summary("level-a"), {task_id: True})
    payload = {
        "summary": _single_task_summary("level-b"),
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


@pytest.mark.parametrize("missing", [True, False], ids=["missing", "null"])
def test_render_refuses_pair_for_missing_or_null_data_seed(tmp_path: Path, missing: bool) -> None:
    """A matching cohort without a real data seed has no trustworthy pairing identity."""
    task_id = "valid-read-0000-clean"
    left = _single_task_summary("a")
    right = _single_task_summary("b")
    if missing:
        left.pop("data_seed")
        right.pop("data_seed")
    else:
        left["data_seed"] = None
        right["data_seed"] = None
    _write_eval(tmp_path / "a.json", left, {task_id: True})
    _write_eval(tmp_path / "b.json", right, {task_id: False})

    summaries = load_summaries(tmp_path)

    assert all(summary["_outcomes"] for summary in summaries)
    assert "Paired McNemar" not in render(summaries)


def test_render_refuses_pair_for_matching_negative_difficulty(tmp_path: Path) -> None:
    """The -1 unset sentinel is not a valid per-trajectory pairing identity."""
    task_id = "valid-read-0000-clean"
    left = _single_task_summary("a", difficulty=-1)
    right = _single_task_summary("b", difficulty=-1)
    _write_eval(tmp_path / "a.json", left, {task_id: True})
    _write_eval(tmp_path / "b.json", right, {task_id: False})

    summaries = load_summaries(tmp_path)

    assert all(summary["_outcomes"] == {} for summary in summaries)
    assert "Paired McNemar" not in render(summaries)


def test_render_adds_top_level_integrity_interval_to_nested_clean_rate() -> None:
    """Catch an integrity-clean column that discards its summary Wilson interval."""
    rich = _summary("rich")
    rich["clean_rate"] = 1.0
    rich["integrity"] = {"clean_rate": 0.5}
    rich["wilson_95"] = {**rich["wilson_95"], "integrity_clean": [0.2, 0.8]}

    assert "50% [20%-80%]" in render([rich])


def _paired_dir(tmp_path: Path, name: str, mutate=None) -> Path:
    """A directory holding one real, pairable pair of evaluations, optionally edited after."""
    directory = tmp_path / name
    directory.mkdir()
    _write_eval(directory / "a.json", _summary("a"), _PAIR)
    _write_eval(directory / "b.json", _summary("b"), {key: not v for key, v in _PAIR.items()})
    if mutate is not None:
        payload = json.loads((directory / "b.json").read_text(encoding="utf-8"))
        mutate(payload)
        (directory / "b.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_load_summaries_depends_on_each_key_at_the_level_the_writer_records_it(
    tmp_path: Path,
) -> None:
    """R38 on ``report.py:47-54``: move a key off its level and the pairing must notice.

    Every failure here is silent by construction -- ``load_summaries`` skips a payload it does
    not recognise and ``_outcomes`` returns ``{}``, so a relevelled key does not raise, it
    deletes a row or a whole run from the report.  That is why each case asserts on what
    disappears rather than on an exception.
    """
    assert "Paired McNemar" in render(load_summaries(_paired_dir(tmp_path, "intact")))

    # ``trajectories`` is a top-level sibling of ``summary``, not a member of it.
    nested = _paired_dir(
        tmp_path, "nested", lambda p: p["summary"].update(trajectories=p.pop("trajectories"))
    )
    assert [s["_outcomes"] == {} for s in load_summaries(nested)] == [False, True]
    assert "Paired McNemar" not in render(load_summaries(nested))

    # ``success_rate`` is the gate: without it at summary level the run is not a summary.
    ungated = _paired_dir(
        tmp_path, "ungated", lambda p: p.update(success_rate=p["summary"].pop("success_rate"))
    )
    assert [s["label"] for s in load_summaries(ungated)] == ["a"]

    # ``difficulty`` is overlaid onto the record by ``write_report``, not nested in the verdict.
    buried = _paired_dir(
        tmp_path,
        "buried",
        lambda p: [r["verdict"].update(difficulty=r.pop("difficulty")) for r in p["trajectories"]],
    )
    assert "Paired McNemar" not in render(load_summaries(buried))

    # ``data_seed`` is a summary key; at the top level the pair has no shared seed identity.
    unseeded = _paired_dir(
        tmp_path, "unseeded", lambda p: p.update(data_seed=p["summary"].pop("data_seed"))
    )
    assert all(s["_outcomes"] for s in load_summaries(unseeded))
    assert "Paired McNemar" not in render(load_summaries(unseeded))


def test_render_preserves_legacy_only_and_mixed_integrity_layouts() -> None:
    """Catch a richer renderer that changes the historical legacy-only columns.

    The mixed half only became mixed under R38.  ``_summary`` used to be hand-written and
    carried no ``integrity`` block, so the "new" run here was a second legacy row and the
    assertion below read ``not in`` -- the mixed layout this test is named for was never
    rendered.  ``summarize`` has recorded an ``integrity`` block for every run since, so the
    real summary raises the column and the legacy row fills it with ``-``.
    """
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
    assert "integrity-clean" in mixed
    # The legacy row has no integrity block, so its cell is the missing-metric dash.
    assert re.search(r"^legacy .*  -  ", mixed, re.MULTILINE)
    assert "1/2 (50%) [9%-91%]" in mixed


def test_load_summaries_names_every_file_it_skipped_and_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A2: a dropped row is reported, because a missing row is invisible in the table.

    Three rejections, each previously a bare ``continue``: a payload that is not an object, a
    payload with no summary object, and a summary with no ``success_rate``. A directory of
    four evaluations rendered as one row and said nothing about the other three, which is
    indistinguishable from a run that only ever produced one.
    """
    _write_eval(tmp_path / "kept.json", _summary("kept", outcomes={"a": True}), {"a": True})
    (tmp_path / "list.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    (tmp_path / "nosummary.json").write_text(json.dumps({"trajectories": []}), encoding="utf-8")
    without_rate = _summary("norate", outcomes={"a": True})
    without_rate.pop("success_rate")
    _write_eval(tmp_path / "norate.json", without_rate, {"a": True})

    summaries = load_summaries(tmp_path)

    assert [summary["_file"] for summary in summaries] == ["kept.json"]
    assert capsys.readouterr().err.splitlines() == [
        "report: skipped list.json: top level is not a JSON object",
        "report: skipped norate.json: summary records no success_rate",
        "report: skipped nosummary.json: no summary object",
    ]


def test_load_summaries_keeps_a_summary_whose_success_rate_the_writer_produced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R38 guard on the skip rule: ``summarize`` is what puts ``success_rate`` in the summary.

    If the writer renames or relevels that key, every artifact becomes a skipped row instead
    of silently rendering nothing -- and this test says so rather than passing on an empty
    directory.
    """
    _write_eval(tmp_path / "kept.json", _summary("kept", outcomes={"a": True}), {"a": True})
    payload = json.loads((tmp_path / "kept.json").read_text(encoding="utf-8"))
    assert "success_rate" in payload["summary"]

    assert len(load_summaries(tmp_path)) == 1
    assert capsys.readouterr().err == ""

    payload["summary"]["rate"] = payload["summary"].pop("success_rate")
    (tmp_path / "kept.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_summaries(tmp_path) == []
    assert capsys.readouterr().err.strip() == (
        "report: skipped kept.json: summary records no success_rate"
    )
