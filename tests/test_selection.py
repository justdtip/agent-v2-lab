from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_llm_lab.pipeline import cli
from local_llm_lab.pipeline.tasks import FAMILIES, LONG_HORIZON_FAMILIES


def test_stage_select_refuses_empty_checkpoint_directory(tmp_path) -> None:
    """Selection with no checkpoint weights must stop before any evaluation is attempted."""
    from local_llm_lab.pipeline.cli import stage_select

    output = tmp_path / "run"
    adapters = output / "adapters"
    adapters.mkdir(parents=True)
    (adapters / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = {"output": output, "select": {"limit": 1, "split": "valid"}}

    with pytest.raises(SystemExit, match="no checkpoint directories.*train"):
        stage_select(config, limit=None, quiet=True)


def _summary(step: int, split: str) -> dict[str, object]:
    """Two deliberately opposed scores: macro favours step 100, micro favours step 200."""
    default_success = 1 if step == 100 else 0
    long_success = 1 if step == 100 else 3
    by_family = {
        family: {
            "successes": long_success if family in LONG_HORIZON_FAMILIES else default_success,
            "tasks": 3 if family in LONG_HORIZON_FAMILIES else 1,
        }
        for family in FAMILIES
    }
    successes = sum(item["successes"] for item in by_family.values())
    return {
        "split": split,
        "tasks": 24,
        "successes": successes,
        "clean_rate": 0.9,
        "valid_action_rate": 0.95,
        "by_family": by_family,
        "rate_counts": {
            "success": {"numerator": successes, "denominator": 24},
            "clean": {"numerator": 22, "denominator": 24},
            "valid_actions": {"numerator": 46, "denominator": 48},
        },
        "wilson_95": {"success": [0.0, 1.0]},
    }


def _row(
    name: str, macro: float, micro: float, clean: float, valid: float, loss: float | None, step: int
) -> dict[str, object]:
    return {
        "name": name,
        "components": {
            "family_macro_success": macro,
            "micro_success": micro,
            "clean_rate": clean,
            "valid_action_rate": valid,
        },
        "val_loss": loss,
        "step": step,
    }
def test_selection_key_orders_all_tiebreakers_and_missing_loss_last() -> None:
    """Catch any departure from macro, micro, clean, valid, loss, then earlier-step ranking."""
    rows = [
        _row("macro", 0.9, 0.1, 0.1, 0.1, None, 999),
        _row("micro", 0.8, 0.9, 0.1, 0.1, 9.0, 999),
        _row("clean", 0.8, 0.8, 0.9, 0.1, 9.0, 999),
        _row("valid", 0.8, 0.8, 0.8, 0.9, 9.0, 999),
        _row("loss", 0.8, 0.8, 0.8, 0.8, 0.1, 999),
        _row("earlier", 0.8, 0.8, 0.8, 0.8, 0.2, 10),
        _row("later", 0.8, 0.8, 0.8, 0.8, 0.2, 20),
        _row("missing", 0.8, 0.8, 0.8, 0.8, None, 1),
    ]

    assert [row["name"] for row in sorted(rows, key=cli._selection_key, reverse=True)] == [
        "macro", "micro", "clean", "valid", "loss", "earlier", "later", "missing"
    ]


def test_stage_select_aggregates_two_cells_and_writes_deterministic_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch a one-cell screen, micro-only family score, log-loss omission, or unstable winner."""
    output = tmp_path / "run"
    adapters = []
    for step in (100, 200):
        adapter = tmp_path / f"step-{step}"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        adapters.append((step, adapter))
    output.mkdir(parents=True)
    (output / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"step": 100, "train_loss": None, "val_loss": 0.2, "tokens": None, "elapsed": 1.0},
                {"step": 200, "train_loss": None, "val_loss": 0.1, "tokens": None, "elapsed": 2.0},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_evaluation(**kwargs):
        calls.append(kwargs)
        step = int(str(kwargs["label"]).split("-")[1])
        return _summary(step, str(kwargs["split"]))

    monkeypatch.setattr(cli, "checkpoint_dirs", lambda _config: adapters)
    monkeypatch.setattr(cli, "run_evaluation", fake_evaluation)
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 4, "max_tokens": 8},
        "select": {
            "screen": [
                {"split": "valid", "difficulty": 1, "per_family": {"default": 1, "long": 3}},
                {"split": "valid2", "difficulty": 2, "per_family": {"default": 1, "long": 3}},
            ]
        },
    }

    best = cli.stage_select(config, limit=None, quiet=True)
    first = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    cli.stage_select(config, limit=None, quiet=True)
    second = json.loads((output / "selection.json").read_text(encoding="utf-8"))

    assert len(calls) == 8
    assert all(call["limit"] is None for call in calls)
    assert {(call["split"], call["difficulty"]) for call in calls} == {
        ("valid", 1),
        ("valid2", 2),
    }
    assert all(call["family_quotas"] == {"default": 1, "long": 3} for call in calls)
    assert best == output / "best-adapter"
    assert (best / "adapter_config.json").is_file()
    assert first == second
    assert first["selected_step"] == 100
    assert first["behavior_best_step"] == 100
    assert first["loss_best_step"] == 200
    assert first["disagreement"] is True
    assert first["model"] == "fake-model" and first["data_seed"] == 17
    assert [cell["difficulty"] for cell in first["screen"]] == [1, 2]
    assert first["checkpoints"][0]["components"]["tasks"] == 48
    assert first["checkpoints"][0]["val_loss"] == pytest.approx(0.2)


def test_validation_losses_ignores_train_rows_and_rejects_malformed_metrics(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    path = output / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 1, "train_loss": 1.0, "val_loss": None, "tokens": 8, "elapsed": 0.1})
        + "\nnot json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"metrics\.jsonl:2: invalid JSON"):
        cli._validation_losses(output)


# --------- R38 slice 3: stage_select's readers against the writers that feed them (issue #70)


def _summarized(successes: int, tasks: int, families: tuple[str, ...] = ("read", "search")) -> dict:
    """An evaluation summary from ``evaluate.summarize`` -- the writer ``stage_select`` reads.

    R38: ``_selection_components`` is fed the return value of ``run_evaluation``, which is
    ``summarize`` plus ``evaluation_metadata``. The other tests in this file hand a dict laid
    out from memory, so they certify their author's belief about the summary rather than the
    summary. Trajectories in, real summary out, and the counts below are readable from the
    arguments: ``successes`` of ``tasks`` succeed, dealt round-robin across ``families``.
    """
    from local_llm_lab.pipeline.evaluate import summarize
    from local_llm_lab.pipeline.runner import Trajectory

    trajectories = [
        Trajectory(
            task_id=f"t{index}",
            family=families[index % len(families)],
            variant="clean",
            label="fake",
            prompt="do it",
            steps=[],
            verdict={"success": index < successes, "clean": True, "calls": 1},
            turns=2,
            valid_turns=2,
            difficulty=1,
        )
        for index in range(tasks)
    ]
    return summarize(trajectories)


def test_selection_components_read_every_key_at_the_level_summarize_writes_it() -> None:
    """R38 on the screen aggregator: the counts it ranks on come from the real writer.

    ``_selection_components`` sums exact numerators and denominators rather than averaging
    rounded rates, so what it depends on is ``rate_counts`` (nested, one level down) and
    ``by_family`` (top level, with ``successes``/``tasks`` inside each bucket). All four
    ranking components are checked against numbers derivable from the arguments above.
    """
    components = cli._selection_components([_summarized(3, 4)])

    assert components["tasks"] == 4 and components["successes"] == 3
    assert components["micro_success"] == pytest.approx(0.75)
    # read: t0 and t2, one of two; search: t1 succeeds, t3 does not. Macro is the mean of the
    # per-family rates (1.0 and 0.5), which is deliberately not the micro rate.
    assert components["family_macro_success"] == pytest.approx(0.75)
    assert components["by_family"] == {
        "read": {"successes": 2, "tasks": 2},
        "search": {"successes": 1, "tasks": 2},
    }
    assert components["clean_rate"] == pytest.approx(1.0)
    assert components["valid_action_rate"] == pytest.approx(1.0)


def test_selection_components_score_zero_macro_when_by_family_moves_off_its_level() -> None:
    """Reported, not changed: the softest read here feeds the FIRST ranking key.

    ``by_family`` is taken with ``.get(..., {})``, so a bucket one level from where
    ``summarize`` puts it is not refused -- every checkpoint scores ``family_macro_success``
    0.0, the comparison silently falls through to micro success, and ``selection.json`` still
    records a macro-first ``criterion`` string. That is a plausible artifact rather than an
    error, which is the case R38 exists to surface. Tightening it is a schema ruling about the
    evaluation summary, not something this reader settles: the sibling probe readers of the
    same artifact are equally permissive about it.
    """
    summary = _summarized(3, 4)
    assert cli._selection_components([summary])["family_macro_success"] == pytest.approx(0.75)

    summary["moved_by_family"] = summary.pop("by_family")

    relevelled = cli._selection_components([summary])

    assert relevelled["family_macro_success"] == 0.0
    assert relevelled["by_family"] == {}
    # The counted components are untouched, so nothing else looks wrong about the row.
    assert relevelled["micro_success"] == pytest.approx(0.75)


def test_selection_components_legacy_fallback_names_counts_no_writer_has_ever_emitted() -> None:
    """The dead half of ``_counts``, found by taking its fallback names to the writer.

    ``_counts`` reads ``rate_counts[name]`` and falls back to a flat pre-``rate_counts``
    layout. Two of the three fallbacks name keys ``summarize`` still emits (``successes``/
    ``tasks`` and ``clean_successes``/``tasks``) and resolve to the same numbers. The third
    names ``valid_turns``/``turns``, which ``summarize`` does not write and which appear in
    none of the eighteen evaluation JSONs under outputs/ -- so on exactly the artifacts the
    fallback exists for, it raises. Loud, not a wrong answer, but dead where it was needed.
    """
    from local_llm_lab.pipeline.evaluate import summarize
    from local_llm_lab.pipeline.runner import Trajectory

    written = summarize([Trajectory(task_id="t", family="read", variant="clean", label="x",
                                    prompt="p", verdict={"success": True}, turns=1, valid_turns=1)])
    assert "valid_turns" not in written and "turns" not in written

    summary = _summarized(3, 4)
    summary["moved_rate_counts"] = summary.pop("rate_counts")

    with pytest.raises(KeyError, match="valid_turns"):
        cli._selection_components([summary])


def _run_select(monkeypatch, tmp_path: Path) -> tuple[Path, list]:
    """Drive ``stage_select`` to the selection.json write with no model and no evaluation."""
    output = tmp_path / "run"
    adapter = tmp_path / "step-100"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    output.mkdir(parents=True)
    stamped: list = []

    monkeypatch.setattr(cli, "checkpoint_dirs", lambda _config: [(100, adapter)])
    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    monkeypatch.setattr(cli, "load_model_spec", lambda name: object())
    monkeypatch.setattr(cli, "run_evaluation", lambda **kwargs: _summarized(3, 4))
    monkeypatch.setattr(
        cli,
        "write_provenance",
        lambda run_dir, *, resolved, spec, extra: stamped.append(extra),
    )
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 2, "max_tokens": 3},
        "select": {"screen": [{"split": "valid", "difficulty": 1, "per_family": {"default": 1}}]},
    }
    cli.stage_select(config, limit=None, quiet=True)
    return output, stamped


def test_stage_select_provenance_records_what_selection_json_holds_not_what_it_meant_to(
    monkeypatch, tmp_path: Path
) -> None:
    """Why the write/re-read looks redundant and is not (cli.py:1041 and :1054).

    The block stamped into provenance is parsed back off disk rather than embedded from the
    dict, so provenance records what was PERSISTED. Proved by making the two differ: the
    writer here lands a document the caller never assembled, and provenance carries that one.
    An optimiser that "simplified" the read-back to ``selection`` would pass every other test
    in this file and quietly let provenance outrank the file it claims to describe.
    """
    real = cli.write_text_atomic
    seen: list = []

    def diverging(path, text):
        seen.append(Path(path))
        real(path, json.dumps({"selected_step": 999}) + "\n")

    monkeypatch.setattr(cli, "write_text_atomic", diverging)

    output, stamped = _run_select(monkeypatch, tmp_path)

    assert seen == [output / "selection.json"]  # and it went through the atomic writer
    persisted = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert stamped == [{"stage": "select", "selection": persisted}]
    assert persisted == {"selected_step": 999}


def test_stage_select_refuses_to_stamp_provenance_over_an_unreadable_selection(
    monkeypatch, tmp_path: Path
) -> None:
    """The read-back is an integrity check, which is the other half of its reason to exist.

    A torn selection.json cannot be parsed, so the stage dies at the read rather than writing
    a provenance that describes a file nobody can open. Worth pinning because `best-adapter/`
    is already copied by this point: the surviving artifact would otherwise be a directory
    that looks finished beside the one file saying which checkpoint it is.
    """
    monkeypatch.setattr(
        cli, "write_text_atomic", lambda path, text: Path(path).write_text(text[:20])
    )

    with pytest.raises(json.JSONDecodeError):
        _run_select(monkeypatch, tmp_path)
