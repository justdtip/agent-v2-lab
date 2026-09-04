from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest


@dataclass(frozen=True)
class _Task:
    task_id: str
    family: str
    prompt: str = "task"
    steps: tuple[object, ...] = ()


def _payload(trajectories, *, data_seed=17):
    return {"data_seed": data_seed, "trajectories": trajectories}


def test_select_patch_cases_intersects_saved_evaluations(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    tasks = {
        "test-aggregate_report-0-clean": _Task("test-aggregate_report-0-clean", "aggregate_report"),
        "test-ledger_reconcile-1-clean": _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"),
    }
    monkeypatch.setattr(
        patch,
        "task_from_id",
        lambda task_id, seed, difficulty: (
            tasks[task_id] if seed == 19 and difficulty == 2 else None
        ),
    )
    passing = _payload(
        [
            {"task_id": key, "verdict": {"success": True}}
            for key in tasks
        ]
        + [{"task_id": "test-other-2-clean", "verdict": {"success": True}}],
        data_seed=11,
    )
    failing = _payload(
        [
            {
                "task_id": key,
                "family": task.family,
                "difficulty": 2,
                "steps": [{"thought": "before"}, {"thought": "drop"}],
                "integrity": {"violations": [{"kind": "value_drop", "step": 1}]},
            }
            for key, task in tasks.items()
        ],
        data_seed=19,
    )

    cases = patch.select_patch_cases(passing, failing, keep_last=2)

    assert [(case.task.task_id, case.decision_step) for case in cases] == [
        ("test-aggregate_report-0-clean", 1),
        ("test-ledger_reconcile-1-clean", 1),
    ]
    assert cases[0].failing_steps[0]["thought"] == "before"


def test_select_patch_cases_rejects_malformed_or_mismatched_payloads() -> None:
    from local_llm_lab.probes import patch

    with pytest.raises(ValueError, match="trajectories"):
        patch.select_patch_cases({}, {}, keep_last=2)
    with pytest.raises(ValueError, match="data_seed"):
        patch.select_patch_cases(_payload([], data_seed="bad"), _payload([]), keep_last=2)


def test_position_groups_are_exact_and_fail_closed() -> None:
    from local_llm_lab.probes import patch

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(char) for char in text]

    tokenizer = Tokenizer()
    text = "SYS|TASK|OLD|NOTE|42|OBS0|OBS1|OBS2|!"
    groups = patch.position_groups(
        tokenizer,
        tokenizer.encode(text),
        system_text="SYS",
        task_text="TASK",
        previous_notes=["OLD", "NOTE|42"],
        note_values=["42"],
        observations=["OBS0", "OBS1", "OBS2"],
    )

    assert tuple(groups) == patch.POSITION_GROUPS
    assert groups["note_value_tokens"] == (18, 19)
    assert groups["last_two_observations"] == (26, 27, 28, 29, 31, 32, 33, 34)
    assert groups["final_token"] == (36,)
    assert set(groups["note_value_tokens"]) <= set(groups["previous_notes"])
    with pytest.raises(ValueError, match="missing"):
        patch.position_groups(
            tokenizer,
            tokenizer.encode("SYS|TASK"),
            system_text="SYS",
            task_text="TASK",
            previous_notes=["MISSING"],
            note_values=[],
            observations=[],
        )


def test_random_positions_are_seeded_unique_and_distinct() -> None:
    from local_llm_lab.probes import patch

    first = patch.random_control_positions(range(12), (2, 3, 4), seed=9, label="case")
    second = patch.random_control_positions(range(12), (2, 3, 4), seed=9, label="case")

    assert first == second
    assert len(first) == 3 and len(set(first)) == 3
    assert set(first).isdisjoint({2, 3, 4})


def test_greedy_generate_uses_cached_masked_forwards(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    class Tokenizer:
        def decode(self, ids):
            return "".join(chr(index) for index in ids)

    class View:
        num_layers = 1

        def __init__(self):
            self.lengths = []

        def make_cache(self):
            return [SimpleNamespace(offset=0)]

        def embed(self, ids):
            self.lengths.append(len(ids[0]) if hasattr(ids, "shape") else len(ids))
            return mx.zeros((1, self.lengths[-1], 1), dtype=mx.float32)

        def masks(self, h, cache):
            assert cache is not None
            return {"mask": h.shape[1]}

        def run_block(self, index, h, masks, cache_i):
            assert index == 0 and masks["mask"] == h.shape[1]
            cache_i.offset += h.shape[1]
            return h

        def final_norm(self, h):
            return h

        def unembed(self, h):
            logits = mx.zeros((1, h.shape[1], 128), dtype=mx.float32)
            logits[..., ord("x")] = 1
            return logits

    view = View()
    calls = []
    monkeypatch.setattr(
        patch,
        "turn_is_complete",
        lambda text: calls.append(text) or len(calls) == 2,
    )

    assert patch.greedy_generate(view, Tokenizer(), [1, 2, 3], max_tokens=5) == "xx"
    assert view.lengths == [3, 1]


def test_task_cell_aggregation_is_over_task_booleans_only() -> None:
    from local_llm_lab.probes import patch

    summary = patch.aggregate_task_flips({"task-a": [True, True], "task-b": [False]})

    assert summary["numerator"] == 1
    assert summary["denominator"] == 2
    assert summary["rate"] == 0.5
    assert len(summary["wilson_95"]) == 2


def test_flip_scoring_requires_the_decision_step_value_drop_to_disappear(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    task = _Task("test-aggregate_report-0-clean", "aggregate_report")
    remaining = SimpleNamespace(violations=(SimpleNamespace(kind="value_drop", step=3),))
    moved = SimpleNamespace(violations=(SimpleNamespace(kind="value_drop", step=2),))
    monkeypatch.setattr(patch, "check_trajectory", lambda *_args, **_kwargs: remaining)
    assert not patch._is_flip(task, [], 3, keep_last=2)
    monkeypatch.setattr(patch, "check_trajectory", lambda *_args, **_kwargs: moved)
    assert patch._is_flip(task, [], 3, keep_last=2)


def test_patch_score_treats_an_unparseable_generation_as_non_flip(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    class Hook:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    case = patch.PatchCase(
        _Task("test-aggregate_report-0-clean", "aggregate_report"),
        0,
        ({"thought": "bad"},),
    )
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "not a turn")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda _raw: (_ for _ in ()).throw(ValueError("bad")))

    assert not patch._score_patch(
        object(), object(), case, layer=1, source_rows=object(), target_positions=(0,),
        failing_ids=[1], keep_last=2, max_tokens=1,
    )


def test_replay_replaces_only_the_immediately_previous_note(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    task = _Task(
        "test-aggregate_report-0-clean",
        "aggregate_report",
        steps=(SimpleNamespace(thought="expert zero"), SimpleNamespace(thought="expert one")),
    )
    case = patch.PatchCase(
        task,
        2,
        (
            {
                "thought": "old zero",
                "action": {"name": "read_file", "arguments": {"path": "a"}},
                "observation": "A",
            },
            {
                "thought": "old one",
                "action": {"name": "read_file", "arguments": {"path": "b"}},
                "observation": "B",
            },
        ),
    )
    monkeypatch.setattr(patch, "render_expert_note", lambda _task, index: f"expert {index}")

    failing, counterfactual = patch.replay_counterfactual(case)

    assert failing[:2] == counterfactual[:2]
    assert failing[2]["content"] == counterfactual[2]["content"]
    assert "old one" in failing[4]["content"]
    assert "expert 1" in counterfactual[4]["content"]
    assert failing[5:] == counterfactual[5:]


def test_patch_cli_is_registered_and_validates_before_loading(monkeypatch, tmp_path) -> None:
    from local_llm_lab.probes import patch

    loads = []
    monkeypatch.setattr(patch, "load_model_spec", lambda name: loads.append(name))
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval",
            str(tmp_path / "missing-b.json"),
            "--failing-eval",
            str(tmp_path / "missing-c.json"),
            "--output",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit):
        patch.main()

    assert loads == []


def test_patch_cli_forwards_registry_spec_and_writes_results(monkeypatch, tmp_path) -> None:
    from local_llm_lab.probes import guard, patch

    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    selected = SimpleNamespace(
        hf_id="fake/hf",
        resolve=lambda *_args: SimpleNamespace(num_layers=4, probe_layers=(1, 2)),
    )
    case = patch.PatchCase(_Task("test-aggregate_report-0-clean", "aggregate_report"), 0, ())
    seen = []
    monkeypatch.setattr(patch, "select_patch_cases", lambda *_args, **_kwargs: [case])
    monkeypatch.setattr(
        patch,
        "load_model_spec",
        lambda name: seen.append(("spec", name)) or selected,
    )
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: seen.append(("guard", None)))
    monkeypatch.setattr(patch, "resolve_policy", lambda name: seen.append(("policy", name)) or None)
    monkeypatch.setattr(
        patch,
        "load_policy",
        lambda name, adapter: seen.append(("load", name, adapter)) or (object(), object()),
    )
    monkeypatch.setattr(patch.ArchitectureView, "from_model", lambda _model: SimpleNamespace())
    monkeypatch.setattr(
        patch,
        "run_patch_probe",
        lambda *_args, **kwargs: seen.append(("probe", kwargs["spec"], kwargs["command"]))
        or {"groups": [], "layers": [], "controls": [], "cells": {}},
    )
    monkeypatch.setattr(patch, "render_markdown", lambda _payload: "# fake")
    argv = [
        "agent-v2-probe-patch",
        "--passing-eval",
        str(passing),
        "--failing-eval",
        str(failing),
        "--output",
        str(tmp_path),
        "--model",
        "qwen35-4b",
        "--policy",
        "base",
        "--layers",
        "1,0.5",
        "--seed",
        "9",
    ]
    monkeypatch.setattr("sys.argv", argv)

    patch.main()

    assert seen[0] == ("spec", "qwen35-4b")
    assert ("policy", "base") in seen and ("load", "fake/hf", None) in seen
    assert seen[-1] == ("probe", selected, argv)
    assert (tmp_path / "patch.json").is_file() and (tmp_path / "patch.md").read_text() == "# fake\n"


def test_patch_cli_rejects_malformed_layers_before_model_loading(monkeypatch, tmp_path) -> None:
    from local_llm_lab.probes import patch

    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    passing.write_text("{}", encoding="utf-8")
    failing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(patch, "select_patch_cases", lambda *_args, **_kwargs: [object()])
    loads = []
    monkeypatch.setattr(patch, "load_policy", lambda *_args: loads.append(True))
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-probe-patch",
            "--passing-eval",
            str(passing),
            "--failing-eval",
            str(failing),
            "--output",
            str(tmp_path),
            "--layers",
            "garbage",
        ],
    )

    with pytest.raises(SystemExit):
        patch.main()

    assert loads == []


def test_patch_probe_uses_named_groups_full_captures_and_distinct_controls(monkeypatch) -> None:
    from local_llm_lab.probes import patch

    cases = [
        patch.PatchCase(
            _Task("test-aggregate_report-0-clean", "aggregate_report"), 0, ({"thought": "bad"},)
        ),
        patch.PatchCase(
            _Task("test-ledger_reconcile-1-clean", "ledger_reconcile"), 0, ({"thought": "bad"},)
        ),
    ]
    groups = {name: (index,) for index, name in enumerate(patch.POSITION_GROUPS)}
    groups["final_token"] = (5,)
    captures = []
    injected = []

    class View:
        num_layers = 1

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text.startswith("failing"):
                return [10 if text.endswith("0") else 110, 11, 12, 13, 14, 15]
            return [20 if text.endswith("0") else 120, 21, 22, 23, 24, 25]

    class Hook:
        def __init__(self, _view, _layer, vector, *, at_positions, replace):
            assert replace
            injected.append((tuple(np.asarray(vector).reshape(-1)), tuple(at_positions)))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(patch.ArchitectureView, "from_model", lambda _model: View())
    monkeypatch.setattr(
        patch,
        "replay_counterfactual",
        lambda case: (
            [{"role": "user", "content": f"f{0 if '-0-' in case.task.task_id else 1}"}],
            [{"role": "user", "content": f"c{0 if '-0-' in case.task.task_id else 1}"}],
        ),
    )
    monkeypatch.setattr(
        patch,
        "build_prompt",
        lambda _tokenizer, messages, **_kwargs: (
            "failing" if messages[0]["content"].startswith("f") else "counter"
        ) + messages[0]["content"][-1],
    )
    monkeypatch.setattr(patch, "position_groups", lambda *_args, **_kwargs: groups)
    def capture(_view, ids, layers, *, positions):
        captures.append((tuple(ids), positions))
        assert positions == "all"
        base = 100 if ids[0] < 100 else 200
        return {
            layer: mx.array([[base + layer + index] for index in range(6)], dtype=mx.float32)
            for layer in layers
        }
    monkeypatch.setattr(patch, "capture_residuals", capture)
    monkeypatch.setattr(patch, "InjectionHook", Hook)
    monkeypatch.setattr(patch, "greedy_generate", lambda *_args, **_kwargs: "raw")
    monkeypatch.setattr(patch, "strip_thinking", lambda raw: (None, raw))
    monkeypatch.setattr(patch, "parse_turn", lambda _raw: SimpleNamespace(thought="note"))
    monkeypatch.setattr(patch, "_is_flip", lambda *_args, **_kwargs: True)
    resolved = SimpleNamespace(as_dict=lambda: {"name": "fake"})

    payload = patch.run_patch_probe(
        object(), Tokenizer(), cases, spec=object(), resolved=resolved, layers=[1], policy="base",
        keep_last=2, max_tokens=1, seed=7, command=["patch"],
    )

    assert len(captures) == 4
    assert len(injected) == len(cases) * len(groups) * 3
    final_vectors = [vector for vector, positions in injected if positions == (5,)]
    assert (106.0,) in final_vectors
    assert (206.0,) in final_vectors
    assert payload["cells"]["1:final_token"]["treatment"]["denominator"] == 2
    assert set(payload["cells"]["1:final_token"]["controls"]) == set(patch.CONTROLS)
