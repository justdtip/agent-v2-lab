from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from local_llm_lab.probes import assistant_axis

TEXTBOOK_PHOTOSYNTHESIS = (
    "Photosynthesis is a process by which green plants, algae, and some bacteria use sunlight, "
    "water, and carbon dioxide to produce oxygen and energy in the form of glucose. The process "
    "occurs in the chloroplasts of plant cells, where chlorophyll, a green pigment, captures "
    "sunlight and converts it into chemical energy.\nOne concrete example of photosynthesis is "
    "a tree. When a tree absorbs sunlight, water, and carbon dioxide, it produces oxygen and "
    "glucose. The oxygen is released into the atmosphere, and the glucose is used by the tree "
    "for energy and growth."
)


def _marker_reply(role: str) -> str:
    return "I speak in my own voice: " + ", ".join(assistant_axis.ROLE_MARKERS[role]) + "."


def _events(output: Path) -> list[dict[str, Any]]:
    """Every event the run log wrote to ``output/events.jsonl``, in order."""
    lines = (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _flat(event: dict[str, Any]) -> dict[str, Any]:
    """One event's top-level keys with its ``fields`` object merged in."""
    flat = {key: value for key, value in event.items() if key != "fields"}
    flat.update(event.get("fields") or {})
    return flat


def test_trajectory_projections_forwards_the_selected_spec_to_prompt_rendering(
    monkeypatch, tmp_path
) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import protocol
    from local_llm_lab.pipeline.evaluate import write_report
    from local_llm_lab.pipeline.runner import Trajectory

    selected = load_model_spec("qwen35-4b")
    eval_path = tmp_path / "eval.json"
    # R38: the record goes through ``Trajectory`` and ``write_report``, the classes that make
    # the file this reader parses. Hand-written, it carried nine of the nineteen fields the
    # dataclass declares and still constructed, so it certified nothing about the seam --
    # ``trajectory_projections`` ends every record with ``Trajectory(**record)``. The summary
    # is left empty on purpose: this reader opens ``payload["trajectories"]`` and nothing else.
    write_report(
        eval_path,
        {},
        [
            Trajectory(
                task_id="fake",
                family="read",
                variant="clean",
                label="x",
                prompt="task",
                # The parse-error step run_task records before breaking (runner.py:439-448):
                # one generated turn to project and then a deliberate end. A step with neither
                # ``action`` nor ``parse_error`` is now a malformed record and raises (ruling
                # on #70), and this test is about which spec reaches build_prompt.
                steps=[{"index": 0, "raw": "note", "parse_error": "boom"}],
                verdict={"success": True},
            )
        ],
    )
    seen = []
    monkeypatch.setattr(
        protocol,
        "build_prompt",
        lambda *_args, spec=None, **_kwargs: seen.append(spec) or "prompt",
    )
    monkeypatch.setattr(
        assistant_axis,
        "response_mean_activations",
        lambda *_args, **_kwargs: {0: np.array([1.0], dtype=np.float32)},
    )

    assistant_axis.trajectory_projections(None, None, eval_path, [1.0], 0, spec=selected)

    assert seen == [selected]


def test_project_loads_and_dispatches_the_selected_spec(monkeypatch, tmp_path: Path) -> None:
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.probes import guard, policies

    selected = SimpleNamespace(name="qwen35-4b", hf_id="fake/hf", policies={})
    seen = []
    args = argparse.Namespace(
        axis=tmp_path / "axis.npz",
        layer=18,
        model="qwen35-4b",
        policy="base",
        eval=tmp_path / "eval.json",
        limit=1,
        output=tmp_path,
        skip_preflight_check=False,
    )
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(assistant_axis, "require_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        assistant_axis,
        "load_axis",
        lambda *_args: ({18: np.array([1.0])}, {"layers": {"18": {}}}),
    )
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(
        policies,
        "resolve_policy",
        lambda name, spec: seen.append(("policy", name, spec)) or None,
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: (None, None, None, None)
    )
    monkeypatch.setattr(
        models,
        "load_model_spec",
        lambda model: seen.append(("load", model)) or selected,
    )

    def intercept(*_args, **kwargs):
        seen.append(("dispatch", kwargs["spec"]))
        raise RuntimeError("stop after dispatch")

    monkeypatch.setattr(assistant_axis, "trajectory_projections", intercept)

    with pytest.raises(RuntimeError, match="stop after dispatch"):
        assistant_axis._project(args, parser)

    assert seen == [
        ("load", "qwen35-4b"),
        ("policy", "base", selected),
        ("dispatch", selected),
    ]
    events = _events(tmp_path)
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "assistant-axis-project"
    assert _flat(events[-1])["status"] == "error"


def test_project_writes_a_run_log_with_identity_and_progress(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """R26 (issue #35, clauses e and g): the projection CLI records what it read by hash
    and emits one progress line per projected trajectory."""
    import hashlib

    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.probes import guard, policies

    selected = SimpleNamespace(name="qwen35-4b", hf_id="fake/hf", policies={})
    axis_path = tmp_path / "axis.npz"
    axis_path.write_bytes(b"axis")
    eval_path = tmp_path / "eval.json"
    eval_path.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        axis=axis_path,
        layer=18,
        model="qwen35-4b",
        policy="base",
        eval=eval_path,
        limit=1,
        output=tmp_path,
        skip_preflight_check=False,
    )
    monkeypatch.setattr(assistant_axis, "require_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(models, "load_model_spec", lambda _name: selected)
    monkeypatch.setattr(policies, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(
        assistant_axis,
        "load_axis",
        lambda *_args: ({18: np.array([1.0])}, {"layers": {"18": {}}}),
    )
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args: (None, None, None, SimpleNamespace(as_dict=dict)),
    )
    monkeypatch.setattr(
        assistant_axis,
        "trajectory_projections",
        lambda *_args, **kwargs: kwargs["progress"](1, 2) or kwargs["progress"](2, 2) or [],
    )
    monkeypatch.setattr(assistant_axis, "summarize_projections", lambda _records: {})
    monkeypatch.setattr(assistant_axis, "render_projection_markdown", lambda *_args: "# fake")

    assistant_axis._project(args, argparse.ArgumentParser())

    out = capsys.readouterr().out
    assert (tmp_path / "run.log").is_file()
    events = _events(tmp_path)
    start = _flat(events[0])
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "assistant-axis-project"
    assert start["model"] == "qwen35-4b"
    assert start["hf_id"] == "fake/hf"
    assert start["policy"] == "base"
    assert start["adapter"] is None
    assert start["layer"] == 18
    assert start["axis_sha256"] == hashlib.sha256(b"axis").hexdigest()
    assert start["eval_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert isinstance(start["git_commit"], str)

    progress = [_flat(event) for event in events if event["kind"] == "progress"]
    assert [(item["step"], item["total"], item["label"]) for item in progress] == [
        (1, 2, "project"),
        (2, 2, "project"),
    ]
    json_path = tmp_path / "trajectories-base-layer18.json"
    wrote = [
        item
        for item in map(_flat, events)
        if item["kind"] == "info" and item["message"] == "wrote"
    ]
    assert [item["path"] for item in wrote] == [str(json_path)]
    assert str(json_path) in out
    assert "[project 1/2 50%]" in out
    assert "# fake" in out.splitlines()  # print(markdown) is still the CLI's own contract


def test_axis_build_uses_actual_depth_selected_spec_and_records_layer_selection(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Catches hard-coded build layers, global policy lookup, or missing diagnostics."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.probes import guard, policies

    selected = SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.167, 0.333, 0.5, 0.667, 0.833, 1.0),
    )
    model = object()
    seen = []
    monkeypatch.setattr(models, "load_model_spec", lambda _name: selected)
    monkeypatch.setattr(assistant_axis, "require_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(
        policies,
        "resolve_policy",
        lambda name, spec: seen.append(("policy", name, spec)) or None,
    )
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args: (
            model,
            object(),
            SimpleNamespace(num_layers=32),
            SimpleNamespace(as_dict=dict),
        ),
    )
    monkeypatch.setattr(assistant_axis, "load_chat_prompts", lambda _count: ["prompt"])

    def build(_model, _tokenizer, _prompts, layers, *_args, **_kwargs):
        assert layers == [5, 11, 16, 21, 27, 32]
        _kwargs["progress"](1, 2, "pirate")
        _kwargs["progress"](2, 2, "chef")
        return {layer: np.ones(1, dtype=np.float32) for layer in layers}, {"layers": {}}


    monkeypatch.setattr(assistant_axis, "build_axis_run", build)
    monkeypatch.setattr(assistant_axis, "save_axis", lambda path, *_args: path)
    monkeypatch.setattr(assistant_axis, "render_build_markdown", lambda *_args: "# fake")
    args = argparse.Namespace(
        model="qwen35-4b",
        policy="base",
        output=tmp_path,
        allow_busy_gpu=False,
        prompts=1,
        role_prompts=1,
        max_tokens=1,
        layers=None,
        min_expression=0.34,
        exemplar=True,
        judge=False,
        skip_preflight_check=False,
    )

    assistant_axis._build(args, argparse.ArgumentParser())

    expected = {
        "source": "registry-default",
        "requested": ["0.167", "0.333", "0.5", "0.667", "0.833", "1.0"],
        "fractions": [0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
        "indices": [5, 11, 16, 21, 27, 32],
        "num_layers": 32,
    }
    payload = json.loads((tmp_path / "axis-base.json").read_text(encoding="utf-8"))
    assert payload["layer_selection"] == expected
    assert seen == [("policy", "base", selected)]

    out = capsys.readouterr().out
    assert (tmp_path / "run.log").is_file()
    events = _events(tmp_path)
    start = _flat(events[0])
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "assistant-axis-build"
    assert start["model"] == "qwen35-4b"
    assert start["hf_id"] == "fake/hf"
    assert start["policy"] == "base"
    assert start["adapter"] is None
    assert start["prompts"] == 1
    assert isinstance(start["git_commit"], str)

    progress = [_flat(event) for event in events if event["kind"] == "progress"]
    # The label is now emitted by the library, which names the phase and the unit.
    assert [(item["step"], item["total"], item["label"]) for item in progress] == [
        (1, 2, "pirate"),
        (2, 2, "chef"),
    ]
    npz_path = tmp_path / "axis-base.npz"
    wrote = [
        item
        for item in map(_flat, events)
        if item["kind"] == "info" and item["message"] == "wrote"
    ]
    assert [item["path"] for item in wrote] == [str(npz_path)]
    assert "default assistant prompts=1" in out
    assert str(npz_path) in out
    assert "[pirate 1/2 50%]" in out
    assert "# fake" in out.splitlines()  # print(markdown) is still the CLI's own contract


def test_axis_build_rejects_malformed_layers_before_gpu_or_loader(
    monkeypatch, tmp_path: Path
) -> None:
    """Catches layer validation occurring after role-rollout model loading."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.probes import guard

    monkeypatch.setattr(models, "load_model_spec", lambda _name: object())
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )
    args = argparse.Namespace(
        model="qwen35-4b",
        policy="base",
        output=tmp_path,
        allow_busy_gpu=False,
        prompts=1,
        role_prompts=1,
        max_tokens=1,
        layers="1,,2",
        min_expression=0.34,
        exemplar=True,
        judge=False,
        skip_preflight_check=False,
    )

    with pytest.raises(SystemExit) as raised:
        assistant_axis._build(args, argparse.ArgumentParser())

    assert raised.value.code == 2


def test_role_prompts_are_strong_and_have_markers() -> None:
    names = [name for name, _system in assistant_axis.ROLES]
    low_and_neutral = {
        "ghost",
        "hermit",
        "leviathan",
        "oracle",
        "trickster",
        "pirate",
        "prophet",
        "wanderer",
        "child",
        "poet",
        "drill sergeant",
        "historian",
        "chef",
        "detective",
        "monk",
        "gambler",
        "nurse",
        "astronaut",
    }
    assert len(names) == 24
    assert low_and_neutral <= set(names)
    for name, system in assistant_axis.ROLES:
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", system) if part]
        assert 2 <= len(sentences) <= 4, name
        assert "never mention being an AI or a language model" in system, name
        assert len(assistant_axis.ROLE_MARKERS[name]) >= 4, name


def test_pirate_heuristic_separates_role_expression_from_textbook_voice() -> None:
    pirate = "Arr, ye and me hearties! Aye, our ship crosses the sea for treasure, matey."
    assert assistant_axis.heuristic_expression_score("pirate", pirate) > 0.9
    assert assistant_axis.heuristic_expression_score("pirate", TEXTBOOK_PHOTOSYNTHESIS) == 0.0
    assert (
        assistant_axis.role_expression_score(
            "pirate",
            assistant_axis.ROLE_TABLE["pirate"].system,
            pirate,
        )
        > 0.9
    )


def test_judge_parser_maps_two_thirds_and_tolerates_surrounding_text() -> None:
    assert assistant_axis.parse_judge_score("I would rate this 2 out of 3.") == pytest.approx(2 / 3)
    assert assistant_axis.parse_judge_score("Score: 0 (not in character)") == 0.0
    assert assistant_axis.parse_judge_score("No numeric verdict") is None


def test_build_axis_refuses_when_too_few_roles_survive(monkeypatch) -> None:
    calls: list[str] = []

    def should_not_capture(*args: Any, **kwargs: Any) -> dict[int, np.ndarray]:
        del args, kwargs
        calls.append("capture")
        return {0: np.ones(4, dtype=np.float32)}

    monkeypatch.setattr(assistant_axis, "response_mean_activations", should_not_capture)
    role_responses = {}
    for index, (name, _system) in enumerate(assistant_axis.ROLES):
        response = _marker_reply(name) if index < 11 else TEXTBOOK_PHOTOSYNTHESIS
        role_responses[name] = [(f"role:{name}:{run}", response) for run in range(3)]

    with pytest.raises(ValueError, match="at least 12 roles.*only 11 survived"):
        assistant_axis.build_axis(
            None,
            None,
            [("default", "assistant response")],
            role_responses,
            [0],
        )
    assert calls == []


def test_build_axis_succeeds_with_synthetic_filtered_vectors(monkeypatch) -> None:
    role_indices = {name: index for index, (name, _system) in enumerate(assistant_axis.ROLES)}

    def fake_response_mean(
        model: Any,
        tokenizer: Any,
        prompt: str,
        response: str,
        layers: list[int],
        *,
        stats: dict[str, int] | None = None,
        dtype: str = "float32",
    ) -> dict[int, np.ndarray]:
        del model, tokenizer, response, dtype
        if stats is not None:
            stats["sequences"] = stats.get("sequences", 0) + 1
        if prompt.startswith("default"):
            vector = np.array([2.0, 0.25, 0.0, 0.0], dtype=np.float32)
        else:
            name = prompt.split(":")[1]
            index = role_indices[name]
            vector = np.array([-1.0 - 0.1 * index, 0.02 * index, 0.1, 0.0], dtype=np.float32)
        return {layer: vector * (1.0 + layer / 100.0) for layer in layers}

    monkeypatch.setattr(assistant_axis, "response_mean_activations", fake_response_mean)
    default = [(f"default:{run}", f"assistant response {run}") for run in range(6)]
    role_responses = {
        name: [(f"role:{name}:{run}", _marker_reply(name)) for run in range(3)]
        for name, _system in assistant_axis.ROLES
    }
    axis, diagnostics = assistant_axis.build_axis(
        None,
        None,
        default,
        role_responses,
        [12, 18, 24],
    )

    assert sorted(axis) == [12, 18, 24]
    assert diagnostics["roles"] == [name for name, _system in assistant_axis.ROLES]
    assert all(entry["kept"] == 3 for entry in diagnostics["role_expression"].values())
    assert diagnostics["layers"]["18"]["axis_norm_fraction"] > 0.0
    assert len(diagnostics["layers"]["18"]["chat_projections"]) == len(default)


class _Tokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return json.dumps(messages) + "<assistant>"


class _Piece:
    text = "generated response"
    token = 0


def test_rollout_jsonl_is_complete_before_axis_vectors_are_computed(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []

    def fake_stream_generate(*args: Any, **kwargs: Any):
        del args, kwargs
        events.append("generate")
        yield _Piece()

    def fake_build_axis(*args: Any, **kwargs: Any):
        del args, kwargs
        events.append("vectors")
        path = tmp_path / "rollouts-base.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 3 + 24 * 3
        assert [record["role"] for record in records[:3]] == ["default-assistant"] * 3
        assert all({"role", "prompt", "response"} <= record.keys() for record in records)
        return {18: np.ones(2, dtype=np.float32)}, {"layers": {}}

    monkeypatch.setattr("mlx_lm.stream_generate", fake_stream_generate)
    monkeypatch.setattr(assistant_axis, "build_axis", fake_build_axis)
    _axis, diagnostics = assistant_axis.build_axis_run(
        None,
        _Tokenizer(),
        ["one", "two", "three"],
        [18],
        tmp_path,
        "base",
        role_prompts=3,
        use_model_judge=False,
    )

    assert events[-1] == "vectors"
    assert events.count("generate") == 3 + 24 * 3
    assert Path(diagnostics["rollouts_path"]) == tmp_path / "rollouts-base.jsonl"


def test_default_assistant_receives_the_same_prompt_list_as_every_role(
    monkeypatch, tmp_path: Path
) -> None:
    """SPEC-004 §4 matched design: the default persona gets the roles' eight prompts.

    Before the fix the default assistant was rolled out against the whole ``--prompts``
    pool (96) while each role saw ``prompts[:role_prompts]`` (8), so the two sides of the
    axis differed in prompt set as well as in persona.
    """
    calls: list[tuple[str | None, list[str]]] = []

    def recording_rollout_role(
        _model: Any,
        _tokenizer: Any,
        _system: str,
        prompts: list[str],
        _max_tokens: int = 192,
        *,
        role: str | None = None,
        exemplar: bool = True,
        rollout_path: Path | None = None,
    ) -> list[tuple[str, str]]:
        del exemplar, rollout_path
        calls.append((role, list(prompts)))
        return [(f"rendered:{prompt}", f"response:{prompt}") for prompt in prompts]

    monkeypatch.setattr(assistant_axis, "rollout_role", recording_rollout_role)
    pool = [f"prompt-{index}" for index in range(12)]

    default, role_responses, _path = assistant_axis.collect_rollouts(
        None, None, pool, tmp_path, "base", role_prompts=4
    )

    shared = pool[:4]
    assert [name for name, _prompts in calls] == ["default-assistant"] + [
        name for name, _system in assistant_axis.ROLES
    ]
    assert [seen for _name, seen in calls] == [shared] * (1 + len(assistant_axis.ROLES))
    assert len(default) == len(shared)
    assert all(len(pairs) == len(shared) for pairs in role_responses.values())


def test_no_role_in_any_group_carries_an_exemplar() -> None:
    """Chief's ruling 2026-09-05: exemplars are dropped from all 24 roles.

    The axis derives from the system prompts alone, so exemplar presence -- previously
    0/6 high, 8/8 low and 6/10 neutral -- is equal at zero across the three groups.
    """
    assert assistant_axis.ROLE_EXEMPLARS == {}
    presence = {"high": 0, "low": 0, "neutral": 0}
    totals = {"high": 0, "low": 0, "neutral": 0}
    for name, role in assistant_axis.ROLE_TABLE.items():
        group = assistant_axis.ROLE_GROUPS[name]
        totals[group] += 1
        presence[group] += int(getattr(role, "exemplar", None) is not None)
    assert totals == {"high": 6, "low": 8, "neutral": 10}
    assert presence == {"high": 0, "low": 0, "neutral": 0}


def test_rollout_role_sends_only_the_system_prompt_and_the_question(
    monkeypatch, tmp_path: Path
) -> None:
    """The dropped one-shot must not survive anywhere in the rendered prompt."""
    prompts_seen: list[str] = []

    def fake_stream_generate(*args: Any, prompt: str, **kwargs: Any):
        del args, kwargs
        prompts_seen.append(prompt)
        yield _Piece()

    monkeypatch.setattr("mlx_lm.stream_generate", fake_stream_generate)
    path = tmp_path / "rollouts.jsonl"

    pairs = assistant_axis.rollout_role(
        None,
        _Tokenizer(),
        assistant_axis.ROLE_TABLE["pirate"].system,
        ["What is gravity?"],
        role="pirate",
        exemplar=True,
        rollout_path=path,
    )

    assert len(pairs) == 1
    messages = json.loads(prompts_seen[0].removesuffix("<assistant>"))
    assert [message["role"] for message in messages] == ["system", "user"]
    # The pirate's former one-shot question and answer, verbatim.
    assert "How do plants make food?" not in prompts_seen[0]
    assert "a wee green sail" not in prompts_seen[0]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["exemplar"] is False and record["role"] == "pirate"


def test_judge_defaults_off_in_the_build_cli_and_is_still_opt_in(
    monkeypatch, tmp_path: Path
) -> None:
    """SPEC-004 §4: heuristic only unless ``--judge`` is passed explicitly."""
    seen: list[bool] = []
    monkeypatch.setattr(assistant_axis, "_build", lambda args, _parser: seen.append(args.judge))
    base = ["agent-v2-probe-axis", "build", "--output", str(tmp_path)]

    monkeypatch.setattr(sys, "argv", list(base))
    assistant_axis.main()
    monkeypatch.setattr(sys, "argv", [*base, "--judge"])
    assistant_axis.main()
    monkeypatch.setattr(sys, "argv", [*base, "--no-judge"])
    assistant_axis.main()

    assert seen == [False, True, False]


def test_build_axis_run_scores_with_the_heuristic_alone_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    """``build_axis_run`` matches ``build_axis``: the same-policy judge is opt-in."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        assistant_axis,
        "collect_rollouts",
        lambda *_args, **_kwargs: (
            [("p", "r")],
            {"pirate": [("p", "r")]},
            tmp_path / "rollouts-base.jsonl",
        ),
    )

    def fake_build_axis(
        _model: Any,
        _tokenizer: Any,
        _default: Any,
        _roles: Any,
        layers: list[int],
        *,
        min_expression: float = 0.34,
        use_model_judge: bool = False,
        capture_dtype: str = "float32",
        progress: Any = None,
    ):
        del min_expression, capture_dtype, progress
        seen["use_model_judge"] = use_model_judge
        return {layer: np.ones(2, dtype=np.float32) for layer in layers}, {"layers": {}}

    monkeypatch.setattr(assistant_axis, "build_axis", fake_build_axis)

    _axis, diagnostics = assistant_axis.build_axis_run(None, None, ["one"], [18], tmp_path, "base")

    assert seen["use_model_judge"] is False
    assert diagnostics["model_judge"] is False


# --------------------------------------------------------------------------- P1 closure


def _closure_rollouts(path: Path) -> None:
    """A six-row stand-in for the 288 saved rollouts: two personas, one expressive reply.

    The pirate reply carries three of its eight markers (0.375): above the ratified 0.34 and
    below the 0.60 the threshold test passes, so one fixture serves both.
    """
    pirate = "Arr, ye scallywags: the ship is away."
    rows = [
        {"role": "default-assistant", "prompt": "q", "response": TEXTBOOK_PHOTOSYNTHESIS},
        {"role": "default-assistant", "prompt": "q", "response": TEXTBOOK_PHOTOSYNTHESIS},
        {"role": "pirate", "prompt": "q", "response": pirate},
        {"role": "pirate", "prompt": "q", "response": TEXTBOOK_PHOTOSYNTHESIS},
        {"role": "ghost", "prompt": "q", "response": TEXTBOOK_PHOTOSYNTHESIS},
        {"role": "ghost", "prompt": "q", "response": TEXTBOOK_PHOTOSYNTHESIS},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_close_writes_the_measured_closure_record(monkeypatch, tmp_path: Path, capsys) -> None:
    """SPEC-004 §4: the record is measured from the saved rollouts, not copied."""
    rollouts = tmp_path / "rollouts-base.jsonl"
    _closure_rollouts(rollouts)
    output = tmp_path / "record"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-v2-probe-axis",
            "close",
            "--rollouts",
            str(rollouts),
            "--output",
            str(output),
        ],
    )

    assistant_axis.main()

    payload = json.loads((output / "closed.json").read_text(encoding="utf-8"))
    assert payload["min_expression"] == pytest.approx(0.34)
    assert payload["records"] == 6
    assert payload["rollouts"]["sha256"] == hashlib.sha256(rollouts.read_bytes()).hexdigest()
    assert payload["rollouts"]["path"] == str(rollouts.resolve())
    assert payload["heuristic"].startswith("src/local_llm_lab/probes/assistant_axis.py:")
    assert payload["decision"] == assistant_axis.CLOSURE_DECISION
    assert isinstance(payload["git_commit"], str) and payload["git_commit"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", payload["date"])

    counts = {entry["role"]: (entry["at_or_above"], entry["total"]) for entry in payload["roles"]}
    assert counts == {"pirate": (1, 2), "ghost": (0, 2)}
    assert payload["best_role"]["role"] == "pirate"
    assert payload["roles_at_zero"] == 1
    assert payload["n_roles"] == 2
    assert payload["default_assistant"]["at_or_above"] == 2
    assert payload["default_assistant"]["marker_backed"] is False

    markdown = (output / "CLOSED.md").read_text(encoding="utf-8")
    assert assistant_axis.CLOSURE_DECISION in markdown
    assert payload["rollouts"]["sha256"] in markdown
    assert payload["heuristic"] in markdown
    assert "| pirate | " in markdown
    assert "1 of 2" in markdown
    assert markdown in capsys.readouterr().out

    events = _events(output)
    assert events[0]["kind"] == "start"
    assert events[0]["run"] == "assistant-axis-close"
    start = _flat(events[0])
    assert start["rollouts_sha256"] == payload["rollouts"]["sha256"]
    assert start["min_expression"] == pytest.approx(0.34)
    progress = [_flat(event) for event in events if event["kind"] == "progress"]
    assert [(item["step"], item["total"], item["label"]) for item in progress] == [
        (1, 3, "role default-assistant"),
        (2, 3, "role pirate"),
        (3, 3, "role ghost"),
    ]
    assert (output / "run.log").is_file()
    assert _flat(events[-1])["status"] == "ok"


def test_close_honours_the_threshold_and_refuses_a_missing_file(
    monkeypatch, tmp_path: Path
) -> None:
    rollouts = tmp_path / "rollouts-base.jsonl"
    _closure_rollouts(rollouts)
    output = tmp_path / "record"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-v2-probe-axis",
            "close",
            "--rollouts",
            str(rollouts),
            "--output",
            str(output),
            "--min-expression",
            "0.60",
        ],
    )

    assistant_axis.main()

    payload = json.loads((output / "closed.json").read_text(encoding="utf-8"))
    assert payload["min_expression"] == pytest.approx(0.60)
    assert {entry["role"]: entry["at_or_above"] for entry in payload["roles"]} == {
        "pirate": 0,
        "ghost": 0,
    }
    assert payload["roles_at_zero"] == 2

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-v2-probe-axis",
            "close",
            "--rollouts",
            str(tmp_path / "absent.jsonl"),
            "--output",
            str(tmp_path / "other"),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        assistant_axis.main()
    assert raised.value.code == 2


def test_axis_verdict_passes_and_fails_on_the_registered_thresholds() -> None:
    passing = {
        "layers": {
            "12": {"pc1_cosine_abs": 0.49, "split_half_cosine": 0.99},
            "18": {"pc1_cosine_abs": 0.50, "split_half_cosine": 0.90},
            "30": {"pc1_cosine_abs": 1.00, "split_half_cosine": 1.00},
        }
    }
    assert assistant_axis.axis_verdict(passing)["status"] == "PASS"
    assert assistant_axis.axis_verdict(passing)["layer"] == 18

    failing = {
        "layers": {
            "12": {"pc1_cosine_abs": 0.80, "split_half_cosine": 0.89},
            "18": {"pc1_cosine_abs": 0.49, "split_half_cosine": 0.99},
        }
    }
    verdict = assistant_axis.axis_verdict(failing)
    assert verdict["status"] == "FAIL"
    assert verdict["pc1_cosine_abs"] == pytest.approx(0.80)


# ------------------------------- SPEC-001 §9/§10 closure: preflight, provenance, R18a/R18b


def _selected_spec(capture_dtype: str = "native") -> SimpleNamespace:
    """A registry stand-in carrying only what the P1 CLI reads off a ``ModelSpec``."""
    return SimpleNamespace(
        name="qwen35-4b",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.5, 1.0),
        probes=SimpleNamespace(capture_dtype=capture_dtype),
    )


def _build_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model="qwen35-4b",
        policy="base",
        output=tmp_path,
        allow_busy_gpu=False,
        prompts=1,
        role_prompts=1,
        max_tokens=1,
        layers=None,
        min_expression=0.34,
        exemplar=True,
        judge=False,
        skip_preflight_check=False,
    )


def test_axis_build_refuses_to_load_without_preflight_evidence(monkeypatch, tmp_path) -> None:
    """SPEC-001 §10: no preflight artifact for the resolved model, no model load."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, preflight
    from local_llm_lab.probes import guard, policies

    monkeypatch.setattr(models, "load_model_spec", lambda _name: _selected_spec())
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", tmp_path / "preflight")
    monkeypatch.setattr(policies, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )

    with pytest.raises(SystemExit) as raised:
        assistant_axis._build(_build_args(tmp_path), argparse.ArgumentParser())

    assert "qwen35-4b" in str(raised.value)
    assert "preflight" in str(raised.value)


def test_axis_project_refuses_to_load_without_preflight_evidence(monkeypatch, tmp_path) -> None:
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, preflight
    from local_llm_lab.probes import guard, policies

    monkeypatch.setattr(models, "load_model_spec", lambda _name: _selected_spec())
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", tmp_path / "preflight")
    monkeypatch.setattr(policies, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )
    args = argparse.Namespace(
        model="qwen35-4b",
        policy="base",
        output=tmp_path,
        allow_busy_gpu=False,
        axis=tmp_path / "axis.npz",
        eval=tmp_path / "eval.json",
        layer=1,
        limit=None,
        skip_preflight_check=False,
    )

    with pytest.raises(SystemExit) as raised:
        assistant_axis._project(args, argparse.ArgumentParser())

    assert "qwen35-4b" in str(raised.value)


def test_axis_build_records_precision_capture_dtype_and_writes_provenance(
    monkeypatch, tmp_path
) -> None:
    """R18a/R18b and SPEC-001 §9 in one completed fake build."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.probes import guard, policies

    selected = _selected_spec()
    block = {"frobenius_relative": 0.004, "elementwise_max": 0.02}
    order: list[str] = []
    monkeypatch.setattr(models, "load_model_spec", lambda _name: selected)
    monkeypatch.setattr(policies, "resolve_policy", lambda _name, _spec: None)
    monkeypatch.setattr(
        assistant_axis,
        "require_preflight",
        lambda spec, **kwargs: order.append(f"preflight:{spec.name}:{kwargs['skip']}"),
    )
    monkeypatch.setattr(assistant_axis, "preflight_precision_block", lambda _spec: dict(block))
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: order.append("guard"))
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args: order.append("load")
        or (object(), object(), SimpleNamespace(num_layers=4), SimpleNamespace(as_dict=dict)),
    )
    monkeypatch.setattr(assistant_axis, "load_chat_prompts", lambda _count: ["prompt"])

    def build(_model, _tokenizer, _prompts, layers, *_args, **kwargs):
        return (
            {layer: np.ones(2, dtype=np.float32) for layer in layers},
            {"layers": {}, "capture_dtype": kwargs["capture_dtype"]},
        )

    monkeypatch.setattr(assistant_axis, "build_axis_run", build)
    monkeypatch.setattr(assistant_axis, "render_build_markdown", lambda *_args: "# fake")

    assistant_axis._build(_build_args(tmp_path), argparse.ArgumentParser())

    assert order == ["preflight:qwen35-4b:False", "guard", "load"]
    diagnostics = json.loads((tmp_path / "axis-base.json").read_text(encoding="utf-8"))
    assert diagnostics["fp32_manual_vs_native"] == block
    assert diagnostics["capture_dtype"] == "native"
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["extra"]["stage"] == "p1-axis-build"
    assert provenance["extra"]["artifacts"] == [
        str(tmp_path / "axis-base.npz"),
        str(tmp_path / "axis-base.json"),
        str(tmp_path / "axis-base.md"),
    ]


def test_build_axis_captures_at_the_requested_dtype_and_records_it(monkeypatch) -> None:
    """R18b: the registry's capture dtype reaches P1's per-response means."""
    seen: list[str] = []

    def fake_response_mean(
        _model, _tokenizer, prompt, _response, layers, *, stats=None, dtype="float32"
    ):
        del stats, prompt
        seen.append(dtype)
        return {layer: np.arange(4, dtype=np.float32) + layer for layer in layers}

    monkeypatch.setattr(assistant_axis, "response_mean_activations", fake_response_mean)
    role_responses = {
        name: [(f"role:{name}:{run}", _marker_reply(name)) for run in range(3)]
        for name, _system in assistant_axis.ROLES
    }

    _axis, diagnostics = assistant_axis.build_axis(
        None,
        None,
        [("default:0", "assistant response")],
        role_responses,
        [1],
        capture_dtype="native",
    )

    assert diagnostics["capture_dtype"] == "native"
    assert set(seen) == {"native"}


def test_save_axis_leaves_no_partial_file_when_the_write_fails(monkeypatch, tmp_path) -> None:
    """R11: an interrupted array write never lands at the destination path."""
    path = tmp_path / "axis-base.npz"

    def explode(target, **_payload):
        """Write a truncated archive the way an interrupted save would, then stop."""
        if isinstance(target, (str, Path)):
            with open(target, "wb") as handle:
                handle.write(b"PK\x03\x04 truncated")
        else:
            target.write(b"PK\x03\x04 truncated")
        raise KeyboardInterrupt("interrupted mid-write")

    monkeypatch.setattr(assistant_axis.np, "savez_compressed", explode)

    with pytest.raises(KeyboardInterrupt):
        assistant_axis.save_axis(path, {1: np.ones(3, dtype=np.float32)}, {"layers": {}})

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_collect_rollouts_reports_the_default_generation_before_the_roles(
    monkeypatch, tmp_path
) -> None:
    """R26(g): the default rollout is a unit of work, and it is in the denominator."""
    calls: list[tuple[int, int, str]] = []
    monkeypatch.setattr(
        assistant_axis,
        "rollout_role",
        lambda *_args, **kwargs: [("prompt", f"reply from {kwargs['role']}")],
    )

    assistant_axis.collect_rollouts(
        None,
        None,
        ["prompt"],
        tmp_path,
        "base",
        role_prompts=1,
        progress=lambda step, total, label: calls.append((step, total, label)),
    )

    # The callback announces the unit it is starting, so the default rollout opens the run at
    # zero and each role's line counts the default in its denominator and its step.
    total = len(assistant_axis.ROLES) + 1
    assert calls[0] == (0, total, "default")
    assert calls[1] == (2, total, assistant_axis.ROLES[0][0])
    assert calls[-1] == (total, total, assistant_axis.ROLES[-1][0])
    assert len(calls) == total


def test_build_axis_reports_progress_through_the_scoring_phase(monkeypatch) -> None:
    """R26(g): the axis phase is not silent; every captured response is one unit."""
    monkeypatch.setattr(
        assistant_axis,
        "response_mean_activations",
        lambda _model, _tokenizer, _prompt, _response, layers, **_kwargs: {
            layer: np.arange(4, dtype=np.float32) + layer for layer in layers
        },
    )
    role_responses = {
        name: [(f"role:{name}:{run}", _marker_reply(name)) for run in range(3)]
        for name, _system in assistant_axis.ROLES
    }
    default = [("default:0", "assistant response")]
    calls: list[tuple[int, int, str]] = []

    assistant_axis.build_axis(
        None,
        None,
        default,
        role_responses,
        [1],
        progress=lambda step, total, label: calls.append((step, total, label)),
    )

    expected_total = len(default) + 3 * len(assistant_axis.ROLES)
    assert calls[0] == (1, expected_total, "capture default")
    assert [step for step, _total, _label in calls] == list(range(1, expected_total + 1))
    assert {total for _step, total, _label in calls} == {expected_total}
