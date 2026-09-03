from __future__ import annotations

import json
import re
from pathlib import Path
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


def test_trajectory_projections_forwards_the_selected_spec_to_prompt_rendering(
    monkeypatch, tmp_path
) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import protocol

    selected = load_model_spec("qwen35-4b")
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "task_id": "fake",
                        "family": "read",
                        "variant": "clean",
                        "label": "x",
                        "prompt": "task",
                        "steps": [{"raw": "note"}],
                        "verdict": {"success": True},
                        "loop_detected": False,
                        "exhausted": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
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


def test_role_prompts_are_strong_and_have_markers_and_exemplars() -> None:
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
    for name, system in assistant_axis.ROLES:
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", system) if part]
        assert 2 <= len(sentences) <= 4, name
        assert "never mention being an AI or a language model" in system, name
        assert len(assistant_axis.ROLE_MARKERS[name]) >= 4, name
    assert len(set(assistant_axis.ROLE_EXEMPLARS) & low_and_neutral) >= 8
    for name, (_question, answer) in assistant_axis.ROLE_EXEMPLARS.items():
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", answer) if part]
        assert 1 <= len(sentences) <= 2, name


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
    ) -> dict[int, np.ndarray]:
        del model, tokenizer, response
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


def test_rollout_role_prepends_the_available_exemplar(monkeypatch, tmp_path: Path) -> None:
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
    assert assistant_axis.ROLE_EXEMPLARS["pirate"][1] in prompts_seen[0]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["exemplar"] is True and record["role"] == "pirate"


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
