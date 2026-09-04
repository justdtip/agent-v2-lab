from __future__ import annotations

import types

import pytest

from local_llm_lab.agent_protocol import Action


def test_render_completion_delegates_with_the_legacy_declaration(monkeypatch) -> None:
    from local_llm_lab.pipeline import branch

    seen = []
    monkeypatch.setattr(
        branch,
        "protocol_render_completion",
        lambda thought, action, *, spec: seen.append((thought, action, spec)) or "canonical\n",
    )
    action = Action("finish", {"answer": "done"})

    assert branch.render_completion("note", action) == "canonical"
    assert seen == [("note", action, branch._LEGACY_SPEC)]


def test_render_completion_preserves_legacy_end_token_bytes() -> None:
    from local_llm_lab.pipeline import branch

    assert branch.render_completion("note", Action("finish", {"answer": "done"})).endswith(
        "<|im_end|>"
    )
    assert not branch.render_completion("note", Action("finish", {"answer": "done"})).endswith(
        "<|im_end|>\n"
    )


def test_continue_forwards_the_active_spec_to_prompt_rendering(monkeypatch) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import branch

    spec = load_model_spec("qwen35-4b")
    prompt_specs = []
    turn = types.SimpleNamespace(action=Action("finish", {"answer": "done"}), thought="done")
    simulator = types.SimpleNamespace(
        execute=lambda _action: "finished",
        verdict=lambda: types.SimpleNamespace(success=True),
    )
    monkeypatch.setattr(
        branch,
        "build_prompt",
        lambda _tokenizer, _messages, **kwargs: prompt_specs.append(kwargs["spec"]) or "prompt",
    )
    monkeypatch.setattr(branch, "generate_turn", lambda *_args: "raw")
    monkeypatch.setattr(branch, "parse_turn", lambda _raw: turn)

    assert branch._continue(
        object(),
        object(),
        object(),
        [{"role": "user", "content": "task"}],
        simulator,
        spec=spec,
        sampler=object(),
        remaining_steps=1,
        max_tokens=4,
        keep_last=2,
    )
    assert prompt_specs == [spec]


def test_mine_pairs_forwards_the_active_spec_to_seed_and_branch_paths(monkeypatch) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.tasks import Task

    spec = load_model_spec("qwen35-4b")
    task = Task("pref-read-0000-clean", "read", "clean", "p", {}, (), "done", frozenset())
    trajectory = types.SimpleNamespace(
        success=True,
        steps=[
            {
                "thought": "seed",
                "raw": "seed raw",
                "action": {"name": "read_file", "arguments": {"path": "a"}},
            }
        ],
    )
    seed_specs = []
    prompt_specs = []
    continuation_specs = []
    simulator = types.SimpleNamespace(
        execute=lambda _action: "observation",
        verdict=lambda: types.SimpleNamespace(success=False),
    )
    monkeypatch.setattr(
        branch,
        "run_task",
        lambda *_args, **kwargs: seed_specs.append(kwargs["spec"]) or trajectory,
    )
    monkeypatch.setattr(branch, "make_sampler", lambda _temperature: object())
    monkeypatch.setattr(
        branch,
        "build_prompt",
        lambda _tokenizer, _messages, **kwargs: prompt_specs.append(kwargs["spec"]) or "prompt",
    )
    monkeypatch.setattr(branch, "generate_turn", lambda *_args: "branch raw")
    monkeypatch.setattr(
        branch,
        "parse_turn",
        lambda _raw: types.SimpleNamespace(
            action=Action("read_file", {"path": "b"}), thought="branch"
        ),
    )
    monkeypatch.setattr(branch.Simulator, "for_task", lambda _task: simulator)
    monkeypatch.setattr(
        branch,
        "_continue",
        lambda *_args, **kwargs: continuation_specs.append(kwargs["spec"]) or False,
    )

    branch.mine_pairs(
        object(),
        object(),
        task,
        spec=spec,
        branches=1,
        temperature=0.9,
        max_steps=2,
        max_tokens=4,
        keep_last=2,
    )

    assert seed_specs == [spec]
    assert prompt_specs == [spec]
    assert continuation_specs == [spec]


def test_branch_pairs_keep_the_seed_step_raw_completion(monkeypatch) -> None:
    """Canonical re-rendering must not replace a saved trajectory's chosen raw bytes."""
    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.tasks import Task

    chosen = branch.render_completion("saved raw note", Action("finish", {"answer": "yes"}))
    rejected = branch.render_completion("wrong", Action("finish", {"answer": "no"}))
    task = Task("valid-read-0000-clean", "read", "clean", "p", {}, (), "yes", frozenset())
    trajectory = types.SimpleNamespace(
        success=True,
        steps=[
            {
                "thought": "canonical note",
                "raw": chosen,
                "action": {"name": "finish", "arguments": {"answer": "yes"}},
            }
        ],
    )
    monkeypatch.setattr(branch, "run_task", lambda *args, **kwargs: trajectory)
    monkeypatch.setattr(branch, "make_sampler", lambda temperature: object())
    monkeypatch.setattr(branch, "build_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(branch, "generate_turn", lambda *args, **kwargs: rejected)

    pairs, _ = branch.mine_pairs(
        None,
        None,
        task,
        spec=branch._LEGACY_SPEC,
        branches=1,
        temperature=1.0,
        max_steps=1,
        max_tokens=20,
        keep_last=2,
    )

    assert pairs[0]["chosen"] == chosen
    assert pairs[0]["chosen"] != branch.render_completion(
        "canonical note", Action("finish", {"answer": "yes"})
    )


def test_branch_pairs_reject_seed_steps_without_raw_completion(monkeypatch) -> None:
    """A missing raw completion is a producer-contract failure, not a re-rendering fallback."""
    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.tasks import Task

    task = Task("valid-read-0000-clean", "read", "clean", "p", {}, (), "yes", frozenset())
    trajectory = types.SimpleNamespace(
        success=True,
        steps=[
            {
                "thought": "canonical note",
                "raw": "",
                "action": {"name": "finish", "arguments": {"answer": "yes"}},
            }
        ],
    )
    monkeypatch.setattr(branch, "run_task", lambda *args, **kwargs: trajectory)
    monkeypatch.setattr(branch, "make_sampler", lambda temperature: object())
    monkeypatch.setattr(branch, "build_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(
        branch,
        "generate_turn",
        lambda *args, **kwargs: branch.render_completion(
            "other", Action("finish", {"answer": "yes"})
        ),
    )

    with pytest.raises(ValueError, match="raw completion"):
        branch.mine_pairs(
            None,
            None,
            task,
            spec=branch._LEGACY_SPEC,
            branches=1,
            temperature=1.0,
            max_steps=1,
            max_tokens=20,
            keep_last=2,
        )
