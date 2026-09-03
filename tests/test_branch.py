from __future__ import annotations

import types

import pytest

from local_llm_lab.agent_protocol import Action


def test_render_completion_delegates_to_the_canonical_protocol_renderer() -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import branch, protocol

    action = Action("finish", {"answer": "done"})

    assert branch.render_completion("note", action) == protocol.render_completion(
        "note", action, spec=load_model_spec(branch.DEFAULT_MODEL)
    )


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
        None, None, task, branches=1, temperature=1.0, max_steps=1, max_tokens=20, keep_last=2
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
            None, None, task, branches=1, temperature=1.0, max_steps=1, max_tokens=20, keep_last=2
        )
