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
        view=object(),
        resolved=object(),
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
        view=object(),
        resolved=object(),
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
            view=object(),
            resolved=object(),
            branches=1,
            temperature=1.0,
            max_steps=1,
            max_tokens=20,
            keep_last=2,
        )


# --------------------------------------------------- R20: view and resolved are required


@pytest.mark.parametrize("omitted", ["view", "resolved"])
def test_mine_pairs_requires_the_active_model_context(omitted: str) -> None:
    """R20: the optionals expire; a caller that omits either one is a TypeError, not a None."""
    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.tasks import Task

    task = Task("pref-read-0000-clean", "read", "clean", "p", {}, (), "done", frozenset())
    context = {"view": object(), "resolved": object()}
    del context[omitted]

    with pytest.raises(TypeError, match=omitted):
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
            **context,
        )


# ------------------------------------------- R2: the turn terminator comes from the run's spec


def _fake_spec(end_of_turn: str):
    """A registry-shaped specification whose declared terminator is not the 3B model's."""
    from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec

    return ModelSpec(
        name="fake-branch",
        hf_id="fake/branch",
        family="fake",
        chat=ChatSpec("unsupported", {}, end_of_turn, ()),
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
    )


def test_branch_completions_end_with_the_active_spec_terminator(monkeypatch) -> None:
    """R2: branch must read end_of_turn from the run's spec, never from the 3B module constant."""
    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.tasks import Task

    terminator = "<end-of-fake-turn>"
    spec = _fake_spec(terminator)
    task = Task("pref-read-0000-clean", "read", "clean", "p", {}, (), "done", frozenset())
    trajectory = types.SimpleNamespace(
        success=True,
        steps=[
            {
                "thought": "seed",
                "raw": f"seed raw{terminator}",
                "action": {"name": "read_file", "arguments": {"path": "a"}},
            }
        ],
    )
    simulator = types.SimpleNamespace(
        execute=lambda _action: "observation",
        verdict=lambda: types.SimpleNamespace(success=False),
    )
    monkeypatch.setattr(branch, "run_task", lambda *_args, **_kwargs: trajectory)
    monkeypatch.setattr(branch, "make_sampler", lambda _temperature: object())
    monkeypatch.setattr(branch, "build_prompt", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(branch, "generate_turn", lambda *_args: "branch raw")
    monkeypatch.setattr(
        branch,
        "parse_turn",
        lambda _raw: types.SimpleNamespace(
            action=Action("read_file", {"path": "b"}), thought="branch"
        ),
    )
    monkeypatch.setattr(branch.Simulator, "for_task", lambda _task: simulator)
    monkeypatch.setattr(branch, "_continue", lambda *_args, **_kwargs: False)

    pairs, _stats = branch.mine_pairs(
        object(),
        object(),
        task,
        spec=spec,
        view=object(),
        resolved=object(),
        branches=1,
        temperature=0.9,
        max_steps=2,
        max_tokens=4,
        keep_last=2,
    )

    assert pairs and pairs[0]["rejected"] == f"branch raw{terminator}"
    assert pairs[0]["rejected"].endswith(terminator)


def test_run_branch_mining_hands_the_resolved_identity_to_its_caller(monkeypatch, tmp_path) -> None:
    """C7: the resolved identity has to leave this function without changing its return shape.

    ``run_branch_mining`` resolves the spec against the loaded model but returns a summary dict
    the CLI's branch command depends on (cli.py:1209), so the entry point that writes provenance
    receives the ResolvedSpec through the sink instead.
    """
    import sys
    from types import ModuleType

    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import branch

    spec = load_model_spec("qwen35-4b")
    resolved = types.SimpleNamespace(as_dict=lambda: {"spec": {"name": spec.name}})
    cleared: list[str] = []
    resolved_sink: list[object] = []

    core = ModuleType("mlx.core")
    core.random = types.SimpleNamespace(seed=lambda _value: None)
    core.clear_cache = lambda: cleared.append("clear_cache")
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    monkeypatch.setattr(branch, "load_model_spec", lambda _name: spec, raising=False)
    monkeypatch.setattr(
        branch, "load_policy", lambda _spec, _adapter: (object(), object(), object(), resolved)
    )
    monkeypatch.setattr(branch, "make_tasks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(branch, "write_jsonl", lambda _path, _rows: "digest")
    monkeypatch.setattr(branch, "mine_pairs", lambda *_a, **_k: pytest.fail("no task to mine"))

    summary = branch.run_branch_mining(
        model_name="qwen35-4b",
        adapter=None,
        split="pref-resolved-identity",
        limit=1,
        branches=1,
        temperature=0.9,
        output=tmp_path,
        transcript_dir=None,
        quiet=True,
        on_resolved=resolved_sink.append,
    )

    assert summary["model"] == resolved.as_dict()
    assert resolved_sink == [resolved]
    assert cleared == ["clear_cache"]


# -------------------------------------------- module main (R21, R26(a), provenance)


def _branch_argv(output, transcripts) -> list[str]:
    return [
        "agent-v2-branch",
        "--split",
        "guard-check",
        "--output",
        str(output),
        "--transcripts",
        str(transcripts),
        "--quiet",
    ]


def test_branch_main_refuses_a_protected_dataset_directory(monkeypatch, tmp_path) -> None:
    """R21: the module main must refuse PROTECTED_DATASETS as hard as the CLI boundary does."""
    import sys

    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.data import ProtectedDatasetError
    from local_llm_lab.project import PROJECT_ROOT

    monkeypatch.setattr(
        branch, "run_branch_mining", lambda **_kwargs: pytest.fail("the R21 guard did not fire")
    )
    monkeypatch.setattr(
        sys, "argv", _branch_argv(PROJECT_ROOT / "data" / "agent_v2", tmp_path / "transcripts")
    )

    with pytest.raises(ProtectedDatasetError):
        branch.main()

    assert not (tmp_path / "transcripts").exists()


def test_branch_main_refuses_an_existing_manifest_with_no_override(monkeypatch, tmp_path) -> None:
    """R21: this stage has no override flag, so an existing dataset is simply refused."""
    import sys

    from local_llm_lab.pipeline import branch
    from local_llm_lab.pipeline.data import DatasetWriteGuardError

    target = tmp_path / "preferences"
    target.mkdir()
    (target / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        branch, "run_branch_mining", lambda **_kwargs: pytest.fail("the R21 guard did not fire")
    )
    monkeypatch.setattr(sys, "argv", _branch_argv(target, tmp_path / "transcripts"))

    with pytest.raises(DatasetWriteGuardError) as error:
        branch.main()

    assert "no overwrite path" in str(error.value)


def test_branch_main_writes_run_log_events_manifest_and_provenance(monkeypatch, tmp_path) -> None:
    """R26(a) plus the missing provenance stamp beside the mined artifact."""
    import json
    import sys

    from local_llm_lab.pipeline import branch

    target = tmp_path / "preferences"
    captured: dict[str, object] = {}
    resolved = types.SimpleNamespace(as_dict=lambda: {"spec": {"name": "resolved-by-the-run"}})

    def fake_run(**kwargs):
        captured.update(kwargs)
        kwargs["on_resolved"](resolved)
        return {"pairs": 0, "branch_points": 0, "seed": 13}

    monkeypatch.setattr(branch, "run_branch_mining", fake_run)
    monkeypatch.setattr(sys, "argv", _branch_argv(target, tmp_path / "transcripts"))

    branch.main()

    assert (target / "run.log").is_file()
    events = [
        json.loads(line)
        for line in (target / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["kind"] for event in events][:1] == ["start"]
    assert len(events) >= 2
    identity = events[0]["fields"]
    assert identity["split"] == "guard-check"
    assert identity["model"] and identity["hf_id"]
    assert callable(captured["progress"])
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "branch"
    assert manifest["split"] == "guard-check"
    assert manifest["seed"] == 13
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["extra"]["stage"] == "branch"
    # C7: the top-level ``model`` block carries the resolved identity the run built against the
    # loaded model -- the shape every stage that holds a ResolvedSpec records there
    # (cli.py:665, probes/patch.py:2408) -- not the static registry spec.
    assert provenance["model"] == resolved.as_dict()


def _interrupt_the_manifest_writer(monkeypatch, limit: int) -> None:
    """Truncate whichever writer the stamp uses and then fail, standing in for a full disk.

    ``runlog.write_text_atomic`` writes through ``runlog``'s own ``os.fdopen`` — the shim the
    sibling lane's ``tests/test_data.py`` interrupts — while a plain stamp writes through
    ``Path.write_text``. Interrupting both keeps the assertion on the outcome the R21 guard
    depends on (no partial sentinel at the destination) rather than on which writer is in use.
    """
    import os
    from pathlib import Path

    from local_llm_lab import runlog as runlog_module

    class _Truncating:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __getattr__(self, name: str):
            return getattr(self._handle, name)

        def write(self, text: str) -> int:
            self._handle.write(text[:limit])
            raise OSError("no space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            self._handle.close()
            return False

    class _InterruptingOS:
        """An ``os`` shim patched into runlog's namespace only, so nothing else is affected."""

        def __init__(self, real) -> None:
            self._real = real

        def __getattr__(self, name: str):
            return getattr(self._real, name)

        def fdopen(self, descriptor, *args, **kwargs):
            return _Truncating(self._real.fdopen(descriptor, *args, **kwargs))

    plain_write_text = Path.write_text

    def truncating_write_text(self, data, *args, **kwargs):
        plain_write_text(self, data[:limit], *args, **kwargs)
        raise OSError("no space left on device")

    monkeypatch.setattr(runlog_module, "os", _InterruptingOS(os))
    monkeypatch.setattr(Path, "write_text", truncating_write_text)


def test_an_interrupted_branch_manifest_write_leaves_the_previous_manifest_intact(
    monkeypatch, tmp_path
) -> None:
    """manifest.json is the R21 guard's own sentinel, so a half-written one is a live hazard.

    ``guard_dataset_write`` (pipeline/data.py:85) decides whether a later write is permitted by
    whether this file is there. With the mined pairs complete and the sentinel truncated, a
    later write would be waved straight over good data, so this stamp must be atomic: either
    the previous manifest survives whole or the new one lands whole.
    """
    from local_llm_lab.pipeline import branch

    target = tmp_path / "preferences"
    target.mkdir()
    original = '{"stage": "branch", "kept": true}\n'
    (target / "manifest.json").write_text(original, encoding="utf-8")

    _interrupt_the_manifest_writer(monkeypatch, 12)
    with pytest.raises(OSError, match="no space left on device"):
        branch._write_stage_manifest(target, {"stage": "branch", "split": "guard-check"})
    monkeypatch.undo()

    assert (target / "manifest.json").read_text(encoding="utf-8") == original
    leftovers = sorted(path.name for path in target.iterdir() if path.name.startswith("."))
    assert leftovers == [], "no partial temporary file may survive at the destination"
