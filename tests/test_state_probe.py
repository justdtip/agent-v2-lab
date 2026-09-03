from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.probes import state_probe


def test_explicit_registered_specs_preserve_legacy_bytes_and_qwen35_template_policy() -> None:
    from local_llm_lab.pipeline.protocol import build_prompt, generation_suffix

    class Tokenizer:
        def __init__(self) -> None:
            self.kwargs = []

        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
            assert not tokenize
            self.kwargs.append(kwargs)
            rendered = "".join(f"{message['role']}:{message['content']}\n" for message in messages)
            if not add_generation_prompt:
                return rendered
            suffix = "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                suffix += "<think>\n\n</think>\n\n"
            return rendered + suffix

    tokenizer = Tokenizer()
    legacy = load_model_spec("qwen25-coder-3b")
    qwen35 = load_model_spec("qwen35-4b")
    messages = [{"role": "user", "content": "hello"}]

    assert build_prompt(tokenizer, messages) == build_prompt(tokenizer, messages, spec=legacy)
    assert build_prompt(tokenizer, messages, spec=qwen35).endswith(generation_suffix(qwen35))
    assert tokenizer.kwargs[-1] == qwen35.chat.template_kwargs


def test_build_probe_dataset_forwards_the_selected_spec_to_prompt_rendering(monkeypatch) -> None:
    selected = load_model_spec("qwen35-4b")
    task = SimpleNamespace(task_id="fake", family="read")
    seen = []

    monkeypatch.setattr(state_probe, "_checkpoint_signature", lambda *_args, **_kwargs: "fake")
    monkeypatch.setattr(
        state_probe,
        "build_rows",
        lambda *_args, **_kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "y"},
                ],
                "metadata": {"step": 0},
            }
        ],
    )
    monkeypatch.setattr(
        state_probe,
        "build_prompt",
        lambda *_args, spec=None, **_kwargs: seen.append(spec) or "prompt",
    )
    monkeypatch.setattr(
        state_probe,
        "capture_residuals",
        lambda *_args, **_kwargs: {0: np.array([1.0])},
    )
    monkeypatch.setattr(state_probe, "_materialize_residuals", lambda captured, *_args: captured)
    monkeypatch.setattr(
        state_probe, "row_labels", lambda *_args: {name: 0.0 for name in state_probe.TARGETS}
    )
    monkeypatch.setattr(state_probe, "mlx_memory_snapshot", lambda *_args: {})

    runtime = SimpleNamespace(reset_peak_memory=lambda: None, clear_cache=lambda: None)
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1])
    state_probe.build_probe_dataset(
        None, tokenizer, [task], [0], spec=selected, mlx_runtime=runtime
    )

    assert seen == [selected]


def test_main_loads_and_dispatches_the_selected_spec(monkeypatch, tmp_path) -> None:
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard, policies

    selected = object()
    seen = []
    monkeypatch.setattr(
        models,
        "load_model_spec",
        lambda model: seen.append(("load", model)) or selected,
    )
    monkeypatch.setattr(
        tasks,
        "make_tasks",
        lambda *_args, **_kwargs: [SimpleNamespace(task_id="t", difficulty=0)],
    )
    monkeypatch.setattr(state_probe, "task_difficulties", lambda *_args: {"t": 0})
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda *_args: None)
    monkeypatch.setattr(evaluate, "load_policy", lambda *_args: (None, None))
    monkeypatch.setattr(state_probe, "artifact_identity", lambda *_args: {})
    monkeypatch.setattr(state_probe, "set_mlx_cache_limit", lambda *_args: 0)

    def intercept(*_args, **kwargs):
        seen.append(("dispatch", kwargs["spec"]))
        raise RuntimeError("stop after dispatch")

    monkeypatch.setattr(state_probe, "build_probe_dataset", intercept)
    monkeypatch.setattr(
        "sys.argv",
        ["state-probe", "--model", "qwen35-4b", "--output", str(tmp_path), "--limit", "1"],
    )

    with pytest.raises(RuntimeError, match="stop after dispatch"):
        state_probe.main()

    assert seen == [("load", "qwen35-4b"), ("dispatch", selected)]
