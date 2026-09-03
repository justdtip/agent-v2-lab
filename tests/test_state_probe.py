from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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
