"""Tests for model registry declarations and resolution seams."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import load_model_spec


@pytest.mark.parametrize("name", ["qwen35-4b", "qwen35-9b"])
def test_hybrid_registry_disables_cache_until_r1(name: str) -> None:
    assert load_model_spec(name).cache_strategy == "none"


class _FakeHybridView:
    num_layers = 4
    hidden_size = 8
    vocab_size = 23
    tie_word_embeddings = False
    cache_trimmable = False

    def layer_kind(self, index: int) -> str:
        return "attention" if index == 3 else "linear_attention"

    def lora_targets(self, policy) -> tuple[str, ...]:
        assert policy == "auto"
        return ("linear_attn.out_proj",)

    def lora_parameter_count(self, keys, rank: int) -> int:
        assert keys == ("linear_attn.out_proj",)
        assert rank == 16
        return 256


@pytest.mark.xfail(strict=True, reason="R1")
def test_unverified_hybrid_snapshot_expectation(monkeypatch) -> None:
    fake_view = _FakeHybridView()
    monkeypatch.setattr(
        ArchitectureView,
        "from_model",
        classmethod(lambda cls, model: fake_view),
    )
    spec = load_model_spec("qwen35-4b")
    tokenizer = SimpleNamespace(snapshot_revision="fake-hybrid-revision")
    resolved = spec.resolve(object(), tokenizer)
    assert resolved.cache_strategy == "snapshot"
