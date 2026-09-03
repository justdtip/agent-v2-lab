"""Tests for model registry declarations and resolution seams."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from local_llm_lab import models
from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import load_model_spec


@pytest.mark.parametrize(
    ("name", "strategy"),
    [
        ("qwen25-coder-3b", "trim"),
        ("qwen35-4b", "auto"),
        ("qwen35-9b", "auto"),
    ],
)
def test_registered_cache_declarations_start_without_equivalence_evidence(
    name: str, strategy: str
) -> None:
    spec = load_model_spec(name)

    assert spec.cache_strategy == strategy
    assert spec.cache_equivalence_verified is None


def _registry_mapping(evidence) -> dict:
    return {
        "name": "test-model",
        "hf_id": "example/test-model",
        "family": "test",
        "chat": {
            "thinking": "unsupported",
            "template_kwargs": {},
            "end_of_turn": "<|im_end|>",
            "extra_stop_tokens": [],
        },
        "lora": {"keys": "attention+mlp", "rank": 16, "scale": 32.0, "dropout": 0.0},
        "train": {},
        "cache": {"strategy": "auto", "equivalence_verified": evidence},
        "probes": {"layer_fractions": [0.5]},
        "memory": {"budget_gib": 22},
        "policies": {},
    }


@pytest.mark.parametrize(
    "evidence",
    [
        "2026-09-03",
        {},
        {"date": "2026-09-03"},
        {"date": "2026-09-03", "sha256": "a" * 64, "source": "extra"},
        {"date": "", "sha256": "a" * 64},
        {"date": "2026-09-03", "sha256": ""},
        {"date": 20260903, "sha256": "a" * 64},
        {"date": "2026-09-03", "sha256": 12},
        {"date": "2026-09-03", "sha256": "g" * 64},
        {"date": "2026-09-03", "sha256": "a" * 63},
    ],
)
def test_cache_equivalence_evidence_rejects_malformed_registry_values(
    tmp_path, monkeypatch, evidence
) -> None:
    (tmp_path / "test-model.yaml").write_text(
        yaml.safe_dump(_registry_mapping(evidence)), encoding="utf-8"
    )
    monkeypatch.setattr(models, "_REGISTRY_DIR", tmp_path)

    with pytest.raises(ValueError, match=r"cache\.equivalence_verified"):
        load_model_spec("test-model")


def test_cache_equivalence_evidence_is_copied_from_registry() -> None:
    evidence = {"date": "2026-09-03", "sha256": "a" * 64}
    raw = _registry_mapping(evidence)

    spec = models._model_spec_from_mapping(raw, source="test")
    evidence["date"] = "changed"

    assert spec.cache_equivalence_verified == {
        "date": "2026-09-03",
        "sha256": "a" * 64,
    }


class _FakeView:
    num_layers = 4
    hidden_size = 8
    vocab_size = 23
    tie_word_embeddings = False

    def __init__(self, cache_trimmable: bool, layer_types: tuple[str, ...]) -> None:
        self.cache_trimmable = cache_trimmable
        self.layer_types = layer_types

    def layer_kind(self, index: int) -> str:
        return self.layer_types[index]

    def lora_targets(self, policy) -> tuple[str, ...]:
        assert policy == "auto"
        return ("linear_attn.out_proj",)

    def lora_parameter_count(self, keys, rank: int) -> int:
        assert keys == ("linear_attn.out_proj",)
        assert rank == 16
        return 256


def _install_view(monkeypatch, cache_trimmable: bool, layer_types: tuple[str, ...]) -> _FakeView:
    fake_view = _FakeView(cache_trimmable, layer_types)
    monkeypatch.setattr(
        ArchitectureView,
        "from_model",
        classmethod(lambda cls, model: fake_view),
    )
    return fake_view


@pytest.mark.parametrize(
    ("cache_trimmable", "layer_types", "evidence", "strategy", "reason"),
    [
        (True, ("attention",) * 4, None, "trim", "auto:trimmable"),
        (
            False,
            ("linear_attention", "linear_attention", "linear_attention", "attention"),
            {"date": "2026-09-03", "sha256": "a" * 64},
            "snapshot",
            "auto:equivalence_verified",
        ),
        (
            False,
            ("linear_attention", "linear_attention", "linear_attention", "attention"),
            None,
            "none",
            "auto:equivalence_unverified",
        ),
    ],
)
def test_auto_cache_resolution_records_safe_branch(
    monkeypatch, cache_trimmable, layer_types, evidence, strategy, reason
) -> None:
    _install_view(monkeypatch, cache_trimmable, layer_types)
    spec = replace(
        load_model_spec("qwen35-4b"),
        cache_strategy="auto",
        cache_equivalence_verified=evidence,
    )
    tokenizer = SimpleNamespace(snapshot_revision="fake-hybrid-revision")
    resolved = spec.resolve(object(), tokenizer)

    assert (resolved.cache_strategy, resolved.cache_strategy_reason) == (strategy, reason)
    assert resolved.as_dict()["cache_strategy_reason"] == reason


@pytest.mark.parametrize("strategy", ["trim", "snapshot", "none"])
def test_explicit_cache_resolution_records_declared_strategy(monkeypatch, strategy: str) -> None:
    _install_view(monkeypatch, False, ("linear_attention",) * 4)
    spec = replace(load_model_spec("qwen35-4b"), cache_strategy=strategy)

    resolved = spec.resolve(object(), SimpleNamespace())

    assert (resolved.cache_strategy, resolved.cache_strategy_reason) == (
        strategy,
        f"explicit:{strategy}",
    )
