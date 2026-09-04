"""Tests for model registry declarations and resolution seams."""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import yaml

from local_llm_lab import models
from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import load_model_spec


@pytest.mark.parametrize(
    ("name", "hf_id", "thinking", "cache_strategy", "lora_keys", "train"),
    [
        (
            "qwen25-coder-3b",
            "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
            "unsupported",
            "trim",
            "attention+mlp",
            {
                "max_seq_length": 2688,
                "batch_size": 2,
                "grad_accumulation_steps": 2,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
            },
        ),
        (
            "qwen35-4b",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            "off",
            "auto",
            "auto",
            {
                "max_seq_length": 2688,
                "batch_size": 2,
                "grad_accumulation_steps": 2,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
            },
        ),
        (
            "qwen35-9b",
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            "off",
            "auto",
            "auto",
            {
                "max_seq_length": 2688,
                "batch_size": 1,
                "grad_accumulation_steps": 4,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
            },
        ),
    ],
)
def test_model_spec_registry_values(
    name: str,
    hf_id: str,
    thinking: str,
    cache_strategy: str,
    lora_keys: str,
    train: dict[str, object],
) -> None:
    from local_llm_lab.models import load_model_spec

    spec = load_model_spec(name)

    assert spec.hf_id == hf_id
    assert spec.chat.thinking == thinking
    assert spec.cache_strategy == cache_strategy
    assert spec.lora.keys == lora_keys
    assert spec.train == train


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


@pytest.mark.parametrize("name", ["qwen25-coder-3b", "qwen35-4b", "qwen35-9b"])
def test_registered_probe_capture_dtype_is_native(name: str) -> None:
    """R18b: native block execution is the registry default for every registered model."""
    spec = load_model_spec(name)

    assert spec.probe_capture_dtype == "native"
    assert spec.probes.capture_dtype == "native"


@pytest.mark.parametrize("declared", ["native", "float32"])
def test_probe_capture_dtype_round_trips_through_the_registry(declared: str) -> None:
    raw = _registry_mapping(None)
    raw["probes"]["capture_dtype"] = declared

    spec = models._model_spec_from_mapping(raw, source="test")

    assert spec.probe_capture_dtype == declared
    assert spec.probes.capture_dtype == declared


def test_probe_capture_dtype_defaults_to_native_when_the_registry_omits_it() -> None:
    raw = _registry_mapping(None)
    assert "capture_dtype" not in raw["probes"]

    assert models._model_spec_from_mapping(raw, source="test").probe_capture_dtype == "native"


@pytest.mark.parametrize("declared", ["float16", "bfloat16", "", 32, None])
def test_probe_capture_dtype_rejects_anything_but_native_or_float32(declared) -> None:
    raw = _registry_mapping(None)
    raw["probes"]["capture_dtype"] = declared

    with pytest.raises(ValueError, match=r"probes\.capture_dtype"):
        models._model_spec_from_mapping(raw, source="test")


def test_probes_view_exposes_the_specs_own_field_names() -> None:
    """SPEC-004 §2 and R17 name ``spec.probes.layer_fractions``; the stored field is unchanged."""
    spec = load_model_spec("qwen35-4b")

    assert spec.probes.layer_fractions == spec.probe_layer_fractions
    assert spec.probes.capture_dtype == spec.probe_capture_dtype


def test_default_spec_for_an_unregistered_model_captures_natively() -> None:
    spec = load_model_spec("example/not-registered")

    assert spec.probe_capture_dtype == "native"
    assert spec.probes.layer_fractions == spec.probe_layer_fractions


@pytest.mark.parametrize("strategy", ["trim", "snapshot", "none"])
def test_explicit_cache_resolution_records_declared_strategy(monkeypatch, strategy: str) -> None:
    _install_view(monkeypatch, False, ("linear_attention",) * 4)
    spec = replace(load_model_spec("qwen35-4b"), cache_strategy=strategy)

    resolved = spec.resolve(object(), SimpleNamespace())

    assert (resolved.cache_strategy, resolved.cache_strategy_reason) == (
        strategy,
        f"explicit:{strategy}",
    )


@pytest.mark.parametrize("name", ["qwen25-coder-3b", "qwen35-4b", "qwen35-9b"])
def test_registry_budget_stays_the_declared_cap_the_device_resolves(name: str) -> None:
    """R32(b): the minimum is resolved at preflight, so the registry keeps its declared intent.

    Rewriting the YAML to this machine's working set would bind every other machine to it and
    lose the distinction the preflight artifact now records.
    """
    spec = load_model_spec(name)

    assert spec.memory_budget_gib == 22.0
    assert not any("device" in field.name for field in fields(spec))
