"""Tests for model registry declarations and resolution seams."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_llm_lab import models
from local_llm_lab.arch import ArchitectureView
from local_llm_lab.models import load_model_spec
from local_llm_lab.project import PROJECT_ROOT  # noqa: F401
from local_llm_lab.runlock import box_state_root as _box_state_root


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
            # R1(b), the Director's decision of 2026-09-07 20:20: the history strategy, verified
            # bit-identically on the real checkpoint (research/records/HISTORY-CACHE-2026-09-07),
            # is this model's default. The other registered models keep R1's `auto`.
            "history",
            "auto",
            {
                "max_seq_length": 2688,
                "batch_size": 2,
                "grad_accumulation_steps": 2,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
                "launch_preflight": {
                    "chunk_size": 256,
                    "max_row_tokens": 2688,
                    "max_batch_size": 1,
                    "max_metal_cache_gib": 2.0,
                },
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
        (
            # The bf16 twin, for lens fitting: the hosted Jacobian lens was fitted on bf16 weights
            # and a comparison against it should not also be a comparison of precisions.
            "gemma3-4b-bf16",
            str((_box_state_root() / "models/gemma-3-4b-it-bf16").resolve()),
            "unsupported",
            "none",
            "auto",
            {
                "max_seq_length": 2688,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
            },
        ),
        (
            # The Gemma 3 pivot's registry entry. `unsupported` because Gemma has no thinking
            # mode; `none` rather than `auto` because 29 of its 34 per-block caches rotate and
            # no equivalence has been measured on this model, and a reuse strategy is an
            # equivalence claim. The train block is UNRULED and exists so no consumer meets a
            # bare key error; it is the Qwen recipe's values and is evidence about another model.
            "gemma3-4b",
            str((_box_state_root() / "models/gemma-3-4b-it-4bit").resolve()),
            "unsupported",
            "none",
            "auto",
            {
                "max_seq_length": 2688,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
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
        ("qwen35-4b", "history"),
        ("qwen35-9b", "auto"),
        ("gemma3-4b", "none"),
    ],
)
def test_registered_cache_declarations_start_without_equivalence_evidence(
    name: str, strategy: str
) -> None:
    """R1: no registered model carries snapshot-equivalence evidence; R1(b): qwen35-4b's declared
    strategy is ``history`` (Director, 2026-09-07 20:20), whose acceptance is a record, not this
    field — the field licenses ``snapshot``, which failed equivalence on the same day."""
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
            "generation_prefix": "<|im_start|>assistant\n",
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


@pytest.mark.parametrize(
    "name", ["qwen25-coder-3b", "qwen35-4b", "qwen35-9b", "gemma3-4b"]
)
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


@pytest.mark.parametrize("name", ["gemma3-4b-cuda-bf16", "gemma3-12b-cuda-bf16"])
def test_the_cuda_gemma_entries_declare_a_float32_lens_fit(name: str) -> None:
    """WS-D, the width rows: there is no width-independent bfloat16 Jacobian at these layers."""
    spec = load_model_spec(name)

    assert spec.probe_lens_fit_dtype == "float32"
    assert spec.probes.lens_fit_dtype == "float32"
    # And it is not the capture's dtype: the ruling separates the two deliberately.
    assert spec.probes.capture_dtype == "native"


def test_lens_fit_dtype_round_trips_and_is_absent_where_undeclared() -> None:
    raw = _registry_mapping(None)
    assert "lens_fit_dtype" not in raw["probes"]
    assert models._model_spec_from_mapping(raw, source="test").probes.lens_fit_dtype is None

    raw["probes"]["lens_fit_dtype"] = "bfloat16"
    assert models._model_spec_from_mapping(raw, source="test").probe_lens_fit_dtype == "bfloat16"


def test_an_unknown_lens_fit_dtype_is_refused_by_name() -> None:
    raw = _registry_mapping(None)
    raw["probes"]["lens_fit_dtype"] = "float16"

    with pytest.raises(ValueError, match="probes.lens_fit_dtype must be one of"):
        models._model_spec_from_mapping(raw, source="test")


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


def test_history_strategy_is_explicit_only_and_does_not_certify_a_checkpoint(monkeypatch):
    _install_view(monkeypatch, False, ("linear_attention",) * 3 + ("attention",))
    raw = _registry_mapping(None)
    raw["cache"]["strategy"] = "history"
    raw["lora"]["keys"] = "auto"
    spec = models._model_spec_from_mapping(raw, source="fixture")
    resolved = spec.resolve(object(), SimpleNamespace())
    assert resolved.cache_strategy == "history"
    assert spec.cache_equivalence_verified is None
    auto = replace(spec, cache_strategy="auto").resolve(object(), SimpleNamespace())
    assert auto.cache_strategy == "none"


@pytest.mark.parametrize(
    "name", ["qwen25-coder-3b", "qwen35-4b", "qwen35-9b", "gemma3-4b"]
)
def test_registry_budget_stays_the_declared_cap_the_device_resolves(name: str) -> None:
    """R32(b): the minimum is resolved at preflight, so the registry keeps its declared intent.

    Rewriting the YAML to this machine's working set would bind every other machine to it and
    lose the distinction the preflight artifact now records.
    """
    spec = load_model_spec(name)

    assert spec.memory_budget_gib == 22.0
    assert not any("device" in field.name for field in fields(spec))


def test_the_gemma_entry_declares_what_the_checkpoint_says_and_no_band() -> None:
    """The pivot's Phase 0 entry, pinned against what was read from the official files.

    Two values were checked against the **weight shapes** rather than against the config, because
    Gemma 3's `text_config` is sparse enough that MLX's own defaults are load-bearing: the
    embedding matrix is 262,208 rows and the query and key projections give 8 and 4 heads. Those
    are properties of the checkpoint, not of this file, so they are not asserted here; what is
    asserted is that this entry does not contradict them and does not invent what it cannot know.

    The turn ending is Gemma's, not ChatML's, which is the whole reason the entry has to exist:
    without it every Gemma run takes `_default_spec` and its `<|im_end|>`.

    No band and no tie-breaks. The band is a ruling and none exists for this model, and the
    periodicity that would derive it -- `sliding_window_pattern`, absent from Gemma's own config
    and defaulted to 6 by MLX -- has no home in this schema at all. Adding a field for it here
    would put a model constant in the registry that the architecture view should be reading from
    the blocks.
    """
    from local_llm_lab.models import load_model_spec

    spec = load_model_spec("gemma3-4b")

    assert spec.name == "gemma3-4b" and spec.family == "gemma3"
    assert spec.chat.end_of_turn == "<end_of_turn>", "Gemma's turn ending, not ChatML's"
    assert spec.chat.extra_stop_tokens == (), (
        "declared empty on purpose: the field is read by nothing (issue 98) and Gemma's "
        "terminators reach the generation loop from its own config"
    )
    assert spec.chat.template_kwargs == {}
    assert spec.probes.live_lens_pairs == (), "no band ruling exists for this model"
    assert spec.memory_budget_gib == 22.0
    assert spec.probe_layer_fractions == (0.167, 0.333, 0.5, 0.667, 0.833, 1.0)


def test_the_gemma_entry_is_what_stands_between_a_run_and_the_chatml_fallback() -> None:
    """Why the entry comes first, stated as a test rather than as a claim in a document.

    `load_model_spec` accepts a bare hf_id and synthesises a spec, and that fallback hands every
    model ChatML's turn ending. A Gemma run without this entry would render every turn with a
    token from another model's vocabulary, and nothing would say so.
    """
    from local_llm_lab.models import load_model_spec

    fallback = load_model_spec("google/gemma-3-4b-it-not-registered")
    registered = load_model_spec("gemma3-4b")

    assert fallback.chat.end_of_turn == "<|im_end|>", "the fallback is ChatML's, for any model"
    assert registered.chat.end_of_turn != fallback.chat.end_of_turn


def test_the_generation_prefix_is_declared_per_model_and_a_registry_file_cannot_omit_it(
    tmp_path, monkeypatch
) -> None:
    """The blocker the Gemma pivot found, and the guard that keeps it from returning.

    `protocol.generation_suffix` held ChatML's assistant marker as a literal and `build_prompt`
    asserted the rendered prompt ends with it, so **every generation-side render raised** on a
    model whose template opens a turn any other way. Gemma's opens with
    `<start_of_turn>model\n`.

    The assertion is deliberately kept: this value is stamped into training manifests and into
    the lens corpus's tokenizer identity, which the prose stage re-derives and hash-compares, so
    deleting the guard would convert a loud failure into a corpus identity that is falsified and
    self-consistent.

    Two lines of defence, and the second exists because the first covers only one construction
    path. The parser refuses a registry file that omits the field, which is asserted below. And
    `ChatSpec` itself takes it **keyword-only with no default**, because a ChatML default on the
    one field whose purpose is to stop a ChatML value being assumed is the same defect one level
    down -- and `_default_spec` builds a `ChatSpec` directly, which is exactly where `<|im_end|>`
    had been hiding.
    """
    import yaml

    from local_llm_lab import models
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.protocol import generation_suffix

    assert load_model_spec("gemma3-4b").chat.generation_prefix == "<start_of_turn>model\n"
    assert load_model_spec("qwen35-4b").chat.generation_prefix == "<|im_start|>assistant\n"
    assert generation_suffix(load_model_spec("gemma3-4b")) == "<start_of_turn>model\n", (
        "no thinking block: Gemma declares `unsupported`, and the empty-think literal is "
        "reached only under `off`"
    )

    with pytest.raises(TypeError, match="generation_prefix"):
        models.ChatSpec("unsupported", {}, "<eot>", ())  # type: ignore[call-arg]

    document = _registry_mapping(None)
    document["chat"].pop("generation_prefix")
    monkeypatch.setattr(models, "_REGISTRY_DIR", tmp_path)
    (tmp_path / "no-prefix.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="generation_prefix"):
        load_model_spec("no-prefix")


def test_a_local_checkpoint_resolves_against_the_project_root_not_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    """A registry path must name the same file from a worktree as from the repository root.

    `models/gemma-3-4b-it-4bit` is a relative path, and a stage that resolved it from wherever the
    process happened to be standing would work from the repository root and fail everywhere else.
    That is the shape of two separate faults this project hit in one day — a `trap` holding a
    relative interpreter path across a `cd`, and this — so the registry resolves it once, at load,
    against the project root.

    The prefix is what marks it, not a filesystem probe: a spec whose meaning depends on what
    happens to exist on the disk it is read from is the same defect one level up. A Hugging Face
    repo id has no prefix and passes through untouched, which the Qwen entries assert above.
    """
    from local_llm_lab.models import load_model_spec

    monkeypatch.chdir(tmp_path)

    for name, directory in (
        ("gemma3-4b", "models/gemma-3-4b-it-4bit"),
        ("gemma3-4b-bf16", "models/gemma-3-4b-it-bf16"),
    ):
        resolved = load_model_spec(name).hf_id
        assert resolved == str((_box_state_root() / directory).resolve())
        assert Path(resolved).is_absolute(), "a loader must not have to guess where this is"

    assert load_model_spec("qwen35-4b").hf_id == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_a_local_checkpoint_resolves_to_the_primary_checkout_not_the_running_one() -> None:
    """A converted checkpoint is a shared artefact, like the lock and the window.

    `models/` is git-ignored and exists once, in the primary checkout, so a worktree resolving
    against its own root finds nothing — which is exactly what happened the first time this ran
    from one. That is the box window's own bug before `box_state_root`, from the same cause and
    fixed by the same reader, and the third instance today of behaviour that depended on where a
    process was standing.
    """
    from pathlib import Path

    from local_llm_lab.models import load_model_spec
    from local_llm_lab.runlock import box_state_root

    resolved = Path(load_model_spec("gemma3-4b").hf_id)

    assert resolved.is_absolute()
    assert resolved == (box_state_root() / "models/gemma-3-4b-it-4bit").resolve()
    assert load_model_spec("qwen35-4b").hf_id == "mlx-community/Qwen3.5-4B-MLX-4bit"
