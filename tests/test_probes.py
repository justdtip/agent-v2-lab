"""Tests for the interpretability probe suite (``src/local_llm_lab/probes``).

Everything here runs against a tiny fake model built with mlx primitives, following the
pattern the J-lens tests in ``test_pipeline.py`` established: the real checkpoint is a 3B
4-bit model whose GPU may be busy, so no test loads it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.pipeline import jlens
from local_llm_lab.probes import adapter_delta, assistant_axis, capture, state_probe, stats

# --------------------------------------------------------------------------- fake model


class _ProbeBlock(nn.Module):
    """A tiny transformer-shaped block: mask-sensitive attention plus a nonlinear MLP.

    Deliberately *uses* ``mask`` (unlike the J-lens test block, which ignores it) so that a
    capture taken with the wrong mask, or with none, gives a different answer and the
    equivalence check below has teeth.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x, mask=None, cache=None):
        del cache
        scores = (self.query(x) @ x.transpose(0, 2, 1)) / math.sqrt(x.shape[-1])
        if mask is not None:
            if isinstance(mask, str):  # create_attention_mask returns "causal"
                length = x.shape[1]
                causal = mx.tril(mx.ones((length, length)))
                scores = mx.where(causal > 0, scores, -1e9)
            else:
                scores = scores + mask
        h = x + 0.1 * (mx.softmax(scores, axis=-1) @ self.value(x))
        return h + 0.1 * mx.tanh(self.out(h))


class _ProbeInner(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = [_ProbeBlock(dim) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim)

    def __call__(self, inputs, cache=None):
        """Mirrors ``Qwen2Model.__call__``: embed, build the mask once, run every block."""
        from mlx_lm.models.base import create_attention_mask

        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, entry in zip(self.layers, cache, strict=True):
            h = layer(h, mask, entry)
        return self.norm(h)


class _ProbeModel(nn.Module):
    def __init__(self, vocab: int = 40, dim: int = 16, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _ProbeInner(vocab, dim, n_layers)


def _model(seed: int = 0) -> _ProbeModel:
    mx.random.seed(seed)
    return _ProbeModel()


_ATTENTION_MASK = "fake-attention-mask"
_SSM_MASK = "fake-ssm-mask"


class _FakeKVCache:
    def __init__(self) -> None:
        self.offset = 0
        self.state = ("keys", "values")

    def is_trimmable(self) -> bool:
        return True


class _FakeArraysCache:
    def __init__(self) -> None:
        self.state = [None, None]

    def is_trimmable(self) -> bool:
        return False


class _ArchitectureAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        scores = self.q_proj(x) @ self.k_proj(x).transpose(0, 2, 1)
        length = x.shape[1]
        scores = mx.where(mx.tril(mx.ones((length, length))) > 0, scores, -1e9)
        return self.o_proj(mx.softmax(scores, axis=-1) @ self.v_proj(x))


class _ArchitectureLinearAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.in_proj_qkvz = nn.Linear(dim, 2 * dim, bias=False)
        self.in_proj_ba = nn.Linear(dim, 2, bias=False)
        self.out_proj = nn.Linear(2 * dim, dim, bias=False)

    def __call__(self, x):
        gate = mx.mean(mx.sigmoid(self.in_proj_ba(x)), axis=-1, keepdims=True)
        return self.out_proj(mx.tanh(self.in_proj_qkvz(x)) * gate)


class _ArchitectureSplitLinearAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(dim, 2 * dim, bias=False)
        self.in_proj_z = nn.Linear(dim, dim, bias=False)
        self.in_proj_b = nn.Linear(dim, 2, bias=False)
        self.in_proj_a = nn.Linear(dim, 2, bias=False)
        self.out_proj = nn.Linear(2 * dim, dim, bias=False)

    def __call__(self, x):
        z = self.in_proj_z(x)
        z = mx.concatenate((z, z), axis=-1)
        gate = mx.mean(
            mx.sigmoid(self.in_proj_b(x) + self.in_proj_a(x)), axis=-1, keepdims=True
        )
        return self.out_proj(mx.tanh(self.in_proj_qkv(x) + z) * gate)


class _ArchitectureMLP(nn.Module):
    def __init__(self, dim: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(mx.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class _ArchitectureBlock(nn.Module):
    def __init__(self, dim: int, *, is_linear: bool, split_linear: bool = False) -> None:
        super().__init__()
        self.is_linear = is_linear
        if is_linear:
            self.linear_attn = (
                _ArchitectureSplitLinearAttention(dim)
                if split_linear
                else _ArchitectureLinearAttention(dim)
            )
        else:
            self.self_attn = _ArchitectureAttention(dim)
        self.mlp = _ArchitectureMLP(dim, intermediate=12)
        self.calls = 0

    def __call__(self, x, mask=None, cache=None):
        del cache
        required_mask = _SSM_MASK if self.is_linear else _ATTENTION_MASK
        if mask != required_mask:
            raise ValueError(f"wrong mask for block kind: {mask!r}")
        self.calls += 1
        attention = self.linear_attn(x) if self.is_linear else self.self_attn(x)
        h = x + 0.05 * attention
        return h + 0.05 * self.mlp(h)


class _ArchitectureText(nn.Module):
    def __init__(
        self, *, vocab: int, dim: int, kinds: tuple[bool, ...], split_linear: bool = False
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = [
            _ArchitectureBlock(dim, is_linear=kind, split_linear=split_linear) for kind in kinds
        ]
        self.norm = nn.RMSNorm(dim)
        self.mask_calls: list[tuple[str, object | None]] = []

    def create_attention_mask(self, h, cache=None):
        del h
        self.mask_calls.append(("attention", cache))
        return _ATTENTION_MASK

    def create_ssm_mask(self, h, cache=None):
        del h
        self.mask_calls.append(("linear_attention", cache))
        return _SSM_MASK


class _ArchitectureFakeBase(nn.Module):
    text: _ArchitectureText

    def make_cache(self) -> list[object]:
        return [
            _FakeArraysCache() if block.is_linear else _FakeKVCache()
            for block in self.text.layers
        ]

    def _unembed(self, h):
        language_model = getattr(self, "language_model", None)
        if language_model is not None and hasattr(language_model, "lm_head"):
            return language_model.lm_head(h)
        return self.text.embed_tokens.as_linear(h)

    def recording_forward(self, ids):
        token_ids = mx.array(ids).astype(mx.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
        h = self.text.embed_tokens(token_ids)
        recorded = {0: h}
        cache = self.make_cache()
        attention_index = next(
            (index for index, block in enumerate(self.text.layers) if not block.is_linear), None
        )
        linear_index = next(
            (index for index, block in enumerate(self.text.layers) if block.is_linear), None
        )
        attention_mask = self.text.create_attention_mask(
            h, None if attention_index is None else cache[attention_index]
        )
        ssm_mask = self.text.create_ssm_mask(
            h, None if linear_index is None else cache[linear_index]
        )
        for index, (block, cache_i) in enumerate(zip(self.text.layers, cache, strict=True)):
            mask = ssm_mask if block.is_linear else attention_mask
            h = block(h, mask=mask, cache=cache_i)
            recorded[index + 1] = h
        return self._unembed(self.text.norm(h)), recorded


class _ArchitectureDenseModel(_ArchitectureFakeBase):
    def __init__(self, *, vocab: int = 23, dim: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _ArchitectureText(
            vocab=vocab, dim=dim, kinds=tuple(False for _ in range(n_layers))
        )
        self.text = self.model


class _ArchitectureLanguageModel(nn.Module):
    def __init__(self, text: _ArchitectureText, *, tied: bool) -> None:
        super().__init__()
        self.model = text
        if not tied:
            self.lm_head = nn.Linear(
                text.embed_tokens.weight.shape[1],
                text.embed_tokens.weight.shape[0],
                bias=False,
            )


class _ArchitectureHybridModel(_ArchitectureFakeBase):
    def __init__(
        self, *, tied: bool = True, vocab: int = 23, dim: int = 8, split_linear: bool = False
    ) -> None:
        super().__init__()
        text = _ArchitectureText(
            vocab=vocab,
            dim=dim,
            kinds=(True, True, True, False),
            split_linear=split_linear,
        )
        self.language_model = _ArchitectureLanguageModel(text, tied=tied)
        self.text = text


def make_dense_fake() -> tuple[_ArchitectureDenseModel, mx.array]:
    mx.random.seed(91)
    return _ArchitectureDenseModel(), mx.array([[2, 5, 1, 7]], dtype=mx.int32)


def make_hybrid_fake(*, tied: bool = True) -> tuple[_ArchitectureHybridModel, mx.array]:
    mx.random.seed(92 if tied else 93)
    return _ArchitectureHybridModel(tied=tied), mx.array([[3, 1, 4, 6]], dtype=mx.int32)


def make_split_hybrid_fake() -> tuple[_ArchitectureHybridModel, mx.array]:
    mx.random.seed(94)
    model = _ArchitectureHybridModel(split_linear=True)
    return model, mx.array([[3, 1, 4, 6]], dtype=mx.int32)


@pytest.mark.parametrize(
    "fake_factory", [make_dense_fake, make_hybrid_fake], ids=["dense", "hybrid"]
)
def test_architecture_view_matches_model_forward(fake_factory) -> None:
    model, ids = fake_factory()
    view = ArchitectureView.from_model(model)
    residuals = view.residuals(ids, tuple(range(view.num_layers + 1)))
    expected, recorded = model.recording_forward(ids)

    assert tuple(residuals) == tuple(range(view.num_layers + 1))
    for layer, hidden in recorded.items():
        assert residuals[layer].dtype == mx.float32
        np.testing.assert_allclose(residuals[layer], hidden, rtol=1e-5, atol=1e-5)
    actual = view.unembed(view.final_norm(residuals[view.num_layers]))
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fake_factory", [make_dense_fake, make_hybrid_fake])
def test_architecture_view_tail_matches_full_tail_at_every_residual(fake_factory) -> None:
    model, ids = fake_factory()
    view = ArchitectureView.from_model(model)
    residuals = view.residuals(ids, tuple(range(view.num_layers + 1)))
    expected = view.final_norm(residuals[view.num_layers])

    for layer in range(view.num_layers + 1):
        actual = view.tail(layer)(residuals[layer])
        assert actual.dtype == mx.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_architecture_view_discovers_all_supported_text_module_locations() -> None:
    dense, _ = make_dense_fake()
    hybrid, _ = make_hybrid_fake()

    assert ArchitectureView.from_model(dense.model).text_module is dense.model
    assert ArchitectureView.from_model(dense).text_module is dense.model
    assert ArchitectureView.from_model(hybrid).text_module is hybrid.language_model.model


def test_hybrid_architecture_view_uses_per_kind_masks_and_caches() -> None:
    model, ids = make_hybrid_fake()
    view = ArchitectureView.from_model(model)
    caches = view.make_cache()
    h = view.embed(ids)
    masks = view.masks(h, caches)

    assert [view.layer_kind(index) for index in range(view.num_layers)] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert masks == {"attention": _ATTENTION_MASK, "linear_attention": _SSM_MASK}
    assert model.text.mask_calls[-2:] == [
        ("attention", caches[3]),
        ("linear_attention", caches[0]),
    ]
    for index, cache_i in enumerate(caches):
        h = view.run_block(index, h, masks, cache_i)

    with pytest.raises(ValueError, match="wrong mask"):
        view.run_block(0, view.embed(ids), {"linear_attention": None}, caches[0])


def test_architecture_view_reports_cache_kinds_and_trimmability() -> None:
    dense, _ = make_dense_fake()
    hybrid, _ = make_hybrid_fake()
    dense_view = ArchitectureView.from_model(dense)
    hybrid_view = ArchitectureView.from_model(hybrid)

    assert all(isinstance(cache, _FakeKVCache) for cache in dense_view.make_cache())
    assert dense_view.cache_trimmable is True
    hybrid_cache = hybrid_view.make_cache()
    assert [type(cache) for cache in hybrid_cache] == [
        _FakeArraysCache,
        _FakeArraysCache,
        _FakeArraysCache,
        _FakeKVCache,
    ]
    assert isinstance(hybrid_cache[0].state, list)
    assert hybrid_view.cache_trimmable is False


def test_architecture_view_supports_tied_and_untied_unembedding() -> None:
    for tied in (True, False):
        model, ids = make_hybrid_fake(tied=tied)
        view = ArchitectureView.from_model(model)
        expected, recorded = model.recording_forward(ids)

        assert view.tie_word_embeddings is tied
        assert view.vocab_size == 23
        actual = view.unembed(view.final_norm(recorded[view.num_layers]))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_architecture_view_runs_only_to_the_deepest_requested_residual() -> None:
    model, ids = make_hybrid_fake()
    view = ArchitectureView.from_model(model)
    residuals = view.residuals(ids, (0, 2))

    assert tuple(residuals) == (0, 2)
    assert [block.calls for block in model.text.layers] == [1, 1, 0, 0]
    with pytest.raises(ValueError, match="layers"):
        view.residuals(ids, ())
    with pytest.raises(ValueError, match="layers"):
        view.residuals(ids, (-1,))
    with pytest.raises(ValueError, match="layers"):
        view.residuals(ids, (view.num_layers + 1,))


def test_architecture_view_lora_target_policies_select_only_existing_modules() -> None:
    dense, _ = make_dense_fake()
    hybrid, _ = make_hybrid_fake()
    dense_view = ArchitectureView.from_model(dense)
    hybrid_view = ArchitectureView.from_model(hybrid)
    dense_keys = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    hybrid_keys = (
        *dense_keys,
        "linear_attn.in_proj_qkvz",
        "linear_attn.in_proj_ba",
        "linear_attn.out_proj",
    )

    assert dense_view.lora_targets("attention+mlp") == dense_keys
    assert dense_view.lora_targets("all-linear") == dense_keys
    assert dense_view.lora_targets("auto") == dense_keys
    assert hybrid_view.lora_targets("attention+mlp") == dense_keys
    assert hybrid_view.lora_targets("all-linear") == hybrid_keys
    assert hybrid_view.lora_targets("auto") == hybrid_keys


def test_architecture_view_validates_explicit_lora_targets_and_counts_shapes() -> None:
    dense, _ = make_dense_fake()
    hybrid, _ = make_hybrid_fake()
    dense_view = ArchitectureView.from_model(dense)
    hybrid_view = ArchitectureView.from_model(hybrid)
    explicit = ("self_attn.q_proj", "mlp.down_proj")

    assert hybrid_view.lora_targets(explicit) == explicit
    # rank * (input + output), summed for every matching module in every block.
    assert dense_view.lora_parameter_count(dense_view.lora_targets("auto"), rank=2) == 992
    assert hybrid_view.lora_parameter_count(hybrid_view.lora_targets("auto"), rank=2) == 956
    assert hybrid_view.lora_parameter_count(explicit, rank=2) == 192
    with pytest.raises(ValueError, match="unknown LoRA target"):
        hybrid_view.lora_targets(("linear_attn.not_real",))
    with pytest.raises(ValueError, match="LoRA policy"):
        hybrid_view.lora_targets("everything")
    with pytest.raises(ValueError, match="rank"):
        hybrid_view.lora_parameter_count(explicit, rank=0)


def test_split_hybrid_lora_targets_and_counts_use_only_actual_modules() -> None:
    model, _ = make_split_hybrid_fake()
    view = ArchitectureView.from_model(model)
    expected = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
    )

    assert view.lora_targets("all-linear") == expected
    assert view.lora_targets("auto") == expected
    # One dense-attention block, four MLPs, and three split linear-attention blocks.
    assert view.lora_parameter_count(expected, rank=2) == 1112
    assert view.lora_parameter_count(
        ("linear_attn.in_proj_qkv", "linear_attn.in_proj_b"), rank=2
    ) == 204


def test_architecture_view_rejects_missing_or_bad_structural_shapes() -> None:
    class BadEmbedding:
        weight = mx.zeros((5,))

        def __call__(self, ids):
            return ids

        def as_linear(self, h):
            return h

    class BadText:
        embed_tokens = BadEmbedding()
        layers = [object()]

        def norm(self, h):
            return h

    with pytest.raises(ValueError, match="embed_tokens.weight"):
        ArchitectureView.from_model(BadText())
    with pytest.raises(ValueError, match="embed_tokens.*layers.*norm"):
        ArchitectureView.from_model(object())


def test_model_spec_resolve_populates_every_field_from_real_architecture_view() -> None:
    from local_llm_lab.models import load_model_spec

    model, _ = make_hybrid_fake(tied=False)
    spec = load_model_spec("qwen35-4b")
    tokenizer = type("Tokenizer", (), {"snapshot_revision": "fake-hybrid-revision"})()
    resolved = spec.resolve(model, tokenizer)

    assert resolved.spec is spec
    assert resolved.num_layers == 4
    assert resolved.hidden_size == 8
    assert resolved.vocab_size == 23
    assert resolved.tie_word_embeddings is False
    assert resolved.layer_types == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    )
    assert resolved.lora_keys == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "linear_attn.in_proj_qkvz",
        "linear_attn.in_proj_ba",
        "linear_attn.out_proj",
    )
    assert resolved.trainable_parameters == 7648
    assert resolved.probe_layers == (1, 1, 2, 3, 3, 4)
    assert resolved.cache_strategy == "snapshot"
    assert resolved.snapshot_revision == "fake-hybrid-revision"
    assert resolved.jvp_method == "untested"


class _ProbeTokenizer:
    """Character-level tokenizer: ids are stable and every prompt is a strict token prefix."""

    def __init__(self, vocab_size: int = 40) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) % self.vocab_size for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(f"<{token_id}>" for token_id in ids)


class _MergingTokenizer(_ProbeTokenizer):
    """Breaks the prefix property: characters merge pairwise, so a prompt of odd length
    tokenises differently on its own than it does inside ``prompt + response``."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        pairs = [text[index : index + 2] for index in range(0, len(text), 2)]
        return [sum(map(ord, pair)) % self.vocab_size for pair in pairs]


# --------------------------------------------------------------------------- capture_residuals


def test_capture_residuals_matches_residual_at_for_every_layer() -> None:
    model = _model()
    ids = [1, 5, 3, 7, 2]
    layers = [0, 1, 3, 4]
    captured = capture.capture_residuals(model, ids, layers, positions="all")
    assert sorted(captured) == layers
    for layer in layers:
        expected = jlens.residual_at(model, ids, layer)[0]
        assert captured[layer].dtype == mx.float32
        assert bool(mx.allclose(captured[layer], expected, atol=1e-6).item())


def test_capture_decomposition_matches_the_models_own_forward_pass() -> None:
    """The check ``jlens`` uses on the real checkpoint, run here on the fake model.

    Capturing after the final block and applying the model's final norm must reproduce
    ``model.model(ids)`` exactly; if the capture used a different attention mask than the one
    the model builds itself, this is where it would show up.
    """
    model = _model(3)
    ids = [4, 9, 2, 7, 1, 6]
    depth = len(model.model.layers)
    captured = capture.capture_residuals(model, ids, [depth], positions="all")
    ours = model.model.norm(captured[depth])
    theirs = model.model(mx.array(ids)[None, :])[0]
    error = float((mx.sqrt(mx.sum((ours - theirs) ** 2) / mx.sum(theirs**2))).item())
    assert error < 1e-5, error


def test_capture_without_the_causal_mask_would_differ() -> None:
    """Guards the guard: an unmasked run really does change the residual stream."""
    model = _model(5)
    ids = [4, 9, 2, 7, 1, 6]
    masked = capture.capture_residuals(model, ids, [2], positions="all")[2]
    h = model.model.embed_tokens(mx.array(ids)[None, :]).astype(mx.float32)
    for block in model.model.layers[:2]:
        h = block(h, None, None).astype(mx.float32)
    assert not bool(mx.allclose(masked, h[0], atol=1e-4).item())


def test_capture_position_variants() -> None:
    model = _model()
    ids = [1, 5, 3, 7, 2]
    every = capture.capture_residuals(model, ids, [2], positions="all")[2]
    last = capture.capture_residuals(model, ids, [2], positions="last")[2]
    picked = capture.capture_residuals(model, ids, [2], positions=[0, 3, -1])[2]
    assert every.shape == (5, 16)
    assert last.shape == (16,)
    assert picked.shape == (3, 16)
    assert bool(mx.allclose(last, every[-1]).item())
    assert bool(mx.allclose(picked[0], every[0]).item())
    assert bool(mx.allclose(picked[1], every[3]).item())
    assert bool(mx.allclose(picked[2], every[-1]).item())


def test_capture_rejects_out_of_range_layers_and_positions() -> None:
    model = _model()
    with pytest.raises(ValueError):
        capture.capture_residuals(model, [1, 2], [5])
    with pytest.raises(ValueError):
        capture.capture_residuals(model, [1, 2], [-1])
    with pytest.raises(ValueError):
        capture.capture_residuals(model, [1, 2], [])
    with pytest.raises(ValueError):
        capture.capture_residuals(model, [1, 2], [1], positions=[7])
    with pytest.raises(ValueError):
        capture.capture_residuals(model, [1, 2], [1], positions="middle")


def test_capture_runs_one_pass_over_the_requested_depth_only() -> None:
    """Blocks deeper than the deepest requested layer are never called."""
    model = _model()
    calls: list[int] = []
    original = model.model.layers[3]

    class _Counting:
        def __call__(self, x, mask=None, cache=None):
            calls.append(3)
            return original(x, mask, cache)

    model.model.layers[3] = _Counting()
    try:
        capture.capture_residuals(model, [1, 2, 3], [0, 1, 2, 3])
        assert calls == []
        capture.capture_residuals(model, [1, 2, 3], [4])
        assert calls == [3]
    finally:
        model.model.layers[3] = original


# ------------------------------------------------------------------- response_mean_activations


def test_response_mean_activations_averages_only_response_positions() -> None:
    model = _model()
    tokenizer = _ProbeTokenizer()
    prompt, response = "prompt text", "the answer"
    stats: dict[str, int] = {}
    means = capture.response_mean_activations(
        model, tokenizer, prompt, response, [1, 3], stats=stats
    )
    assert stats == {"sequences": 1}
    joint = tokenizer.encode(prompt + response)
    start = len(tokenizer.encode(prompt))
    every = capture.capture_residuals(model, joint, [1, 3], positions="all")
    for layer in (1, 3):
        expected = mx.mean(every[layer][start:], axis=0)
        assert means[layer].shape == (16,)
        assert bool(mx.allclose(means[layer], expected, atol=1e-6).item())
        assert not bool(mx.allclose(means[layer], mx.mean(every[layer], axis=0), atol=1e-4).item())


def test_response_mean_activations_repairs_and_counts_a_broken_prefix() -> None:
    model = _model()
    stats: dict[str, int] = {}
    means = capture.response_mean_activations(
        model, _MergingTokenizer(), "odd", "response", [2], stats=stats
    )
    assert stats == {"sequences": 1, "prefix_mismatch": 1}
    assert means[2].shape == (16,)


def test_response_mean_activations_rejects_an_empty_response() -> None:
    with pytest.raises(ValueError):
        capture.response_mean_activations(_model(), _ProbeTokenizer(), "prompt", "", [1])


# --------------------------------------------------------------------------- InjectionHook


class _StubCache:
    """Just enough of an mlx-lm KV cache for the hook: the pre-call sequence offset."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset


def test_injection_hook_adds_alpha_times_vector_at_every_position() -> None:
    model = _model()
    ids = [1, 5, 3, 7, 2]
    baseline = capture.capture_residuals(model, ids, [2, 3], positions="all")
    vector = mx.arange(16).astype(mx.float32) / 16.0
    with capture.InjectionHook(model, layer=1, vector=vector, alpha=2.5) as hook:
        steered = capture.capture_residuals(model, ids, [2, 3], positions="all")
    # Layer 2 is the output of block 1 plus the injection, exactly.
    delta = steered[2] - baseline[2]
    assert bool(mx.allclose(delta, mx.broadcast_to(2.5 * vector, delta.shape), atol=1e-5).item())
    assert hook.calls == 1
    assert hook.injected == 1
    # And the perturbation propagates: deeper layers are not merely shifted by alpha * v.
    assert not bool(mx.allclose(steered[3] - baseline[3], delta, atol=1e-4).item())


def test_injection_hook_restores_the_original_block_even_after_an_error() -> None:
    model = _model()
    original = model.model.layers[2]
    vector = mx.ones((16,))
    with capture.InjectionHook(model, 2, vector):
        assert model.model.layers[2] is not original
    assert model.model.layers[2] is original
    with pytest.raises(RuntimeError):  # noqa: SIM117 - the nesting is the point
        with capture.InjectionHook(model, 2, vector):
            raise RuntimeError("boom")
    assert model.model.layers[2] is original


def test_injection_hook_honours_from_and_at_positions() -> None:
    model = _model()
    ids = [1, 5, 3, 7, 2]
    baseline = capture.capture_residuals(model, ids, [1], positions="all")
    vector = mx.ones((16,)).astype(mx.float32)
    with capture.InjectionHook(model, 0, vector, alpha=3.0, positions=("from", 2)):
        after = capture.capture_residuals(model, ids, [1], positions="all")
    delta = after[1] - baseline[1]
    assert bool(mx.allclose(delta[:2], mx.zeros((2, 16)), atol=1e-6).item())
    assert bool(mx.allclose(delta[2:], 3.0 * mx.ones((3, 16)), atol=1e-5).item())
    with capture.InjectionHook(model, 0, vector, alpha=3.0, positions=3):
        single = capture.capture_residuals(model, ids, [1], positions="all")
    delta = single[1] - baseline[1]
    assert bool(mx.allclose(delta[3], 3.0 * mx.ones((16,)), atol=1e-5).item())
    assert float(mx.sum(mx.abs(delta[:3])).item()) == 0.0
    assert float(mx.sum(mx.abs(delta[4:])).item()) == 0.0


def test_injection_hook_uses_the_cache_offset_for_absolute_positions() -> None:
    """Under a KV cache a call sees only a slice, so ``("at", 6)`` must land on the token whose
    absolute index is 6 -- the case that matters for ``TurnCache`` and ``stream_generate``."""
    model = _model()
    vector = mx.ones((16,)).astype(mx.float32)
    hook = capture.InjectionHook(model, 0, vector, alpha=2.0, positions=("at", 6))
    x = mx.zeros((1, 3, 16))
    with hook:
        wrapper = model.model.layers[0]
        plain = wrapper.block(x, None, None)
        shifted = wrapper(x, None, _StubCache(offset=5))
        missed = wrapper(x, None, _StubCache(offset=20))
    assert bool(mx.allclose(shifted[0, 1] - plain[0, 1], 2.0 * vector, atol=1e-5).item())
    assert float(mx.sum(mx.abs(shifted[0, 0] - plain[0, 0])).item()) == 0.0
    assert bool(mx.allclose(missed, plain, atol=1e-6).item())
    assert hook.calls == 2
    assert hook.injected == 1


def test_injection_hook_validates_its_arguments() -> None:
    model = _model()
    with pytest.raises(ValueError):
        capture.InjectionHook(model, 99, mx.ones((16,)))
    with pytest.raises(ValueError):
        capture.InjectionHook(model, 0, mx.ones((16,)), positions="middle")


# --------------------------------------------------------------------------- lora_block_mask


class _FakeLoRALinear(nn.Module):
    """``y = W x + scale * (x @ lora_a) @ lora_b``, the shape ``mlx_lm.tuner.lora.LoRALinear``
    has (lora.py:95-98), so zeroing ``lora_b`` must remove the update exactly."""

    def __init__(self, dim: int, rank: int = 2, scale: float = 4.0) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.scale = scale
        self.lora_a = mx.random.normal((dim, rank))
        self.lora_b = mx.random.normal((rank, dim))

    def __call__(self, x):
        return self.linear(x) + self.scale * ((x @ self.lora_a) @ self.lora_b)


class _FakeAdaptedBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _FakeLoRALinear(dim)
        self.mlp = _FakeLoRALinear(dim)

    def __call__(self, x, mask=None, cache=None):
        del mask, cache
        return self.mlp(self.self_attn(x))


class _FakeAdaptedInner(nn.Module):
    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.layers = [_FakeAdaptedBlock(dim) for _ in range(n_layers)]


class _FakeAdaptedModel(nn.Module):
    def __init__(self, dim: int = 8, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _FakeAdaptedInner(dim, n_layers)


def test_lora_block_mask_zeroes_the_excluded_layers_and_restores_them() -> None:
    mx.random.seed(11)
    model = _FakeAdaptedModel()
    saved = [(block.self_attn.lora_b, block.mlp.lora_b) for block in model.model.layers]
    x = mx.random.normal((1, 3, 8))
    before = [block(x) for block in model.model.layers]

    with capture.lora_block_mask(model, keep_layers={1, 2}) as masked:
        assert masked == 4  # two modules in each of the two masked blocks
        for index, block in enumerate(model.model.layers):
            for module in (block.self_attn, block.mlp):
                zeroed = float(mx.sum(mx.abs(module.lora_b)).item()) == 0.0
                assert zeroed == (index not in {1, 2})
        # A masked block now equals its base linear path alone.
        assert bool(
            mx.allclose(
                model.model.layers[0](x),
                model.model.layers[0].mlp.linear(model.model.layers[0].self_attn.linear(x)),
                atol=1e-5,
            ).item()
        )
        assert bool(mx.allclose(model.model.layers[1](x), before[1], atol=1e-6).item())

    for index, block in enumerate(model.model.layers):
        assert bool(mx.array_equal(block.self_attn.lora_b, saved[index][0]).item())
        assert bool(mx.array_equal(block.mlp.lora_b, saved[index][1]).item())
        assert bool(mx.allclose(block(x), before[index], atol=1e-6).item())


def test_lora_block_mask_restores_after_an_error() -> None:
    mx.random.seed(12)
    model = _FakeAdaptedModel()
    saved = model.model.layers[0].mlp.lora_b
    with pytest.raises(RuntimeError), capture.lora_block_mask(model, keep_layers=set()):
        raise RuntimeError("boom")
    assert bool(mx.array_equal(model.model.layers[0].mlp.lora_b, saved).item())


# --------------------------------------------------------------------------- strip_state_fields

_PENDING_RE = re.compile(r"pending: ([^.]*(?:\.[^\s][^.]*)*)\.")
_BUCKET_RE = re.compile(
    r"(?:approved|held \(skip\)|held skipped|first half|second half): ([0-9][0-9, ]*)"
)


def _long_family_notes(family: str) -> list[str]:
    from local_llm_lab.pipeline.tasks import make_tasks

    tasks = [task for task in make_tasks("test", 36) if task.family == family]
    assert tasks, family
    return [step.thought for task in tasks for step in task.steps]


@pytest.mark.parametrize(
    "family", ["ledger_reconcile", "conditional_update", "aggregate_report", "batch_update"]
)
def test_strip_state_fields_removes_pending_lists_and_bucket_contents(family: str) -> None:
    stripped_any = False
    for note in _long_family_notes(family):
        stripped = capture.strip_state_fields(note)
        pending = _PENDING_RE.search(note)
        if pending:
            stripped_any = True
            assert "pending: [stripped]" in stripped
            for name in (part.strip() for part in pending.group(1).split(",")):
                assert name and name not in stripped, (name, stripped)
        for bucket in _BUCKET_RE.finditer(note):
            stripped_any = True
            for value in (part.strip() for part in bucket.group(1).split(",")):
                if not value:
                    continue
                # A bucket value must not survive as a standalone number; digits inside a
                # retained filename ("invoice-4-143.txt") are not a leak.
                assert not re.search(rf"(?<![\w.-]){re.escape(value)}(?![\w.-])", stripped), (
                    value,
                    stripped,
                )
    # batch_update carries its queue in "Next:"/"Remaining after this:" fields, which the
    # design's field list does not name, so nothing is stripped there.
    assert stripped_any == (family != "batch_update")


def test_strip_state_fields_replaces_every_named_field() -> None:
    note = (
        "Invoices read: 2 of 6. approved: 178, 40; held (skip): 38. "
        "Reading invoice-2-537.txt; pending: invoice-3-599.txt, invoice-4-143.txt."
    )
    assert capture.strip_state_fields(note) == (
        "Invoices read: 2 of 6. approved: [stripped]; held (skip): [stripped]. "
        "Reading invoice-2-537.txt; pending: [stripped]."
    )


def test_strip_state_fields_keeps_prose_after_a_running_maximum() -> None:
    note = "threshold=57. highest so far: service-3=88 (final), above threshold, so throttle service-3.ini."
    assert capture.strip_state_fields(note) == (
        "threshold=57. highest so far: [stripped], above threshold, so throttle service-3.ini."
    )


def test_strip_state_fields_handles_the_summing_step_spellings() -> None:
    assert capture.strip_state_fields(
        "All 6 invoices read. approved: 178, 40; held skipped: 38. Summing approved amounts."
    ) == (
        "All 6 invoices read. approved: [stripped]; held skipped: [stripped]. Summing approved amounts."
    )
    assert capture.strip_state_fields(
        "first half complete: 18 + 14; second half complete: 72 + 30. Computing the first subtotal."
    ) == (
        "first half complete: [stripped]; second half complete: [stripped]. Computing the first subtotal."
    )


def test_strip_state_fields_leaves_other_text_byte_identical() -> None:
    for note in (
        "Plan: list the ledger, read every invoice. Listing.",
        "Search matched 2 files; record-429-0.txt is already read, so reading the new one: record-791-1.txt.",
        "Visited 3 node(s), no Result yet. Node 2 points to lab/test/0006/chain/node-3-358.txt; following it.",
        "",
    ):
        assert capture.strip_state_fields(note) == note


# --------------------------------------------------------------------------- adapter delta


def _lora_linear(dim_in: int = 12, dim_out: int = 7, rank: int = 3, scale: float = 4.0):
    """A real ``mlx_lm.tuner.lora.LoRALinear`` over a random ``nn.Linear``, with random factors."""
    from mlx_lm.tuner.lora import LoRALinear

    mx.random.seed(21)
    base = nn.Linear(dim_in, dim_out, bias=False)
    lora = LoRALinear.from_base(base, r=rank, scale=scale)
    lora.lora_a = mx.random.normal((dim_in, rank))
    lora.lora_b = mx.random.normal((rank, dim_out))
    return lora, base


def test_delta_orientation_matches_mlx_lms_own_lora_linear() -> None:
    """``delta_from_factors`` must be the same update mlx-lm's LoRALinear actually applies."""
    lora, base = _lora_linear()
    delta = adapter_delta.delta_from_factors(lora.lora_a, lora.lora_b, lora.scale)
    assert delta.shape == base.weight.shape  # (out, in), the base weight's orientation
    x = mx.random.normal((5, 12))
    difference = lora(x) - base(x)
    assert bool(mx.allclose(difference, x @ delta.T, atol=1e-4).item())


def test_delta_orientation_matches_the_fused_weight() -> None:
    """The same claim read off ``LoRALinear.fuse`` (mlx_lm/tuner/lora.py:52-53)."""
    lora, base = _lora_linear()
    delta = adapter_delta.delta_from_factors(lora.lora_a, lora.lora_b, lora.scale)
    fused = lora.fuse()
    assert bool(mx.allclose(fused.weight, base.weight + delta, atol=1e-4).item())


def _synthetic_info(left: np.ndarray, right: np.ndarray, scale: float = 1.0) -> dict:
    """``info`` in ``load_adapter_deltas`` shape for a hand-built update ``scale * left @ right``."""
    return {
        "module": "synthetic",
        "layer": 0,
        "type": "down_proj",
        "scale": scale,
        "rank": int(left.shape[1]),
        "shape": [int(right.shape[1]), int(left.shape[0])],
        "lora_a": left,
        "lora_b": right,
    }


def test_spectrum_is_exact_against_a_dense_svd() -> None:
    rng = np.random.default_rng(5)
    lora_a = rng.normal(size=(20, 4)).astype(np.float32)
    lora_b = rng.normal(size=(4, 13)).astype(np.float32)
    info = _synthetic_info(lora_a, lora_b, scale=3.0)
    dense = 3.0 * (lora_a @ lora_b).T
    expected = np.linalg.svd(dense, compute_uv=False)[:4]
    assert np.allclose(adapter_delta.spectrum(info), expected, atol=1e-4)
    assert adapter_delta.delta_norm(info) == pytest.approx(float(np.linalg.norm(dense)), rel=1e-4)
    vectors = adapter_delta.left_singular_vectors(info, k=2)
    assert vectors.shape == (13, 2)
    # Left singular vectors span the same directions as a dense SVD's (up to sign).
    dense_u = np.linalg.svd(dense)[0][:, :2]
    assert np.allclose(np.abs(np.diag(dense_u.T @ vectors)), 1.0, atol=1e-3)


def test_effective_rank_90_on_synthetic_rank_three_matrices() -> None:
    rng = np.random.default_rng(7)
    for _ in range(5):
        left = rng.normal(size=(30, 3)).astype(np.float32)
        right = rng.normal(size=(3, 24)).astype(np.float32)
        info = _synthetic_info(left, right)
        values = adapter_delta.spectrum(info, top=16)
        assert int(np.sum(values > 1e-4)) == 3
        assert adapter_delta.effective_rank_90(values) <= 3
    # A flat spectrum needs almost all of its components to reach 90% of the energy.
    assert adapter_delta.effective_rank_90(np.ones(10)) == 9
    assert adapter_delta.effective_rank_90(np.array([1.0, 0.0, 0.0])) == 1
    assert adapter_delta.effective_rank_90(np.zeros(4)) == 0


def test_principal_angle_cosines_are_one_for_identical_and_zero_for_orthogonal_subspaces() -> None:
    rng = np.random.default_rng(11)
    basis = np.linalg.qr(rng.normal(size=(20, 8)))[0]
    same = adapter_delta.principal_angle_cosines(basis[:, :4], basis[:, :4])
    assert np.allclose(same, 1.0, atol=1e-6)
    orthogonal = adapter_delta.principal_angle_cosines(basis[:, :4], basis[:, 4:])
    assert np.allclose(orthogonal, 0.0, atol=1e-6)
    # A rotation inside the same span is still the same subspace.
    rotated = basis[:, :4] @ np.linalg.qr(rng.normal(size=(4, 4)))[0]
    assert np.allclose(adapter_delta.principal_angle_cosines(basis[:, :4], rotated), 1.0, atol=1e-6)


def test_module_layer_and_type_parse_adapter_keys() -> None:
    name = "model.layers.17.mlp.down_proj"
    assert adapter_delta.module_layer(name) == 17
    assert adapter_delta.module_type(name) == "down_proj"
    assert adapter_delta.module_layer("model.embed_tokens") is None


def test_load_adapter_deltas_reads_a_written_adapter(tmp_path) -> None:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(3)
    tensors = {
        "model.layers.0.self_attn.q_proj.lora_a": rng.normal(size=(8, 2)).astype(np.float32),
        "model.layers.0.self_attn.q_proj.lora_b": rng.normal(size=(2, 6)).astype(np.float32),
        "model.layers.1.mlp.down_proj.lora_a": rng.normal(size=(9, 2)).astype(np.float32),
        "model.layers.1.mlp.down_proj.lora_b": rng.normal(size=(2, 4)).astype(np.float32),
    }
    save_file(tensors, str(tmp_path / "adapters.safetensors"))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"lora_parameters": {"scale": 32.0, "rank": 2}})
    )

    deltas = adapter_delta.load_adapter_deltas(tmp_path)
    assert sorted(deltas) == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.down_proj",
    ]
    delta, info = deltas["model.layers.1.mlp.down_proj"]
    assert info == {
        **info,
        "layer": 1,
        "type": "down_proj",
        "scale": 32.0,
        "rank": 2,
        "shape": [4, 9],
    }
    assert delta.shape == (4, 9)  # (out, in)
    expected = (
        32.0
        * (
            tensors["model.layers.1.mlp.down_proj.lora_a"]
            @ tensors["model.layers.1.mlp.down_proj.lora_b"]
        ).T
    )
    assert np.allclose(np.array(delta), expected, atol=1e-4)

    report = adapter_delta.analyse_adapter(tmp_path)
    assert [record["module"] for record in report["modules"]] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.down_proj",
    ]
    assert all(record["relative_norm"] is None for record in report["modules"])
    assert sum(report["effective_rank_histogram"].values()) == 2
    assert "P5" in adapter_delta.render_markdown(
        {"runs": [{**report, "label": "x"}], "has_base": False}
    )


def test_compare_adapters_scores_a_copy_at_one_and_a_random_run_far_below(tmp_path) -> None:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(13)
    name = "model.layers.0.mlp.down_proj"

    def write(directory: Path, seed: int) -> Path:
        local = np.random.default_rng(seed)
        directory.mkdir()
        save_file(
            {
                f"{name}.lora_a": local.normal(size=(64, 4)).astype(np.float32),
                f"{name}.lora_b": local.normal(size=(4, 48)).astype(np.float32),
            },
            str(directory / "adapters.safetensors"),
        )
        return directory

    del rng
    first = write(tmp_path / "runA", 1)
    copy = write(tmp_path / "runB", 1)
    other = write(tmp_path / "runC", 99)
    comparison = adapter_delta.compare_adapters([first, copy, other], top=4)
    scores = comparison["mean_cosine_by_type"]
    assert scores["runA|runB"]["all"] == pytest.approx(1.0, abs=1e-5)
    assert scores["runA|runC"]["all"] < 0.6
    assert comparison["modules"][0]["module"] == name


# --------------------------------------------------------------------------- row_labels

_HIGHEST_RE = re.compile(r"highest so far: (?:none|service-\d+=(\d+))")
_FIRST_HALF_RE = re.compile(r"first half: ([^;]*);")


def _clean_tasks(count: int = 60):
    from local_llm_lab.pipeline.tasks import make_tasks

    return make_tasks("test", count)  # the test split is generated without recovery variants


def test_row_labels_pending_count_matches_the_notes_pending_list() -> None:
    checked = 0
    for task in _clean_tasks():
        for index, step in enumerate(task.steps):
            listed = _PENDING_RE.search(step.thought)
            if not listed:
                continue
            checked += 1
            body = listed.group(1).strip()
            expected = 0 if body == "none" else len(body.split(","))
            assert state_probe.row_labels(task, index)["pending_count"] == expected, (
                task.task_id,
                index,
                step.thought,
            )
    assert checked > 50


def test_row_labels_running_max_and_first_bucket_match_their_notes() -> None:
    checked = 0
    for task in _clean_tasks():
        for index, step in enumerate(task.steps):
            labels = state_probe.row_labels(task, index)
            highest = _HIGHEST_RE.search(step.thought)
            if highest:
                checked += 1
                assert labels["running_max"] == (
                    float(highest.group(1)) if highest.group(1) else 0.0
                )
            first = _FIRST_HALF_RE.search(step.thought)
            if first:
                checked += 1
                body = first.group(1).strip().replace(" (full)", "")
                assert labels["first_bucket_count"] == (
                    0 if body == "none" else len(body.split(","))
                )
    assert checked > 40


def test_row_labels_phase_matches_the_batch_update_note_prefixes() -> None:
    checked = 0
    for task in _clean_tasks():
        if task.family != "batch_update":
            continue
        for index, step in enumerate(task.steps):
            if step.action.name == "finish":  # "Phase verify complete" ends the verify phase
                assert state_probe.row_labels(task, index)["phase"] == "other"
                continue
            # "Phase apply complete ...; phase verify begins." announces verify, so a
            # "<phase> begins" declaration wins over the sentence's opening words.
            lowered = step.thought.lower()
            stated = next(
                (
                    phase
                    for phase in ("inspect", "apply", "verify")
                    if f"phase {phase} begins" in lowered
                ),
                None,
            ) or next(
                (
                    phase
                    for phase in ("inspect", "apply", "verify")
                    if lowered.startswith(f"phase {phase}")
                ),
                None,
            )
            if stated is None:
                continue
            checked += 1
            assert state_probe.row_labels(task, index)["phase"] == stated, (task.task_id, index)
    assert checked > 20


def test_row_labels_phase_is_other_or_inspect_outside_batch_update() -> None:
    for task in _clean_tasks(24):
        if task.family == "batch_update":
            continue
        for index in range(len(task.steps)):
            assert state_probe.row_labels(task, index)["phase"] in ("inspect", "other")
    # A read loop is inspection; an isolated read is not.
    ledger = next(task for task in _clean_tasks(24) if task.family == "ledger_reconcile")
    phases = [state_probe.row_labels(ledger, index)["phase"] for index in range(len(ledger.steps))]
    assert phases[1] == "inspect"  # first invoice of the read loop
    assert phases[0] == "other"  # list_files
    assert phases[-1] == "other"  # finish


def test_row_labels_prev_error_matches_build_rows_recovery_metadata() -> None:
    """``build_rows`` already replays the simulator to mark corrective decisions; the probe's
    own replay must agree with it, including on the recovery variants of the train split."""
    from local_llm_lab.pipeline.data import build_rows
    from local_llm_lab.pipeline.tasks import make_tasks

    rows = 0
    errors = 0
    for task in make_tasks("train", 48):
        for row in build_rows(task):
            labels = state_probe.row_labels(task, row["metadata"]["step"])
            rows += 1
            errors += int(bool(labels["prev_error"]))
            assert bool(labels["prev_error"]) == bool(row["metadata"]["recovery"])
    assert rows > 200
    assert errors > 0  # the check would be vacuous if no row followed an error


def test_row_labels_next_tool_is_the_expert_action() -> None:
    task = _clean_tasks(12)[7]
    for index, step in enumerate(task.steps):
        assert state_probe.row_labels(task, index)["next_tool"] == step.action.name


def test_row_labels_rejects_a_step_out_of_range() -> None:
    task = _clean_tasks(1)[0]
    with pytest.raises(ValueError):
        state_probe.row_labels(task, len(task.steps))


# --------------------------------------------------------------------------- probe fitting


def _planted_dataset(seed: int = 4, n_tasks: int = 40, per_task: int = 4, dim: int = 16):
    """Features with a planted linear signal in three of the six targets.

    The three others are left undefined, which also exercises the "skip a target with no
    labelled rows" path.
    """
    rng = np.random.default_rng(seed)
    n = n_tasks * per_task
    features = rng.normal(size=(n, dim))
    task_ids = np.array([f"task-{index // per_task:03d}" for index in range(n)])
    pending = features @ rng.normal(size=dim) + 0.1 * rng.normal(size=n)
    phase = np.array(["inspect", "other", "verify"])[np.argmax(features[:, :3], axis=1)]
    error = (features @ rng.normal(size=dim) > 0).astype(float)
    labels = {
        "pending_count": pending,
        "phase": phase.astype(str),
        "prev_error": error,
        "next_tool": np.array([""] * n),
        "running_max": np.full(n, np.nan),
        "first_bucket_count": np.full(n, np.nan),
    }
    return state_probe.ProbeDataset(
        layers=[0],
        features={0: features.astype(np.float32)},
        labels=labels,
        task_ids=task_ids,
        meta={"rows": n, "tasks": n_tasks, "layers": [0], "strip": False},
    )


def test_split_by_task_never_shares_a_task_between_halves() -> None:
    task_ids = np.array([f"t{index // 3}" for index in range(30)])
    train, test = state_probe.split_by_task(task_ids, seed=1)
    assert len(train) + len(test) == 30
    assert not (set(task_ids[train]) & set(task_ids[test]))
    assert 0 < len(test) < 30
    # Deterministic for a fixed seed.
    assert np.array_equal(train, state_probe.split_by_task(task_ids, seed=1)[0])


def test_ridge_recovers_a_planted_linear_signal_and_the_control_does_not() -> None:
    dataset = _planted_dataset()
    results = state_probe.fit_probes(dataset, ["pending_count"], [0], seed=3)
    entry = results["targets"]["pending_count"]["layers"]["0"]
    assert entry["probe"]["r2"] > 0.9, entry
    assert entry["shuffled_control"]["r2"] < 0.2, entry
    assert entry["probe"]["mae"] < entry["shuffled_control"]["mae"]


def test_logistic_recovers_planted_classes_and_the_control_does_not() -> None:
    dataset = _planted_dataset()
    results = state_probe.fit_probes(dataset, ["phase", "prev_error"], [0], seed=3)
    phase = results["targets"]["phase"]["layers"]["0"]
    assert phase["probe"]["accuracy"] > 0.8, phase
    assert phase["probe"]["macro_f1"] > 0.75, phase
    assert phase["shuffled_control"]["accuracy"] < 0.6, phase
    binary = results["targets"]["prev_error"]["layers"]["0"]
    assert binary["probe"]["auc"] > 0.9, binary
    assert 0.25 < binary["shuffled_control"]["auc"] < 0.75, binary
    assert results["targets"]["phase"]["classes"] == ["inspect", "other", "verify"]


def test_fit_probes_skips_targets_with_no_labelled_rows() -> None:
    dataset = _planted_dataset()
    results = state_probe.fit_probes(dataset, ["running_max", "next_tool"], [0], seed=3)
    assert results["targets"]["running_max"]["skipped"]
    assert results["targets"]["next_tool"]["skipped"]
    assert results["targets"]["running_max"]["rows"] == 0


def test_render_markdown_reports_the_control_next_to_every_result() -> None:
    dataset = _planted_dataset()
    results = state_probe.fit_probes(dataset, ["pending_count", "phase"], [0], seed=3)
    markdown = state_probe.render_markdown(results, "fake")
    assert "shuffled" in markdown
    assert "pending_count (regression)" in markdown
    assert "phase (categorical)" in markdown


def test_logistic_fit_converges_on_a_separable_problem() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(200, 5))
    classes = (features[:, 0] + features[:, 1] > 0).astype(int)
    weights = state_probe.logistic_fit(features, classes, 2, l2=1e-3, steps=600)
    predicted = np.argmax(state_probe._predict_logits(features, weights), axis=1)
    assert float(np.mean(predicted == classes)) > 0.95


def test_ridge_fit_agrees_in_its_primal_and_dual_forms() -> None:
    rng = np.random.default_rng(6)
    features = rng.normal(size=(40, 8))  # n > d, so the primal branch is taken
    targets = features @ rng.normal(size=8) + 0.05 * rng.normal(size=40)
    weights = state_probe.ridge_fit(features, targets, alpha=1.0)
    wide = np.hstack([features, np.zeros((40, 100))])  # n < d, dual branch, same fit on the span
    wide_weights = state_probe.ridge_fit(wide, targets, alpha=1.0)
    assert np.allclose(
        state_probe._predict_linear(features, weights),
        state_probe._predict_linear(wide, wide_weights),
        atol=1e-6,
    )


# ---------------------------------------------- the position confound and its three controls
#
# The first P2 run found that a lookup on (family, step index) predicted every target as well
# as the activation probe (design §5, "Confound found on first run"). These tests pin the three
# corrections: the position-only baseline, the determinism statistic that says whether a given
# dataset carries the confound, and the within-position fit that removes it by construction.


def _cell_frame(n_tasks: int = 48, per_task: int = 6):
    """Task ids, families, step indices and difficulties for a grid of (family, step) cells."""
    task_ids = np.array([f"task-{task:03d}" for task in range(n_tasks) for _ in range(per_task)])
    family = np.array(
        [("alpha", "beta")[task % 2] for task in range(n_tasks) for _ in range(per_task)]
    )
    step = np.array([step for _task in range(n_tasks) for step in range(per_task)])
    difficulty = np.array([task % 3 for task in range(n_tasks) for _ in range(per_task)])
    return task_ids, family, step, difficulty


def _cell_dataset(features, regression, categorical, frame):
    """A one-layer dataset carrying a regression target and a categorical one."""
    task_ids, family, step, difficulty = frame
    n = len(task_ids)
    labels = {
        "pending_count": np.asarray(regression, dtype=np.float64),
        "phase": np.asarray(categorical).astype(str),
        "prev_error": np.array([""] * n),
        "next_tool": np.array([""] * n),
        "running_max": np.full(n, np.nan),
        "first_bucket_count": np.full(n, np.nan),
    }
    return state_probe.ProbeDataset(
        layers=[0],
        features={0: np.asarray(features, dtype=np.float32)},
        labels=labels,
        task_ids=task_ids,
        family=family,
        step_index=step,
        difficulty=difficulty,
        meta={"rows": n, "tasks": len(set(task_ids.tolist())), "layers": [0], "strip": False},
    )


def _pure_position_dataset(seed: int = 11, dim: int = 8):
    """Labels are an exact function of (family, step); the activations are pure noise."""
    frame = _cell_frame()
    _task_ids, family, step, _difficulty = frame
    rng = np.random.default_rng(seed)
    regression = 10.0 * step + np.where(family == "alpha", 0.0, 3.0)
    categorical = np.array([f"{name}{index % 3}" for name, index in zip(family, step, strict=True)])
    return _cell_dataset(rng.normal(size=(len(family), dim)), regression, categorical, frame)


def _decoupled_state_dataset(seed: int = 12, dim: int = 8):
    """Labels are a latent state independent of position, encoded linearly in the activations."""
    frame = _cell_frame()
    _task_ids, family, _step, _difficulty = frame
    rng = np.random.default_rng(seed)
    n = len(family)
    state = rng.normal(size=n)
    direction = rng.normal(size=dim)
    features = np.outer(state, direction) + 0.05 * rng.normal(size=(n, dim))
    return _cell_dataset(features, state, np.where(state > 0, "high", "low"), frame)


def _position_plus_state_dataset(seed: int = 13, dim: int = 8, *, state_in_features: bool = True):
    """A large position signal plus a small state signal, both linear in the activations."""
    frame = _cell_frame()
    _task_ids, family, step, _difficulty = frame
    rng = np.random.default_rng(seed)
    n = len(family)
    position = 10.0 * step + np.where(family == "alpha", 0.0, 3.0)
    state = rng.normal(size=n)
    directions = np.linalg.qr(rng.normal(size=(dim, 2)))[0]
    features = np.outer(position / 10.0, directions[:, 0]) + 0.02 * rng.normal(size=(n, dim))
    if state_in_features:
        features = features + 0.6 * np.outer(state, directions[:, 1])
    categorical = np.where(state > 0, "high", "low")
    return _cell_dataset(features, position + state, categorical, frame)


def test_position_baseline_matches_a_label_that_is_a_pure_function_of_position() -> None:
    dataset = _pure_position_dataset()
    train, test = state_probe.split_by_task(dataset.task_ids, seed=7)
    regression = state_probe.position_baseline(dataset, "pending_count", train, test)
    assert regression["r2"] == pytest.approx(1.0)
    assert regression["mae"] == pytest.approx(0.0)
    categorical = state_probe.position_baseline(dataset, "phase", train, test)
    assert categorical["accuracy"] == pytest.approx(1.0)
    assert categorical["macro_f1"] == pytest.approx(1.0)


def test_a_probe_on_noise_cannot_beat_the_position_baseline() -> None:
    """The confound in its pure form: the baseline is perfect, so every margin is negative."""
    results = state_probe.fit_probes(_pure_position_dataset(), ["pending_count", "phase"], [0])
    for target in ("pending_count", "phase"):
        entry = results["targets"][target]
        assert entry["layers"]["0"]["margin"] <= 0, entry
    assert "does not beat position baseline" in state_probe.render_markdown(results, "fake")


def test_position_baseline_is_near_chance_when_state_is_independent_of_position() -> None:
    dataset = _decoupled_state_dataset()
    results = state_probe.fit_probes(dataset, ["pending_count", "phase"], [0])
    regression = results["targets"]["pending_count"]
    assert regression["position_baseline"]["r2"] < 0.15
    assert regression["layers"]["0"]["probe"]["r2"] > 0.9
    assert regression["layers"]["0"]["margin"] > 0.7
    categorical = results["targets"]["phase"]
    assert categorical["position_baseline"]["accuracy"] < 0.7
    assert categorical["layers"]["0"]["margin"] > 0.25
    assert "probe beats position baseline" in state_probe.render_markdown(results, "fake")


def test_position_determinism_is_one_on_a_pure_function_and_zero_when_decoupled() -> None:
    pure = state_probe.position_determinism(_pure_position_dataset(), ["pending_count", "phase"])
    assert pure["pending_count"]["constant_fraction"] == 1.0
    assert pure["pending_count"]["explained"] == pytest.approx(1.0)
    assert pure["phase"]["constant_fraction"] == 1.0
    assert pure["pending_count"]["cells"] == 12  # two families x six steps

    decoupled = state_probe.position_determinism(
        _decoupled_state_dataset(), ["pending_count", "phase"]
    )
    assert decoupled["pending_count"]["constant_fraction"] == 0.0
    assert abs(decoupled["pending_count"]["explained"]) < 0.1
    assert decoupled["phase"]["constant_fraction"] == 0.0
    assert decoupled["phase"]["explained"] < 0.7


def test_within_position_probe_recovers_the_state_the_raw_margin_hides() -> None:
    """Features carry a big position signal and a small state signal.

    Against the raw label the probe looks like a position decoder -- its margin over the
    baseline is a rounding error -- but with the cell mean removed the state is still there.
    """
    dataset = _position_plus_state_dataset()
    results = state_probe.fit_probes(dataset, ["pending_count"], [0], within_position=True)
    entry = results["targets"]["pending_count"]
    layer = entry["layers"]["0"]
    assert entry["position_baseline"]["r2"] > 0.99  # position alone explains almost everything
    assert layer["probe"]["r2"] > 0.99
    assert layer["margin"] < 0.05  # so the margin says almost nothing was added
    assert layer["within_position"]["r2"] > 0.7  # but the state is decodable within the cell


def test_within_position_probe_finds_nothing_when_the_state_is_not_in_the_features() -> None:
    dataset = _position_plus_state_dataset(state_in_features=False)
    results = state_probe.fit_probes(dataset, ["pending_count"], [0], within_position=True)
    layer = results["targets"]["pending_count"]["layers"]["0"]
    assert layer["probe"]["r2"] > 0.99  # the position signal is still fully decodable
    assert layer["within_position"]["r2"] < 0.2


def test_within_position_categorical_scores_only_cells_with_more_than_one_class() -> None:
    pure = state_probe.fit_probes(_pure_position_dataset(), ["phase"], [0], within_position=True)
    assert pure["targets"]["phase"]["layers"]["0"]["within_position"]["skipped"]
    mixed = state_probe.fit_probes(
        _position_plus_state_dataset(), ["phase"], [0], within_position=True
    )
    assert mixed["targets"]["phase"]["layers"]["0"]["within_position"]["accuracy"] > 0.8


def test_within_position_cell_eligibility_never_looks_at_test_labels() -> None:
    values = np.array([0.0, 1.0, 0.0, 1.0])
    keys = ["alpha|0", "alpha|0", "beta|0", "beta|0"]
    # Each training cell is constant. The held-out labels would make both cells look variable
    # if eligibility were incorrectly computed over the complete dataset.
    eligible = state_probe._varying_training_cells(values, keys, np.array([0, 2]), "regression")
    assert eligible == set()


def test_within_position_excludes_cells_not_supported_by_training() -> None:
    dataset = _position_plus_state_dataset()
    _train, test = state_probe.split_by_task(
        dataset.task_ids, seed=20260903, difficulty=dataset.difficulty
    )
    held_out_task = dataset.task_ids[test[0]]
    mask = dataset.task_ids == held_out_task
    dataset.family[mask] = "heldout-only"

    results = state_probe.fit_probes(dataset, ["pending_count"], [0], within_position=True)
    within = results["targets"]["pending_count"]["layers"]["0"]["within_position"]
    assert not within.get("skipped"), within
    assert within["excluded_test_rows"] >= int(mask.sum())
    assert within["n_test"] < results["targets"]["pending_count"]["n_test"]


def test_split_by_task_puts_every_difficulty_on_both_sides() -> None:
    task_ids, _family, _step, difficulty = _cell_frame()
    train, test = state_probe.split_by_task(task_ids, seed=5, difficulty=difficulty)
    assert not (set(task_ids[train]) & set(task_ids[test]))
    assert set(difficulty[train].tolist()) == {0, 1, 2}
    assert set(difficulty[test].tolist()) == {0, 1, 2}
    # Without the argument every task is one group, which is the original behaviour.
    plain_train, plain_test = state_probe.split_by_task(task_ids, seed=5)
    assert len(plain_train) + len(plain_test) == len(task_ids)
    assert not (set(task_ids[plain_train]) & set(task_ids[plain_test]))


def test_fit_probes_breaks_results_down_by_difficulty() -> None:
    results = state_probe.fit_probes(_decoupled_state_dataset(), ["pending_count"], [0])
    entry = results["targets"]["pending_count"]
    by_difficulty = entry["layers"]["0"]["probe"]["by_difficulty"]
    assert set(by_difficulty) == {"0", "1", "2"}
    assert sum(group["n"] for group in by_difficulty.values()) == entry["n_test"]
    assert set(entry["position_baseline"]["by_difficulty"]) == {"0", "1", "2"}
    assert "By difficulty" in state_probe.render_markdown(results, "fake")


def test_mix_difficulty_plan_spans_three_difficulties_and_keeps_the_recovery_variants() -> None:
    from local_llm_lab.pipeline.tasks import make_tasks

    levels = {}
    variants = {}
    for split, perturb in state_probe.MIX_PLAN:
        # 24 tasks is two passes over the twelve families, so the variant cycle advances.
        tasks = make_tasks(split, 24) if perturb is None else make_tasks(split, 24, perturb=perturb)
        difficulties = state_probe.task_difficulties(split, tasks)
        assert set(difficulties) == {task.task_id for task in tasks}
        levels[split] = sorted(set(difficulties.values()))
        variants[split] = {task.variant for task in tasks}
    assert levels["train"] == [0]
    assert levels["p2mix"] == [0, 1]  # a rollout-style name alternates, per tasks.difficulty
    assert levels["test"] == [2]
    assert variants["train"] - {"clean"}  # the recovery variants are present
    assert variants["p2mix"] - {"clean"}
    assert variants["test"] == {"clean"}


def test_mixing_difficulties_breaks_the_position_to_state_mapping() -> None:
    """The mixed design adds real within-position support, including pending_count."""
    from local_llm_lab.pipeline.tasks import make_tasks

    def labelled(plan, count: int = 120):
        tasks = []
        difficulties = {}
        for split, perturb in plan:
            made = (
                make_tasks(split, count)
                if perturb is None
                else make_tasks(split, count, perturb=perturb)
            )
            tasks.extend(made)
            difficulties.update(state_probe.task_difficulties(split, made))
        return state_probe.build_label_dataset(tasks, difficulties)

    targets = ["pending_count", "first_bucket_count", "running_max"]
    train_only = state_probe.position_determinism(labelled([("train", None)]), targets)
    mixed_dataset = labelled(list(state_probe.MIX_PLAN))
    mixed = state_probe.position_determinism(mixed_dataset, targets)
    assert sorted(set(mixed_dataset.difficulty.tolist())) == [0, 1, 2]
    assert mixed["pending_count"]["constant_fraction"] < (
        train_only["pending_count"]["constant_fraction"] - 0.05
    )
    assert mixed["pending_count"]["explained"] < train_only["pending_count"]["explained"] - 0.03
    assert train_only["first_bucket_count"]["constant_fraction"] > 0.8
    assert mixed["first_bucket_count"]["constant_fraction"] < 0.3
    assert (
        mixed["running_max"]["constant_fraction"] < train_only["running_max"]["constant_fraction"]
    )

    gate = state_probe.validate_mixed_design(mixed_dataset, list(state_probe.TARGETS))
    assert gate["status"] == "PASS", gate
    assert gate["targets"]["pending_count"]["eligible_cells"] >= 30
    assert gate["targets"]["pending_count"]["n_test"] >= 300


def test_mixed_design_gate_rejects_three_difficulties_without_within_cell_variation() -> None:
    gate = state_probe.validate_mixed_design(
        _pure_position_dataset(),
        ["pending_count", "phase"],
        min_cells=1,
        min_train_rows=1,
        min_test_rows=1,
    )
    assert gate["difficulty_levels"] == [0, 1, 2]
    assert gate["status"] == "FAIL"
    assert gate["targets"]["pending_count"]["eligible_cells"] == 0
    assert gate["targets"]["phase"]["eligible_cells"] == 0


def test_mix_difficulty_forces_within_position_analysis() -> None:
    assert state_probe.resolve_within_position(True, None) is True
    assert state_probe.resolve_within_position(False, None) is False
    assert state_probe.resolve_within_position(False, True) is True
    with pytest.raises(ValueError, match="requires within-position"):
        state_probe.resolve_within_position(True, False)


def test_dataset_round_trip_keeps_the_per_row_position_arrays(tmp_path) -> None:
    dataset = _pure_position_dataset()
    path = state_probe.save_dataset(dataset, tmp_path / "state.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a complete file must not warn
        reloaded = state_probe.load_dataset(path)
    assert reloaded.family.tolist() == dataset.family.tolist()
    assert reloaded.step_index.tolist() == dataset.step_index.tolist()
    assert reloaded.difficulty.tolist() == dataset.difficulty.tolist()
    assert reloaded.cells.tolist() == dataset.cells.tolist()


def test_load_dataset_derives_position_keys_from_a_file_written_before_them(tmp_path) -> None:
    path = tmp_path / "old.npz"
    task_ids = np.array(
        [
            "train-read-0000-clean",
            "train-read-0000-clean",
            "train-aggregate_report-0011-transient",
        ]
    )
    np.savez_compressed(
        path,
        layer_0=np.zeros((3, 4), dtype=np.float32),
        label_pending_count=np.array([1.0, 0.0, 2.0]),
        task_ids=task_ids,
        meta=np.array(json.dumps({"rows": 3, "layers": [0]})),
    )
    with pytest.warns(UserWarning, match="family"):
        dataset = state_probe.load_dataset(path)
    assert dataset.family.tolist() == ["read", "read", "aggregate_report"]
    assert dataset.step_index.tolist() == [0, 1, 0]
    assert dataset.difficulty.tolist() == [-1, -1, -1]  # unknown, so no difficulty breakdown


_BASE_CAPTURE = Path(__file__).resolve().parents[1] / "outputs/probes/state/state-base.npz"


@pytest.mark.skipif(not _BASE_CAPTURE.is_file(), reason="the base capture has not been made")
def test_position_baseline_reproduces_the_confound_on_the_captured_base_dataset() -> None:
    """The regression test for the confound itself, on the real 737-row base capture.

    Read-only: this loads the saved activations, never the model. §5 reports a position-only
    R^2 of 0.907 for pending_count against the probe's 0.874, which is why the margin, not the
    probe's own R^2, is the reportable number.
    """
    with pytest.warns(UserWarning):  # captured before the per-row arrays existed
        dataset = state_probe.load_dataset(_BASE_CAPTURE)
    train, test = state_probe.split_by_task(dataset.task_ids, seed=20260903)
    baseline = state_probe.position_baseline(dataset, "pending_count", train, test)
    assert baseline["r2"] > 0.85, baseline
    determinism = state_probe.position_determinism(dataset, ["pending_count", "first_bucket_count"])
    assert determinism["first_bucket_count"]["constant_fraction"] == 1.0
    assert determinism["pending_count"]["explained"] > 0.9


# --------------------------------------------------------------------------- statistics


def test_mann_whitney_u_separates_a_known_split() -> None:
    low = np.arange(10.0)
    high = np.arange(10.0) + 100.0
    result = stats.mann_whitney_u(high, low)
    assert result["u"] == 100.0  # every pair favours "high"
    assert result["effect"] == 1.0
    assert result["p"] < 1e-3
    # Identical samples: no separation, and a p-value that says so.
    same = stats.mann_whitney_u(low, low.copy())
    assert same["effect"] == 0.5
    assert same["p"] > 0.9


def test_mann_whitney_u_handles_ties_and_empty_samples() -> None:
    tied = stats.mann_whitney_u(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 2.0]))
    assert 0.0 <= tied["effect"] <= 1.0
    assert 0.0 <= tied["p"] <= 1.0
    empty = stats.mann_whitney_u(np.array([]), np.array([1.0]))
    assert math.isnan(empty["u"])


def test_auc_and_ranks() -> None:
    assert stats.ranks(np.array([3.0, 1.0, 2.0])).tolist() == [3.0, 1.0, 2.0]
    assert stats.ranks(np.array([1.0, 1.0])).tolist() == [1.5, 1.5]
    assert stats.auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert stats.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0
    assert math.isnan(stats.auc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_macro_f1_and_r2_edge_cases() -> None:
    classes = np.array(["a", "b"])
    assert stats.macro_f1(np.array(["a", "b"]), np.array(["a", "b"]), classes) == 1.0
    assert stats.macro_f1(np.array(["a", "a"]), np.array(["a", "b"]), classes) == pytest.approx(
        1 / 3
    )
    assert stats.r2(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 1.0
    assert math.isnan(stats.r2(np.array([1.0, 1.0]), np.array([1.0, 1.0])))
    assert stats.mae(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == 1.0


# --------------------------------------------------------------------------- probe dataset


class _ChatTokenizer(_ProbeTokenizer):
    """Adds a chat template, and keeps only the last few words so the fake model stays cheap."""

    window = 48

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert not tokenize
        rendered = "".join(f"<|{message['role']}|>\n{message['content']}\n" for message in messages)
        return rendered + ("<|assistant|>\n" if add_generation_prompt else "")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        words = text.split()[-self.window :] or ["empty"]
        return [sum(map(ord, word)) % self.vocab_size for word in words]


def test_build_prompt_with_a_wide_window_does_not_re_hide_observations() -> None:
    """``build_probe_dataset`` re-renders rows that ``build_rows`` already windowed; windowing
    twice would re-stub a stub and silently change the text, so the call must be a no-op."""
    from local_llm_lab.pipeline.protocol import build_prompt, hidden_observation

    tokenizer = _ChatTokenizer()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {
            "role": "tool",
            "name": "read_file",
            "content": hidden_observation("read_file", "a\nb\nc"),
        },
        {"role": "assistant", "content": "note"},
        {"role": "tool", "name": "read_file", "content": "x\ny"},
    ]
    wide = build_prompt(tokenizer, messages, keep_last=len(messages))
    assert wide == tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    assert "3 line(s)" in wide  # the original stub, not a stub of the stub
    assert "1 line(s)" not in build_prompt(tokenizer, messages, keep_last=len(messages))


def test_build_probe_dataset_captures_one_row_per_supervised_step(tmp_path) -> None:
    from local_llm_lab.pipeline.data import build_rows
    from local_llm_lab.pipeline.tasks import make_tasks

    model = _model()
    tokenizer = _ChatTokenizer()
    tasks = [task for task in make_tasks("test", 12) if task.family in ("read", "synthesis")][:2]
    expected = sum(len(build_rows(task)) for task in tasks)
    seen: list[int] = []
    difficulties = state_probe.task_difficulties("test", make_tasks("test", 12))
    dataset = state_probe.build_probe_dataset(
        model,
        tokenizer,
        tasks,
        [1, 3],
        difficulties=difficulties,
        progress=lambda n, total, rows: seen.append(rows),
    )
    assert len(dataset) == expected
    assert dataset.features[1].shape == (expected, 16)
    assert dataset.features[3].shape == (expected, 16)
    assert dataset.labels["next_tool"].tolist() == [
        step.action.name
        for task in tasks
        for step, row in zip(
            [task.steps[row["metadata"]["step"]] for row in build_rows(task)],
            build_rows(task),
            strict=True,
        )
    ]
    assert set(dataset.task_ids.tolist()) == {task.task_id for task in tasks}
    assert dataset.meta["strip"] is False
    assert seen == [len(build_rows(tasks[0])), expected]

    # The per-row position coordinates the baseline needs come from the task and the row's
    # metadata, not from parsing the task id.
    assert dataset.family.tolist() == [task.family for task in tasks for _ in build_rows(task)]
    assert dataset.step_index.tolist() == [
        row["metadata"]["step"] for task in tasks for row in build_rows(task)
    ]
    assert dataset.difficulty.tolist() == [
        difficulties[task.task_id] for task in tasks for _ in build_rows(task)
    ]
    assert dataset.meta["difficulties"] == {"2": expected}

    path = state_probe.save_dataset(dataset, tmp_path / "state.npz")
    reloaded = state_probe.load_dataset(path)
    assert len(reloaded) == len(dataset)
    assert np.allclose(reloaded.features[3], dataset.features[3])
    assert reloaded.labels["phase"].tolist() == dataset.labels["phase"].tolist()
    assert reloaded.meta == dataset.meta
    assert reloaded.family.tolist() == dataset.family.tolist()
    assert reloaded.step_index.tolist() == dataset.step_index.tolist()
    assert reloaded.difficulty.tolist() == dataset.difficulty.tolist()


def test_build_probe_dataset_strip_changes_the_captured_activations() -> None:
    """The stripped condition must actually reach the model, not just the note object."""
    from local_llm_lab.pipeline.tasks import make_tasks

    model = _model()
    tokenizer = _ChatTokenizer()
    tasks = [task for task in make_tasks("test", 12) if task.family == "ledger_reconcile"][:1]
    intact = state_probe.build_probe_dataset(model, tokenizer, tasks, [2])
    stripped = state_probe.build_probe_dataset(model, tokenizer, tasks, [2], strip=True)
    assert len(intact) == len(stripped)
    assert stripped.meta["strip"] is True
    differences = np.abs(intact.features[2] - stripped.features[2]).max(axis=1)
    assert float(differences.max()) > 1e-4
    # The labels are ground truth from the generator, so stripping must not move them.
    assert intact.labels["pending_count"].tolist() == stripped.labels["pending_count"].tolist()


def test_build_probe_dataset_materializes_all_layers_together(monkeypatch) -> None:
    """A row must be evaluated as one graph before any activation is copied to NumPy."""
    from local_llm_lab.pipeline.tasks import make_tasks

    class GuardedArray:
        def __init__(self, values: np.ndarray, group: list[GuardedArray]) -> None:
            self.values = values
            self.group = group
            self.evaluated = False

        def __array__(self, dtype=None, copy=None):
            assert all(value.evaluated for value in self.group), "row was not evaluated together"
            return np.array(self.values, dtype=dtype, copy=True if copy is None else copy)

    class Runtime:
        def eval(self, *values) -> None:
            assert len(values) == 2
            assert len({id(value.group) for value in values}) == 1
            for value in values:
                value.evaluated = True

        def clear_cache(self) -> None:
            pass

        def reset_peak_memory(self) -> None:
            pass

        def get_active_memory(self) -> int:
            return 0

        def get_cache_memory(self) -> int:
            return 0

        def get_peak_memory(self) -> int:
            return 0

    sources: list[GuardedArray] = []

    def fake_capture(_model, _token_ids, layers, positions="last"):
        assert positions == "last"
        group: list[GuardedArray] = []
        for layer in layers:
            group.append(GuardedArray(np.full(16, layer, dtype=np.float32), group))
        sources.extend(group)
        return dict(zip(layers, group, strict=True))

    monkeypatch.setattr(state_probe, "capture_residuals", fake_capture)
    tasks = [task for task in make_tasks("test", 12) if task.family == "read"]
    dataset = state_probe.build_probe_dataset(
        object(),
        _ChatTokenizer(),
        tasks,
        [1, 3],
        mlx_runtime=Runtime(),
    )
    assert np.all(dataset.features[1] == 1.0)
    assert np.all(dataset.features[3] == 3.0)
    for source in sources:
        source.values[:] = -1.0
    assert np.all(dataset.features[1] == 1.0), "features must not retain MLX-backed storage"


def test_build_probe_dataset_reclaims_cache_and_reports_each_task(monkeypatch) -> None:
    """A second task exceeds the simulated budget unless the first task's cache was cleared."""
    from local_llm_lab.pipeline.tasks import make_tasks

    class Runtime:
        def __init__(self) -> None:
            self.cached = 0
            self.peak = 0

        def eval(self, *values) -> None:
            del values
            # Each two-row task peaks at 300 MiB. Without a task-boundary clear, the
            # second task crosses the simulated 512 MiB allocator budget.
            self.cached += 150 * 2**20
            self.peak = max(self.peak, self.cached)
            if self.cached > 512 * 2**20:
                raise MemoryError("simulated MLX allocator exhaustion")

        def clear_cache(self) -> None:
            self.cached = 0

        def reset_peak_memory(self) -> None:
            self.peak = self.cached

        def get_active_memory(self) -> int:
            return 2 * 2**30

        def get_cache_memory(self) -> int:
            return self.cached

        def get_peak_memory(self) -> int:
            return self.peak

    def fake_capture(_model, _token_ids, layers, positions="last"):
        del positions
        return {layer: np.full(16, layer, dtype=np.float32) for layer in layers}

    monkeypatch.setattr(state_probe, "capture_residuals", fake_capture)
    tasks = [task for task in make_tasks("test", 24) if task.family == "read"]
    telemetry = []
    dataset = state_probe.build_probe_dataset(
        object(),
        _ChatTokenizer(),
        tasks,
        [1],
        mlx_runtime=Runtime(),
        memory_progress=telemetry.append,
    )
    assert len(dataset) > 0
    assert len(telemetry) == len(tasks)
    assert all(sample["cache_bytes"] == 0 for sample in telemetry)
    assert all(sample["active_bytes"] == 2 * 2**30 for sample in telemetry)


def test_build_probe_dataset_preserves_row_level_prompt_statistics(monkeypatch) -> None:
    from local_llm_lab.pipeline.data import build_rows
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.pipeline.tasks import make_tasks

    def fake_capture(_model, _token_ids, layers, positions="last"):
        del positions
        return {layer: mx.full((16,), layer, dtype=mx.float32) for layer in layers}

    monkeypatch.setattr(state_probe, "capture_residuals", fake_capture)
    tokenizer = _ChatTokenizer()
    tasks = [task for task in make_tasks("test", 24) if task.family == "read"]
    expected_lengths = [
        len(
            tokenizer.encode(
                build_prompt(
                    tokenizer,
                    row["messages"][:-1],
                    keep_last=len(row["messages"][:-1]),
                )
            )
        )
        for task in tasks
        for row in build_rows(task)
    ]
    dataset = state_probe.build_probe_dataset(
        object(),
        tokenizer,
        tasks,
        [1],
    )
    assert dataset.meta["mean_prompt_tokens"] == pytest.approx(np.mean(expected_lengths))
    assert dataset.meta["max_prompt_tokens"] == max(expected_lengths)


def test_build_probe_dataset_resumes_completed_task_checkpoints(monkeypatch, tmp_path) -> None:
    """An interrupted capture reuses its atomic task shard and captures only later tasks."""
    from local_llm_lab.pipeline.tasks import make_tasks

    model = _model()
    tokenizer = _ChatTokenizer()
    tasks = [task for task in make_tasks("test", 24) if task.family == "read"]
    checkpoint_dir = tmp_path / "checkpoints"

    def interrupt_after_first(number, _total, _rows):
        if number == 1:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        state_probe.build_probe_dataset(
            model,
            tokenizer,
            tasks,
            [1, 3],
            checkpoint_dir=checkpoint_dir,
            checkpoint_context={"model": "fake", "policy": "base"},
            progress=interrupt_after_first,
        )

    shards = sorted(checkpoint_dir.glob("*.npz"))
    assert len(shards) == 1
    first = state_probe.load_dataset(shards[0])
    first_mtime = shards[0].stat().st_mtime_ns

    def later_capture(_model, _token_ids, layers, positions="last"):
        del positions
        return {layer: mx.full((16,), 99.0, dtype=mx.float32) for layer in layers}

    monkeypatch.setattr(state_probe, "capture_residuals", later_capture)
    telemetry = []
    resumed = state_probe.build_probe_dataset(
        model,
        tokenizer,
        tasks,
        [1, 3],
        checkpoint_dir=checkpoint_dir,
        checkpoint_context={"model": "fake", "policy": "base"},
        memory_progress=telemetry.append,
    )
    assert shards[0].stat().st_mtime_ns == first_mtime
    assert np.allclose(resumed.features[1][: len(first)], first.features[1])
    assert np.all(resumed.features[1][len(first) :] == 99.0)
    assert resumed.meta["resumed_tasks"] == 1
    assert len(list(checkpoint_dir.glob("*.npz"))) == len(tasks)
    assert telemetry[0]["resumed"] is True
    assert telemetry[0]["peak_bytes"] is None


def test_checkpoint_task_hash_is_stable_across_python_hash_seeds() -> None:
    script = """
from local_llm_lab.pipeline.tasks import make_tasks
from local_llm_lab.probes.state_probe import _checkpoint_signature
task = next(task for task in make_tasks('train', 12) if len(task.required_tools) > 1)
print(_checkpoint_signature(
    task, [1, 3], strip=False, keep_last=2, difficulty=0, context={}
)['task_sha256'])
"""
    hashes = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("1", "3")
    }
    assert len(hashes) == 1


def test_artifact_identity_changes_when_local_weights_change(tmp_path) -> None:
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    weights = artifact / "adapter.safetensors"
    config = artifact / "adapter_config.json"
    weights.write_bytes(b"weights-one")
    config.write_text('{"rank": 8}\n', encoding="utf-8")
    before = state_probe.artifact_identity(artifact)
    weights.write_bytes(b"weights-two")
    after = state_probe.artifact_identity(artifact)
    assert before.startswith("sha256:")
    assert before != after


def test_artifact_identity_accepts_a_partial_cached_huggingface_snapshot(monkeypatch) -> None:
    from types import SimpleNamespace

    import huggingface_hub

    revision = SimpleNamespace(commit_hash="abc123", refs=frozenset({"main"}))
    repository = SimpleNamespace(repo_id="org/model", repo_type="model", revisions=[revision])
    monkeypatch.setattr(
        huggingface_hub,
        "scan_cache_dir",
        lambda: SimpleNamespace(repos=[repository]),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *_args, **_kwargs: pytest.fail("must not require a complete snapshot"),
    )
    assert state_probe.artifact_identity("org/model") == "hf:org/model@abc123"


def test_state_probe_cli_bounds_and_releases_mlx_before_fitting(monkeypatch, tmp_path) -> None:
    from local_llm_lab.pipeline import evaluate
    from local_llm_lab.probes import guard, policies

    events = []

    class TrackedModel:
        def __del__(self):
            events.append("model released")

    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: events.append("idle checked"))
    monkeypatch.setattr(policies, "resolve_policy", lambda _policy: None)
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda _model, _adapter: (events.append("model loaded") or TrackedModel(), object()),
    )
    monkeypatch.setattr(state_probe, "artifact_identity", lambda value: f"identity:{value}")
    monkeypatch.setattr(
        state_probe,
        "build_probe_dataset",
        lambda *_args, **_kwargs: events.append("dataset captured") or _pure_position_dataset(),
    )
    monkeypatch.setattr(
        state_probe,
        "fit_probes",
        lambda *_args, **_kwargs: (
            events.append("probes fitted") or {"targets": {}, "meta": {}, "seed": 1}
        ),
    )
    monkeypatch.setattr(state_probe, "render_markdown", lambda *_args: "report")

    previous_limit = 987654321

    def set_limit(value):
        events.append(("cache limit", value))
        return previous_limit

    monkeypatch.setattr(mx, "set_cache_limit", set_limit)
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("cache cleared"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-v2-probe-state",
            "--model",
            "fake/model",
            "--limit",
            "1",
            "--layers",
            "0",
            "--targets",
            "pending_count",
            "--mlx-cache-limit-mib",
            "64",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    state_probe.main()

    assert events.index(("cache limit", 64 * 2**20)) < events.index("model loaded")
    assert events.index("model released") < events.index("probes fitted")
    assert events.index("cache cleared") < events.index("probes fitted")
    assert events.index(("cache limit", previous_limit)) < events.index("probes fitted")

    events.clear()

    def fail_capture(*_args, **_kwargs):
        events.append("capture failed")
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(state_probe, "build_probe_dataset", fail_capture)
    with pytest.raises(RuntimeError, match="simulated capture failure"):
        state_probe.main()
    assert events.index("model released") > events.index("capture failed")
    assert events.index("cache cleared") > events.index("capture failed")
    assert events.index(("cache limit", previous_limit)) > events.index("capture failed")
    assert "probes fitted" not in events


def test_save_dataset_keeps_previous_file_when_atomic_write_fails(monkeypatch, tmp_path) -> None:
    dataset = _pure_position_dataset()
    path = state_probe.save_dataset(dataset, tmp_path / "state.npz")
    original = path.read_bytes()

    def broken_save(handle, **_payload):
        handle.write(b"incomplete replacement")
        handle.flush()
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(state_probe.np, "savez_compressed", broken_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        state_probe.save_dataset(dataset, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_set_mlx_cache_limit_uses_mib_and_rejects_nonpositive_values() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.limit = 3

        def set_cache_limit(self, value: int) -> int:
            previous, self.limit = self.limit, value
            return previous

    runtime = Runtime()
    assert state_probe.set_mlx_cache_limit(runtime, 512) == 3
    assert runtime.limit == 512 * 2**20
    with pytest.raises(ValueError, match="positive"):
        state_probe.set_mlx_cache_limit(runtime, 0)


# --------------------------------------------------------------------------- assistant axis


def test_roles_are_twenty_four_and_each_half_spans_every_group() -> None:
    names = [name for name, _prompt in assistant_axis.ROLES]
    assert len(names) == 24
    assert len(set(names)) == 24
    assert all(
        prompt.startswith("You are") and prompt.endswith(".") for _n, prompt in assistant_axis.ROLES
    )
    high = {"consultant", "reviewer", "analyst", "librarian", "tutor", "engineer"}
    low = {"ghost", "hermit", "leviathan", "oracle", "trickster", "pirate", "prophet", "wanderer"}
    for half in (names[:12], names[12:]):
        assert set(half) & high and set(half) & low and set(half) - high - low
    assert assistant_axis.DEFAULT_SYSTEM.startswith("You are a concise, capable assistant")


class _PlantedModel:
    """A stand-in whose "activation" for a response is a planted direction plus noise.

    ``build_axis`` only ever reaches the model through ``response_mean_activations``, so
    patching that one function is enough to test the axis arithmetic on vectors whose answer
    is known in advance.
    """

    def __init__(self, dim: int = 24, seed: int = 1) -> None:
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.direction = np.zeros(dim, dtype=np.float32)
        self.direction[0] = 1.0
        # A shared "role-ness" offset, orthogonal to the planted axis so it does not fight it.
        style = self.rng.normal(size=dim)
        style[0] = 0.0
        self.role_style = (style / np.linalg.norm(style)).astype(np.float32)

    def activation(self, response: str, layers):
        import mlx.core as mx

        # Default-assistant responses sit at +1 along the planted direction, roles at -1, and
        # every role also carries a shared "role-ness" offset so PC1 has something to find.
        assistant = response.startswith("assistant")
        base = self.direction if assistant else -self.direction
        role_index = 0 if assistant else int(response.split(":")[0].removeprefix("role"))
        offset = 0.0 if assistant else (0.1 + 0.02 * role_index) * self.role_style
        noise = 0.02 * self.rng.normal(size=self.dim)
        vector = (base + offset + noise).astype(np.float32)
        return {layer: mx.array(vector * (1.0 + 0.01 * layer)) for layer in layers}


@pytest.fixture()
def planted(monkeypatch):
    model = _PlantedModel()

    def fake_response_mean(model_, tokenizer, prompt, response, layers, *, stats=None):
        del model_, tokenizer, prompt
        if stats is not None:
            stats["sequences"] = stats.get("sequences", 0) + 1
        return model.activation(response, layers)

    monkeypatch.setattr(assistant_axis, "response_mean_activations", fake_response_mean)
    return model


def test_build_axis_recovers_a_planted_direction_with_a_stable_pc1(planted) -> None:
    default = [(f"p{index}", f"assistant reply {index}") for index in range(20)]
    roles = {
        f"role{index}": [(f"p{k}", f"role{index}: reply {k}") for k in range(5)]
        for index in range(24)
    }
    axis, diagnostics = assistant_axis.build_axis(None, None, default, roles, [6, 18])
    assert sorted(axis) == [6, 18]
    for layer in (6, 18):
        direction = np.array(axis[layer])
        assert assistant_axis.project(planted.direction, direction) > 0.9
        entry = diagnostics["layers"][str(layer)]
        assert entry["split_half_cosine"] > 0.99
        assert entry["chat_projection_mean"] > 0.9
        assert 0.0 <= entry["pc1_cosine_abs"] <= 1.0
        assert len(entry["role_projections"]) == 24
    assert diagnostics["n_default_responses"] == 20


def test_build_axis_pc1_aligns_when_roles_vary_along_the_axis(monkeypatch) -> None:
    """The paper's check: when role variation *is* the axis, |cos(axis, PC1)| is near one."""
    import mlx.core as mx

    dim = 16
    rng = np.random.default_rng(2)
    # Equal-magnitude entries: per-dimension standardisation rescales each coordinate, so a
    # one-hot or wildly uneven direction would survive centring but not standardisation.
    direction = rng.choice([-1.0, 1.0], size=dim).astype(np.float32) / math.sqrt(dim)

    def fake_response_mean(model_, tokenizer, prompt, response, layers, *, stats=None):
        del model_, tokenizer, prompt, stats
        if response.startswith("assistant"):
            offset = 1.0
        else:
            offset = -1.0 - 0.4 * int(response.split(":")[0].removeprefix("role"))
        vector = offset * direction + 0.01 * rng.normal(size=dim)
        return {layer: mx.array(vector.astype(np.float32)) for layer in layers}

    monkeypatch.setattr(assistant_axis, "response_mean_activations", fake_response_mean)
    default = [(f"p{index}", f"assistant {index}") for index in range(8)]
    roles = {f"role{index}": [("p", f"role{index}: r")] for index in range(24)}
    _axis, diagnostics = assistant_axis.build_axis(None, None, default, roles, [0])
    assert diagnostics["layers"]["0"]["pc1_cosine_abs"] > 0.9


def test_project_is_a_cosine() -> None:
    assert assistant_axis.project(np.array([2.0, 0.0]), np.array([5.0, 0.0])) == pytest.approx(1.0)
    assert assistant_axis.project(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert assistant_axis.project(np.array([-1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(
        -1.0
    )
    assert assistant_axis.project(np.zeros(3), np.ones(3)) == 0.0


def test_save_and_load_axis_round_trips(tmp_path) -> None:
    import mlx.core as mx

    axis = {6: mx.array(np.arange(4, dtype=np.float32)), 18: mx.array(np.ones(4, dtype=np.float32))}
    diagnostics = {"layers": {"6": {"axis_norm": 1.0}}, "roles": ["a"]}
    path = assistant_axis.save_axis(tmp_path / "axis.npz", axis, diagnostics)
    loaded, meta = assistant_axis.load_axis(path)
    assert sorted(loaded) == [6, 18]
    assert loaded[6].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert meta == diagnostics


def test_rollout_role_generates_greedily_through_stream_generate(monkeypatch) -> None:
    seen: list[str] = []

    class _Piece:
        def __init__(self, text: str) -> None:
            self.text = text
            self.token = 0

    def fake_stream_generate(model, tokenizer, *, prompt, max_tokens, sampler):
        del model, tokenizer, sampler
        seen.append(prompt)
        for piece in ("Ahoy", " there", f" [{max_tokens}]"):
            yield _Piece(piece)

    monkeypatch.setattr("mlx_lm.stream_generate", fake_stream_generate)
    pairs = assistant_axis.rollout_role(
        None, _ChatTokenizer(), "You are a pirate.", ["Hello", "Again"], max_tokens=7
    )
    assert [response for _prompt, response in pairs] == ["Ahoy there [7]", "Ahoy there [7]"]
    assert all("You are a pirate." in prompt for prompt in seen)
    assert all(prompt.endswith("<|assistant|>\n") for prompt in seen)
    assert pairs[0][0] == seen[0]


def test_slope_is_least_squares_and_nan_for_one_turn() -> None:
    assert assistant_axis._slope([0.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert assistant_axis._slope([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert math.isnan(assistant_axis._slope([0.5]))


def _eval_payload(tmp_path: Path) -> Path:
    """A miniature evaluation JSON in exactly the layout ``pipeline.evaluate`` writes."""

    def trajectory(task_id, success, steps, **extra):
        record = {
            "task_id": task_id,
            "family": "read",
            "variant": "clean",
            "label": "fake",
            "prompt": f"do {task_id}",
            "steps": steps,
            "verdict": {"success": success, "reasons": [] if success else ["wrong answer: x"]},
            "parse_error": None,
            "turns": len(steps),
            "valid_turns": len(steps),
            "elapsed_seconds": 1.0,
            "faults": [],
            "generated_tokens": 10,
            "loop_detected": False,
            "exhausted": False,
        }
        record.update(extra)
        return record

    def step(index, name="read_file"):
        return {
            "index": index,
            "thought": f"note {index}",
            "action": {"name": name, "arguments": {"path": "a.txt"}},
            "observation": f"contents {index}",
            "raw": f"note {index}\n```json\n{{}}\n```",
        }

    payload = {
        "summary": {},
        "trajectories": [
            trajectory("t-good", True, [step(0), step(1), step(2, "finish")]),
            trajectory("t-loop", False, [step(0), step(1)], loop_detected=True),
            trajectory(
                "t-parse",
                False,
                [{"index": 0, "raw": "no call here", "parse_error": "boom"}],
                parse_error="boom",
            ),
        ],
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload))
    return path


def test_trajectory_projections_rebuild_prompts_like_the_runner(tmp_path) -> None:
    from local_llm_lab.agent_protocol import Action
    from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT, assistant_message, tool_message

    model = _model()
    tokenizer = _ChatTokenizer()
    axis = np.array(model.model.embed_tokens.weight[3], dtype=np.float32)
    records = assistant_axis.trajectory_projections(
        model, tokenizer, _eval_payload(tmp_path), axis, layer=2
    )
    assert [record["task_id"] for record in records] == ["t-good", "t-loop", "t-parse"]
    good, loop, parse = records
    assert good["turns"] == 3  # the finish turn is projected, but nothing is appended after it
    assert good["success"] is True and good["failure_reason"] is None
    assert loop["failure_reason"] == "repetition loop"
    assert parse["failure_reason"] == "parse error"
    assert parse["turns"] == 1
    assert all(-1.0 <= value <= 1.0 for record in records for value in record["projections"])
    assert good["min"] == pytest.approx(min(good["projections"]))
    assert good["mean"] == pytest.approx(float(np.mean(good["projections"])))

    # The second turn's prompt must be exactly what the runner would have built.
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.probes.capture import response_mean_activations

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "do t-good"},
        assistant_message("note 0", Action("read_file", {"path": "a.txt"})),
        tool_message("read_file", "contents 0"),
    ]
    expected = assistant_axis.project(
        response_mean_activations(
            model, tokenizer, build_prompt(tokenizer, messages), "note 1\n```json\n{}\n```", [2]
        )[2],
        axis,
    )
    assert good["projections"][1] == pytest.approx(expected, abs=1e-6)


def test_summarize_projections_separates_outcomes_and_reports_the_u_test() -> None:
    records = [
        {
            "mean": 0.9 - 0.01 * index,
            "slope": 0.01,
            "min": 0.8,
            "success": True,
            "failure_reason": None,
            "turns": 3,
        }
        for index in range(10)
    ] + [
        {
            "mean": 0.2 - 0.01 * index,
            "slope": -0.02,
            "min": 0.05,
            "success": False,
            "failure_reason": "repetition loop" if index % 2 else "step budget exhausted",
            "turns": 4,
        }
        for index in range(10)
    ]
    summary = assistant_axis.summarize_projections(records)
    assert summary["success"]["n"] == 10
    assert summary["failure"]["n"] == 10
    assert summary["success"]["mean"]["mean"] > summary["failure"]["mean"]["mean"]
    test = summary["mann_whitney_success_vs_failure"]["mean"]
    assert test["effect"] == 1.0
    assert test["p"] < 0.001
    assert set(summary["by_failure_reason"]) == {"repetition loop", "step budget exhausted"}
    markdown = assistant_axis.render_projection_markdown(summary, records, "fake")
    assert "Mann-Whitney" in markdown and "repetition loop" in markdown


def test_render_build_markdown_contains_the_sanity_checks(planted) -> None:
    default = [(f"p{index}", f"assistant {index}") for index in range(6)]
    roles = {f"role{index}": [("p", f"role{index}: r")] for index in range(24)}
    _axis, diagnostics = assistant_axis.build_axis(None, None, default, roles, [6])
    markdown = assistant_axis.render_build_markdown(diagnostics, "base")
    assert "split-half" in markdown
    assert "cos(axis, PC1)" in markdown
    assert "role0" in markdown


# --------------------------------------------------------------------------- guards and policies


def test_gpu_users_reports_matching_processes(monkeypatch) -> None:
    import subprocess

    from local_llm_lab.probes import guard

    class _Completed:
        stdout = "111 uv run agent-pipeline eval\n999 grep agent-pipeline\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())
    monkeypatch.setattr("os.getpid", lambda: 999)
    assert guard.gpu_users() == ["111 uv run agent-pipeline eval"]

    class _Empty:
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Empty())
    assert guard.gpu_users() == []


def test_require_idle_gpu_aborts_when_busy_and_passes_when_overridden(monkeypatch) -> None:
    import argparse

    from local_llm_lab.probes import guard

    monkeypatch.setattr(guard, "gpu_users", lambda *a, **k: ["111 agent-v2-eval"])
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        guard.require_idle_gpu(parser, argparse.Namespace(allow_busy_gpu=False), "loading a model")
    guard.require_idle_gpu(parser, argparse.Namespace(allow_busy_gpu=True), "loading a model")
    monkeypatch.setattr(guard, "gpu_users", lambda *a, **k: [])
    guard.require_idle_gpu(parser, argparse.Namespace(allow_busy_gpu=False), "loading a model")


def test_resolve_policy_maps_names_and_rejects_unknown(tmp_path) -> None:
    from local_llm_lab.probes.policies import POLICY_NAMES, resolve_policy

    assert POLICY_NAMES == ("base", "A", "B", "C")
    assert resolve_policy("base") is None
    assert resolve_policy("B").name == "best-adapter"
    assert resolve_policy("B").parent.name == "agent-v2b"
    assert resolve_policy(str(tmp_path)) == tmp_path.resolve()
    with pytest.raises(ValueError):
        resolve_policy("nope")


def test_run_label_distinguishes_identically_named_adapter_directories() -> None:
    assert adapter_delta.run_label(Path("outputs/agent-v2b/best-adapter")) == "agent-v2b"
    assert adapter_delta.run_label(Path("outputs/agent-v2c/adapters")) == "agent-v2c"
    assert adapter_delta.run_label(Path("outputs/custom-run")) == "custom-run"


def test_random_subspace_cosine_is_small_in_a_large_ambient_space() -> None:
    assert adapter_delta.random_subspace_cosine(512, 4, trials=2) < 0.25
    assert adapter_delta.random_subspace_cosine(8, 8, trials=2) == pytest.approx(1.0, abs=1e-6)


def test_strip_state_fields_removes_batch_update_queue() -> None:
    from local_llm_lab.probes.capture import strip_state_fields

    note = (
        "Phase apply. Next: worker-2.ini mode=audit -> mode=fast. "
        "Remaining after this: worker-3.ini mode=observe -> mode=fast; worker-4.ini mode=safe -> mode=strict."
    )
    out = strip_state_fields(note)
    assert "worker-2.ini" not in out and "worker-3.ini" not in out and "worker-4.ini" not in out
    assert "audit" not in out and "observe" not in out
    assert out.startswith("Phase apply. Next: [stripped]. Remaining after this: [stripped]")
    verify = "Phase verify. Next: worker-1.ini, expect mode=fast. Remaining after this: none."
    out2 = strip_state_fields(verify)
    assert "worker-1.ini" not in out2 and "expect mode" not in out2
    assert out2 == "Phase verify. Next: [stripped]. Remaining after this: [stripped]."
