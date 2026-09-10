"""Backend-neutral architecture contracts, with no backend import or model load."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.arch_base import (
    ArchitectureViewBase,
    _validate_hidden_span,
    _validate_scored_position,
)


class Linear:
    def __init__(self, output=7, input=3):
        self.weight = SimpleNamespace(shape=(output, input))


class OtherLinear(Linear):
    pass


class Wrapped:
    def __init__(self, linear):
        self.linear = linear


class Block:
    def __init__(self, entries, *, recurrent=False):
        self.entries = entries
        self.is_linear = recurrent

    def named_modules(self):
        yield "", self
        yield from self.entries


class View(ArchitectureViewBase):
    def __init__(self, blocks):
        self.blocks = blocks
        self.num_layers = len(blocks)

    def layer_kind(self, index):
        return "linear_attention" if self.blocks[index].is_linear else "attention"

    @staticmethod
    def _linear_types():
        return (Linear, OtherLinear)

    @staticmethod
    def _linear_dimensions(module):
        return module.weight.shape


def _view():
    q = Linear()
    return View([
        Block([
            ("mlp.down_proj", Linear()),
            ("self_attn.q_proj", Wrapped(q)),
            ("self_attn.q_proj.linear", q),
            ("self_attn.q_proj.lora_a", Linear()),
            ("self_attn.v_proj", OtherLinear()),
            ("self_attn.in_proj_qkvz", Linear()),
        ]),
        Block([("linear_attn.in_proj_qkvz", Linear(5, 2))], recurrent=True),
    ])


def test_policy_order_wrapper_paths_and_hybrid_auto_stay_unchanged():
    view = _view()
    assert view.lora_targets("attention+mlp") == (
        "self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj",
    )
    assert view.lora_targets("auto") == (
        "self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj",
        "linear_attn.in_proj_qkvz",
    )
    assert view.lora_parameter_count(("self_attn.q_proj", "linear_attn.in_proj_qkvz"), 2) == 34
    assert not any(".linear" in path or ".lora_a" in path for _, path, _ in view._linear_modules())


def test_explicit_targets_preserve_user_order_and_count_each_block():
    view = View([Block([("q_proj", Linear())]), Block([("q_proj", Linear(9, 2))])])
    assert view.lora_targets(("q_proj",)) == ("q_proj",)
    assert view.lora_parameter_count(("q_proj",), 2) == 42


@pytest.mark.parametrize("policy", [(), ("missing",), ("self_attn.q_proj",) * 2, "unknown"])
def test_invalid_lora_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        _view().lora_targets(policy)


@pytest.mark.parametrize("rank", [True, False, 0, -1, 1.0])
def test_lora_rank_must_be_positive_integer(rank):
    with pytest.raises(ValueError, match="positive integer"):
        _view().lora_parameter_count(("self_attn.q_proj",), rank)


@pytest.mark.parametrize("value", [True, False, -1, 2, 1.0, "1"])
def test_block_index_contract(value):
    with pytest.raises(ValueError, match="block index must lie"):
        _view()._validate_block_index(value)


def test_valid_block_indices_cache_cardinality_and_kind_selection():
    view = _view()
    assert [view._validate_block_index(i) for i in (0, 1)] == [0, 1]
    assert view._cache_entries(None) == [None, None]
    caches = [object(), object()]
    assert view._cache_entries(caches) == caches
    assert view._first_cache(caches, "linear_attention") is caches[1]
    assert view._first_cache(caches, "absent") is None
    with pytest.raises(ValueError, match="one entry per block"):
        view._cache_entries([None])


@pytest.mark.parametrize("position", [-3, -1, 0, 2])
def test_scored_position_retains_python_negative_index(position):
    assert _validate_scored_position(position, 3) == position


@pytest.mark.parametrize("position", [True, False, -4, 3, 0.0, "1"])
def test_invalid_scored_position(position):
    with pytest.raises(ValueError):
        _validate_scored_position(position, 3)


def test_hidden_span_is_half_open_and_keeps_offsets():
    assert _validate_hidden_span((0, 3), 3) == (0, 3)
    assert _validate_hidden_span([1, 2], 3) == (1, 2)


@pytest.mark.parametrize("span", [None, (), (1,), (0, 1, 2), (True, 2), (0, 2.0),
                                  (-1, 2), (1, 1), (2, 1), (0, 4)])
def test_invalid_hidden_span_fails_closed(span):
    with pytest.raises(ValueError):
        _validate_hidden_span(span, 3)


def test_mlx_view_inherits_the_shared_public_contract():
    assert issubclass(ArchitectureView, ArchitectureViewBase)
    for name in ("lora_targets", "lora_parameter_count", "_validate_block_index", "_cache_entries"):
        assert getattr(ArchitectureView, name) is getattr(ArchitectureViewBase, name)
    assert str(inspect.signature(ArchitectureView.lora_targets)) == (
        "(self, policy: 'str | tuple[str, ...]') -> 'tuple[str, ...]'"
    )


def test_mlx_backend_hooks_keep_quantized_dimension_count(monkeypatch):
    import sys
    from types import ModuleType

    class QuantizedLinear(Linear):
        bits = 4

    nn = ModuleType("mlx.nn")
    nn.Linear = Linear
    nn.QuantizedLinear = QuantizedLinear
    mlx = ModuleType("mlx")
    mlx.nn = nn
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    view = ArchitectureView.__new__(ArchitectureView)
    view.blocks = [Block([("q_proj", Wrapped(QuantizedLinear()))])]
    view.num_layers = 1
    assert view.lora_targets("auto") == ("q_proj",)
    # MLX packs eight 4-bit inputs in each stored word: input dimension is 3 * 8.
    assert view.lora_parameter_count(("q_proj",), 2) == 2 * (7 + 24)
