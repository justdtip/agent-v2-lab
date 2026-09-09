"""Tiny CPU contracts for the upstream-backed native torch capture seam."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from local_llm_lab.upstream_ref import UpstreamUnavailable, load_upstream  # noqa: E402

try:
    _upstream = load_upstream()
except UpstreamUnavailable as error:
    pytest.skip(str(error), allow_module_level=True)
pytest.importorskip("jlens", reason="the selected upstream Jacobian-lens reference is unavailable")
ActivationRecorder = _upstream.fitting.ActivationRecorder
from torch import nn  # noqa: E402

from local_llm_lab.torch_capture import TorchCapture  # noqa: E402


class Block(nn.Module):
    def __init__(self, value, tuple_output=False):
        super().__init__()
        self.value, self.tuple_output = value, tuple_output

    def forward(
        self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None
    ):
        out = hidden_states * self.value
        return (out, "preserved") if self.tuple_output else out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 3)
        self.layers = nn.ModuleList([Block(2), Block(3, tuple_output=True)])
        self.fail = False

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                attention_mask=f"mask-{i}",
                position_embeddings=(i, i + 1),
                past_key_values=past_key_values,
            )
            if isinstance(h, tuple):
                h = h[0]
        if self.fail:
            raise RuntimeError("deliberate model failure")
        return SimpleNamespace(logits=h)


class Sink:
    def __init__(self):
        self.rows, self.outputs = {}, []

    def residual(self, layer, offset, h):
        self.rows[layer] = (offset, h)

    def output(self, offset, ids, logits):
        self.outputs.append((offset, ids, logits))


def fixture():
    torch.set_num_threads(1)
    model, sink = Model(), Sink()
    view = SimpleNamespace(
        model=model,
        layers=model.layers,
        num_layers=2,
        hidden_size=3,
        _unwrap_cache=lambda value: value,
    )
    return view, sink, torch.tensor([[1, 2, 3]])


def test_upstream_recorder_and_native_sink_preserve_graph_and_block_convention():
    view, sink, ids = fixture()
    assert issubclass(TorchCapture, ActivationRecorder)
    expected = view.model(ids).logits
    with TorchCapture(view, sink, layers=(0, 1, 2)) as capture:
        actual = capture(ids).logits
        assert torch.equal(actual, expected)
        assert torch.equal(sink.rows[0][1] * 2, sink.rows[1][1])
        assert torch.equal(sink.rows[1][1] * 3, sink.rows[2][1])
        assert capture.activations[1] is sink.rows[2][1]
        assert capture.layer_inputs[1]["attention_mask"] == "mask-1"
        assert capture.layer_inputs[1]["position_embeddings"] == (1, 2)
        assert sink.rows[0][1].dtype == actual.dtype == torch.float32
        assert sink.outputs == [(0, [1, 2, 3], actual)]
        actual.sum().backward()
        assert view.model.embed.weight.grad is not None
    assert not capture.activations
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in view.layers)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_replacement_intervention_reaches_downstream_and_only_selected_position(layer):
    view, sink, ids = fixture()
    expected = view.model(ids).logits
    donor = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
    with TorchCapture(view, sink, layers=(0, 1, 2)) as capture:
        capture.intervene(layer, 1, lambda h: donor)
        actual = capture(ids).logits
        multiplier = {0: 6, 1: 3, 2: 1}[layer]
        assert torch.equal(actual[0, 1], donor * multiplier)
        assert torch.equal(actual[0, (0, 2), :], expected[0, (0, 2), :])
        actual.sum().backward()
        assert torch.equal(donor.grad, torch.full_like(donor, multiplier))


def test_injection_compatibility_and_absolute_cache_positions():
    view, sink, ids = fixture()
    cache = SimpleNamespace(get_seq_length=lambda: 5)
    with TorchCapture(view, sink, layers=(1, 2), injection=(1, 6, [1.0, 2.0, 3.0])) as capture:
        actual = capture(ids, cache=cache).logits
        assert torch.equal(
            actual[0, 1], (view.model.embed(ids)[0, 1] * 2 + torch.tensor([1.0, 2.0, 3.0])) * 3
        )
        assert sink.rows[1][0] == sink.outputs[0][0] == 5
    with (
        TorchCapture(view, sink, layers=(1,), injection=(1, 2, [0.0, 0.0, 0.0])) as capture,
        pytest.raises(ValueError, match="already cached"),
    ):
        capture(ids, cache=cache)


def test_direct_model_call_is_observed_and_hooks_removed_after_failure():
    view, sink, ids = fixture()
    capture = TorchCapture(view, sink, layers=(2,))
    with pytest.raises(RuntimeError, match="deliberate"), capture:
        view.model(input_ids=ids)
        view.model.fail = True
        view.model(input_ids=ids)
    assert len(sink.outputs) == 1
    assert not capture.activations and not capture.layer_inputs
    assert not view.model._forward_hooks and not view.model._forward_pre_hooks
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in view.layers)


@pytest.mark.parametrize("fn", [lambda h: h[:1], lambda h: h.double(), lambda h: None])
def test_intervention_refuses_shape_or_dtype_change(fn):
    view, sink, ids = fixture()
    with TorchCapture(view, sink, layers=(1,)) as capture:
        capture.intervene(1, 0, fn)
        with pytest.raises(ValueError, match="replacement"):
            capture(ids)


def test_unsupported_heads_and_invalid_indices_fail_before_hooks():
    view, sink, ids = fixture()
    with pytest.raises(NotImplementedError, match="head"):
        TorchCapture(view, sink, layers=(1,), attention_blocks=(0,))
    with pytest.raises(ValueError, match="layers"):
        TorchCapture(view, sink, layers=(True,))
    with TorchCapture(view, sink, layers=(1,)) as capture:
        with pytest.raises(ValueError, match="intervention"):
            capture.intervene(3, 0, lambda h: h)
        with pytest.raises(ValueError, match="batch"):
            capture(ids.expand(2, -1))


def test_upstream_index_resolution_matches_sink_residual_indices():
    from jlens.fitting import _check_layer_indices

    view, sink, ids = fixture()
    source_blocks, target_block = _check_layer_indices(None, None, view.num_layers)
    layers = tuple(block + 1 for block in (*source_blocks, target_block))
    with TorchCapture(view, sink, layers=layers) as capture:
        capture(ids)
        for block in (*source_blocks, target_block):
            assert capture.activations[block] is sink.rows[block + 1][1]


def test_interchange_can_be_registered_after_entry_at_an_unrequested_layer():
    view, sink, ids = fixture()
    frozen_support = torch.tensor([1.0, 0.0, 1.0])
    donor = torch.tensor([5.0, 6.0, 7.0])
    expected = view.model(ids).logits
    with TorchCapture(view, sink, layers=(2,)) as capture:
        capture.intervene(1, 0, lambda h: h + frozen_support * (donor - h))
        result = capture(ids).logits
        assert result[0, 0, 0] == donor[0] * 3
        assert result[0, 0, 2] == donor[2] * 3
        assert result[0, 0, 1] == expected[0, 0, 1]
        assert set(sink.rows) == {2}
        assert "hidden_states" not in capture.layer_inputs[0]


def test_cache_decode_does_not_apply_intervention_twice_and_fresh_prefill_resets():
    view, sink, ids = fixture()
    calls = []
    offset = [0]
    cache = SimpleNamespace(get_seq_length=lambda: offset[0])
    with TorchCapture(view, sink, layers=(2,)) as capture:
        capture.intervene(1, 1, lambda h: calls.append(True) or h)
        capture(ids, cache=cache)
        offset[0] = 3
        capture(ids[:, :1], cache=cache)
        assert calls == [True]
        offset[0] = 0
        capture(ids, cache=cache)
        assert calls == [True, True]


def test_partial_hook_installation_failure_cleans_up(monkeypatch):
    view, sink, _ = fixture()
    capture = TorchCapture(view, sink, layers=(1, 2))

    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(view.layers[1], "register_forward_pre_hook", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        capture.__enter__()
    assert not capture._active
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in view.layers)


def test_capture_reentry_uses_fresh_hooks_and_refuses_call_outside_context():
    view, sink, ids = fixture()
    capture = TorchCapture(view, sink, layers=(2,))
    with pytest.raises(RuntimeError, match="entered"):
        capture(ids)
    for _ in range(2):
        with capture:
            capture(ids)
            with pytest.raises(RuntimeError, match="nested"):
                capture.__enter__()
    assert len(sink.outputs) == 2


def test_direct_model_positional_cache_reports_absolute_offset():
    view, sink, ids = fixture()
    cache = SimpleNamespace(get_seq_length=lambda: 8)
    with TorchCapture(view, sink, layers=(2,)):
        view.model(ids, cache)
    assert sink.rows[2][0] == sink.outputs[0][0] == 8


def test_wrapper_declares_no_cache_and_preserves_explicit_cache_choice():
    view, sink, ids = fixture()
    choices = []
    handle = view.model.register_forward_pre_hook(
        lambda module, args, kwargs: choices.append(kwargs.get("use_cache")), with_kwargs=True
    )
    try:
        with TorchCapture(view, sink, layers=(2,)) as capture:
            capture(ids)
            capture(ids, past_key_values=SimpleNamespace(get_seq_length=lambda: 0))
            capture(ids, use_cache=False, past_key_values=SimpleNamespace(get_seq_length=lambda: 0))
        assert choices == [False, True, False]
    finally:
        handle.remove()


def test_block_kwargs_are_flat_and_exclude_positional_hidden_alias():
    view, sink, ids = fixture()

    class ExtraKwargsBlock(nn.Module):
        def forward(self, h, **kwargs):
            return h * 2

    view.model.layers[0] = ExtraKwargsBlock()
    with TorchCapture(view, sink, layers=(0, 2)) as capture:
        capture(ids)
        assert capture.layer_inputs[0] == {
            "attention_mask": "mask-0",
            "position_embeddings": (0, 1),
            "past_key_values": None,
        }
