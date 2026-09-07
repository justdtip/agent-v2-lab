"""Native arithmetic tests use a tiny installed Qwen3.5 model, never a checkpoint.

These are implementation tests, not the research checkpoint's acceptance gate.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from local_llm_lab.arch import ArchitectureView, NativeCapture


def tiny_model():
    mx.random.seed(37)
    # The installed Metal recurrence tiles keys in groups of 32; smaller keys are invalid.
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
    )
    model = TextModel(args)
    model.eval()
    return model


class Sink:
    def __init__(self):
        self.residuals = []
        self.heads = []
        self.logits = []

    def residual(self, layer, offset, h):
        self.residuals.append((layer, offset, np.array(h)))

    def attention(self, block, target, weights, written, total):
        self.heads.append((block, target, np.array(weights), np.array(written), np.array(total)))

    def output(self, offset, ids, logits):
        self.logits.append((offset, list(ids), np.array(logits)))


def test_native_observer_preserves_uncached_and_cached_logits_and_restores_modules():
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    originals = list(model.layers)
    ids = mx.array([[1, 2, 3, 4]])
    native_cache = model.make_cache()
    native = [
        np.array(model(ids, cache=native_cache)),
        np.array(model(mx.array([[5]]), cache=native_cache)),
    ]
    sink = Sink()
    with NativeCapture(view, sink, layers=(3, 4), attention_blocks=(3,)) as captured:
        cache = model.make_cache()
        got = [
            np.array(captured(ids, cache=cache)),
            np.array(captured(mx.array([[5]]), cache=cache)),
        ]
        for a, b in zip(native, got, strict=True):
            np.testing.assert_array_equal(a, b)
    assert all(a is b for a, b in zip(originals, model.layers, strict=True))
    assert [entry[0] for entry in sink.logits] == [0, 4]
    for _, target, weights, written, total in sink.heads:
        assert weights.shape == (2, target + 1)
        np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-6)
        np.testing.assert_allclose(written.sum(0), total, atol=1e-6)
    assert {layer for layer, _, _ in sink.residuals} == {3, 4}


def test_zero_injection_is_exact_and_nonzero_source_changes_later_positions():
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    ids = mx.array([[1, 2, 3, 4]])
    baseline = np.array(model(ids))
    with NativeCapture(
        view, Sink(), layers=(2, 4), injection=(2, 1, np.zeros(64, dtype=np.float32))
    ) as wrapped:
        np.testing.assert_array_equal(np.array(wrapped(ids)), baseline)
    direction = np.arange(64, dtype=np.float32) / 64
    with NativeCapture(view, Sink(), layers=(2, 4), injection=(2, 1, direction)) as wrapped:
        changed = np.array(wrapped(ids))
    np.testing.assert_array_equal(changed[:, 0], baseline[:, 0])
    assert not np.array_equal(changed[:, -1], baseline[:, -1])


def test_native_wrappers_restore_even_when_a_sink_raises():
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    blocks = list(model.layers)
    sink = Sink()

    def fail(*args):
        raise RuntimeError("capture writer failed")

    sink.residual = fail
    with (
        pytest.raises(RuntimeError, match="capture writer failed"),
        NativeCapture(view, sink, layers=(3,)) as wrapped,
    ):
        wrapped(mx.array([[1, 2]]))
    assert all(a is b for a, b in zip(blocks, model.layers, strict=True))


def test_session_handles_lookahead_and_replays_recorded_forward_chunks(tmp_path):
    from types import SimpleNamespace

    from local_llm_lab.pipeline.live_lens.instruments import LensMaps
    from local_llm_lab.pipeline.live_lens.session import CaptureSession, LensReadout, replay

    model = tiny_model()
    view = ArchitectureView.from_model(model)
    reader = LensReadout(view, LensMaps({3: np.eye(64)}, "fixture", 64, 4))
    rows = []
    session = CaptureSession(
        view, reader, rows.append, layers=(3, 4), attention_blocks=(3,), top_k=3, audit_modulus=1
    )
    tokenizer = SimpleNamespace(bos_token=None, encode=lambda text, **kw: [1, 2, 3])
    with session.generation(model, tokenizer, "fixture", turn_cache=None) as captured:
        cache = model.make_cache()
        first = captured(mx.array([[1, 2, 3]]), cache=cache)
        native_next = int(mx.argmax(first[0, -1]).item())
        captured(mx.array([[native_next]]), cache=cache)  # stream_generate lookahead
        session.emitted(native_next)
    rank_rows = [r for r in rows if r["kind"] == "rank"]
    assert [(r["position"], r["horizon"]) for r in rank_rows] == [(2, 1), (2, 1)]
    emitted = [r for r in rows if r["kind"] == "emitted"]
    assert emitted[0]["position"] == 3
    assert emitted[0]["token_id"] == native_next
    assert all(r["rank"] == 1 for r in rank_rows if r["layer"] == 4)
    assert replay(view, rows, atol=0.0, rtol=0.0)["max_abs_error"] == 0.0
    for row in (r for r in rows if r["kind"] == "head" and "audit" in r):
        from local_llm_lab.pipeline.live_lens.records import transport_overlap

        audit = row["audit"]
        recomputed = transport_overlap(
            row["written_top"],
            audit["source_top"],
            audit["attention"],
            source_positions=audit["positions"],
            target_position=row["position"],
        )
        assert recomputed["score"] == row["score"]
    with session.generation(model, tokenizer, "fixture", turn_cache=None):
        assert not session.ranks.buffer


def test_final_layer_lens_readout_preserves_native_precision():
    from local_llm_lab.pipeline.live_lens.instruments import LensMaps
    from local_llm_lab.pipeline.live_lens.session import LensReadout

    for dtype in (mx.float32, mx.bfloat16):
        model = tiny_model()
        model.set_dtype(dtype)
        view = ArchitectureView.from_model(model)
        sink = Sink()
        # Keep the native dtype rather than converting through NumPy (which lacks bf16).
        values = {}
        sink.residual = lambda layer, offset, h, values=values: values.update({layer: h})
        sink.output = lambda *args: None
        reader = LensReadout(view, LensMaps({}, "identity", 64, 4))
        with NativeCapture(view, sink, layers=(4,)) as wrapped:
            logits = wrapped(mx.array([[1, 2, 3]]))
        # Preserve batch shape too: matrix-vector and matrix-matrix reductions can round
        # differently even when norm, unembedding, and residual dtype are identical.
        np.testing.assert_array_equal(
            np.array(reader.logits(values[4], 4)), np.array(logits.astype(mx.float32))
        )


def test_two_capture_owners_cannot_wrap_the_same_model():
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    with (
        NativeCapture(view, Sink(), layers=(4,)),
        pytest.raises(RuntimeError, match="nested"),
        NativeCapture(view, Sink(), layers=(4,)),
    ):
        pass
    with NativeCapture(view, Sink(), layers=(4,)) as wrapped:
        wrapped(mx.array([[1]]))


def test_injection_rejects_an_already_cached_source_and_accepts_rebuilt_state():
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache=cache)
    with NativeCapture(view, Sink(), layers=(4,), injection=(2, 1, np.zeros(64))) as wrapped:
        with pytest.raises(ValueError, match="fresh prefill"):
            wrapped(mx.array([[4]]), cache=cache)
        # Rebuild both recurrent and attention states; matching zero intervention is exact.
        rebuilt = model.make_cache()
        actual = np.array(wrapped(mx.array([[1, 2, 3]]), cache=rebuilt))
    expected = np.array(model(mx.array([[1, 2, 3]]), cache=model.make_cache()))
    np.testing.assert_array_equal(actual, expected)


def test_attention_only_mode_skips_head_projection_but_preserves_native_logits():
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    ids = mx.array([[1, 2, 3]])
    expected = np.array(model(ids))
    sink = Sink()
    rows = []
    sink.attention = lambda block, target, weights, written, total: rows.append(written)
    with NativeCapture(view, sink, layers=(3, 4), attention_blocks=(3,),
                       head_vectors=False) as wrapped:
        np.testing.assert_array_equal(np.array(wrapped(ids)), expected)
    assert rows == [None]
