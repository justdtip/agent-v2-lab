"""Real tiny-model replay integration; no checkpoint, and requires the free MLX slot.

Requirements 3.4/7: source, ledger and forward partitions survive a changed lens.
These tests do not establish real-checkpoint or hosted-pilot acceptance.
"""

import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_live_lens_native import tiny_model

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.pipeline.lens_fitting.replay import read_source, replay_record
from local_llm_lab.pipeline.live_lens.instruments import LensMaps
from local_llm_lab.pipeline.live_lens.session import CaptureSession, LensReadout, RecordWriter


def captured_fixture(tmp_path):
    """Two different prompts, split prefill and native lookahead, over hybrid state."""
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    layers = tuple(range(1, view.num_layers + 1))
    lens = LensMaps(
        {layer: np.eye(view.hidden_size, dtype=np.float32) for layer in layers[:-1]},
        "tiny-identity-fixture",
        view.hidden_size,
        view.num_layers,
    )
    prompts = {"first": [1, 2, 3], "second": [2, 3, 4, 5]}
    tokenizer = SimpleNamespace(bos_token=None, encode=lambda text, **kwargs: prompts[text])
    source = tmp_path / "source.jsonl"
    with RecordWriter(source, {"model": "tiny", "lens_sha256": lens.sha256}) as writer:
        session = CaptureSession(view, LensReadout(view, lens), writer, layers=layers, top_k=3)
        for prompt, ids in prompts.items():
            session.set_context(kind="chat", messages=[{"role": "user", "content": prompt}])
            with session.generation(model, tokenizer, prompt, turn_cache=None) as captured:
                cache = view.make_cache()
                captured(mx.array([ids[:2]]), cache=cache)
                logits = captured(mx.array([ids[2:]]), cache=cache)
                for _ in range(9):
                    token = int(mx.argmax(logits[0, -1]).item())
                    # Preserve the generator's forward-before-emission ordering.
                    logits = captured(mx.array([[token]]), cache=cache)
                    session.emitted(token)
    events, _ = read_source(source)
    return view, tokenizer, lens, layers, events


@pytest.mark.parametrize("changed_lens", [False, True])
def test_native_replay_preserves_two_turns_and_partitions_under_new_lens(tmp_path, changed_lens):
    """§3.4: actual native controls stay exact; only requested lens readings change."""
    view, tokenizer, lens, layers, events = captured_fixture(tmp_path)
    if changed_lens:
        lens = LensMaps(
            {layer: -matrix for layer, matrix in lens.maps.items()},
            "tiny-negative-fixture",
            view.hidden_size,
            view.num_layers,
        )
    originals = list(view.model.layers)
    target = tmp_path / "replayed.jsonl"
    with RecordWriter(target, {"model": "tiny", "lens_sha256": lens.sha256}) as writer:
        result = replay_record(view, tokenizer, lens, events, writer, layers=layers)
    actual, _ = read_source(target)
    control_kinds = {"forward", "emitted", "end_turn"}
    assert [r for r in actual if r["kind"] in control_kinds] == [
        r for r in events if r["kind"] in control_kinds
    ]
    assert sum(r["kind"] == "begin_turn" for r in actual) == 2
    assert result["forwards"] == sum(r["kind"] == "forward" for r in events)
    assert all(a is b for a, b in zip(originals, view.model.layers, strict=True))
    expected_readings = [r for r in events if r["kind"] == "reading"]
    actual_readings = [r for r in actual if r["kind"] == "reading"]
    if changed_lens:
        assert expected_readings != actual_readings
    else:
        assert expected_readings == actual_readings
    assert {r["horizon"] for r in actual if r["kind"] == "rank"} == {1, 4, 8}


def test_native_hash_failure_preserves_first_cause_and_restores_wrappers(tmp_path):
    """§3.4: an exact forward-hash failure remains the diagnostic and aborts output."""
    view, tokenizer, lens, layers, events = captured_fixture(tmp_path)
    bad = copy.deepcopy(events)
    next(r for r in bad if r["kind"] == "forward")["logits_sha256"] = "0" * 64
    originals = list(view.model.layers)
    target = tmp_path / "failed.jsonl"
    with (
        pytest.raises(ValueError, match=r"forward\.logits_sha256"),
        RecordWriter(target, {"model": "tiny"}) as writer,
    ):
        replay_record(view, tokenizer, lens, bad, writer, layers=layers)
    assert all(a is b for a, b in zip(originals, view.model.layers, strict=True))
    with pytest.raises(ValueError, match="incomplete"):
        read_source(target)


def test_the_final_layer_readout_is_run_and_checked_against_the_model_s_own_logits(tmp_path):
    """The check the layer-34 rank-1 rate was mistaken for, and it is a different check.

    `output` reports the model's own softmax at the final layer, which is right: a reader of a
    final-layer row should be able to assume it is the model's distribution. What was wrong was
    that the readout was *only* substituted there, so a 100% rank-1 rate at the final layer said
    that decoding is greedy and positions line up, and said nothing about the residual tap or the
    unembedding — while a record and a published page both described it as the instrument proving
    itself.

    Run for real, the identity branch of `LensReadout.logits` exercises the tap and
    `view.native_readout`, so agreement with the native logits covers everything the readout does
    except the fitted maps. Every forward records its own figure so the gate can be tightened onto
    observed behaviour rather than guessed.
    """
    view, _, _, _, events = captured_fixture(tmp_path)
    errors = [
        event["final_readout_max_abs_error"]
        for event in events
        if event.get("kind") == "forward"
    ]
    assert errors, "every forward records the final-layer agreement"
    assert all(error is not None for error in errors), (
        "and records it as a number, because a missing figure reads as a passing one"
    )
    assert max(errors) < 1e-3, (
        f"the identity branch should reproduce the model's own logits; worst was {max(errors):.3g}"
    )


def test_a_readout_that_disagrees_at_the_final_layer_stops_the_run(tmp_path):
    """The negative control: the check must be able to fail, on the defect it exists to catch.

    A tap at the wrong position or an unembedding applied without its norm moves logits by whole
    units. This breaks the identity branch by exactly that much and asserts the run stops, because
    a gate that cannot fail is not a gate — the lesson of the residual comparator that compared a
    broken loop against itself.
    """
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    layers = tuple(range(1, view.num_layers + 1))
    lens = LensMaps(
        {layer: np.eye(view.hidden_size, dtype=np.float32) for layer in layers[:-1]},
        "tiny-identity-fixture",
        view.hidden_size,
        view.num_layers,
    )

    class Skewed(LensReadout):
        """Correct everywhere except the one branch the check covers."""

        def logits(self, residual, layer):
            value = super().logits(residual, layer)
            return value + 5.0 if layer == self.view.num_layers else value

    tokenizer = SimpleNamespace(bos_token=None, encode=lambda text, **kwargs: [1, 2, 3])
    session = CaptureSession(view, Skewed(view, lens), lambda row: None, layers=layers, top_k=3)
    session.set_context(kind="chat", messages=[{"role": "user", "content": "first"}])
    with (
        pytest.raises(ValueError, match="disagrees with the model's own logits"),
        session.generation(model, tokenizer, "first", turn_cache=None) as captured,
    ):
        captured(mx.array([[1, 2, 3]]), cache=view.make_cache())


def test_every_emitted_token_carries_its_span_label_at_write_time(tmp_path):
    """The pre-registration's facet is a property of the data, not of a later analysis.

    A segmentation computed after a record is read is a boundary chosen after seeing the answer,
    which is the thing a pre-registration exists to prevent. So the label is written with the
    token.
    """
    _, _, _, _, events = captured_fixture(tmp_path)
    emitted = [event for event in events if event.get("kind") == "emitted"]
    assert emitted, "the fixture generates tokens"
    assert all("span" in event for event in emitted), "every emitted token is labelled"
    assert {event["span"] for event in emitted} <= {
        "note",
        "call_skeleton",
        "call_argument",
        "chat_prose",
    }


def test_the_span_scanner_puts_the_fused_quote_and_slash_in_the_argument() -> None:
    """The one token the `update-0028` finding turns on, and the boundary rule it forces.

    Gemma emits `' "/'` as a single token: a space, the quote that opens the path, and the path's
    first character. Assigning a token by its first character would file that whole token as
    skeleton and take the finding's own decision point out of the argument facet. A token any part
    of which carries argument content is an argument token.

    The closing `'"}}'` is the other side of the same rule: it begins inside the value and carries
    none of it, so it is skeleton.
    """
    from local_llm_lab.pipeline.live_lens.session import SpanLabeller

    labeller = SpanLabeller("agentic")
    pieces = [
        "Progress", " note", ":", " ok", "\n", "```", "json", "\n",
        '{"', "name", '":', ' "', "read", "_", "file", '",',
        ' "', "arguments", '":', ' {"', "path", '":',
        ' "/', "test", "/", "config", ".", "ini", '"}}',
    ]
    labels = [labeller.feed(piece) for piece in pieces]
    by_piece = dict(zip(pieces, labels, strict=True))

    assert by_piece["Progress"] == "note"
    assert by_piece["```"] == "call_skeleton", "the fence is convention, not deliberation"
    assert by_piece["file"] == "call_skeleton", "the tool name is fixed by the calling convention"
    assert by_piece[' "/'] == "call_argument"
    assert by_piece["config"] == "call_argument"
    assert by_piece['"}}'] == "call_skeleton", "the terminator carries none of the value"


def test_a_chat_episode_labels_every_token_prose_and_malformed_output_still_labels() -> None:
    """Two ends the scanner must not fall off.

    A chat turn has no calls, and this model produces malformed JSON often enough that a scanner
    which raised on it would silently shrink one facet of the primary comparison.
    """
    from local_llm_lab.pipeline.live_lens.session import SpanLabeller

    chat = SpanLabeller("chat")
    assert {chat.feed(piece) for piece in ("Hello", " there", '{"', '"')} == {"chat_prose"}

    broken = SpanLabeller("agentic")
    labels = [broken.feed(piece) for piece in ("note", "```", 'json {"path": "a', "\\", '"')]
    assert len(labels) == 5, "an unterminated string does not stop the labelling"
    assert "call_argument" in labels


def test_the_final_layer_gate_fires_at_decode_and_not_only_at_prefill(tmp_path):
    """Where 98% of the gate's observations come from, it must still be able to fail.

    Stage two records `final_readout_max_abs_error` on every forward: exactly 0.0 at all 608
    decode steps and 0.4375 to 0.75 at the 13 prefills. Bit-identical agreement at decode is the
    strongest validation of the tap this programme has, *provided the comparison is real there* —
    and the existing negative control only ever drove a single prefill call, so it proved the gate
    can fail at prefill and said nothing about the branch supplying almost all the evidence.

    The Chief asked the question and it is the right one: a clean number is a reason for suspicion
    proportional to how much you wanted it. So the skew here is withheld until prefill is over and
    applied only to decode rows, which fails if and only if the gate runs at decode.
    """
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    layers = (view.num_layers,)
    lens = LensMaps(
        {layer: np.eye(view.hidden_size, dtype=np.float32) for layer in range(1, view.num_layers)},
        "tiny-identity-fixture",
        view.hidden_size,
        view.num_layers,
    )

    class SkewedAfterPrefill(LensReadout):
        """Correct for the first `spare` final-layer reads, wrong afterwards.

        The prefill's rows are consumed first, so the skew lands on the decode step and nowhere
        else. If the gate only compared at prefill this readout would never be caught.
        """

        def __init__(self, view, lens, spare):
            super().__init__(view, lens)
            self.remaining = spare

        def logits(self, residual, layer):
            value = super().logits(residual, layer)
            if layer != self.view.num_layers:
                return value
            if self.remaining > 0:
                self.remaining -= 1
                return value
            return value + 5.0

    prompt = [1, 2, 3]
    tokenizer = SimpleNamespace(bos_token=None, encode=lambda text, **kwargs: prompt)
    readout = SkewedAfterPrefill(view, lens, spare=len(prompt))
    session = CaptureSession(view, readout, lambda row: None, layers=layers, top_k=3)
    session.set_context(kind="chat", messages=[{"role": "user", "content": "first"}])

    with (
        pytest.raises(ValueError, match="disagrees with the model's own logits"),
        session.generation(model, tokenizer, "first", turn_cache=None) as captured,
    ):
        cache = view.make_cache()
        logits = captured(mx.array([prompt]), cache=cache)
        assert readout.remaining == 0, "the prefill consumed exactly the unskewed reads"
        token = int(mx.argmax(logits[0, -1]).item())
        captured(mx.array([[token]]), cache=cache)


def test_the_final_layer_comparison_uses_the_captured_residual_and_not_the_logits_it_checks(
    tmp_path,
):
    """The other half of the same worry: 0.0 could mean agreement or mean comparing a thing to itself.

    If the gate compared the model's logits against a value derived from those same logits, exact
    agreement would be trivially true and the gate would have no power anywhere. It does not: the
    checked value comes from the captured layer-34 residual through the readout's identity branch,
    which is a different array reached by a different computation.

    Asserted by making the *residual* wrong while leaving the readout correct. A tap on the wrong
    tensor is the defect this is really guarding against, and it must be caught.
    """
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    layers = (view.num_layers,)
    lens = LensMaps(
        {layer: np.eye(view.hidden_size, dtype=np.float32) for layer in range(1, view.num_layers)},
        "tiny-identity-fixture",
        view.hidden_size,
        view.num_layers,
    )
    tokenizer = SimpleNamespace(bos_token=None, encode=lambda text, **kwargs: [1, 2, 3])
    session = CaptureSession(view, LensReadout(view, lens), lambda row: None, layers=layers, top_k=3)
    session.set_context(kind="chat", messages=[{"role": "user", "content": "first"}])

    with (
        pytest.raises(ValueError, match="disagrees with the model's own logits"),
        session.generation(model, tokenizer, "first", turn_cache=None) as captured,
    ):
        original = session.residual

        def corrupted(layer, offset, hidden):
            return original(layer, offset, hidden + 3.0)

        session.residual = corrupted
        captured(mx.array([[1, 2, 3]]), cache=view.make_cache())
