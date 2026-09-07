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
