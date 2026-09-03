from __future__ import annotations

import numpy as np

from local_llm_lab.probes import adapter_delta


def test_adapter_direction_readouts_derive_negative_direction_by_jvp_linearity(monkeypatch) -> None:
    calls: list[float] = []

    info = {"type": "down_proj", "layer": 0, "module": "layers.0.down_proj"}
    monkeypatch.setattr(
        adapter_delta,
        "load_adapter_deltas",
        lambda _path: {"layers.0.down_proj": (None, info)},
    )
    monkeypatch.setattr(
        adapter_delta,
        "left_singular_vectors",
        lambda _info, k: np.ones((2, k), dtype=np.float32),
    )
    monkeypatch.setattr(
        adapter_delta,
        "spectrum",
        lambda _info, top: np.ones((top,), dtype=np.float32),
    )
    monkeypatch.setattr(adapter_delta, "_as_mx", lambda vector: vector)

    class JLens:
        DEFAULT_CORPUS = ("a",)

        @staticmethod
        def encode(_tokenizer, _text):
            return [1]

        @staticmethod
        def jlens_map(_view, _layer, probe, _corpus):
            calls.append(float(probe[0]))
            return probe, {"method": "forward"}

        @staticmethod
        def logit_lens(*_args, **_kwargs):
            return []

        readout = logit_lens

    monkeypatch.setattr(adapter_delta, "_jlens_module", lambda: JLens)
    records = adapter_delta.readout_update_directions(object(), object(), "adapter", [0])

    assert len(records) == 2  # +v and -v need no duplicate corpus JVP.
    assert calls == [1.0, 1.0]


def test_adapter_direction_readouts_default_to_parsed_adapter_module_types(monkeypatch) -> None:
    calls: list[float] = []
    down = {"type": "down_proj", "layer": 0, "module": "layers.0.down_proj"}
    up = {"type": "up_proj", "layer": 0, "module": "layers.0.up_proj"}
    monkeypatch.setattr(
        adapter_delta,
        "load_adapter_deltas",
        lambda _path: {"layers.0.down_proj": (None, down), "layers.0.up_proj": (None, up)},
    )
    monkeypatch.setattr(
        adapter_delta,
        "left_singular_vectors",
        lambda _info, k: np.ones((2, k), dtype=np.float32),
    )
    monkeypatch.setattr(
        adapter_delta, "spectrum", lambda _info, top: np.ones((top,), dtype=np.float32))
    monkeypatch.setattr(adapter_delta, "_as_mx", lambda vector: vector)

    class JLens:
        DEFAULT_CORPUS = ("a",)

        @staticmethod
        def encode(_tokenizer, _text):
            return [1]

        @staticmethod
        def jlens_map(_view, _layer, probe, _corpus):
            calls.append(float(probe[0]))
            return probe, {"method": "forward"}

        @staticmethod
        def logit_lens(*_args, **_kwargs):
            return []

        readout = logit_lens

    monkeypatch.setattr(adapter_delta, "_jlens_module", lambda: JLens)
    records = adapter_delta.readout_update_directions(
        object(), object(), "adapter", [0], directions=1
    )

    assert [record["type"] for record in records] == ["down_proj", "up_proj"]
    assert calls == [1.0, 1.0]

    calls.clear()
    filtered = adapter_delta.readout_update_directions(
        object(), object(), "adapter", [0], types=("up_proj",), directions=1
    )

    assert [record["type"] for record in filtered] == ["up_proj"]
    assert calls == [1.0]
