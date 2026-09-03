from __future__ import annotations

import numpy as np

from local_llm_lab.probes import adapter_delta


def test_adapter_direction_readouts_derive_negative_direction_by_jvp_linearity(monkeypatch) -> None:
    calls: list[float] = []

    info = {
        "type": "down_proj",
        "layer": 0,
        "module": "layers.0.down_proj",
        "shape": [2, 8],
    }
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
    view = type("View", (), {"hidden_size": 2})()
    records = adapter_delta.readout_update_directions(view, object(), "adapter", [0])

    assert len(records) == 2  # +v and -v need no duplicate corpus JVP.
    assert calls == [1.0, 1.0]


def test_adapter_direction_readouts_default_to_residual_sized_adapter_outputs(monkeypatch) -> None:
    calls: list[float] = []
    down = {
        "type": "residual_update",
        "layer": 0,
        "module": "layers.0.reducer",
        "shape": [2, 8],
    }
    same_type_expansion = {
        "type": "residual_update",
        "layer": 0,
        "module": "layers.1.reducer",
        "shape": [8, 2],
    }
    expansion = {
        "type": "expansion",
        "layer": 0,
        "module": "layers.0.expander",
        "shape": [8, 2],
    }
    square = {
        "type": "square_residual",
        "layer": 0,
        "module": "layers.0.square",
        "shape": [2, 2],
    }
    monkeypatch.setattr(
        adapter_delta,
        "load_adapter_deltas",
        lambda _path: {
            "layers.0.expander": (None, expansion),
            "layers.0.reducer": (None, down),
            "layers.0.square": (None, square),
            "layers.1.reducer": (None, same_type_expansion),
        },
    )
    monkeypatch.setattr(
        adapter_delta,
        "left_singular_vectors",
        lambda info, k: np.ones((info["shape"][0], k), dtype=np.float32),
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
    view = type("View", (), {"hidden_size": 2})()
    records = adapter_delta.readout_update_directions(
        view, object(), "adapter", [0], directions=1
    )

    assert [record["module"] for record in records] == ["layers.0.reducer"]
    assert calls == [1.0]

    calls.clear()
    incompatible = adapter_delta.readout_update_directions(
        view, object(), "adapter", [0], types=("expansion",), directions=1
    )

    assert incompatible == []
    assert calls == []

    square_records = adapter_delta.readout_update_directions(
        view, object(), "adapter", [0], types=("square_residual",), directions=1
    )

    assert [record["module"] for record in square_records] == ["layers.0.square"]
    assert calls == [1.0]
