from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_llm_lab.pipeline import evaluate
from local_llm_lab.probes import adapter_delta, capture


def test_adapter_direction_readouts_derive_negative_direction_by_jvp_linearity(monkeypatch) -> None:
    calls: list[float] = []
    j_lens_inputs: list[np.ndarray] = []

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

        @staticmethod
        def readout(_view, vector, _tokenizer, **_kwargs):
            j_lens_inputs.append(np.asarray(vector))
            return []

    monkeypatch.setattr(adapter_delta, "_jlens_module", lambda: JLens)
    view = type("View", (), {"hidden_size": 2})()
    records = adapter_delta.readout_update_directions(view, object(), "adapter", [0])

    assert len(records) == 2  # +v and -v need no duplicate corpus JVP.
    assert calls == [1.0, 1.0]
    np.testing.assert_allclose(j_lens_inputs[1], -j_lens_inputs[0])


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
        "type": "residual_update",
        "layer": 1,
        "module": "layers.1.square",
        "shape": [2, 2],
    }
    monkeypatch.setattr(
        adapter_delta,
        "load_adapter_deltas",
        lambda _path: {
            "layers.0.expander": (None, expansion),
            "layers.0.reducer": (None, down),
            "layers.1.square": (None, square),
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
        view, object(), "adapter", [0, 1], directions=1
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
        view, object(), "adapter", [1], types=("residual_update",), directions=1
    )

    assert [record["module"] for record in square_records] == ["layers.1.square"]
    assert calls == [1.0]


def test_layer_blocks_balances_the_remainder_without_dropping_a_layer() -> None:
    """Catches floor-sized blocks that omit or duplicate the non-divisible tail."""
    assert adapter_delta._layer_blocks(7, 3) == [(0, 1, 2), (3, 4), (5, 6)]

    with pytest.raises(ValueError, match="between 1 and num_layers"):
        adapter_delta._layer_blocks(7, 0)
    with pytest.raises(ValueError, match="between 1 and num_layers"):
        adapter_delta._layer_blocks(7, 8)


def test_block_ablation_uses_leave_one_out_masks_and_exact_screen_counts(monkeypatch) -> None:
    """Catches block-only masks, per-cell rate averaging, and unstable tie selection."""
    view = type("View", (), {"num_layers": 7})()
    active_keep: list[tuple[int, ...]] = []
    mask_entries: list[tuple[int, ...]] = []
    mask_exits: list[tuple[int, ...]] = []

    @contextmanager
    def fake_mask(received_view, keep_layers):
        assert received_view is view
        keep = tuple(keep_layers)
        active_keep.append(keep)
        mask_entries.append(keep)
        try:
            yield 7 - len(keep)
        finally:
            mask_exits.append(active_keep.pop())

    monkeypatch.setattr(capture, "lora_block_mask", fake_mask)
    screen = [
        {"split": "valid", "difficulty": 1, "per_family": {"default": 1}},
        {"split": "valid2", "difficulty": 2, "per_family": {"default": 1}},
    ]

    def cell(overall: int, ledger: int, batch: int) -> dict[str, Any]:
        return {
            "successes": overall,
            "tasks": 4,
            "by_family": {
                "ledger_reconcile": {"successes": ledger, "tasks": 2},
                "batch_update": {"successes": batch, "tasks": 2},
            },
        }

    summaries = {
        "full_adapter": [cell(3, 2, 1), cell(3, 1, 2)],
        "empty_adapter": [cell(1, 1, 0), cell(1, 0, 1)],
        "remove_block_0": [cell(2, 1, 1), cell(2, 1, 1)],
        "remove_block_1": [cell(3, 2, 1), cell(3, 1, 2)],
        "remove_block_2": [cell(2, 1, 1), cell(2, 1, 1)],
    }
    calls: list[tuple[str, int, str, tuple[int, ...]]] = []

    def fake_evaluate(name: str, cell_index: int, cell_spec: dict[str, Any]):
        calls.append((name, cell_index, cell_spec["split"], active_keep[0]))
        return summaries[name][cell_index]

    result = adapter_delta.run_block_ablation(
        view,
        blocks=3,
        screen=screen,
        evaluate_condition=fake_evaluate,
    )

    assert result["blocks"] == [[0, 1, 2], [3, 4], [5, 6]]
    assert mask_entries == [
        (0, 1, 2, 3, 4, 5, 6),
        (),
        (3, 4, 5, 6),
        (0, 1, 2, 5, 6),
        (0, 1, 2, 3, 4),
    ]
    assert mask_exits == mask_entries
    assert calls[0] == ("full_adapter", 0, "valid", (0, 1, 2, 3, 4, 5, 6))
    assert calls[-1] == ("remove_block_2", 1, "valid2", (0, 1, 2, 3, 4))

    full = result["controls"]["full_adapter"]
    assert full["overall"] == {
        "successes": 6,
        "tasks": 8,
        "success_rate": 0.75,
        "wilson_95": pytest.approx((0.40926987910258916, 0.9285223111419724)),
    }
    assert full["by_family"]["ledger_reconcile"] == {
        "successes": 3,
        "tasks": 4,
        "success_rate": 0.75,
        "wilson_95": pytest.approx((0.3006360524426366, 0.9544139373553637)),
    }
    assert result["conditions"][0]["overall"] == {
        "successes": 4,
        "tasks": 8,
        "success_rate": 0.5,
        "wilson_95": pytest.approx((0.21521252682444186, 0.7847874731755582)),
    }
    assert result["most_costly_removal"] == {
        "condition": "remove_block_0",
        "block_index": 0,
        "removed_layers": [0, 1, 2],
        "success_rate_cost": 0.25,
    }


def test_block_ablation_exits_the_mask_when_a_screen_evaluation_raises(monkeypatch) -> None:
    """Catches a missing context exit that would leave later conditions masked."""
    view = type("View", (), {"num_layers": 7})()
    active_keep: list[tuple[int, ...]] = []
    restored: list[tuple[int, ...]] = []

    @contextmanager
    def fake_mask(_view, keep_layers):
        active_keep.append(tuple(keep_layers))
        try:
            yield 0
        finally:
            restored.append(active_keep.pop())

    monkeypatch.setattr(capture, "lora_block_mask", fake_mask)

    def fail_evaluation(_name, _cell_index, _cell):
        raise RuntimeError("screen failed")

    with pytest.raises(RuntimeError, match="screen failed"):
        adapter_delta.run_block_ablation(
            view,
            blocks=3,
            screen=[{"split": "valid"}],
            evaluate_condition=fail_evaluation,
        )

    assert active_keep == []
    assert restored == [(0, 1, 2, 3, 4, 5, 6)]


def test_reused_policy_loader_is_restored_after_an_exception(monkeypatch) -> None:
    """Catches process-global loader leakage after one ablation condition fails."""
    def original(*_args, **_kwargs):
        return "fresh-model", "fresh-tokenizer"

    monkeypatch.setattr(evaluate, "load_policy", original)
    model = object()
    tokenizer = object()

    with (
        pytest.raises(RuntimeError, match="evaluation failed"),
        adapter_delta._reuse_loaded_policy(model, tokenizer),
    ):
        assert evaluate.load_policy("ignored", Path("ignored")) == (model, tokenizer)
        raise RuntimeError("evaluation failed")

    assert evaluate.load_policy is original


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ablate"],
        ["--ablate", "--adapters", "plural", "--adapter", "one", "--screen", "screen"],
        ["--adapters", "plural", "--adapter", "one"],
        ["--adapters", "plural", "--blocks", "3"],
        ["--adapters", "plural", "--screen", "screen"],
        ["--ablate", "--adapter", "one", "--screen", "screen", "--no-base"],
        ["--ablate", "--adapter", "one", "--screen", "screen", "--top", "3"],
        [
            "--ablate",
            "--adapter",
            "one",
            "--screen",
            "screen",
            "--readout-layers",
            "1",
        ],
    ],
)
def test_cli_rejects_missing_and_cross_mode_ablation_arguments(
    monkeypatch, tmp_path, arguments
) -> None:
    """Catches either mode accepting the other mode's adapter, screen, or block flags."""
    from local_llm_lab.probes import guard

    monkeypatch.setattr(
        guard,
        "require_idle_gpu",
        lambda *_args: pytest.fail("validation reached the GPU guard"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["agent-v2-probe-delta", *arguments, "--output", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as raised:
        adapter_delta.main()

    assert raised.value.code == 2


def test_ablation_cli_uses_one_loaded_policy_and_writes_resolved_metadata(
    monkeypatch, tmp_path
) -> None:
    """Catches checkpoint reloads, missing screen plumbing, and incomplete final metadata."""
    from local_llm_lab import models
    from local_llm_lab.probes import guard

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    screen_path = tmp_path / "screen.yaml"
    screen_path.write_text(
        "seed: 23\n"
        "keep_last: 2\n"
        "select:\n"
        "  screen:\n"
        "    - {split: valid, difficulty: 1, per_family: {default: 1}}\n"
        "    - {split: valid2, difficulty: 2, per_family: {default: 1}}\n"
        "eval: {max_steps: 5, max_tokens: 9}\n",
        encoding="utf-8",
    )
    output = tmp_path / "ablation"
    model = object()
    tokenizer = object()
    view = type("View", (), {"num_layers": 7})()
    load_calls: list[tuple[str, Path | None]] = []
    evaluation_calls: list[dict[str, Any]] = []
    guard_calls: list[str] = []

    class Resolved:
        @staticmethod
        def as_dict():
            return {"spec": {"name": "fake-model"}, "num_layers": 7}

    class Spec:
        hf_id = "fake/hf"

        @staticmethod
        def resolve(received_model, received_tokenizer):
            assert (received_model, received_tokenizer) == (model, tokenizer)
            return Resolved()

    def fake_load_policy(model_name: str, adapter_path: Path | None):
        load_calls.append((model_name, adapter_path))
        return model, tokenizer

    @contextmanager
    def fake_mask(received_view, keep_layers):
        assert received_view is view
        yield 7 - len(tuple(keep_layers))

    def fake_run_evaluation(**kwargs):
        assert evaluate.load_policy("ignored", None) == (model, tokenizer)
        evaluation_calls.append(kwargs)
        return {
            "successes": 1,
            "tasks": 2,
            "by_family": {"ledger_reconcile": {"successes": 1, "tasks": 2}},
        }

    def fake_require_idle_gpu(_parser, _args, action):
        guard_calls.append(action)

    def fake_load_model_spec(name):
        return Spec() if name == "fake-model" else None

    monkeypatch.setattr(guard, "require_idle_gpu", fake_require_idle_gpu)
    monkeypatch.setattr(models, "load_model_spec", fake_load_model_spec)
    monkeypatch.setattr(evaluate, "load_policy", fake_load_policy)
    monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(capture, "lora_block_mask", fake_mask)
    monkeypatch.setattr(
        adapter_delta.ArchitectureView,
        "from_model",
        classmethod(lambda _cls, received_model: view if received_model is model else None),
    )
    argv = [
        "agent-v2-probe-delta",
        "--ablate",
        "--adapter",
        str(adapter),
        "--screen",
        str(screen_path),
        "--model",
        "fake-model",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    adapter_delta.main()

    assert load_calls == [("fake/hf", adapter)]
    assert evaluate.load_policy is fake_load_policy
    assert guard_calls == ["loading the adapter policy for block ablation"]
    assert len(evaluation_calls) == 16
    assert evaluation_calls[0]["model_name"] == "fake/hf"
    assert evaluation_calls[0]["split"] == "valid"
    assert evaluation_calls[0]["difficulty"] == 1
    assert evaluation_calls[0]["family_quotas"] == {"default": 1}
    assert evaluation_calls[0]["seed"] == 23
    assert evaluation_calls[0]["keep_last"] == 2
    assert evaluation_calls[0]["max_steps"] == 5
    assert evaluation_calls[0]["max_tokens"] == 9
    assert evaluation_calls[0]["output"].parent == output / "evaluations" / "full_adapter"

    payload = json.loads((output / "ablation.json").read_text(encoding="utf-8"))
    assert payload["command"] == " ".join(argv)
    assert payload["model"] == {"spec": {"name": "fake-model"}, "num_layers": 7}
    assert payload["adapter"] == str(adapter.resolve())
    assert payload["screen"] == {
        "path": str(screen_path.resolve()),
        "cells": [
            {"split": "valid", "difficulty": 1, "per_family": {"default": 1}},
            {"split": "valid2", "difficulty": 2, "per_family": {"default": 1}},
        ],
    }
    assert payload["block_count"] == 6
    assert payload["controls"]["full_adapter"]["kept_layers"] == [0, 1, 2, 3, 4, 5, 6]
    assert payload["conditions"][0]["removed_layers"] == [0, 1]
    markdown = (output / "ablation.md").read_text(encoding="utf-8")
    assert "Full adapter" in markdown
    assert "Empty adapter" in markdown
    assert "ledger_reconcile" in markdown
