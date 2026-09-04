from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_llm_lab.models import ResolvedSpec, load_model_spec
from local_llm_lab.pipeline import cli, evaluate
from local_llm_lab.probes import adapter_delta, capture
from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.provenance import write_provenance


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


def test_static_delta_cli_logs_its_identity_and_progress_without_a_base_model(
    monkeypatch, tmp_path
) -> None:
    """R26(e)/(g): --no-base needs no model, so the whole static path runs on written files.

    Catches a static CLI that writes its artifacts without the log pair, and a per-adapter
    unit that reports no progress.
    """
    from safetensors.numpy import save_file

    from local_llm_lab.probes import guard

    adapter = tmp_path / "agent-v2b" / "best-adapter"
    adapter.mkdir(parents=True)
    rng = np.random.default_rng(5)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.lora_a": rng.normal(size=(8, 2)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.lora_b": rng.normal(size=(2, 6)).astype(np.float32),
            "model.layers.1.mlp.down_proj.lora_a": rng.normal(size=(9, 2)).astype(np.float32),
            "model.layers.1.mlp.down_proj.lora_b": rng.normal(size=(2, 4)).astype(np.float32),
        },
        str(adapter / "adapters.safetensors"),
    )
    # R38: the config the static path reads is the trainer's own, not a two-key stand-in.
    _write_adapter_config(adapter, _resolved_spec("fake/base"))
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("--no-base reached the GPU guard")
    )
    output = tmp_path / "delta"
    argv = [
        "agent-v2-probe-delta",
        "--adapters",
        str(adapter),
        "--no-base",
        "--model",
        "fake-model",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    adapter_delta.main()

    assert (output / "delta.json").is_file() and (output / "delta.md").is_file()
    assert (output / "run.log").is_file()
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start = events[0]
    assert start["kind"] == "start" and start["run"] == "adapter-delta"
    assert start["command"] == argv
    identity = dict(start["fields"])
    assert identity.pop("git_commit")  # a commit or "unknown"; never absent
    assert identity == {
        "model": "fake-model",
        "adapters": [str(adapter.resolve())],
        "readout_layers": [],
        "top": 16,
        "no_base": True,
    }
    assert [
        (event["step"], event["total"], event["label"])
        for event in events
        if event["kind"] == "progress"
    ] == [(1, 1, "adapter agent-v2b")]
    wrote = next(event for event in events if event["message"] == "wrote")
    assert wrote["fields"] == {
        "json": str(output / "delta.json"),
        "md": str(output / "delta.md"),
    }
    assert events[-1]["kind"] == "end" and events[-1]["status"] == "ok"


def test_block_ablation_reports_progress_per_condition_without_touching_the_record(
    monkeypatch,
) -> None:
    """R26(g): one line per evaluated condition — the two controls, then each removed block."""

    view = type("View", (), {"num_layers": 7})()

    @contextmanager
    def fake_mask(_view, keep_layers):
        yield 7 - len(tuple(keep_layers))

    monkeypatch.setattr(capture, "lora_block_mask", fake_mask)
    screen = [{"split": "valid", "difficulty": 1, "per_family": {"default": 1}}]

    def fake_evaluate(_name, _cell_index, _cell):
        return {
            "successes": 1,
            "tasks": 2,
            "by_family": {"ledger_reconcile": {"successes": 1, "tasks": 2}},
        }

    seen: list[tuple[int, int, str]] = []
    with_progress = adapter_delta.run_block_ablation(
        view,
        blocks=3,
        screen=screen,
        evaluate_condition=fake_evaluate,
        progress=lambda step, total, label: seen.append((step, total, label)),
    )
    without_progress = adapter_delta.run_block_ablation(
        view,
        blocks=3,
        screen=screen,
        evaluate_condition=fake_evaluate,
        progress=None,
    )

    assert seen == [
        (1, 5, "condition full_adapter"),
        (2, 5, "condition empty_adapter"),
        (3, 5, "condition remove_block_0"),
        (4, 5, "condition remove_block_1"),
        (5, 5, "condition remove_block_2"),
    ]
    assert json.dumps(with_progress, sort_keys=True) == json.dumps(
        without_progress, sort_keys=True
    )


def _delta_payload() -> dict[str, Any]:
    def module(layer: int, name: str) -> dict[str, Any]:
        return {
            "module": f"model.layers.{layer}.{name}",
            "layer": layer,
            "type": name,
            "shape": [4, 4],
            "rank": 8,
            "scale": 32.0,
            "delta_norm": 1.0 + layer,
            "singular_values": [1.0, 0.5],
            "effective_rank_90": 2,
            "relative_norm": None,
        }

    return {
        "model": "fake-model",
        "has_base": False,
        "runs": [
            {
                "label": "run-a",
                "adapter": "/tmp/a",
                "scale": 32.0,
                "modules": [module(0, "q_proj"), module(1, "down_proj")],
                "effective_rank_histogram": {"2": 2},
            },
            {
                "label": "run-b",
                "adapter": "/tmp/b",
                "scale": 32.0,
                "modules": [module(0, "q_proj")],
                "effective_rank_histogram": {"2": 1},
            },
        ],
        "comparison": None,
        "readouts": None,
    }
def test_reused_policy_loader_is_restored_after_an_exception(monkeypatch) -> None:
    """Catches process-global loader leakage after one ablation condition fails."""
    def original(*_args, **_kwargs):
        return "fresh-model", "fresh-tokenizer", "fresh-view", "fresh-resolved"

    monkeypatch.setattr(evaluate, "load_policy", original)
    model = object()
    tokenizer = object()
    view = object()
    resolved = object()

    with (
        pytest.raises(RuntimeError, match="evaluation failed"),
        adapter_delta._reuse_loaded_policy(model, tokenizer, view, resolved),
    ):
        assert evaluate.load_policy("ignored", Path("ignored")) == (
            model,
            tokenizer,
            view,
            resolved,
        )
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

    resolved = Resolved()

    class Spec:
        name = "fake-model"
        hf_id = "fake/hf"

        @staticmethod
        def resolve(received_model, received_tokenizer):
            assert (received_model, received_tokenizer) == (model, tokenizer)
            return resolved

    def fake_load_policy(given, adapter_path: Path | None):
        load_calls.append((given.hf_id, adapter_path))
        return model, tokenizer, view, given.resolve(model, tokenizer)

    @contextmanager
    def fake_mask(received_view, keep_layers):
        assert received_view is view
        yield 7 - len(tuple(keep_layers))

    def fake_run_evaluation(**kwargs):
        assert evaluate.load_policy("ignored", None) == (model, tokenizer, view, resolved)
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
    monkeypatch.setattr(adapter_delta, "require_preflight", lambda *_args, **_kwargs: None)
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
    assert evaluation_calls[0]["spec"].hf_id == "fake/hf"
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

    # R26(e)/(g), issue #35: the run writes its own log pair; the start event identifies it
    # and one progress line lands per condition.
    assert (output / "run.log").is_file()
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start = events[0]
    assert start["kind"] == "start" and start["run"] == "block-ablation"
    assert start["command"] == argv
    identity = dict(start["fields"])
    assert identity.pop("git_commit")  # a commit or "unknown"; never absent
    assert identity == {
        "model": "fake-model",
        "hf_id": "fake/hf",
        "adapter": str(adapter.resolve()),
        "screen": str(screen_path.resolve()),
        "screen_sha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
        "blocks": 6,
    }
    assert [event["label"] for event in events if event["kind"] == "progress"] == [
        "condition full_adapter",
        "condition empty_adapter",
        *(f"condition remove_block_{index}" for index in range(6)),
    ]
    assert events[-1]["kind"] == "end" and events[-1]["status"] == "ok"
    assert "ledger_reconcile" in markdown


# --------------------------- SPEC-001 §8/§9/§10 closure: base identity, preflight, provenance


def _resolved_spec(hf_id: str, *, revision: str | None = None) -> ResolvedSpec:
    """A ``ResolvedSpec`` for ``hf_id``, as ``ModelSpec.resolve`` would return one.

    ``resolve`` reads an ``ArchitectureView`` over loaded weights, which this suite must never
    touch, so the dataclass is constructed directly -- but it *is* the real class over a real
    registry declaration, so ``as_dict`` lays the record out exactly as ``write_provenance``
    stores it, and ``lora_config`` reads the fields it really reads.  The architecture numbers
    are a small stand-in; nothing under test looks at them.
    """
    spec = replace(load_model_spec("qwen35-4b"), hf_id=hf_id)
    return ResolvedSpec(
        spec=spec,
        num_layers=4,
        hidden_size=8,
        vocab_size=32,
        tie_word_embeddings=True,
        layer_types=("full_attention",) * 4,
        lora_keys=("self_attn.q_proj", "mlp.down_proj"),
        trainable_parameters=64,
        probe_layers=(1, 2, 3),
        cache_strategy="none",
        cache_strategy_reason="explicit:none",
        snapshot_revision=revision,
        jvp_method="untested",
    )


def _write_adapter_config(directory: Path, resolved: ResolvedSpec) -> None:
    """``adapter_config.json`` as a real training run leaves it (R38).

    mlx-lm writes this file, not us: ``mlx_lm.lora.train_model`` hands ``vars(args)`` to
    ``mlx_lm.utils.save_config`` before the first step.  So the fixture takes the route
    ``stage_train`` takes -- a real arm config through ``lora_config``, merged over the
    library's own ``CONFIG_DEFAULTS``, into ``_effective_lora_args`` -- and then lets mlx-lm's
    own serialiser write it.  Pinning the shape to the library rather than to a checked-in
    sample means a release that moves ``scale`` out of ``lora_parameters`` breaks this suite
    instead of silently handing :func:`adapter_scale` the 20.0 default.  Every step is a dict
    operation: no weights are loaded and no array is evaluated.
    """
    import importlib

    from mlx_lm.utils import save_config

    config = cli.load_config(PROJECT_ROOT / "configs" / "agent_v2b_qwen35_4b.yaml")
    config["output"] = directory.parent
    lora = {
        **importlib.import_module("mlx_lm.lora").CONFIG_DEFAULTS,
        **cli.lora_config(config, resolved),
    }
    save_config(vars(cli._effective_lora_args(lora, {})), directory / "adapter_config.json")


def _adapter_dir(directory: Path, *, base: str | None = None, revision: str | None = None) -> Path:
    """One adapter directory, optionally declaring the base it was trained on.

    ``base`` is threaded through the registry declaration rather than poked into the written
    JSON, because that is the only route by which a real run's ``adapter_config.json`` gets a
    ``model`` field.  ``revision`` is written by the real :func:`write_provenance` into the run
    directory above the adapter, which is where the pipeline puts it.
    """
    from safetensors.numpy import save_file

    name = "model.layers.0.mlp.down_proj"
    local = np.random.default_rng(3)
    directory.mkdir(parents=True)
    save_file(
        {
            f"{name}.lora_a": local.normal(size=(8, 2)).astype(np.float32),
            f"{name}.lora_b": local.normal(size=(2, 6)).astype(np.float32),
        },
        str(directory / "adapters.safetensors"),
    )
    resolved = _resolved_spec(base or "fake/unnamed-base", revision=revision)
    _write_adapter_config(directory, resolved)
    if base is None:
        # mlx-lm always records ``model``; drop it here to reach the provenance fallback,
        # which is the only way an adapter directory can lack one.
        config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        del config["model"]
        (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    if revision is not None:
        write_provenance(
            directory.parent,
            resolved=resolved,
            spec=resolved.spec,
            extra={"stage": "test-fixture"},
        )
    return directory


def test_compare_adapters_refuses_two_adapters_trained_on_different_bases(tmp_path) -> None:
    """SPEC-001 §8: cross-run comparison asserts an identical base, by name."""
    left = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base-one")
    right = _adapter_dir(tmp_path / "agent-v2c" / "best-adapter", base="fake/base-two")

    with pytest.raises(ValueError) as raised:
        adapter_delta.compare_adapters([left, right], top=2)

    message = str(raised.value)
    assert "agent-v2b" in message and "agent-v2c" in message
    assert "fake/base-one" in message and "fake/base-two" in message


def test_compare_adapters_refuses_two_snapshot_revisions_of_one_base(tmp_path) -> None:
    left = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base", revision="aaa")
    right = _adapter_dir(tmp_path / "agent-v2c" / "best-adapter", base="fake/base", revision="bbb")

    with pytest.raises(ValueError) as raised:
        adapter_delta.compare_adapters([left, right], top=2)

    message = str(raised.value)
    assert "agent-v2b" in message and "agent-v2c" in message
    assert "aaa" in message and "bbb" in message


def test_compare_adapters_accepts_one_base_and_records_the_identity(tmp_path) -> None:
    left = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base", revision="aaa")
    right = _adapter_dir(tmp_path / "agent-v2c" / "best-adapter", base="fake/base", revision="aaa")

    comparison = adapter_delta.compare_adapters([left, right], top=2)

    assert comparison["base_identity"] == {
        "agent-v2b": {"hf_id": "fake/base", "snapshot_revision": "aaa"},
        "agent-v2c": {"hf_id": "fake/base", "snapshot_revision": "aaa"},
    }


# ------------------------------------------------ R38: each read against its real writer


def test_adapter_scale_reads_the_scale_the_trainer_nests_under_lora_parameters(tmp_path) -> None:
    """R38 on ``adapter_delta.py:85``, against the file mlx-lm's trainer writes.

    The recipe's ``train.scale`` is 32.0 and the library's own default is 20.0, so a reader
    looking anywhere but ``lora_parameters.scale`` reports the default and every relative-norm
    number in P5 comes out 1.6x small.  Lifting the key one level up is the move that tells
    the two apart: only a read at the writer's level notices.
    """
    adapter = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base")
    config_path = adapter / "adapter_config.json"

    assert adapter_delta.adapter_scale(adapter) == 32.0

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["scale"] = config["lora_parameters"].pop("scale")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert adapter_delta.adapter_scale(adapter) == adapter_delta.DEFAULT_SCALE


def test_base_identity_reads_the_hf_id_the_trainer_writes_as_model(tmp_path) -> None:
    """R38 on ``adapter_delta.py:104``: the field's name is load-bearing, so rename it."""
    adapter = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base")
    config_path = adapter / "adapter_config.json"

    assert adapter_delta.adapter_base_identity(adapter)["hf_id"] == "fake/base"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model"] = config.pop("model")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert adapter_delta.adapter_base_identity(adapter)["hf_id"] is None


def test_base_identity_reads_the_revision_at_the_level_write_provenance_records_it(
    tmp_path,
) -> None:
    """R38 on ``adapter_delta.py:113``: ``snapshot_revision`` lives inside the model block."""
    run = tmp_path / "agent-v2b"
    adapter = _adapter_dir(run / "best-adapter", base="fake/base", revision="aaa")
    provenance_path = run / "provenance.json"

    assert adapter_delta.adapter_base_identity(adapter)["snapshot_revision"] == "aaa"

    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    record["snapshot_revision"] = record["model"].pop("snapshot_revision")
    provenance_path.write_text(json.dumps(record), encoding="utf-8")

    assert adapter_delta.adapter_base_identity(adapter)["snapshot_revision"] is None


def test_base_identity_reads_the_flat_model_block_a_resolved_less_stage_writes(tmp_path) -> None:
    """R38 on ``adapter_delta.py:113-118``: ``write_provenance`` has two shapes, not one.

    ``select``, ``eval`` and ``rollout`` (``pipeline/cli.py:1042``, ``:1112``, ``:1165``) pass
    ``resolved=None`` and write into ``config["output"]`` -- the very directory that holds
    ``adapters/`` and ``best-adapter/``, and the one this reader consults as the adapter's
    parent.  Each overwrites whatever ``train`` left there, so the flat ``asdict(spec)`` block
    is what an adapter's parent provenance normally holds by the time P5 reads it.  That block
    states ``hf_id`` outright at its top level and the reader was looking only one level down,
    under ``spec``; it also has no ``snapshot_revision`` key at all, which is why an unresolved
    stage can only ever yield an unknown revision.
    """
    adapter = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base=None)
    spec = replace(load_model_spec("qwen35-4b"), hf_id="fake/base")
    write_provenance(
        adapter.parent, resolved=None, spec=spec, extra={"stage": "select"}
    )
    record = json.loads((adapter.parent / "provenance.json").read_text(encoding="utf-8"))
    assert "snapshot_revision" not in record["model"] and "spec" not in record["model"]

    assert adapter_delta.adapter_base_identity(adapter) == {
        "hf_id": "fake/base",
        "snapshot_revision": None,
    }


def test_static_delta_cli_refuses_to_load_without_preflight_evidence(monkeypatch, tmp_path) -> None:
    """SPEC-001 §10: the base-model arm of P5 is gated like every other model load."""
    from local_llm_lab.pipeline import preflight
    from local_llm_lab.probes import guard

    adapter = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base")
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", tmp_path / "preflight")
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-v2-probe-delta",
            "--adapters",
            str(adapter),
            "--model",
            "qwen35-4b",
            "--output",
            str(tmp_path / "delta"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        adapter_delta.main()

    assert "qwen35-4b" in str(raised.value)
    assert "preflight" in str(raised.value)


def test_ablation_cli_refuses_to_load_without_preflight_evidence(monkeypatch, tmp_path) -> None:
    from local_llm_lab.pipeline import preflight
    from local_llm_lab.probes import guard

    adapter = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base")
    screen_path = tmp_path / "screen.yaml"
    screen_path.write_text(
        "keep_last: 2\n"
        "select:\n"
        "  screen:\n"
        "    - {split: valid, difficulty: 1, per_family: {default: 1}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", tmp_path / "preflight")
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-v2-probe-delta",
            "--ablate",
            "--adapter",
            str(adapter),
            "--screen",
            str(screen_path),
            "--model",
            "qwen35-4b",
            "--output",
            str(tmp_path / "ablation"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        adapter_delta.main()

    assert "qwen35-4b" in str(raised.value)


def test_static_delta_cli_records_the_precision_block_and_writes_provenance(
    monkeypatch, tmp_path
) -> None:
    """R18a and SPEC-001 §9 on the arm that loads nothing."""
    from local_llm_lab.probes import guard

    adapter = _adapter_dir(tmp_path / "agent-v2b" / "best-adapter", base="fake/base")
    block = {"frobenius_relative": 0.004, "elementwise_max": 0.02}
    monkeypatch.setattr(adapter_delta, "preflight_precision_block", lambda _spec: dict(block))
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("--no-base reached the GPU guard")
    )
    output = tmp_path / "delta"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-v2-probe-delta",
            "--adapters",
            str(adapter),
            "--no-base",
            "--model",
            "fake-model",
            "--output",
            str(output),
        ],
    )

    adapter_delta.main()

    payload = json.loads((output / "delta.json").read_text(encoding="utf-8"))
    assert payload["fp32_manual_vs_native"] == block
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["extra"]["stage"] == "p5-adapter-delta"
    assert provenance["extra"]["artifacts"] == [
        str(output / "delta.json"),
        str(output / "delta.md"),
    ]
