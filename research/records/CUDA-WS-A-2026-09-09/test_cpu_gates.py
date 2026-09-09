"""Tiny serialized-checkpoint tests; no real checkpoint or runtime window is used."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name("cpu_gates.py")
spec = importlib.util.spec_from_file_location("ws_a_cpu_gates", SOURCE)
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)


def tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    config = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        sliding_window=1024,
        max_position_embeddings=2048,
    )
    torch.manual_seed(123)
    reference = transformers.Gemma3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    checkpoint = {
        f"language_model.{key}": value.detach().clone()
        for key, value in reference.state_dict().items()
        if key != "lm_head.weight"
    }
    checkpoint["vision_tower.fixture.weight"] = torch.ones(2, dtype=torch.bfloat16)
    safetensors.save_file(
        checkpoint, str(tmp_path / "model.safetensors"), metadata={"format": "pt"}
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gemma3", "text_config": config.to_dict()})
    )
    return reference


def test_hf_loads_every_text_tensor_and_preserves_native_tying(tmp_path):
    torch = pytest.importorskip("torch")
    reference = tiny_checkpoint(tmp_path)
    loaded, audit = gates.load_text_model(gates.checkpoint_metadata(tmp_path))
    assert audit["loading_info"]["missing_keys"] == []
    assert set(audit["loading_info"]["unexpected_keys"]) == {"vision_tower.fixture.weight"}
    assert set(loaded.state_dict()) == set(reference.state_dict())
    for key, expected in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected), key
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_missing_text_weight_fails_closed(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    tiny_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    state = safetensors.load_file(str(path))
    del state["language_model.model.layers.0.mlp.down_proj.weight"]
    safetensors.save_file(state, str(path), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="missing_keys"):
        gates.load_text_model(gates.checkpoint_metadata(tmp_path))


def test_unknown_nontext_tensor_fails_closed(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    tiny_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    state = safetensors.load_file(str(path))
    state["foreign.weight"] = state.pop("vision_tower.fixture.weight")
    safetensors.save_file(state, str(path), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="unrecognised"):
        gates.checkpoint_metadata(tmp_path)


def test_ids_reject_short_and_boolean_data(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="1,400"):
        gates.read_ids(path)
    path.write_text(json.dumps([True] * 1400))
    with pytest.raises(ValueError, match="booleans"):
        gates.read_ids(path)
    path.write_text(json.dumps({"token_ids": list(range(1400))}))
    ids, provenance = gates.read_ids(path)
    assert ids == list(range(1400))
    assert provenance["sha256"] == gates.sha256(path)


def test_window_requires_matching_nonce_and_live_holder(monkeypatch):
    window = SimpleNamespace(
        nonce="ours",
        holder_state="running",
        seat="test",
        pid=123,
        path=Path("window.json"),
        purpose="tiny test",
    )
    lock = SimpleNamespace(
        read_window=lambda: window,
        WINDOW_HOLDER_ENV="TEST_WINDOW_NONCE",
        blocking_window=lambda: None,
    )
    monkeypatch.setenv("TEST_WINDOW_NONCE", "ours")
    assert gates.assert_own_window(lock)["holder_state"] == "running"
    monkeypatch.setenv("TEST_WINDOW_NONCE", "foreign")
    with pytest.raises(RuntimeError, match="nonce"):
        gates.assert_own_window(lock)
    monkeypatch.setenv("TEST_WINDOW_NONCE", "ours")
    window.holder_state = "not running"
    with pytest.raises(RuntimeError, match="running"):
        gates.assert_own_window(lock)


def test_cli_requires_explicit_execution_before_any_output(tmp_path):
    with pytest.raises(SystemExit):
        gates.main(
            [
                "--checkpoint",
                str(tmp_path),
                "--token-ids",
                str(tmp_path / "ids.json"),
                "--output",
                str(tmp_path / "out.json"),
                "--cap-gib",
                "10.656005859375",
                "--projected-peak-gib",
                "9.0",
                "--source-commit",
                "test",
            ]
        )
    assert not (tmp_path / "out.json").exists()


def test_source_commit_must_match_actual(monkeypatch):
    from local_llm_lab import runlog

    observed = []

    def actual(root):
        observed.append(root)
        return "a" * 40

    monkeypatch.setattr(runlog, "git_commit", actual)
    assert gates.verified_source_commit("a" * 40) == "a" * 40
    assert observed == [gates.ROOT]
    with pytest.raises(ValueError, match="actual"):
        gates.verified_source_commit("b" * 40)
    monkeypatch.setattr(runlog, "git_commit", lambda _: "unknown")
    with pytest.raises(ValueError, match="unknown"):
        gates.verified_source_commit("unknown")


def test_structure_checks_each_header_dimension_and_span():
    config = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=16,
        vocab_size=64,
        layer_types=["sliding_attention", "full_attention"],
    )
    measured = {
        "num_layers": 2,
        "hidden_size": 16,
        "vocab_size": 64,
        "attention_spans": ["sliding", "global"],
    }
    assert gates.verify_structure(measured, config)["expected_from_header_config"] == measured
    for field in measured:
        damaged = {**measured, field: None}
        with pytest.raises(ValueError, match=field):
            gates.verify_structure(damaged, config)


def test_completed_normal_measurement_survives_rotary_failure(monkeypatch):
    measured, saved = {}, []
    monkeypatch.setattr(gates, "compare_retained", lambda *args: {"by_layer": {0: 0.0}})

    def fail(*args):
        raise ValueError("rotary failure")

    monkeypatch.setattr(gates, "rotary_rounding", fail)
    with pytest.raises(ValueError, match="rotary failure"):
        gates.measure_length(
            None, [1, 2], lambda phase: saved.append(json.loads(json.dumps(measured))), measured
        )
    assert measured["normal"] == {"status": "measured", "by_layer": {0: 0.0}}
    assert measured["rotary_rounding"]["status"] == "failed"
    assert measured["mask_dispatch_control"]["status"] == "unexecuted"
    assert any(row["normal"]["status"] == "measured" for row in saved)


def test_completed_normal_measurement_survives_memory_stop(monkeypatch):
    measured = {}
    monkeypatch.setattr(gates, "compare_retained", lambda *args: {"by_layer": {0: 0.0}})

    def checkpoint(phase):
        if phase == "normal":
            assert measured["normal"]["status"] == "measured"
            raise RuntimeError("memory stop")

    with pytest.raises(RuntimeError, match="memory stop"):
        gates.measure_length(None, [1, 2], checkpoint, measured)
    assert measured["normal"]["status"] == "measured"
    assert measured["rotary_rounding"]["status"] == "unexecuted"


def test_atomic_record_failure_preserves_previous_json(tmp_path, monkeypatch):
    from local_llm_lab import runlog

    path = tmp_path / "record.json"
    gates.write_report(path, {"completed": "normal"})

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(runlog.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        gates.write_report(path, {"completed": "rotary"})
    assert json.loads(path.read_text()) == {"completed": "normal"}


def test_final_high_water_is_written_after_forward_failure(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    report = {"memory": [{"phase": "before_forward", "process_peak_bytes": 10}]}
    monkeypatch.setattr(gates, "memory_reading", lambda: {"process_peak_bytes": 99})
    with pytest.raises(ValueError, match="forward failed"):
        try:
            raise ValueError("forward failed")
        finally:
            gates.finish_report(
                path, report, time.monotonic(), preserving_exception=sys.exc_info()[0] is not None
            )
    written = json.loads(path.read_text())
    assert written["memory"][-1] == {"phase": "final", "process_peak_bytes": 99}
    assert written["elapsed_seconds"] >= 0


def test_final_measurement_and_write_errors_cannot_mask_original(tmp_path, monkeypatch):
    report = {"memory": []}

    def fail_measurement():
        raise OSError("memory reading failed")

    def fail_write(*args):
        raise OSError("record write failed")

    monkeypatch.setattr(gates, "memory_reading", fail_measurement)
    monkeypatch.setattr(gates, "write_report", fail_write)
    with pytest.raises(ValueError, match="original load failure"):
        try:
            raise ValueError("original load failure")
        finally:
            gates.finish_report(
                tmp_path / "record.json",
                report,
                time.monotonic(),
                preserving_exception=sys.exc_info()[0] is not None,
            )
    assert report["final_memory_error"]["message"] == "memory reading failed"


def test_final_save_error_is_not_hidden_without_an_original_error(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "memory_reading", lambda: {"process_peak_bytes": 1})

    def fail_write(*args):
        raise OSError("record write failed")

    monkeypatch.setattr(gates, "write_report", fail_write)
    with pytest.raises(OSError, match="record write failed"):
        gates.finish_report(
            tmp_path / "record.json", {"memory": []}, time.monotonic(), preserving_exception=False
        )
