"""Tiny serialized-checkpoint tests; no real checkpoint or runtime window is used."""

from __future__ import annotations

import copy
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
    config = transformers.AutoConfig.for_model(
        "gemma3_text",
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
    reference = (
        transformers.AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
    )
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
        json.dumps(
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "text_config": config.to_dict(),
            }
        )
    )
    return reference


def test_hf_loads_every_text_tensor_and_preserves_native_tying(tmp_path):
    torch = pytest.importorskip("torch")
    reference = tiny_checkpoint(tmp_path)
    from local_llm_lab.hf_text import load_text_causal_lm

    loaded, audit = load_text_causal_lm(tmp_path, dtype="bfloat16", device="cpu")
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
    from local_llm_lab.hf_text import load_text_causal_lm

    with pytest.raises(ValueError, match="missing_keys"):
        load_text_causal_lm(tmp_path, dtype="bfloat16", device="cpu")


def test_shared_loader_discovers_foreign_towers_from_headers(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    tiny_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    state = safetensors.load_file(str(path))
    state["foreign.weight"] = state.pop("vision_tower.fixture.weight")
    safetensors.save_file(state, str(path), metadata={"format": "pt"})
    from local_llm_lab.hf_text import checkpoint_metadata, load_text_causal_lm

    metadata = checkpoint_metadata(tmp_path)
    assert metadata["other"] == {"foreign.weight"}
    loaded, report = load_text_causal_lm(tmp_path, dtype="bfloat16", device="cpu")
    assert report["unexpected_keys"] == ["foreign.weight"]
    assert not hasattr(loaded, "foreign")


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


def install_probe_stub(monkeypatch, report):
    def probe(view, ids, *, checkpoint):
        checkpoint("native_dtype_loop", {"status": "running"})
        checkpoint(
            "native_dtype_loop", valid_measurement(64)["precision_probe"]["native_dtype_loop"]
        )
        return report

    monkeypatch.setitem(
        sys.modules,
        "research.acceptance.torch_seam",
        SimpleNamespace(residual_precision_probe=probe),
    )


def test_completed_probe_survives_rotary_failure(monkeypatch):
    measured, saved = {}, []
    install_probe_stub(monkeypatch, {"status": "measured", "tokens": 64})

    def fail(*args):
        raise ValueError("rotary failure")

    monkeypatch.setattr(gates, "rotary_rounding", fail)
    with pytest.raises(ValueError, match="rotary failure"):
        gates.measure_length(
            SimpleNamespace(num_layers=1),
            list(range(64)),
            lambda phase: saved.append(copy.deepcopy(measured)),
            measured,
        )
    assert measured["precision_probe"]["status"] == "measured"
    assert measured["rotary_rounding"]["status"] == "failed"
    assert any(row["precision_probe"]["status"] == "measured" for row in saved)


def test_completed_probe_phase_survives_memory_stop(monkeypatch):
    measured = {}
    install_probe_stub(monkeypatch, {"status": "measured", "tokens": 64})

    def checkpoint(phase):
        if measured["phases"].get(phase, {}).get("status") == "measured":
            raise RuntimeError("memory stop")

    with pytest.raises(RuntimeError, match="memory stop"):
        gates.measure_length(SimpleNamespace(num_layers=1), list(range(64)), checkpoint, measured)
    assert measured["phases"]["native_dtype_loop"]["status"] == "measured"
    assert measured["phases"]["native_dtype_loop"]["max_abs"] == 0.0
    assert measured["rotary_rounding"]["status"] == "unexecuted"


def valid_measurement(length):
    def arm(error):
        return {
            "status": "measured",
            "max_abs": error,
            "max_norm_relative": error,
            "by_layer": {
                i: {"max_abs": error, "native_max_abs": 1.0, "max_norm_relative": error}
                for i in range(2)
            },
        }

    native = {**arm(0.0), "exact": True}
    controls = {
        name: arm(0.2)
        for name in ("mask_dispatch", "hook_site_off_by_one", "entry_transform_omission")
    }
    if length == 64:
        controls["mask_dispatch"] = arm(0.0)
    return {
        "tokens": length,
        "rotary_rounding": {"status": "measured", "max_abs": 0.002},
        "precision_probe": {
            "status": "measured",
            "tokens": length,
            "native_dtype_loop": native,
            "promoted_fp32_loop": arm(0.1),
            "controls": controls,
        },
    }


def test_gate_accepts_exact_seam_without_gating_cross_precision_floor():
    # A 10% floor is deliberately above the old bound; it is descriptive, not accepted accuracy.
    rows = [valid_measurement(length) for length in gates.LENGTHS]
    outcome = gates.assess_gate2(rows, num_layers=1)
    assert outcome["status"] == "pass"
    assert outcome["cross_precision_bound"] is None


@pytest.mark.parametrize("problem", ["nonzero", "missing_site", "nonfinite", "invalid_zero_scale"])
def test_gate_rejects_invalid_same_dtype_evidence(problem):
    rows = [valid_measurement(length) for length in gates.LENGTHS]
    native = rows[1]["precision_probe"]["native_dtype_loop"]
    if problem == "nonzero":
        native["by_layer"][0]["max_abs"] = 1e-30
    elif problem == "missing_site":
        del native["by_layer"][0]
    elif problem == "nonfinite":
        native["by_layer"][0]["max_norm_relative"] = float("nan")
    else:
        native["by_layer"][0]["native_max_abs"] = 0.0
        native["by_layer"][0]["max_abs"] = 1e-30
    assert gates.assess_gate2(rows, num_layers=1)["status"] == "fail"


@pytest.mark.parametrize("problem", ["missing_control", "equal_floor", "below_floor", "short_mask"])
def test_gate_rejects_controls_that_do_not_establish_the_required_difference(problem):
    rows = [valid_measurement(length) for length in gates.LENGTHS]
    controls = rows[1]["precision_probe"]["controls"]
    if problem == "missing_control":
        del controls["entry_transform_omission"]
    elif problem == "short_mask":
        mask = rows[0]["precision_probe"]["controls"]["mask_dispatch"]
        mask.update(max_abs=1e-30, max_norm_relative=1e-30)
        for row in mask["by_layer"].values():
            row.update(max_abs=1e-30, max_norm_relative=1e-30)
    else:
        relative = 0.1 if problem == "equal_floor" else 0.01
        controls["mask_dispatch"].update(max_abs=relative, max_norm_relative=relative)
        for row in controls["mask_dispatch"]["by_layer"].values():
            row.update(max_abs=relative, max_norm_relative=relative)
    assert gates.assess_gate2(rows, num_layers=1)["status"] == "fail"


def test_gate_refuses_missing_or_duplicate_lengths():
    assert gates.assess_gate2([valid_measurement(64)], num_layers=1)["status"] == "incomplete"
    assert (
        gates.assess_gate2([valid_measurement(64), valid_measurement(64)], num_layers=1)["status"]
        == "incomplete"
    )


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


def test_gate_refuses_inconsistent_control_aggregate():
    rows = [valid_measurement(length) for length in gates.LENGTHS]
    control = rows[1]["precision_probe"]["controls"]["entry_transform_omission"]
    for row in control["by_layer"].values():
        row["max_norm_relative"] = 0.0
    assert gates.assess_gate2(rows, num_layers=1)["status"] == "fail"


def test_gate_accepts_zero_reference_only_when_error_and_relative_are_zero():
    rows = [valid_measurement(length) for length in gates.LENGTHS]
    for measured in rows:
        probe = measured["precision_probe"]
        for arm in (
            probe["native_dtype_loop"],
            probe["promoted_fp32_loop"],
            *probe["controls"].values(),
        ):
            arm["by_layer"][0] = {"max_abs": 0.0, "native_max_abs": 0.0, "max_norm_relative": 0.0}
    assert gates.assess_gate2(rows, num_layers=1)["status"] == "pass"


def test_native_seam_failure_stops_before_precision_floor(monkeypatch):
    measured, phases = {}, []

    def probe(view, ids, *, checkpoint):
        arm = valid_measurement(64)["precision_probe"]["native_dtype_loop"]
        arm["by_layer"][1]["max_abs"] = 1e-30
        checkpoint("native_dtype_loop", arm)
        pytest.fail("precision floor must not start after native seam failure")

    monkeypatch.setitem(
        sys.modules,
        "research.acceptance.torch_seam",
        SimpleNamespace(residual_precision_probe=probe),
    )
    with pytest.raises(RuntimeError, match="native dtype seam failed"):
        gates.measure_length(
            SimpleNamespace(num_layers=1), list(range(64)), phases.append, measured
        )
    assert measured["acceptance"]["status"] == "fail"
    assert measured["rotary_rounding"]["status"] == "unexecuted"
    assert phases[-1] == "native_dtype_exactness_failure"
