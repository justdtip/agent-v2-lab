"""Tiny serialized-checkpoint tests; no real checkpoint or runtime window is used."""

from __future__ import annotations

import builtins
import copy
import gc
import importlib.util
import json
import sys
import time
import weakref
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
    checkpoint["vision_tower.vision_model.embeddings.patch_embedding.weight"] = torch.zeros(
        4, 3, 2, 2, dtype=torch.bfloat16
    )
    checkpoint["multi_modal_projector.mm_input_projection_weight"] = torch.zeros(
        4, 16, dtype=torch.bfloat16
    )
    safetensors.save_file(
        checkpoint, str(tmp_path / "model.safetensors"), metadata={"format": "pt"}
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "text_config": config.to_dict(),
                "vision_config": {"model_type": "siglip_vision_model", "hidden_size": 4},
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
    assert set(audit["loading_info"]["unexpected_keys"]) == {
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "multi_modal_projector.mm_input_projection_weight",
    }
    assert audit["unexpected_keys"]["count"] == 2
    assert audit["unexpected_keys"]["prefixes"] == ["multi_modal_projector", "vision_tower"]
    assert audit["architecture"] == "Gemma3ForCausalLM" and audit["wrapper"] is True
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
    state["foreign.weight"] = state.pop(
        "vision_tower.vision_model.embeddings.patch_embedding.weight"
    )
    safetensors.save_file(state, str(path), metadata={"format": "pt"})
    from local_llm_lab.hf_text import checkpoint_metadata, load_text_causal_lm

    metadata = checkpoint_metadata(tmp_path)
    assert metadata["other"] == {
        "foreign.weight",
        "multi_modal_projector.mm_input_projection_weight",
    }
    loaded, report = load_text_causal_lm(tmp_path, dtype="bfloat16", device="cpu")
    assert report["unexpected_keys"]["count"] == 2
    assert report["unexpected_keys"]["prefixes"] == ["foreign", "multi_modal_projector"]
    assert set(report["loading_info"]["unexpected_keys"]) == metadata["other"]
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


def valid_measurement(length, *, num_layers=1):
    def arm(error, actual_dtype="torch.bfloat16"):
        return {
            "status": "measured",
            "max_abs": error,
            "max_norm_relative": error,
            "by_layer": {
                i: {
                    "max_abs": error,
                    "native_max_abs": 1.0,
                    "max_norm_relative": error,
                    "actual_dtype": actual_dtype,
                    "native_dtype": "torch.bfloat16",
                }
                for i in range(num_layers + 1)
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
            "promoted_fp32_loop": arm(0.1, "torch.float32"),
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
            arm["by_layer"][0].update(max_abs=0.0, native_max_abs=0.0, max_norm_relative=0.0)
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


def valid_float32_control(length, error=1e-3, *, num_layers=1):
    return {
        "status": "measured",
        "tokens": length,
        "max_abs": error,
        "max_norm_relative": error,
        "by_layer": {
            layer: {
                "max_abs": error,
                "native_max_abs": 1.0,
                "max_norm_relative": error,
                "actual_dtype": "torch.float32",
                "native_dtype": "torch.float32",
            }
            for layer in range(num_layers + 1)
        },
    }


def test_float32_control_uses_its_declared_bound_separately_from_bf16_exactness():
    measurements = [valid_float32_control(length) for length in (64, 1400)]
    # Exactly the registered boundary is valid; this comparison need not be bit-identical.
    assert gates.assess_float32_control(measurements, num_layers=1)["status"] == "pass"
    assert all(row["max_abs"] != 0 for row in measurements)


@pytest.mark.parametrize(
    "problem",
    [
        "above_bound",
        "missing_site",
        "bf16_actual",
        "bf16_native",
        "missing_dtype",
        "nonfinite",
        "inconsistent_relative",
        "inconsistent_max",
        "not_measured",
    ],
)
def test_float32_control_fails_closed_on_bad_precision_or_evidence(problem):
    measurements = [valid_float32_control(length) for length in (64, 1400)]
    arm = measurements[1]
    row = arm["by_layer"][1]
    if problem == "above_bound":
        row["max_abs"] = row["max_norm_relative"] = 0.001000001
        arm["max_abs"] = arm["max_norm_relative"] = row["max_abs"]
    elif problem == "missing_site":
        del arm["by_layer"][1]
    elif problem == "bf16_actual":
        row["actual_dtype"] = "torch.bfloat16"
    elif problem == "bf16_native":
        row["native_dtype"] = "torch.bfloat16"
    elif problem == "missing_dtype":
        del row["native_dtype"]
    elif problem == "nonfinite":
        row["max_abs"] = float("nan")
    elif problem == "inconsistent_relative":
        row["max_norm_relative"] = 0.0
    elif problem == "inconsistent_max":
        arm["max_abs"] = 0.0
    else:
        arm["status"] = "running"
    assert gates.assess_float32_control(measurements, num_layers=1)["status"] == "fail"


@pytest.mark.parametrize("lengths", [(64,), (1400,), (64, 64), (64, 1400, 1400), (64, 1400, 2)])
def test_float32_control_requires_each_declared_length_exactly_once(lengths):
    measurements = [valid_float32_control(length) for length in lengths]
    assert gates.assess_float32_control(measurements, num_layers=1)["status"] == "incomplete"


def test_number_annotation_is_recursive_preserves_boolean_and_null_and_is_idempotent():
    original = {
        "count": 3,
        "error": 0.125,
        "passed": True,
        "missing": None,
        "nested": {"basis": "same frozen inputs", "tokens": [64, 1400]},
        "existing": {"value": 9.8, "basis": "laptop-basis"},
        "expected": {"num_layers": 34, "known": True},
        "expected_from_header_config": {"hidden_size": 2560},
        "native_dtype_bound": 0.0,
        "projected_peak_gib": 9.8,
    }
    saved = copy.deepcopy(original)
    annotated = gates.annotate_numbers(original)
    assert original == saved
    assert annotated["count"] == {"value": 3, "basis": "measured-here"}
    assert annotated["error"] == {"value": 0.125, "basis": "measured-here"}
    assert annotated["passed"] is True and annotated["missing"] is None
    assert annotated["nested"]["method"] == "same frozen inputs"
    assert "basis" not in annotated["nested"]
    assert annotated["nested"]["tokens"] == [
        {"value": 64, "basis": "measured-here"},
        {"value": 1400, "basis": "measured-here"},
    ]
    assert annotated["existing"] == original["existing"]
    assert annotated["expected"]["num_layers"] == {"value": 34, "basis": "expected"}
    assert annotated["expected"]["known"] is True
    assert annotated["expected_from_header_config"]["hidden_size"]["basis"] == "expected"
    assert annotated["native_dtype_bound"] == {"value": 0.0, "basis": "expected"}
    assert annotated["projected_peak_gib"] == {"value": 9.8, "basis": "expected"}
    assert gates.annotate_numbers(annotated) == annotated


def test_number_annotation_respects_an_explicit_laptop_basis():
    annotated = gates.annotate_numbers({"max_abs": 0.0124, "bound": 1e-3}, basis="laptop-basis")
    assert annotated["max_abs"] == {"value": 0.0124, "basis": "laptop-basis"}
    assert annotated["bound"] == {"value": 1e-3, "basis": "expected"}


def test_write_report_annotates_only_the_opted_in_schema_without_changing_live_values(tmp_path):
    path = tmp_path / "annotated.json"
    report = {"number_schema": "value-basis-v1", "tokens": 64, "passed": False, "missing": None}
    gates.write_report(path, report)
    encoded = json.loads(path.read_text())
    assert encoded["tokens"] == {"value": 64, "basis": "measured-here"}
    assert encoded["passed"] is False and encoded["missing"] is None
    assert report["tokens"] == 64  # Live gate arithmetic still consumes ordinary numbers.
    generic = {"tokens": 64, "basis": "unrelated generic report"}
    gates.write_report(path, generic)
    assert json.loads(path.read_text()) == generic


def test_number_schema_still_refuses_nonfinite_json_before_replacing_a_good_record(tmp_path):
    path = tmp_path / "annotated.json"
    gates.write_report(path, {"number_schema": "value-basis-v1", "max_abs": 0.0})
    previous = path.read_text()
    with pytest.raises(ValueError):
        gates.write_report(path, {"number_schema": "value-basis-v1", "max_abs": float("nan")})
    assert path.read_text() == previous


def footprint_metadata(elements):
    return {
        "text": {"model.embed_tokens.weight": {"shape": [elements], "dtype": "BF16"}},
        # Deliberately understated: the fixture cap is derived from shapes after fp32 promotion.
        "text_bytes": 1,
    }


def test_cpu_fixture_cap_is_promoted_fp32_bytes_computed_before_loading():
    assert gates.CPU_FIXTURE_MAX_FP32_BYTES == 1024**2
    elements = gates.CPU_FIXTURE_MAX_FP32_BYTES // 4
    assert (
        gates.validate_execution_device(
            "cpu", fixture_cpu=True, metadata=footprint_metadata(elements)
        )
        == gates.CPU_FIXTURE_MAX_FP32_BYTES
    )
    with pytest.raises(RuntimeError, match="fixture|CPU|cpu"):
        gates.validate_execution_device(
            "cpu", fixture_cpu=True, metadata=footprint_metadata(elements + 1)
        )


@pytest.mark.parametrize(
    "selected,fixture_cpu",
    [
        ("cpu", False),
        ("mps", False),
        ("mps", True),
        ("cuda", True),
        ("cuda:2", True),
    ],
)
def test_cpu_real_models_mps_and_cuda_fixture_mode_are_refused(selected, fixture_cpu):
    with pytest.raises(RuntimeError):
        gates.validate_execution_device(
            selected, fixture_cpu=fixture_cpu, metadata=footprint_metadata(16)
        )


def test_cuda_scope_can_be_validated_without_running_cuda_or_using_fixture_bound():
    # This only validates metadata and scope; there is no torch import or device call.
    elements = gates.CPU_FIXTURE_MAX_FP32_BYTES
    assert (
        gates.validate_execution_device(
            "cuda:0", fixture_cpu=False, metadata=footprint_metadata(elements)
        )
        == elements * 4
    )


def test_tiny_fixture_mirrors_official_wrapper_declaration_and_both_foreign_towers(tmp_path):
    tiny_checkpoint(tmp_path)
    from local_llm_lab.hf_text import checkpoint_metadata

    metadata = checkpoint_metadata(tmp_path)
    config = metadata["config"]
    assert config["model_type"] == "gemma3"
    assert config["architectures"] == ["Gemma3ForConditionalGeneration"]
    assert config["text_config"]["model_type"] == "gemma3_text"
    assert config["vision_config"]["model_type"] == "siglip_vision_model"
    assert metadata["wrapper"] is True and metadata["safetensors_format"] == ["pt"]
    assert metadata["other_prefixes"] == ["multi_modal_projector", "vision_tower"]
    assert "language_model.model.embed_tokens.weight" in metadata["tensors"]
    assert "language_model.lm_head.weight" not in metadata["tensors"]
    assert gates.validate_execution_device("cpu", fixture_cpu=True, metadata=metadata) <= 1024**2


def resume_arguments():
    return {
        "source_commit": "a" * 40,
        "fingerprint": "b" * 64,
        "metadata": {"sha256": {"config.json": "c" * 64, "model.safetensors": "d" * 64}},
        "runtime": {"backend": "torch", "device": "cpu", "determinism": "pinned"},
        "inputs": {"token_sha256": "e" * 64, "lengths": [64, 1400]},
    }


def test_resume_uses_existing_identity_and_records_and_reads_numeric_wrappers(tmp_path):
    from research.acceptance.gate_records import GateRecords, Identity

    store = gates.resume_store(tmp_path / "units", **resume_arguments())
    assert isinstance(store, GateRecords) and isinstance(store.identity, Identity)
    store.write(2, {"status": "pass", "measurement": valid_float32_control(64)})
    reused = gates.resume_store(tmp_path / "units", **resume_arguments()).completed(2)
    assert reused["status"] == "pass"
    assert reused["measurement"]["tokens"] == 64
    assert reused["measurement"]["by_layer"]["1"]["max_norm_relative"] == 1e-3


@pytest.mark.parametrize(
    "changed", ["source_commit", "fingerprint", "checkpoint", "device", "tokens"]
)
def test_resume_refuses_mismatched_source_checkpoint_device_or_inputs(tmp_path, changed):
    arguments = resume_arguments()
    store = gates.resume_store(tmp_path / "units", **arguments)
    store.write(2, {"status": "pass", "measurement": valid_float32_control(64)})
    changed_arguments = copy.deepcopy(arguments)
    if changed in ("source_commit", "fingerprint"):
        changed_arguments[changed] = "f" * len(changed_arguments[changed])
    elif changed == "checkpoint":
        changed_arguments["metadata"]["sha256"]["model.safetensors"] = "f" * 64
    elif changed == "device":
        changed_arguments["runtime"]["device"] = "cuda:0"  # Identity data only, no device call.
    else:
        changed_arguments["inputs"]["token_sha256"] = "f" * 64
    resumed = gates.resume_store(tmp_path / "units", **changed_arguments)
    assert resumed.completed(2) is None
    assert 2 in resumed.refusals


@pytest.mark.parametrize("status", ["running", "measured", "fail", "unexecuted"])
def test_resume_never_uses_a_unit_that_did_not_complete_and_pass(tmp_path, status):
    store = gates.resume_store(tmp_path / "units", **resume_arguments())
    store.write(2, {"status": status, "measurement": valid_float32_control(64)})
    assert store.completed(2) is None
    assert 2 in store.refusals


def prepare_main_fixture(tmp_path, monkeypatch, *, selected="cpu"):
    """Actual tiny shared-loader objects; fake phase readings keep CLI tests inexpensive."""
    pytest.importorskip(
        "local_llm_lab.arch_torch", reason="tiny gate integration requires the selected upstream"
    )
    from local_llm_lab import device, hf_text, runlock

    root = tmp_path / "checkpoint"
    root.mkdir()
    tiny_checkpoint(root)
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps([index % 64 for index in range(1400)]))
    output = tmp_path / "device-record.json"
    events, loads, measures, model_references = [], [], [], []
    runtime = {
        "backend": "torch",
        "device": selected,
        "determinism": "pinned",
        "pinned": {"seed": 0, "deterministic": True, "attn_implementation": "eager"},
        "torch": "fixture-2.14",
        "threads": 1,
        "cuda_available": False,
        "extra_runtime_reading": {"marker": "keep full describe result"},
    }

    def require_pin(name):
        assert any(event[0] == "pin" for event in events), f"{name} happened before pin"
        events.append((name,))

    def select(prefer=None):
        events.append(("select", prefer))
        return selected

    def pin(seed=0, *, attention="eager", **kwargs):
        events.append(("pin", seed, attention))
        return copy.deepcopy(runtime)

    def describe():
        require_pin("describe")
        return copy.deepcopy(runtime)

    def information(*args, **kwargs):
        require_pin("device_info")
        return {"backend": "torch", "device": selected, "memory_size": 16 * gates.GIB}

    def memory(*args, **kwargs):
        require_pin("memory")
        return {
            "rss_bytes": 1024**2,
            "process_peak_bytes": 1024**2,
            "device_peak_bytes": 1024**2,
            "allocated_bytes": 1024**2,
        }

    def budget(*args, **kwargs):
        require_pin("budget")
        return 8 * gates.GIB

    def forbidden_box_claim(*args, **kwargs):
        raise AssertionError("tiny CPU fixture tests require neither a window nor a model lock")

    original_load = hf_text.load_text_causal_lm

    def load(path, **kwargs):
        require_pin("load")
        if loads:
            gc.collect()
            assert model_references[-1]() is None, "bf16 model/view survives into float32 load"
        model, audit = original_load(path, **kwargs)
        loads.append(kwargs["dtype"])
        model_references.append(weakref.ref(model))
        assert kwargs["device"] == "cpu"
        return model, audit

    def bf16(view, ids, checkpoint, measurement):
        measures.append(("bfloat16", len(ids)))
        measurement.update(valid_measurement(len(ids), num_layers=view.num_layers))
        checkpoint("fixture_bf16_complete")
        return measurement

    def fp32(view, ids, checkpoint, measurement):
        measures.append(("float32", len(ids)))
        measurement.update(valid_float32_control(len(ids), num_layers=view.num_layers))
        checkpoint("fixture_float32_complete")
        return measurement

    monkeypatch.setattr(device, "select", select)
    monkeypatch.setattr(device, "pin", pin)
    monkeypatch.setattr(device, "describe", describe)
    monkeypatch.setattr(device, "device_info", information)
    monkeypatch.setattr(device, "budget", budget)
    monkeypatch.setattr(device, "clear_cache", lambda *args, **kwargs: require_pin("clear_cache"))
    monkeypatch.setattr(device, "reset_peak", lambda *args, **kwargs: require_pin("reset_peak"))
    monkeypatch.setattr(gates, "memory_reading", memory)
    monkeypatch.setattr(gates, "verified_source_commit", lambda declared: declared)
    monkeypatch.setattr(gates, "source_fingerprint", lambda: ("b" * 64, {"fixture": "c" * 64}))
    monkeypatch.setattr(gates, "assert_own_window", forbidden_box_claim)
    monkeypatch.setattr(runlock, "hold_model_run_lock", forbidden_box_claim)
    monkeypatch.setattr(hf_text, "load_text_causal_lm", load)
    monkeypatch.setattr(gates, "measure_length", bf16)
    monkeypatch.setattr(gates, "measure_float32", fp32)
    # Restore even if main installs a temporary import guard or changes process environment.
    monkeypatch.setattr(builtins, "__import__", builtins.__import__)
    monkeypatch.setenv("LLL_BACKEND", "torch")
    monkeypatch.setenv("LLL_DEVICE", "cpu")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    argv = [
        "--execute",
        "--checkpoint",
        str(root),
        "--token-ids",
        str(ids_path),
        "--output",
        str(output),
        "--source-commit",
        "a" * 40,
        "--cap-gib",
        "1",
        "--projected-peak-gib",
        "0.1",
        "--fixture-cpu",
    ]
    return SimpleNamespace(
        argv=argv,
        output=output,
        root=root,
        events=events,
        loads=loads,
        measures=measures,
        runtime=runtime,
        model_references=model_references,
    )


def test_main_selects_and_pins_before_device_readings_and_serial_shared_loads(
    tmp_path, monkeypatch
):
    setup = prepare_main_fixture(tmp_path, monkeypatch)
    assert gates.main(setup.argv) == 0
    assert ("select", None) in setup.events  # --device is optional; the shared selector decides.
    assert setup.loads == ["bfloat16", "float32"]
    assert setup.measures == [
        ("bfloat16", 64),
        ("bfloat16", 1400),
        ("float32", 64),
        ("float32", 1400),
    ]
    encoded = json.loads(setup.output.read_text())
    assert encoded["number_schema"] == "value-basis-v1"
    assert encoded["measurements"][0]["tokens"] == {"value": 64, "basis": "measured-here"}
    written = gates.plain_numbers(encoded)
    assert written["device_description"] == setup.runtime
    assert written["source_commit"] == "a" * 40
    expected_config_sha = gates.sha256(setup.root / "config.json")
    assert written["checkpoint"]["sha256"]["config.json"] == expected_config_sha
    assert written["gates"]["2"]["status"] == "pass"
    assert written["float32_control"]["status"] == "pass"
    assert all(written["gates"][key]["status"] == "unexecuted" for key in ("3", "4"))
    assert [row["tokens"] for row in written["float32_measurements"]] == [64, 1400]


@pytest.mark.parametrize(
    "selected,fixture_cpu,oversized",
    [
        ("cpu", False, False),
        ("cpu", True, True),
        ("mps", False, False),
    ],
)
def test_main_refuses_laptop_model_scope_before_the_shared_loader(
    tmp_path, monkeypatch, selected, fixture_cpu, oversized
):
    from local_llm_lab import hf_text

    setup = prepare_main_fixture(tmp_path, monkeypatch, selected=selected)
    args = [value for value in setup.argv if value != "--fixture-cpu"]
    args += ["--device", selected]
    if fixture_cpu:
        args.append("--fixture-cpu")
    if oversized:
        metadata = hf_text.checkpoint_metadata(setup.root)
        metadata["text"]["model.embed_tokens.weight"]["shape"] = [1024**2]
        monkeypatch.setattr(hf_text, "checkpoint_metadata", lambda path: metadata)

    def forbidden_load(*args, **kwargs):
        pytest.fail("laptop scope must be refused before any checkpoint tensor load")

    monkeypatch.setattr(hf_text, "load_text_causal_lm", forbidden_load)
    with pytest.raises(RuntimeError, match="CPU|CUDA|fixture|MPS|mps"):
        gates.main(args)
    assert setup.loads == [] and setup.measures == []


def test_main_resume_reuses_completed_passing_units(tmp_path, monkeypatch):
    setup = prepare_main_fixture(tmp_path, monkeypatch)
    assert gates.main(setup.argv) == 0
    before = list(setup.measures)
    assert gates.main([*setup.argv, "--resume"]) == 0
    assert setup.measures == before
    unit_directory = setup.output.parent / (setup.output.stem + "-gates")
    stored_units = sorted(unit_directory.glob("gate-*.json"))
    assert stored_units
    for path in stored_units:
        assert json.loads(path.read_text())["status"] == "pass"

    # Preserve the identity and claimed pass, but invalidate the actual exactness evidence.
    first = unit_directory / "gate-01.json"
    tampered = gates.plain_numbers(json.loads(first.read_text()))
    native = tampered["measurement"]["precision_probe"]["native_dtype_loop"]
    native.update(exact=False, max_abs=1e-5, max_norm_relative=1e-5)
    native["by_layer"]["0"].update(max_abs=1e-5, max_norm_relative=1e-5)
    gates.write_report(first, tampered)
    assert gates.main([*setup.argv, "--resume"]) == 0
    assert setup.measures == [*before, ("bfloat16", 64)]
    written = gates.plain_numbers(json.loads(setup.output.read_text()))
    assert "reused" not in written["resume"]["1"]
    assert all("reused" in written["resume"][key] for key in ("2", "3", "4"))

    second = unit_directory / "gate-02.json"
    tampered = gates.plain_numbers(json.loads(second.read_text()))
    control = tampered["measurement"]["precision_probe"]["controls"]["mask_dispatch"]
    control["by_layer"]["1"]["actual_dtype"] = "torch.float32"
    gates.write_report(second, tampered)
    assert gates.main([*setup.argv, "--resume"]) == 0
    assert setup.measures == [*before, ("bfloat16", 64), ("bfloat16", 1400)]
    written = gates.plain_numbers(json.loads(setup.output.read_text()))
    assert "reused" not in written["resume"]["2"]
    assert all("reused" in written["resume"][key] for key in ("1", "3", "4"))


def test_float32_measurement_uses_promoted_arm_without_repeating_controls(monkeypatch):
    measurement, events = {}, []
    expected = valid_float32_control(64)

    def probe(view, ids, *, checkpoint, include_controls):
        assert include_controls is False
        checkpoint("native_dtype_loop", {"status": "measured", "max_abs": 0.0})
        checkpoint("promoted_fp32_loop", expected)
        return {
            "status": "measured",
            "native_dtype_loop": {"status": "measured", "max_abs": 0.0},
            "promoted_fp32_loop": expected,
        }

    monkeypatch.setitem(
        sys.modules,
        "research.acceptance.torch_seam",
        SimpleNamespace(residual_precision_probe=probe),
    )
    result = gates.measure_float32(
        SimpleNamespace(num_layers=1), list(range(64)), events.append, measurement
    )
    assert result is measurement and result["status"] == "measured"
    assert result["max_abs"] == 1e-3 and result["by_layer"] == expected["by_layer"]
    assert result["tokens"] == 64
    assert events == ["native_dtype_loop", "promoted_fp32_loop", "float32_comparison_complete"]


@pytest.mark.parametrize(
    "arm_name,field,dtype",
    [
        ("native_dtype_loop", "actual_dtype", "torch.float32"),
        ("native_dtype_loop", "native_dtype", "torch.float32"),
        ("promoted_fp32_loop", "actual_dtype", "torch.bfloat16"),
        ("promoted_fp32_loop", "native_dtype", "torch.float32"),
        ("mask_dispatch", "actual_dtype", "torch.float32"),
        ("entry_transform_omission", "native_dtype", "torch.float32"),
    ],
)
def test_bf16_gate_rejects_wrong_declared_comparison_precision(arm_name, field, dtype):
    rows = [valid_measurement(length) for length in (64, 1400)]
    probe = rows[1]["precision_probe"]
    arm = probe.get(arm_name, probe["controls"].get(arm_name))
    arm["by_layer"][1][field] = dtype
    assert gates.assess_gate2(rows, num_layers=1)["status"] == "fail"


def test_source_verification_failure_never_imports_or_calls_device_backend(tmp_path, monkeypatch):
    touched = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "local_llm_lab.device"
            or (name == "local_llm_lab" and "device" in (fromlist or ()))
            or name.split(".")[0] in ("mlx", "mlx_lm")
        ):
            touched.append(name)
            raise AssertionError("source refusal must precede all device/backend imports")
        return original_import(name, globals, locals, fromlist, level)

    def forbidden_device(*args, **kwargs):
        touched.append("device function")
        raise AssertionError("source refusal must precede device readings")

    # If another test imported the device module, block calls through that cached object too.
    cached_device = sys.modules.get("local_llm_lab.device")
    if cached_device is not None:
        for name in ("select", "pin", "describe", "backend", "device_info", "peak", "working_set"):
            monkeypatch.setattr(cached_device, name, forbidden_device)

    def reject_source(declared):
        raise ValueError("source verification refused")

    monkeypatch.setattr(gates, "verified_source_commit", reject_source)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    output = tmp_path / "refused.json"
    args = [
        "--execute",
        "--fixture-cpu",
        "--checkpoint",
        str(tmp_path),
        "--token-ids",
        str(tmp_path / "ids.json"),
        "--output",
        str(output),
        "--source-commit",
        "a" * 40,
        "--cap-gib",
        "1",
        "--projected-peak-gib",
        "0.1",
    ]
    with pytest.raises(ValueError, match="source verification refused"):
        gates.main(args)
    written = gates.plain_numbers(json.loads(output.read_text()))
    assert touched == []
    assert "final_memory_error" not in written
    assert written["memory"][-1]["phase"] == "final"
    assert written["error"]["message"] == "source verification refused"


def test_real_official_layout_fixture_runs_bf16_and_fp32_measurements_at_both_lengths(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    architecture = pytest.importorskip(
        "local_llm_lab.arch_torch", reason="tiny real probe requires the selected upstream"
    )
    TorchArchitectureView = architecture.TorchArchitectureView
    from local_llm_lab.hf_text import load_text_causal_lm
    from research.acceptance import torch_seam

    tiny_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(42)
    ids = torch.randint(0, 64, (1400,), generator=generator).tolist()
    original_probe = torch_seam.residual_precision_probe
    omitted_controls = []

    def observed_probe(view, ids, **kwargs):
        result = original_probe(view, ids, **kwargs)
        if kwargs.get("include_controls") is False:
            assert all(arm["status"] == "unexecuted" for arm in result["controls"].values())
            assert result["native_dtype_loop"]["max_abs"] == 0.0
            assert result["promoted_fp32_loop"]["max_abs"] == 0.0
            omitted_controls.append(len(ids))
        return result

    monkeypatch.setattr(torch_seam, "residual_precision_probe", observed_probe)
    bf16, fp32 = [], []
    for dtype, destination in (("bfloat16", bf16), ("float32", fp32)):
        model, audit = load_text_causal_lm(tmp_path, dtype=dtype, device="cpu")
        assert audit["architecture"] == "Gemma3ForCausalLM"
        view = TorchArchitectureView.from_model(model)
        with torch.no_grad():
            for length in (64, 1400):
                row = {}
                phases = []
                operation = gates.measure_length if dtype == "bfloat16" else gates.measure_float32
                operation(view, ids[:length], phases.append, row)
                assert phases
                destination.append(row)
        del view, model
        gc.collect()
    assert gates.assess_gate2(bf16, num_layers=3)["status"] == "pass"
    assert gates.assess_float32_control(fp32, num_layers=3)["status"] == "pass"
    assert omitted_controls == [64, 1400]


def test_resume_source_refusal_preserves_previous_report_bytes(tmp_path, monkeypatch):
    setup = prepare_main_fixture(tmp_path, monkeypatch)
    assert gates.main(setup.argv) == 0
    previous = setup.output.read_bytes()

    def reject(declared):
        raise ValueError("source commit does not match")

    monkeypatch.setattr(gates, "verified_source_commit", reject)
    with pytest.raises(ValueError, match="source commit"):
        gates.main([*setup.argv, "--resume"])
    assert setup.output.read_bytes() == previous


def test_explicit_host_cap_cannot_raise_shared_budget(tmp_path, monkeypatch):
    setup = prepare_main_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="host cap.*R47"):
        gates.main([*setup.argv, "--host-cap-gib", "9"])
    assert setup.loads == [] and setup.measures == []


def test_resume_retains_original_attempt_and_model_memory(tmp_path, monkeypatch):
    setup = prepare_main_fixture(tmp_path, monkeypatch)
    assert gates.main(setup.argv) == 0
    previous = setup.output.read_bytes()
    previous_sha = gates.sha256(setup.output)
    directory = setup.output.parent / (setup.output.stem + "-gates")
    resources = {
        str(index): gates.plain_numbers(
            json.loads((directory / f"gate-{index:02}.json").read_text())
        )["memory"]
        for index in range(1, 5)
    }
    assert all(resources.values())
    assert gates.main([*setup.argv, "--resume"]) == 0
    resumed = gates.plain_numbers(json.loads(setup.output.read_text()))
    assert Path(resumed["previous_attempt"]["path"]).read_bytes() == previous
    assert resumed["previous_attempt"]["sha256"] == previous_sha
    assert resumed["resumed_resources"] == resources
    assert setup.loads == ["bfloat16", "float32"]
    assert "loading" not in resumed


def test_resume_missing_original_memory_reruns_only_that_unit(tmp_path, monkeypatch):
    setup = prepare_main_fixture(tmp_path, monkeypatch)
    assert gates.main(setup.argv) == 0
    directory = setup.output.parent / (setup.output.stem + "-gates")
    path = directory / "gate-01.json"
    record = gates.plain_numbers(json.loads(path.read_text()))
    del record["memory"]
    gates.write_report(path, record)
    before = list(setup.measures)
    assert gates.main([*setup.argv, "--resume"]) == 0
    assert setup.measures == [*before, ("bfloat16", 64)]


@pytest.mark.parametrize("problem", ["missing", "wrong_phase", "above_cap", "negative", "nan"])
def test_resume_resource_evidence_requires_completed_unit_within_caps(problem):
    readings = [{"phase": "bfloat16/64/assessed", "device_peak_bytes": 5, "process_peak_bytes": 8}]
    assert gates.valid_resource_evidence(
        readings, assessed_phase="bfloat16/64/assessed", device_cap=10, host_cap=10
    )
    if problem == "missing":
        readings.clear()
    elif problem == "wrong_phase":
        readings[0]["phase"] = "before_load"
    else:
        readings[0]["device_peak_bytes"] = {"above_cap": 11, "negative": -1, "nan": float("nan")}[
            problem
        ]
    assert not gates.valid_resource_evidence(
        readings, assessed_phase="bfloat16/64/assessed", device_cap=10, host_cap=10
    )


def test_resume_source_fingerprint_includes_upstream_but_excludes_output_reports(
    tmp_path, monkeypatch
):
    from local_llm_lab import upstream_ref

    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("fixture")
    (root / "uv.lock").write_text("fixture")
    script = root / "script.py"
    script.write_text("fixture")
    reference = tmp_path / "reference"
    (reference / "jlens").mkdir(parents=True)
    source = reference / "jlens" / "hooks.py"
    source.write_text("first implementation")
    monkeypatch.setattr(gates, "ROOT", root)
    monkeypatch.setattr(gates, "__file__", str(script))
    monkeypatch.setattr(
        upstream_ref,
        "load_upstream",
        lambda: SimpleNamespace(
            path=reference, provenance=lambda: {"commit": "a", "path": str(reference)}
        ),
    )
    first, hashes = gates.source_fingerprint()
    assert hashes["upstream/jlens/hooks.py"] == gates.sha256(source)
    (root / "output.json").write_text('{"run": 1}')
    assert gates.source_fingerprint()[0] == first
    source.write_text("changed implementation")
    assert gates.source_fingerprint()[0] != first
