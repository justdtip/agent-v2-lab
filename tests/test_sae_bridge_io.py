"""Tiny file fixtures; importing a model framework is never part of this suite."""

import hashlib
import json
import runpy
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors import SafetensorError
from safetensors.numpy import save_file

from local_llm_lab.pipeline.sae_bridge.assets import (
    decoder_from_files,
    read_bf16_matrix,
    validate_reference_config,
)
from local_llm_lab.pipeline.sae_bridge.readout import acceptance, write_readout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16_file(path, values, key="weight"):
    values = np.asarray(values, dtype=np.float32)
    bits = (values.view(np.uint32) >> 16).astype("<u2")
    header = json.dumps(
        {key: {"dtype": "BF16", "shape": list(values.shape), "data_offsets": [0, bits.nbytes]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + bits.tobytes())


def test_bf16_reader_exact_bits_across_chunks(tmp_path):
    path = tmp_path / "weight.safetensors"
    values = np.array([[0.0, -0.0], [1.5, -2.0], [256.0, 0.125]], dtype=np.float32)
    bf16_file(path, values)
    actual = read_bf16_matrix(path, "weight", expected_sha256=sha(path), chunk_rows=1)
    np.testing.assert_array_equal(actual.view(np.uint32), values.view(np.uint32))
    assert actual.dtype == np.float32


def test_bf16_reader_refuses_wrong_hash_dtype_and_nonfinite(tmp_path):
    path = tmp_path / "weight.safetensors"
    bf16_file(path, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="hash"):
        read_bf16_matrix(path, "weight", expected_sha256="0" * 64)
    save_file({"weight": np.ones((1, 2), dtype=np.float32)}, str(path))
    with pytest.raises(ValueError, match="BF16"):
        read_bf16_matrix(path, "weight", expected_sha256=sha(path))
    bf16_file(path, [[np.inf, 1.0]])
    with pytest.raises(ValueError, match="nonfinite"):
        read_bf16_matrix(path, "weight", expected_sha256=sha(path))


def test_bf16_reader_refuses_malformed_container(tmp_path):
    path = tmp_path / "bad.safetensors"
    bf16_file(path, [[1.0, 2.0]])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(SafetensorError):
        read_bf16_matrix(path, "weight", expected_sha256=sha(path))


def test_reference_config_requires_unquantised_bf16_and_dimensions():
    config = {
        "torch_dtype": "bfloat16",
        "text_config": {"num_hidden_layers": 3, "hidden_size": 2, "vocab_size": 5},
    }
    assert validate_reference_config(config) == (3, 2, 5)
    for change in [{"torch_dtype": "float32"}, {"quantization": {"bits": 4}}]:
        with pytest.raises(ValueError):
            validate_reference_config(config | change)
    with pytest.raises(ValueError):
        validate_reference_config(
            config | {"text_config": config["text_config"] | {"quantization_config": {"bits": 4}}}
        )


@pytest.fixture
def dictionary(tmp_path):
    config = tmp_path / "config.json"
    params = tmp_path / "params.safetensors"
    metadata = {
        "model_name": "test/base",
        "hf_hook_point_in": "model.layers.1.output",
        "hf_hook_point_out": "model.layers.1.output",
        "width": 3,
        "l0": 2,
        "architecture": "jump_relu",
        "type": "sae",
        "affine_connection": False,
    }
    config.write_text(json.dumps(metadata))
    save_file(
        {
            "w_dec": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32),
            "w_enc": np.ones((2, 3), np.float32),
        },
        str(params),
    )
    return config, params, metadata


def load_dictionary(config, params, **changes):
    options = dict(
        expected_config_sha256=sha(config),
        expected_params_sha256=sha(params),
        base="test/base",
        probe_layer=2,
        hidden_size=2,
        width=3,
        l0=2,
        coordinate_convention="gemma_scope2_raw_resid_post_no_rescale",
    )
    return decoder_from_files(config, params, **(options | changes))


def test_decoder_transposes_once_without_normalising(dictionary):
    config, params, _ = dictionary
    actual = load_dictionary(config, params)
    np.testing.assert_array_equal(actual, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])


@pytest.mark.parametrize(
    "change",
    [
        {"base": "different/base"},
        {"probe_layer": 1},
        {"width": 4},
        {"l0": 1},
        {"coordinate_convention": "unit_norm"},
        {"expected_params_sha256": "0" * 64},
    ],
)
def test_decoder_preconditions_fail_closed(dictionary, change):
    config, params, _ = dictionary
    with pytest.raises(ValueError):
        load_dictionary(config, params, **change)


def test_decoder_rejects_wrong_output_hook_even_if_input_matches(dictionary):
    config, params, metadata = dictionary
    config.write_text(json.dumps(metadata | {"hf_hook_point_out": "model.layers.0.output"}))
    with pytest.raises(ValueError, match="hook"):
        load_dictionary(config, params)


def test_acceptance_control_bites_through_actual_composition():
    w = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    j = np.eye(2, dtype=np.float32)
    wrong = -j
    d = np.array([[1.0, 2.0], [0.0, 1.0]], np.float32)
    evidence, jd = acceptance(
        w, j, wrong, d, feature_id=0, k=1, rtol=1e-4, atol=1e-3, relative_l2_limit=1e-4
    )
    assert evidence["passed"]
    assert evidence["positive"]["passed"]
    assert not evidence["wrong_layer"]["passed"]
    assert evidence["topk_symmetric_difference"] == 2
    np.testing.assert_array_equal(jd, d)
    unmet, _ = acceptance(
        w, j, j, d, feature_id=0, k=1, rtol=1e-4, atol=1e-3, relative_l2_limit=1e-4
    )
    assert not unmet["passed"]
    assert unmet["topk_symmetric_difference"] == 0


def test_artifact_is_exclusive_stream_and_carries_missing_labels(tmp_path):
    path = tmp_path / "readout.jsonl"
    w = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    jd = np.eye(2, dtype=np.float32)
    meta = {"lens_sha256": "x", "label_status": "unavailable_no_verified_dictionary_mapping"}
    write_readout(path, w, jd, k=1, metadata=meta, token_piece=lambda t: f"token{t}")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["type"] == "header"
    assert "not token probabilities" in rows[0]["interpretation"]
    assert [row["feature_id"] for row in rows[1:-1]] == [0, 1]
    assert rows[1]["label"] is None
    assert rows[1]["tokens"][0]["piece"] == "token0"
    assert rows[-1]["type"] == "complete"
    assert rows[-1]["features"] == 2
    previous = "0" * 64
    for row in rows:
        digest = row.pop("sha256")
        assert row["previous_sha256"] == previous
        assert (
            digest
            == hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        )
        previous = digest
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_readout(path, w, jd, k=1, metadata=meta, token_piece=str)
    assert path.read_bytes() == before


def test_no_completion_footer_on_interrupted_readout(tmp_path):
    path = tmp_path / "incomplete.jsonl"
    w = np.eye(2, dtype=np.float32)

    def broken(_):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        write_readout(path, w, w, k=1, metadata={}, token_piece=broken)
    assert not any(json.loads(line)["type"] == "complete" for line in path.read_text().splitlines())


def test_driver_tiny_end_to_end_and_wrong_precision_guard(tmp_path, dictionary, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers import models as tokenizer_models

    from local_llm_lab import models
    from local_llm_lab.pipeline.live_lens.instruments import LENS_IDENTITY_KEY

    script = Path(__file__).resolve().parents[1] / "scripts/sae_bridge.py"
    main = runpy.run_path(str(script))["main"]
    main.__globals__["require_window"] = lambda: None  # Only this tiny in-process fixture.
    monkeypatch.setattr(
        models,
        "load_model_spec",
        lambda _: SimpleNamespace(base="test/base", training=None, hf_id=str(tmp_path)),
    )
    config = {
        "model_type": "gemma3",
        "torch_dtype": "bfloat16",
        "text_config": {"num_hidden_layers": 3, "hidden_size": 2, "vocab_size": 3},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensor_key = "language_model.model.embed_tokens.weight"
    shard = tmp_path / "model.safetensors"
    bf16_file(shard, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], key=tensor_key)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {tensor_key: shard.name}}))
    tokenizer_path = tmp_path / "tokenizer.json"
    Tokenizer(tokenizer_models.WordLevel({"a": 0, "b": 1, "c": 2}, unk_token="a")).save(
        str(tokenizer_path)
    )
    identity = {"base": "test/base", "num_layers": 3, "training": None}
    lens = tmp_path / "lens.npz"
    np.savez(
        lens,
        J0=-np.eye(2, dtype=np.float32),
        J1=np.eye(2, dtype=np.float32),
        **{LENS_IDENTITY_KEY: np.frombuffer(json.dumps(identity).encode(), dtype=np.uint8)},
    )
    _, params, metadata = dictionary
    dictionary_config = tmp_path / "dictionary.json"
    dictionary_config.write_text(json.dumps(metadata))
    reg = {
        "reference_model": "fixture",
        "model_identity": identity,
        "hidden_size": 2,
        "reference_config_sha256": sha(tmp_path / "config.json"),
        "readout": {
            "index": index.name,
            "index_sha256": sha(index),
            "key": tensor_key,
            "shard": shard.name,
            "shard_sha256": sha(shard),
        },
        "tokenizer_sha256": sha(tokenizer_path),
        "lens": {"path": str(lens), "sha256": sha(lens), "fitting_precision": "bf16"},
        "probe_layer": 2,
        "wrong_probe_layer": 1,
        "feature_id": 0,
        "k": 1,
        "dictionary": {
            "suite": "resid_post_all",
            "config": str(dictionary_config),
            "config_sha256": sha(dictionary_config),
            "params": str(params),
            "params_sha256": sha(params),
            "width": 3,
            "l0": 2,
            "coordinate_convention": "gemma_scope2_raw_resid_post_no_rescale",
        },
        "tolerances": {"rtol": 1e-4, "atol": 1e-3, "relative_l2_limit": 1e-4},
        "projected_peak_gib": 6,
        "peak_basis": "tiny synthetic fixture",
    }
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps(reg))
    assert main(["--registration", str(registration)]) == 0
    benchmark_dir = tmp_path / "benchmark"
    assert (
        main(["--registration", str(registration), "--execute", "--output", str(benchmark_dir)])
        == 0
    )
    result = json.loads((benchmark_dir / "result.json").read_text())
    assert result["model_forward_count"] == 0
    assert result["instrument"] == reg
    evidence = json.loads((benchmark_dir / "acceptance.json").read_text())
    assert evidence["registration_sha256"] == sha(registration)
    assert evidence["instrument"] == reg
    output = tmp_path / "readout"
    assert (
        main(
            [
                "--registration",
                str(registration),
                "--execute",
                "--mode",
                "readout",
                "--benchmark",
                str(benchmark_dir / "result.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads((output / "features.jsonl").read_text().splitlines()[-1])["features"] == 3
    reg["lens"]["fitting_precision"] = "4bit"
    with pytest.raises(ValueError, match="BF16"):
        main.__globals__["load_inputs"](reg)
    reg["lens"]["fitting_precision"] = "bf16"
    reg["tolerances"]["atol"] = 1000  # The control cannot fail these deliberately loose bounds.
    reg["tolerances"]["relative_l2_limit"] = 1000
    registration.write_text(json.dumps(reg))
    failed = tmp_path / "failed"
    assert main(["--registration", str(registration), "--execute", "--output", str(failed)]) == 2
    evidence = json.loads((failed / "acceptance.json").read_text())
    assert not evidence["passed"]
    assert evidence["registration_sha256"] == sha(registration)
    assert evidence["instrument"] == reg
    assert not (failed / "result.json").exists()
