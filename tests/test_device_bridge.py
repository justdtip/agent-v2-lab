"""File-only runner: actual containers and captures, with numeric entry points guarded."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_device_lens_admission import BASE, api, digest, fit, snapshot, write_json
from tokenizers import Tokenizer, models, pre_tokenizers


def runner():
    name = "local_llm_lab.probes.device_bridge"
    assert importlib.util.find_spec(name), "device A1/A2 runner is absent"
    return importlib.import_module(name)


@pytest.fixture
def files(tmp_path, request):
    cp = snapshot(tmp_path)
    vocab = {
        s: i
        for i, s in enumerate(["[UNK]", "prompt", "note", "{", '"', "name", ":", "read_file", "}"])
    }
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(cp / "tokenizer.json"))
    save_file(
        {
            "model.embed_tokens.weight": np.arange(18, dtype=np.float32).reshape(9, 2),
            "model.norm.weight": np.zeros(2, np.float32),
        },
        cp / "model.safetensors",
        metadata={"format": "pt"},
    )
    source, maps = fit(tmp_path, cp)
    if getattr(request, "param", None) == "distinct_control":
        maps["J3"] = -np.eye(2, dtype=np.float32)
        np.savez_compressed(source / "exact-maps.npz", **maps)
    # The fit hash manifest identifies checkpoint tensors/config, not tokenizer auxiliaries.
    man = json.loads((source / "manifest.json").read_text())
    man["load_report_sha256"].pop("tokenizer.json")
    write_json(source / "manifest.json", man)
    meta = api().admit_device_lens(source, checkpoint=cp, model_base=BASE)
    sae = tmp_path / "dictionary"
    sae.mkdir()
    conf = {
        "architecture": "jump_relu",
        "type": "sae",
        "width": 1,
        "l0": 1,
        "model_name": BASE,
        "hf_hook_point_in": "model.layers.1.output",
        "hf_hook_point_out": "model.layers.1.output",
    }
    write_json(sae / "config.json", conf)
    save_file(
        {
            "w_enc": np.array([[1], [0]], np.float32),
            "w_dec": np.array([[1, 0]], np.float32),
            "b_enc": np.zeros(1, np.float32),
            "threshold": np.zeros(1, np.float32),
            "b_dec": np.zeros(2, np.float32),
        },
        sae / "params.safetensors",
    )
    # Gain is identity and rows increase with token id: independently known top tokens.
    save_file({"top_tokens": np.array([[8, 7]], np.int64)}, sae / "examples.safetensors")
    receipt = {
        "repo": "fixture/sae",
        "folder": "resid_post_all/layer_1_width_1_l0_small",
        "config": conf,
        "files": {
            p.name: {"algorithm": "sha256", "digest": digest(p), "bytes": p.stat().st_size}
            for p in sae.iterdir()
        },
    }
    write_json(sae / "DIGEST.json", receipt)
    corpus = tmp_path / "corpus.jsonl"
    row = {
        "prompt": "prompt " * 20,
        "completion": 'note {"name": "read_file"}',
        "metadata": {"task_id": "one", "step": 0, "family": "fixture", "variant": "base"},
    }
    corpus.write_text(json.dumps(row) + "\n")
    pids = tok.encode(row["prompt"], add_special_tokens=False).ids
    completion = tok.encode(row["completion"], add_special_tokens=False)
    act = next(
        i
        for i, (a, b) in enumerate(completion.offsets)
        if a <= row["completion"].index("read_file") < b
    )
    index = {
        "i": 0,
        **row["metadata"],
        "P_note": len(pids) - 1,
        "P_act": len(pids) + act - 1,
        "n_prompt_tokens": len(pids),
        "n_note_tokens": act,
    }
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "index.jsonl").write_text(json.dumps(index) + "\n")
    for side in ("note", "act"):
        np.save(capture / f"residual_{side}.npy", np.array([[[10, 1]] * 3], np.float32))
    manifest = {
        "schema_version": 1,
        "model": BASE,
        "checkpoint": "/original/device/path",
        "load_report_sha256": meta["checkpoint_files_sha256"],
        "decisions": 1,
        "rows_from": 0,
        "rows_to": 1,
        "corpus_decisions": 1,
        "memmap_row": "global index minus rows_from",
        "precision": "float32",
        "width": 1,
        "repo_layers": [1, 2, 3],
        "positions": ["P_note: last prompt token", "P_act: token before the tool-name token"],
        "tf32": {"matmul_allow_tf32": False, "float32_matmul_precision": "highest"},
    }
    write_json(capture / "manifest.json", manifest)
    events = [
        {"event": "corpus", "corpus_sha256": digest(corpus)},
        {"event": "loaded", "maps_sha256": meta["source_archive_sha256"]},
        {"event": "done", "decisions": 1},
    ]
    (capture / "progress.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    cells = [
        {
            "row": 0,
            "position": p,
            "token_index": index[p],
            "token_id": completion.ids[0 if p == "P_note" else act],
            "context_tokens": index["P_act"] + 2,
        }
        for p in ("P_note", "P_act")
    ]
    positions = {
        "schema_version": 1,
        "capture_files_sha256": {p.name: digest(p) for p in capture.iterdir()},
        "corpus_sha256": digest(corpus),
        "tokenizer_sha256": digest(cp / "tokenizer.json"),
        "cells": cells,
        "reduction": "per-position, no reduction",
        "endpoint": "pre-final-norm",
    }
    from local_llm_lab.probes import sae_bridge as B

    positions["domain_of_validity"] = {
        "statement": "Fixture-only declaration; no empirical claim about a device lens.",
        "lens_sha256": meta["npz_sha256"],
        "nu_sha256": B.nu_digest(meta["nu"]),
        "positions": sorted(c["token_index"] for c in cells),
        "context_tokens": sorted({c["context_tokens"] for c in cells}),
    }
    write_json(tmp_path / "positions.json", positions)
    pairs = []
    for cell in cells:
        reading = {
            "positions": [cell["token_index"]],
            "reduction": positions["reduction"],
            "endpoint": positions["endpoint"],
            "context_tokens": cell["context_tokens"],
        }
        pair = B.reading_identity(
            meta["nu"],
            lens_sha256=meta["npz_sha256"],
            capture_dtype="float32",
            capture_batch=1,
            reading=reading,
        )
        pairs.append({"pair": pair, "relative": 0.004, "basis": "fixture measurement"})
    write_json(tmp_path / "pairings.json", {BASE: {"measured_pairings": {"2": pairs}}})
    config = {
        "schema_version": 1,
        "model_base": BASE,
        "capture_model": BASE,
        "checkpoint_sha256": meta["checkpoint_files_sha256"],
        "lens_sha256": meta["npz_sha256"],
        "dictionary_repo": receipt["repo"],
        "dictionary_folder": receipt["folder"],
        "error_budget": {
            "raw_reconstruction_threshold": 0.5,
            "lens_score_error_threshold": 0.5,
            "denominator_floor": 1e-8,
            "near_zero_policy": "refuse",
        },
        "a1": {
            "k": 2,
            "chunk": 1,
            "control_layer": 3,
            "check_features": [0],
            "two_product_tolerance": 1e-4,
            "maximum_control_overlap": 0.9,
            "convention_threshold": 0.5,
        },
        "a2": {"k": 1, "identity_tolerance": 1e-3},
    }
    write_json(tmp_path / "config.json", config)
    return {
        "model_snapshot": cp,
        "dictionary_dir": sae,
        "dictionary_receipt": sae / "DIGEST.json",
        "lens_archive": source / "admitted-maps.npz",
        "capture_dir": capture,
        "positions": tmp_path / "positions.json",
        "corpus": corpus,
        "pairings": tmp_path / "pairings.json",
        "config": tmp_path / "config.json",
        "output_dir": tmp_path / "output",
        "stage": "a2",
    }


NUMERIC = (
    "encode",
    "decode",
    "raw_reconstruction_budget",
    "decompose_position",
    "feature_scores",
    "feature_score_column",
    "negative_control",
    "bridge_provenance",
    "load_dictionary",
    "load_unembedding",
)


def forbid_numeric(monkeypatch):
    from local_llm_lab.probes import sae_bridge as B

    def forbidden(*args, **kwargs):
        pytest.fail("numeric helper ran before admission completed")

    for name in NUMERIC:
        monkeypatch.setattr(B, name, forbidden)


def test_runner_exists():
    assert callable(runner().run_bridge)


def test_dry_run_checks_real_files_and_all_cells_before_any_numeric_helper(files, monkeypatch):
    forbid_numeric(monkeypatch)
    result = runner().run_bridge(**files, dry_run=True)
    assert result["status"] == "admitted-dry-run"
    assert set(result["checked"]) >= {
        "checkpoint",
        "lens",
        "dictionary",
        "capture",
        "positions",
        "pairings",
        "config",
    }
    assert len(result["provenance"]["readings"]) == 2
    assert "a2" not in result
    assert "no Neuronpedia source maps to resid_post_all" in result["provenance"]["labels_reason"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("checkpoint", "checkpoint"),
        ("lens", "hash|sha256"),
        ("dictionary", "digest"),
        ("receipt_scope", "folder"),
        ("manifest", "capture.*checkpoint"),
        ("array", "capture.*hash"),
        ("index", "P_act"),
        ("token", "token_id"),
        ("context", "context_tokens"),
        ("corpus", "corpus"),
        ("tokenizer", "tokenizer"),
        ("pairing", "context_tokens"),
        ("pairing_missing", "pairing"),
        ("budget", "error_budget|threshold"),
        ("precision", "precision"),
        ("layers", "repo_layers"),
        ("incomplete", "done"),
    ],
)
def test_bad_artifact_refuses_by_name_before_numeric(files, monkeypatch, mutation, match):
    forbid_numeric(monkeypatch)
    positions = json.loads(files["positions"].read_text())
    manifest = json.loads((files["capture_dir"] / "manifest.json").read_text())
    if mutation in ("checkpoint", "lens", "dictionary", "array", "corpus", "tokenizer"):
        path = {
            "checkpoint": files["model_snapshot"] / "model.safetensors",
            "lens": files["lens_archive"],
            "dictionary": files["dictionary_dir"] / "params.safetensors",
            "array": files["capture_dir"] / "residual_act.npy",
            "corpus": files["corpus"],
            "tokenizer": files["model_snapshot"] / "tokenizer.json",
        }[mutation]
        with path.open("ab") as f:
            f.write(b"changed")
    elif mutation == "receipt_scope":
        r = json.loads(files["dictionary_receipt"].read_text())
        r["folder"] = "another/layer"
        write_json(files["dictionary_receipt"], r)
    elif mutation in ("manifest", "precision", "layers"):
        if mutation == "manifest":
            manifest["load_report_sha256"]["config.json"] = "0" * 64
        elif mutation == "precision":
            manifest["precision"] = "bfloat16"
        else:
            manifest["repo_layers"] = [1, 1, 3]
        write_json(files["capture_dir"] / "manifest.json", manifest)
        positions["capture_files_sha256"]["manifest.json"] = digest(
            files["capture_dir"] / "manifest.json"
        )
    elif mutation == "index":
        p = files["capture_dir"] / "index.jsonl"
        r = json.loads(p.read_text())
        r["P_act"] += 1
        p.write_text(json.dumps(r) + "\n")
        positions["capture_files_sha256"]["index.jsonl"] = digest(p)
    elif mutation in ("token", "context"):
        positions["cells"][-1]["token_id" if mutation == "token" else "context_tokens"] += 1
    elif mutation == "incomplete":
        p = files["capture_dir"] / "progress.jsonl"
        p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
        positions["capture_files_sha256"]["progress.jsonl"] = digest(p)
    elif mutation.startswith("pairing"):
        r = json.loads(files["pairings"].read_text())
        if mutation == "pairing_missing":
            r[BASE]["measured_pairings"] = {}
        else:
            r[BASE]["measured_pairings"]["2"][-1]["pair"]["context_tokens"] += 1
        write_json(files["pairings"], r)
    elif mutation == "budget":
        r = json.loads(files["config"].read_text())
        del r["error_budget"]["lens_score_error_threshold"]
        write_json(files["config"], r)
    write_json(files["positions"], positions)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        runner().run_bridge(**files, dry_run=True)
    assert not files["output_dir"].exists()


def test_a2_computes_both_errors_and_writes_complete_provenance(files):
    result = runner().run_bridge(**files)
    assert result["status"] == "complete"
    assert len(result["a2"]) == 2
    for row in result["a2"]:
        assert row["raw_reconstruction_share"] == pytest.approx(1 / np.sqrt(101))
        assert "lens_score_error_share" in row
    assert result["provenance"]["checkpoint"]["sha256"]
    assert result["provenance"]["labels"] == "unlabelled"
    assert result["interpretation_limits"][0].startswith("A large feature contribution")
    assert json.loads((files["output_dir"] / "result.json").read_text()) == result


def test_a1_dry_run_requires_shipped_receipt_and_control_without_capture(files, monkeypatch):
    forbid_numeric(monkeypatch)
    files.update(stage="a1", capture_dir=None, positions=None, corpus=None, pairings=None)
    result = runner().run_bridge(**files, dry_run=True)
    assert "examples" in result["checked"]


def test_a1_records_discriminator_two_products_and_failed_control(files):
    files.update(stage="a1", capture_dir=None, positions=None, corpus=None, pairings=None)
    result = runner().run_bridge(**files)
    assert result["status"] == "refused-numeric-check"
    assert result["a1"]["two_products"]["passed"]
    assert result["a1"]["convention_check"]["raw_overlap"] == 1
    assert not result["a1"]["negative_control"]["passed"]
    assert result["a1"]["readout_heading"] == "Our top-token readout at repository layer 2"


@pytest.mark.parametrize("files", ["distinct_control"], indirect=True)
def test_a1_success_with_distinct_layer_control(files):
    files.update(stage="a1", capture_dir=None, positions=None, corpus=None, pairings=None)
    result = runner().run_bridge(**files)
    assert result["status"] == "complete"
    assert result["a1"]["negative_control"]["mean_overlap"] == 0
    assert result["a1"]["two_products"]["top_token_membership_matches"]


def test_a1_missing_examples_receipt_refuses_in_dry_run(files, monkeypatch):
    forbid_numeric(monkeypatch)
    receipt = json.loads(files["dictionary_receipt"].read_text())
    del receipt["files"]["examples.safetensors"]
    write_json(files["dictionary_receipt"], receipt)
    files.update(stage="a1", capture_dir=None, positions=None, corpus=None, pairings=None)
    with pytest.raises(ValueError, match="digest receipt missing examples"):
        runner().run_bridge(**files, dry_run=True)


@pytest.mark.parametrize("stage", ["a1", "a2"])
def test_final_dictionary_a1_admits_and_absent_final_capture_refuses(files, monkeypatch, stage):
    forbid_numeric(monkeypatch)
    path = files["dictionary_dir"] / "config.json"
    config = json.loads(path.read_text())
    config.update(
        hf_hook_point_in="model.layers.3.output", hf_hook_point_out="model.layers.3.output"
    )
    write_json(path, config)
    receipt = json.loads(files["dictionary_receipt"].read_text())
    receipt["config"] = config
    receipt["folder"] = receipt["folder"].replace("layer_1", "layer_3")
    receipt["files"]["config.json"].update(digest=digest(path), bytes=path.stat().st_size)
    write_json(files["dictionary_receipt"], receipt)
    config = json.loads(files["config"].read_text())
    config["dictionary_folder"] = receipt["folder"]
    write_json(files["config"], config)
    files["stage"] = stage
    if stage == "a1":
        files.update(capture_dir=None, positions=None, corpus=None, pairings=None)
        assert runner().run_bridge(**files, dry_run=True)["status"] == "admitted-dry-run"
    else:
        with pytest.raises(ValueError, match="repo_layers missing dictionary layer 4"):
            runner().run_bridge(**files, dry_run=True)


def test_two_product_check_rejects_wrong_top_ids_even_with_correct_scores(files, monkeypatch):
    from local_llm_lab.probes import sae_bridge as B

    original = B.feature_scores

    def wrong_top(dictionary, unembedding, lens_map, **kwargs):
        if lens_map is not None:
            ids = np.array([[0, 1]])
            scores = B.feature_score_column(dictionary, unembedding, lens_map, 0)[ids]
            return ids, scores
        return original(dictionary, unembedding, lens_map, **kwargs)

    monkeypatch.setattr(B, "feature_scores", wrong_top)
    files.update(stage="a1", capture_dir=None, positions=None, corpus=None, pairings=None)
    result = runner().run_bridge(**files)
    assert not result["a1"]["two_products"]["passed"]


@pytest.mark.parametrize(
    "field",
    [
        "fit_dtype",
        "fit_width",
        "capture_dtype",
        "capture_width",
        "lens_side",
        "nu_sha256",
        "lens_sha256",
        "positions",
        "reduction",
        "endpoint",
        "context_tokens",
    ],
)
def test_each_pairing_field_is_bound_before_numbers(files, monkeypatch, field):
    forbid_numeric(monkeypatch)
    table = json.loads(files["pairings"].read_text())
    for record in table[BASE]["measured_pairings"]["2"]:
        value = record["pair"][field]
        record["pair"][field] = (
            value + 1 if type(value) is int else [999] if isinstance(value, list) else "changed"
        )
    write_json(files["pairings"], table)
    with pytest.raises(ValueError, match=field):
        runner().run_bridge(**files, dry_run=True)


@pytest.mark.parametrize("statement", [None, 42, {}, "  "])
def test_missing_domain_statement_refuses_outside_fit_context(files, monkeypatch, statement):
    forbid_numeric(monkeypatch)
    seal = json.loads(files["positions"].read_text())
    seal["domain_of_validity"]["statement"] = statement
    write_json(files["positions"], seal)
    with pytest.raises(ValueError, match="domain_of_validity"):
        runner().run_bridge(**files, dry_run=True)


def test_failed_discriminator_stops_before_lens_readout(files, monkeypatch):
    from local_llm_lab.probes import sae_bridge as B

    shipped = files["dictionary_dir"] / "examples.safetensors"
    save_file({"top_tokens": np.array([[2, 3]], np.int64)}, shipped)
    receipt = json.loads(files["dictionary_receipt"].read_text())
    receipt["files"][shipped.name]["digest"] = digest(shipped)
    receipt["files"][shipped.name]["bytes"] = shipped.stat().st_size
    write_json(files["dictionary_receipt"], receipt)
    original = B.feature_scores

    def no_lens_after_stop(dictionary, unembedding, lens_map, **kwargs):
        assert lens_map is None, "lens readout ran after discriminator stopped"
        return original(dictionary, unembedding, lens_map, **kwargs)

    monkeypatch.setattr(B, "feature_scores", no_lens_after_stop)
    files.update(stage="a1", capture_dir=None, positions=None, corpus=None, pairings=None)
    result = runner().run_bridge(**files)
    assert result["status"] == "refused-numeric-check"
    assert result["a1"]["convention_check"]["stop"]
    assert "top_tokens" not in result["a1"]


def test_index_cannot_redirect_readout_to_unverified_weights(files, monkeypatch):
    forbid_numeric(monkeypatch)
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    write_json(
        files["model_snapshot"] / "model.safetensors.index.json",
        {"weight_map": {name: "../unverified.safetensors" for name in names}},
    )
    with pytest.raises(ValueError, match="checkpoint.*index"):
        runner().run_bridge(**files, dry_run=True)


def test_cli_dry_run_reads_explicit_paths_and_returns_own_exit_code(files):
    import os
    import sys
    from pathlib import Path

    from local_llm_lab import spawn

    args = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/device_sae_bridge.py"),
    ]
    for key, value in files.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = spawn.run([*args, "--dry-run"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "admitted-dry-run"
