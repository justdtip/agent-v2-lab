"""T5 CPU contract: real tiny Torch forwards and sealed synthetic workspace captures."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from test_device_bridge import files as workspace_files
from test_device_lens_admission import BASE, api, digest, fit, write_json

from local_llm_lab import spawn
from local_llm_lab.probes import device_bridge as D
from local_llm_lab.probes import sae_bridge as B

files = workspace_files


def measurement():
    name = "local_llm_lab.probes.measure_pairings"
    assert importlib.util.find_spec(name), "T5 pairing measurement producer is absent"
    return importlib.import_module(name)


class Block(torch.nn.Module):
    def forward(self, h):
        return h + 1 + (h.shape[0] - 1) / 8


class Tiny(torch.nn.Module):
    """Two coordinates identify input token and position; batch changes the arithmetic path."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(4)])
        self.input_device = "cpu"

    def forward(self, ids=None, *, input_ids=None, output_attentions=False, logits_to_keep=None):
        ids = input_ids if input_ids is not None else ids
        h = torch.stack(
            (ids.float() + 1, torch.arange(ids.shape[1]).expand_as(ids).float() + 1), -1
        )
        for block in self.layers:
            h = block(h)
        return h


def reseal(args):
    seal = D._json(args["positions"])
    seal["capture_files_sha256"] = {
        name: digest(args["capture_dir"] / name) for name in D.CAPTURE_FILES
    }
    write_json(args["positions"], seal)


@pytest.fixture
def case(files, tmp_path):
    corpus = D._rows(files["corpus"])
    # Include corpus explicitly; never follow the device's historical path on the laptop.
    manifest = D._json(files["capture_dir"] / "manifest.json")
    manifest["device"] = "cpu"
    manifest["sample"] = []
    write_json(files["capture_dir"] / "manifest.json", manifest)
    index_path = files["capture_dir"] / "index.jsonl"
    index = D._rows(index_path)[0]
    index["in_sample"] = False
    index_path.write_text(json.dumps(index) + "\n")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(files["model_snapshot"] / "tokenizer.json"))
    row = corpus[0]
    ids = tok.encode(row["prompt"], add_special_tokens=False).ids
    ids += tok.encode(row["completion"], add_special_tokens=False).ids
    cells = D._json(files["positions"])["cells"]
    for cell in cells:
        position = cell["token_index"]
        # Independent arithmetic oracle for the width-1 model, not the new recording helper.
        h = np.array([ids[position] + 1, position + 1], dtype=np.float32)
        np.save(
            files["capture_dir"] / f"residual_{cell['position'][2:]}.npy",
            np.stack([h + layer for layer in [1, 2, 3]])[None],
        )
    reseal(files)
    checkpoint_manifest = tmp_path / "checkpoint-hashes.json"
    write_json(checkpoint_manifest, manifest["load_report_sha256"])
    return {
        "model_snapshot": files["model_snapshot"],
        "checkpoint_manifest": checkpoint_manifest,
        "lens_archive": files["lens_archive"],
        "lens_sidecar": files["lens_archive"].with_suffix(".json"),
        "capture_dir": files["capture_dir"],
        "positions": files["positions"],
        "corpus": files["corpus"],
        "layers": [1, 2, 3],
        "output": tmp_path / "measured.json",
    }


def install_tiny(monkeypatch, case, model=None):
    M = measurement()
    calls = []

    def load(snapshot, manifest):
        calls.append(snapshot)
        tiny = model or Tiny()
        return M.TorchResidualPath(tiny, capture_model=tiny), {
            "checkpoint_sha256": D._json(case["checkpoint_manifest"]),
            "torch_version": torch.__version__,
            "device": "cpu",
            "attention": "eager",
        }

    monkeypatch.setattr(M, "_load_path", load)
    return calls


def test_intact_capture_produces_cell_layer_registrations(case, monkeypatch):
    calls = install_tiny(monkeypatch, case)
    M = measurement()
    result = M.measure_pairings(**case)
    assert len(calls) == 1
    assert D._json(case["output"]) == result
    B._supplied_pairing_table(result)
    tables = D._pairing_tables(case["output"])
    cells = D._json(case["positions"])["cells"]
    for layer in case["layers"]:
        records = result[BASE]["measured_pairings"][str(layer)]
        assert len(records) == len(cells)
        for cell, record in zip(cells, records, strict=True):
            assert set(record) == {"pair", "relative", "basis"}
            assert set(record["pair"]) == set(B.PAIR_FIELDS)
            assert record["pair"]["positions"] == [cell["token_index"]]
            assert record["pair"]["fit_width"] == 2
            # width 2 shifts each coordinate by layer/8 relative to width 1.
            stored = np.load(case["capture_dir"] / f"residual_{cell['position'][2:]}.npy")
            expected = np.sqrt(2) * layer / 8 / np.linalg.norm(stored[0, layer - 1].astype(float))
            assert record["relative"] == pytest.approx(expected)
            selected = D._cell_pairing(tables, BASE, layer, record["pair"], cell=cell)
            assert (
                B.path_pairing(layer, base=BASE, identity=record["pair"], pairing_table=selected)[0]
                == record
            )
    provenance = result["_provenance"]
    assert provenance["positions_sha256"] == digest(case["positions"])
    assert provenance["checkpoint_sha256"] == D._json(case["checkpoint_manifest"])
    assert len(provenance["source_commit"]) == 40
    assert provenance["runtime"]["torch_version"] == torch.__version__


@pytest.mark.parametrize(
    "damage", ["checkpoint", "sidecar", "capture", "tokenizer", "positions", "manifest"]
)
def test_hash_mismatch_before_model_load(case, monkeypatch, damage):
    calls = install_tiny(monkeypatch, case)
    path = {
        "checkpoint": case["model_snapshot"] / "config.json",
        "sidecar": case["lens_sidecar"],
        "capture": case["capture_dir"] / "index.jsonl",
        "tokenizer": case["model_snapshot"] / "tokenizer.json",
        "positions": case["positions"],
        "manifest": case["capture_dir"] / "manifest.json",
    }[damage]
    if damage == "sidecar":
        data = D._json(path)
        data["nu_sha256"] = "0" * 64
        write_json(path, data)
    elif damage == "positions":
        data = D._json(path)
        data["capture_files_sha256"]["manifest.json"] = "0" * 64
        write_json(path, data)
    else:
        path.write_text(path.read_text() + ("broken JSON" if damage == "manifest" else " "))
    with pytest.raises(ValueError, match="hash_mismatch"):
        measurement().measure_pairings(**case)
    assert not calls
    assert not case["output"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("row", 1),
        ("position", "P_other"),
        ("token_index", 0),
        ("token_id", 100),
        ("context_tokens", 2),
    ],
)
def test_bad_cell_before_model_load(case, monkeypatch, field, value):
    calls = install_tiny(monkeypatch, case)
    data = D._json(case["positions"])
    data["cells"][-1][field] = value
    write_json(case["positions"], data)
    with pytest.raises(ValueError, match="cell_outside_capture"):
        measurement().measure_pairings(**case)
    assert not calls


@pytest.mark.parametrize(
    "value,reason", [(123.0, "capture_residual_mismatch"), (float("nan"), "non_finite_values")]
)
def test_resealed_bad_residual_refuses(case, monkeypatch, value, reason):
    install_tiny(monkeypatch, case)
    path = case["capture_dir"] / "residual_act.npy"
    data = np.load(path)
    data[0, -1, -1] = value
    np.save(path, data)
    reseal(case)
    with pytest.raises(ValueError, match=reason):
        measurement().measure_pairings(**case)
    assert not case["output"].exists()


def test_existing_output_is_never_touched(case, monkeypatch):
    calls = install_tiny(monkeypatch, case)
    case["output"].write_bytes(b"preserve")
    with pytest.raises(ValueError, match="existing_output"):
        measurement().measure_pairings(**case)
    assert case["output"].read_bytes() == b"preserve"
    assert not calls


def change_fit(case, tmp_path, widths):
    chunks = []
    for i, width in enumerate(widths):
        source, _ = fit(
            tmp_path, case["model_snapshot"], name=f"chunk{i}", rows=(i * 2, i * 2 + 1), width=width
        )
        manifest = D._json(source / "manifest.json")
        manifest["load_report_sha256"].pop("tokenizer.json")
        write_json(source / "manifest.json", manifest)
        chunks.append(source)
    if len(chunks) > 1:
        source = tmp_path / "merge"
        api().merge_device_lens_chunks(source, chunks)
    meta = api().admit_device_lens(source, checkpoint=case["model_snapshot"], model_base=BASE)
    case["lens_archive"] = source / "admitted-maps.npz"
    case["lens_sidecar"] = source / "admitted-maps.json"
    progress = case["capture_dir"] / "progress.jsonl"
    events = D._rows(progress)
    for event in events:
        if event["event"] == "loaded":
            event["maps_sha256"] = meta["source_archive_sha256"]
    progress.write_text("".join(json.dumps(e) + "\n" for e in events))
    reseal(case)


@pytest.mark.parametrize("widths", [[1], [32], [16, 16, 8]])
def test_schedules_from_admission_keep_each_chunk(case, monkeypatch, tmp_path, widths):
    change_fit(case, tmp_path, widths)
    install_tiny(monkeypatch, case)
    result = measurement().measure_pairings(**case)
    for layer in case["layers"]:
        for record, receipt in zip(
            result[BASE]["measured_pairings"][str(layer)],
            result["_provenance"]["measurements"][str(layer)],
            strict=True,
        ):
            assert record["pair"]["fit_width"] == (widths[0] if len(widths) == 1 else widths)
            terms = receipt["chunk_terms"]
            assert [term["forward_batch"] for term in terms] == widths
            assert record["relative"] == max(term["relative"] for term in terms)
            if widths == [1]:
                assert record["relative"] == 0
            else:
                assert all(term["relative"] > 0 for term in terms)
            if len(widths) == 3:
                assert terms[0]["relative"] == terms[1]["relative"]
                assert terms[0]["relative"] > terms[2]["relative"]


def test_equal_length_prompts_keep_separate_cell_measurements(case, monkeypatch, files):
    row = D._rows(case["corpus"])[0]
    row2 = {**row, "metadata": {**row["metadata"], "task_id": "two"}}
    case["corpus"].write_text(json.dumps(row) + "\n" + json.dumps(row2) + "\n")
    index = D._rows(case["capture_dir"] / "index.jsonl")[0]
    second = {**index, "i": 1, "task_id": "two"}
    (case["capture_dir"] / "index.jsonl").write_text(
        json.dumps(index) + "\n" + json.dumps(second) + "\n"
    )
    manifest = D._json(case["capture_dir"] / "manifest.json")
    manifest.update(decisions=2, rows_to=2, corpus_decisions=2)
    write_json(case["capture_dir"] / "manifest.json", manifest)
    for name in ("note", "act"):
        path = case["capture_dir"] / f"residual_{name}.npy"
        np.save(path, np.repeat(np.load(path), 2, axis=0))
    progress = case["capture_dir"] / "progress.jsonl"
    events = D._rows(progress)
    events[0]["corpus_sha256"] = digest(case["corpus"])
    events[-1]["decisions"] = 2
    progress.write_text("".join(json.dumps(e) + "\n" for e in events))
    seal = D._json(case["positions"])
    seal["corpus_sha256"] = digest(case["corpus"])
    seal["cells"] += [{**cell, "row": 1} for cell in seal["cells"]]
    write_json(case["positions"], seal)
    reseal(case)
    install_tiny(monkeypatch, case)
    result = measurement().measure_pairings(**case)
    B._supplied_pairing_table(result)
    records = result[BASE]["measured_pairings"]["2"]
    assert records[0]["pair"] == records[2]["pair"]
    tables = D._pairing_tables(case["output"])
    for cell, record in zip(seal["cells"], records, strict=True):
        selected = D._cell_pairing(tables, BASE, 2, record["pair"], cell=cell)
        assert selected[BASE]["measured_pairings"]["2"] == record
    with pytest.raises(ValueError, match="exact cell provenance"):
        D._cell_pairing(tables, BASE, 2, records[0]["pair"])
    with pytest.raises(ValueError, match="no measurement for this cell"):
        D._cell_pairing(tables, BASE, 2, records[0]["pair"], cell={**seal["cells"][0], "row": 9})
    # The real A2 admission consumes the written table, then keeps its separate domain gate.
    files["pairings"] = case["output"]
    D.run_bridge(**files, dry_run=True)


def test_nonfinite_fit_forward_refuses_and_removes_hooks(case, monkeypatch):
    class Nonfinite(Block):
        def forward(self, h):
            return super().forward(h) if h.shape[0] == 1 else h * float("nan")

    model = Tiny()
    model.layers[-2] = Nonfinite()
    install_tiny(monkeypatch, case, model)
    with pytest.raises(ValueError, match="non_finite_values"):
        measurement().measure_pairings(**case)
    assert all(not layer._forward_hooks for layer in model.layers)
    assert not case["output"].exists()


def test_bitwise_check_distinguishes_signed_zero(case, monkeypatch):
    class Zero(Block):
        def forward(self, h):
            return torch.zeros_like(h)

    model = Tiny()
    model.layers[0] = Zero()
    case["layers"] = [1]
    for name in ("note", "act"):
        path = case["capture_dir"] / f"residual_{name}.npy"
        array = np.load(path)
        array[0, 0] = -0.0
        np.save(path, array)
    reseal(case)
    install_tiny(monkeypatch, case, model)
    with pytest.raises(ValueError, match="capture_residual_mismatch"):
        measurement().measure_pairings(**case)


@pytest.mark.parametrize("layers", [[], [4], [2, 2], [True]])
def test_bad_layer_request_precedes_model_load(case, monkeypatch, layers):
    calls = install_tiny(monkeypatch, case)
    case["layers"] = layers
    with pytest.raises(ValueError, match="cell_outside_capture"):
        measurement().measure_pairings(**case)
    assert not calls


def test_domain_statement_is_left_for_chief_and_runner_still_requires_it(case, monkeypatch, files):
    seal = D._json(case["positions"])
    seal.pop("domain_of_validity")
    write_json(case["positions"], seal)
    install_tiny(monkeypatch, case)
    result = measurement().measure_pairings(**case)
    assert "domain_of_validity" not in result["_provenance"]
    files["pairings"] = case["output"]
    with pytest.raises(ValueError, match="domain_of_validity statement"):
        D.run_bridge(**files, dry_run=True)


def test_runner_refuses_measurements_from_another_capture(case, monkeypatch, files):
    install_tiny(monkeypatch, case)
    measurement().measure_pairings(**case)
    manifest = D._json(case["capture_dir"] / "manifest.json")
    manifest["other_run"] = "same shape, another capture"
    write_json(case["capture_dir"] / "manifest.json", manifest)
    reseal(case)
    files["pairings"] = case["output"]
    with pytest.raises(ValueError, match="pairing provenance files_sha256 mismatch"):
        D.run_bridge(**files, dry_run=True)


def test_stripping_t5_provenance_never_downgrades_to_legacy_matching(case, monkeypatch, files):
    install_tiny(monkeypatch, case)
    result = measurement().measure_pairings(**case)
    result.pop("_provenance")
    write_json(case["output"], result)
    files["pairings"] = case["output"]
    with pytest.raises(ValueError, match="T5.*provenance"):
        D.run_bridge(**files, dry_run=True)


def test_capture_uses_outer_forward_with_exact_sample_and_kept_positions(case, monkeypatch):
    class CaptureSensitive(Tiny):
        def __init__(self):
            super().__init__()
            self.capture_calls = []

        def forward(
            self, ids=None, *, input_ids=None, output_attentions=False, logits_to_keep=None
        ):
            if input_ids is not None:
                assert output_attentions is True
                assert logits_to_keep.tolist() == [19, 24]
                self.capture_calls.append(input_ids.shape[0])
            return super().forward(ids, input_ids=input_ids)

    model = CaptureSensitive()
    index_path = case["capture_dir"] / "index.jsonl"
    index = D._rows(index_path)[0]
    index["in_sample"] = True
    index_path.write_text(json.dumps(index) + "\n")
    manifest = D._json(case["capture_dir"] / "manifest.json")
    manifest["sample"] = [0]
    write_json(case["capture_dir"] / "manifest.json", manifest)
    reseal(case)
    install_tiny(monkeypatch, case, model)
    measurement().measure_pairings(**case)
    assert model.capture_calls == [1, 1]


def test_zero_norm_refuses_undefined_relative_term(case, monkeypatch):
    class Zero(Block):
        def forward(self, h):
            return torch.zeros_like(h)

    model = Tiny()
    model.layers[0] = Zero()
    case["layers"] = [1]
    for name in ("note", "act"):
        path = case["capture_dir"] / f"residual_{name}.npy"
        array = np.load(path)
        array[0, 0] = 0.0
        np.save(path, array)
    reseal(case)
    install_tiny(monkeypatch, case, model)
    with pytest.raises(ValueError, match="non_finite_values: undefined relative"):
        measurement().measure_pairings(**case)


def test_runtime_adapter_loads_bf16_then_promotes_and_records_actual_flags(case, monkeypatch):
    M = measurement()
    from local_llm_lab import device
    from local_llm_lab.pipeline.lens_fitting import upstream

    model = Tiny()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)))
    loads = []

    def load(snapshot, **kwargs):
        loads.append((snapshot, kwargs))
        return model, {
            "sha256": D._json(case["checkpoint_manifest"]),
            "attn_implementation": "eager",
            "device": "cpu",
        }

    monkeypatch.setattr(M.hf_text, "load_text_causal_lm", load)
    monkeypatch.setattr(device, "pin", lambda **kwargs: None)
    monkeypatch.setattr(
        upstream,
        "load_upstream",
        lambda: SimpleNamespace(provenance=lambda: {"commit": "fixture-upstream"}),
    )
    module = ModuleType("jlens.hf")
    module.HFLensModel = lambda received, tokenizer: received
    monkeypatch.setitem(sys.modules, "jlens.hf", module)
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    try:
        path, runtime = M._load_path(case["model_snapshot"], {"device": "cpu"})
        assert loads == [
            (
                case["model_snapshot"],
                {"dtype": "bfloat16", "attn_implementation": "eager", "device": "cpu"},
            )
        ]
        assert model.weight.dtype == torch.float32
        assert not model.weight.requires_grad
        assert not model.training
        assert path.read([1, 2, 3], 1, [4], 1)[4].tolist() == [7, 6]
        assert runtime["torch_version"] == torch.__version__
        assert runtime["kernel_flags"] == {
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        assert runtime["upstream"]["commit"] == "fixture-upstream"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous[0]
        torch.backends.cudnn.allow_tf32 = previous[1]
        torch.set_float32_matmul_precision(previous[2])


def test_unknown_underscore_metadata_remains_compatible(files):
    table = D._json(files["pairings"])
    table["_provenance"] = "legacy opaque note"
    write_json(files["pairings"], table)
    D.run_bridge(**files, dry_run=True)


def test_cli_has_explicit_inputs_without_loading_a_model():
    script = Path(__file__).resolve().parents[1] / "scripts/measure_pairings.py"
    assert script.is_file(), "T5 CLI is absent"
    result = spawn.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for flag in (
        "model-snapshot",
        "checkpoint-manifest",
        "lens-archive",
        "lens-sidecar",
        "capture-dir",
        "positions",
        "corpus",
        "layers",
        "output",
    ):
        assert f"--{flag}" in result.stdout
