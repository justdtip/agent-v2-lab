"""Device file admission crosses the archive, metadata, and real loader boundaries."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from local_llm_lab import spawn

BASE = "google/tiny-it"
ROOT = Path(__file__).resolve().parents[1]


def api():
    name = "local_llm_lab.probes.device_lens_admission"
    assert importlib.util.find_spec(name) is not None, "device-lens admission API is absent"
    return importlib.import_module(name)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot(tmp_path, *, base=BASE):
    root = tmp_path / "checkpoint"
    root.mkdir(exist_ok=True)
    write_json(
        root / "config.json",
        {
            "_name_or_path": base,
            "num_hidden_layers": 4,
            "hidden_size": 2,
        },
    )
    save_file(
        {"model.embed_tokens.weight": np.ones((3, 2), np.float32)},
        root / "model.safetensors",
        metadata={"format": "pt"},
    )
    return root


def fit(tmp_path, checkpoint, *, name="fit", rows=(0, 1), width=2, offset=0):
    root = tmp_path / name
    root.mkdir()
    maps = {
        "J1": np.array([[1, 2], [0, 1]], np.float32) + offset,
        "J2": np.array([[2, 0], [3, 1]], np.float32) + offset,
        "J3": np.array([[0, 4], [1, 2]], np.float32) + offset,
    }
    np.savez_compressed(root / "exact-maps.npz", **maps)
    corpus = {"manifest": "prose.json", "split": "fit", "rows": list(rows)}
    nu = {
        "schema_version": 1,
        "decoder_depth": 4,
        "estimator": "upstream-exact-autograd",
        "corpus": corpus,
        "endpoint": {
            "source_layers_repo": [1, 2, 3],
            "source_layers_upstream": [0, 1, 2],
            "target_layer_repo": 4,
            "target_layer_upstream": 3,
            "target_is_pre_final_norm": True,
        },
        "pair_weighting": {
            "n_prompts": len(rows),
            "n_skipped": 0,
            "skipped": [],
            "per_prompt": "equal weight per prompt regardless of n_valid_positions",
            "source_target_pairs": "causal, p' >= p, within the selected set",
        },
        "position_weighting": {
            "dim_batch": width,
            "max_seq_len": 128,
            "skip_first": 16,
            "source_reduction": "mean over selected source positions",
            "target_reduction": "sum over selected target positions, not normalised",
            "source_and_target_tied": True,
            "n_valid_positions": {"min": 111, "max": 111, "total": 111 * len(rows)},
            "seq_len": {"min": 128, "max": 128},
            "probe": [
                {
                    "first": 16,
                    "last": 126,
                    "selected": 111,
                    "seq_len": 128,
                    "runs": [[16, 127]],
                    "contiguous": True,
                }
            ],
        },
        "precision": {
            "dtype": "float32",
            "declared_dtype": "float32",
            "stored_dtype": "float32",
            "forward_batch": width,
            "anchor_batch": width,
            "capture_dtype": "native",
        },
        "upstream": {"commit": "fixture-upstream"},
    }
    man = {
        "schema_version": 1,
        "n_rows": len(rows),
        "n_layers": 4,
        "d_model": 2,
        "corpus": corpus,
        "checkpoint": str(checkpoint),
        "precision": "float32",
        "dim_batch": width,
        "forward_batch": width,
        "max_seq_len": 128,
        "load_report_sha256": {p.name: digest(p) for p in checkpoint.iterdir()},
    }
    write_json(root / "manifest.json", man)
    write_json(root / "nu.json", nu)
    return root, maps


def test_real_loader_reads_repository_layer_without_shift_and_preserves_sources(tmp_path):
    cp = snapshot(tmp_path)
    source, maps = fit(tmp_path, cp)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    meta = api().admit_device_lens(source, checkpoint=cp, model_base=BASE)
    archive = source / "admitted-maps.npz"
    lens, read_meta = api().load_admitted_device_lens(archive, checkpoint=cp, model_base=BASE)
    residual = np.array([2, 3], np.float32)
    np.testing.assert_array_equal(lens.apply(residual, 2), [4, 9])
    assert not np.array_equal(lens.apply(residual, 2), residual @ maps["J3"].T)
    assert set(lens.maps) == {1, 2, 3}
    assert meta == read_meta
    assert meta["model"] == {"base": BASE, "num_layers": 4, "endpoint": "identity"}
    assert lens.identity.endpoint == "identity"
    assert meta["source_archive_sha256"] == digest(source / "exact-maps.npz")
    original_nu = json.loads((source / "nu.json").read_text())
    assert meta["source_nu"]["content"] == original_nu
    admitted_nu = json.loads(json.dumps(meta["nu"]))
    assert admitted_nu["precision"].pop("fit_dtype") == "float32"
    assert admitted_nu == original_nu
    assert meta["fit_widths"][0]["forward_batch"] == 2
    assert all((source / name).read_bytes() == content for name, content in originals.items())


@pytest.mark.parametrize(
    "depth,base", [(34, "google/gemma-3-4b-it"), (48, "google/gemma-3-12b-it")]
)
def test_full_device_depth_admission_preserves_each_repository_layer(tmp_path, depth, base):
    """Real device layer counts, tiny file tensors: no checkpoint model is instantiated."""
    from local_llm_lab.pipeline.live_lens.instruments import LensIdentity, LensMaps

    cp = snapshot(tmp_path, base=base)
    config = json.loads((cp / "config.json").read_text())
    config["num_hidden_layers"] = depth
    write_json(cp / "config.json", config)
    source, _ = fit(tmp_path, cp)
    maps = {f"J{k}": np.array([[k, 1], [0, k + 1]], np.float32) for k in range(1, depth)}
    np.savez_compressed(source / "exact-maps.npz", **maps)
    nu = json.loads((source / "nu.json").read_text())
    nu["decoder_depth"] = depth
    nu["endpoint"].update(
        source_layers_repo=list(range(1, depth)),
        source_layers_upstream=list(range(depth - 1)),
        target_layer_repo=depth,
        target_layer_upstream=depth - 1,
    )
    write_json(source / "nu.json", nu)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["n_layers"] = depth
    write_json(source / "manifest.json", manifest)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}

    metadata = api().admit_device_lens(source, checkpoint=cp, model_base=base)
    archive = source / "admitted-maps.npz"
    lens = LensMaps.load(
        archive,
        expected_sha256=metadata["npz_sha256"],
        hidden_size=2,
        num_layers=depth,
        identity=LensIdentity(base=base, num_layers=depth, endpoint="identity"),
    )
    with np.load(archive, allow_pickle=False) as stored:
        assert set(stored.files) == {f"J{k}" for k in range(depth - 1)}
    assert set(lens.maps) == set(range(1, depth))
    assert lens.identity.as_dict() == {"base": base, "num_layers": depth, "endpoint": "identity"}
    residual = np.array([2, 3], np.float32)
    # This checks every boundary, including 9 -> 10 and the final stored map, against a
    # hand-computed result that is distinct for each repository layer.
    for k in range(1, depth):
        np.testing.assert_array_equal(lens.apply(residual, k), [2 * k + 3, 3 * k + 3])
        np.testing.assert_array_equal(lens.maps[k], maps[f"J{k}"])
    np.testing.assert_array_equal(lens.apply(residual, depth), residual)
    assert all((source / name).read_bytes() == content for name, content in originals.items())


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("shifted", "keys"),
        ("missing", "keys"),
        ("shape", "shape"),
        ("nonfinite", "finite"),
        ("width", "anchor_batch"),
        ("count", "n_prompts"),
        ("endpoint", "endpoint"),
        ("checkpoint", "checkpoint"),
        ("unknown_base", "base"),
        ("schema", "schema_version"),
        ("conflicting_fit_dtype", "fit_dtype"),
    ],
)
def test_invalid_source_refuses_before_writing(tmp_path, mutation, match):
    cp = snapshot(tmp_path)
    source, maps = fit(tmp_path, cp)
    nu = json.loads((source / "nu.json").read_text())
    if mutation == "shifted":
        maps = {f"J{i}": maps[f"J{i + 1}"] for i in range(3)}
    elif mutation == "missing":
        del maps["J2"]
    elif mutation == "shape":
        maps["J2"] = np.ones((3, 3), np.float32)
    elif mutation == "nonfinite":
        maps["J2"][0, 0] = np.nan
    elif mutation == "width":
        del nu["precision"]["anchor_batch"]
    elif mutation == "count":
        nu["pair_weighting"]["n_prompts"] = 3
    elif mutation == "endpoint":
        nu["endpoint"]["target_layer_repo"] = 3
    elif mutation == "checkpoint":
        with (cp / "model.safetensors").open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "unknown_base":
        config = json.loads((cp / "config.json").read_text())
        del config["_name_or_path"]
        write_json(cp / "config.json", config)
    elif mutation == "schema":
        nu["schema_version"] = 2
    elif mutation == "conflicting_fit_dtype":
        nu["precision"]["fit_dtype"] = "bfloat16"
    write_json(source / "nu.json", nu)
    np.savez_compressed(source / "exact-maps.npz", **maps)
    with pytest.raises(ValueError, match=match):
        api().admit_device_lens(source, checkpoint=cp, model_base=BASE)
    assert not (source / "admitted-maps.npz").exists()


@pytest.mark.parametrize("field", ["npz_sha256", "model", "nu", "source_archive_sha256"])
def test_tampered_admitted_sidecar_refuses(tmp_path, field):
    cp = snapshot(tmp_path)
    source, _ = fit(tmp_path, cp)
    api().admit_device_lens(source, checkpoint=cp, model_base=BASE)
    sidecar = source / "admitted-maps.json"
    meta = json.loads(sidecar.read_text())
    if field == "model":
        meta[field]["base"] = "google/other-it"
    elif field == "nu":
        meta[field]["position_weighting"]["skip_first"] = 0
    else:
        meta[field] = "0" * 64
    write_json(sidecar, meta)
    with pytest.raises(ValueError, match="hash|identity|declaration|provenance"):
        api().load_admitted_device_lens(
            source / "admitted-maps.npz", checkpoint=cp, model_base=BASE
        )


def test_wrong_requested_identity_and_existing_output_refuse(tmp_path):
    cp = snapshot(tmp_path)
    source, _ = fit(tmp_path, cp)
    with pytest.raises(ValueError, match="base|identity"):
        api().admit_device_lens(source, checkpoint=cp, model_base="google/other-it")
    api().admit_device_lens(source, checkpoint=cp, model_base=BASE)
    with pytest.raises(FileExistsError):
        api().admit_device_lens(source, checkpoint=cp, model_base=BASE)


def test_existing_source_sidecar_must_name_this_archive(tmp_path):
    cp = snapshot(tmp_path)
    source, _ = fit(tmp_path, cp)
    write_json(
        source / "exact-maps.json",
        {"npz_sha256": "0" * 64, "model": {"base": BASE, "num_layers": 4}},
    )
    with pytest.raises(ValueError, match="sidecar.*hash|sidecar.*file"):
        api().admit_device_lens(source, checkpoint=cp, model_base=BASE)


def test_chunk_merger_preserves_full_declarations_weights_and_admits(tmp_path):
    cp = snapshot(tmp_path)
    first, a = fit(tmp_path, cp, name="c1", rows=(0, 1), width=2)
    second, b = fit(tmp_path, cp, name="c2", rows=(3,), width=1, offset=3)
    merged = tmp_path / "merged"
    api().merge_device_lens_chunks(merged, [first, second])
    nu = json.loads((merged / "nu.json").read_text())
    assert nu["corpus"]["rows"] == [0, 1, 3]
    assert nu["precision"]["forward_batch"] == [2, 1]
    assert nu["position_weighting"]["dim_batch"] == [2, 1]
    chunks = nu["merge"]["chunks"]
    assert [chunk["weight"] for chunk in chunks] == [2 / 3, 1 / 3]
    assert chunks[0]["nu"] == json.loads((first / "nu.json").read_text())
    assert chunks[0]["archive_sha256"] == digest(first / "exact-maps.npz")
    with np.load(merged / "exact-maps.npz") as archive:
        for key in a:
            np.testing.assert_allclose(archive[key], a[key] + 1, rtol=0, atol=1e-6)
    meta = api().admit_device_lens(merged, checkpoint=cp, model_base=BASE)
    assert [row["forward_batch"] for row in meta["fit_widths"]] == [2, 1]
    assert [row["n_prompts"] for row in meta["fit_widths"]] == [2, 1]
    api().load_admitted_device_lens(merged / "admitted-maps.npz", checkpoint=cp, model_base=BASE)


@pytest.mark.parametrize(
    "kind",
    ["overlap", "different_corpus", "different_positions", "different_checkpoint", "missing_width"],
)
def test_chunk_merger_refuses_incompatible_populations_and_declarations(tmp_path, kind):
    cp = snapshot(tmp_path)
    a, _ = fit(tmp_path, cp, name="c1", rows=(0, 1))
    b, _ = fit(tmp_path, cp, name="c2", rows=(1,) if kind == "overlap" else (3,))
    nu = json.loads((b / "nu.json").read_text())
    man = json.loads((b / "manifest.json").read_text())
    if kind == "different_corpus":
        nu["corpus"]["manifest"] = man["corpus"]["manifest"] = "different.json"
    elif kind == "different_positions":
        nu["position_weighting"]["skip_first"] = 8
    elif kind == "different_checkpoint":
        man["load_report_sha256"]["model.safetensors"] = "0" * 64
    elif kind == "missing_width":
        del nu["precision"]["anchor_batch"]
    write_json(b / "nu.json", nu)
    write_json(b / "manifest.json", man)
    out = tmp_path / "bad-merge"
    with pytest.raises(ValueError, match="overlap|corpus|position|checkpoint|anchor_batch"):
        api().merge_device_lens_chunks(out, [a, b])
    assert not out.exists()


def test_cli_admits_real_fixture_files(tmp_path):
    cp = snapshot(tmp_path)
    source, _ = fit(tmp_path, cp)
    result = spawn.run(
        [
            sys.executable,
            str(ROOT / "scripts/admit_device_lens.py"),
            str(source),
            "--checkpoint",
            str(cp),
            "--model-base",
            BASE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["npz_sha256"] == digest(source / "admitted-maps.npz")


def test_merger_script_emits_promised_nu(tmp_path):
    cp = snapshot(tmp_path)
    a, _ = fit(tmp_path, cp, name="a", rows=(0,))
    b, _ = fit(tmp_path, cp, name="b", rows=(1,))
    target = ROOT / "scripts/merge_device_lens_chunks.py"
    result = spawn.run(
        [sys.executable, str(target), str(tmp_path / "out"), str(a), str(b)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/nu.json").is_file()


def test_merger_preserves_original_float32_accumulation_order(tmp_path):
    """Changing the merger's arithmetic would create different fitted matrices."""
    cp = snapshot(tmp_path)
    chunks = []
    for i, value in enumerate((100_000_000, 1, -100_000_000)):
        chunk, _ = fit(tmp_path, cp, name=f"c{i}", rows=(i,))
        np.savez_compressed(
            chunk / "exact-maps.npz",
            **{f"J{k}": np.full((2, 2), value, dtype=np.float32) for k in (1, 2, 3)},
        )
        chunks.append(chunk)
    api().merge_device_lens_chunks(tmp_path / "out", chunks)
    # Upstream accumulates the stored dtype in input order: (1e8 + 1) + -1e8 == 0.
    with np.load(tmp_path / "out/exact-maps.npz") as archive:
        np.testing.assert_array_equal(archive["J1"], np.zeros((2, 2), np.float32))


def test_merged_archive_cannot_claim_different_weighted_values(tmp_path):
    cp = snapshot(tmp_path)
    first, _ = fit(tmp_path, cp, name="c1", rows=(0,))
    second, _ = fit(tmp_path, cp, name="c2", rows=(1,), offset=2)
    out = tmp_path / "merged"
    api().merge_device_lens_chunks(out, [first, second])
    with np.load(out / "exact-maps.npz") as archive:
        maps = {key: archive[key] + 5 for key in archive.files}
    np.savez_compressed(out / "exact-maps.npz", **maps)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["exact_maps_sha256"] = digest(out / "exact-maps.npz")
    write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="weighted.*map|merged.*map"):
        api().admit_device_lens(out, checkpoint=cp, model_base=BASE)


@pytest.mark.parametrize("merged", [False, True])
def test_actual_writer_precision_declaration_reaches_bridge_identity(tmp_path, merged):
    from local_llm_lab.probes import sae_bridge as bridge

    cp = snapshot(tmp_path)
    source, _ = fit(tmp_path, cp, name="first", rows=(0,), width=2)
    if merged:
        second, _ = fit(tmp_path, cp, name="second", rows=(1,), width=1)
        target = tmp_path / "merged"
        api().merge_device_lens_chunks(target, [source, second])
        source = target
    raw = (source / "nu.json").read_bytes()
    # The actual declare_nu writer names dtype, declared_dtype, and stored_dtype;
    # it does not write the bridge's fit_dtype spelling.
    assert "fit_dtype" not in json.loads(raw)["precision"]
    meta = api().admit_device_lens(source, checkpoint=cp, model_base=BASE)
    lens, meta = api().load_admitted_device_lens(
        source / "admitted-maps.npz", checkpoint=cp, model_base=BASE
    )
    identity = bridge.reading_identity(
        meta["nu"], lens_sha256=lens.sha256, capture_dtype="float32", capture_batch=1
    )
    assert identity["fit_dtype"] == "float32"
    assert identity["fit_width"] == ([2, 1] if merged else 2)
    assert identity["nu_sha256"] == meta["nu_sha256"]
    assert meta["source_nu"]["content"] == json.loads(raw)
    assert meta["source_nu"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert (source / "nu.json").read_bytes() == raw
