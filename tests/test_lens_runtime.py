"""Requirements §§4, 7, 12, 13: pure pre-load validation and exact provenance."""

import importlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.live_lens.instruments import LensIdentity

#: The identity every toy lens in this file is fitted for (issue 99). Three layers because
#: `write_lens` requires all and only the nonfinal maps, and these fixtures write two.
_TOY = LensIdentity("example/tiny", 3)


def api():
    return importlib.import_module("local_llm_lab.pipeline.lens_fitting.runtime")


def snapshot(tmp_path):
    root = tmp_path / "models--example--tiny" / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"hidden_size": 2, "num_hidden_layers": 3}')
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
    return root


def test_snapshot_identity_hashes_exact_shards_and_revision(tmp_path):
    """§4: an identity binds exact loaded config/weights, not another cached revision."""
    runtime = api()
    root = snapshot(tmp_path)
    first = runtime.snapshot_identity(root, hf_id="example/tiny")
    assert first["hf_id"] == "example/tiny"
    assert first["resolved_revision"] == "a" * 40
    assert first["snapshot_path"] == str(root)
    assert {r["name"] for r in first["files"]} == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    (root / "model-00002-of-00002.safetensors").write_bytes(b"different")
    assert (
        first["snapshot_sha256"]
        != runtime.snapshot_identity(root, hf_id="example/tiny")["snapshot_sha256"]
    )
    with pytest.raises(ValueError, match="repository"):
        runtime.snapshot_identity(root, hf_id="other/repository")


def test_snapshot_rejects_missing_indexed_weight_before_loading(tmp_path):
    """§4: partial snapshots cannot masquerade as a complete loaded checkpoint."""
    runtime = api()
    root = snapshot(tmp_path)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-absent.safetensors"}})
    )
    with pytest.raises(ValueError, match="shard"):
        runtime.snapshot_identity(root, hf_id="example/tiny")


def test_prepare_rejects_model_mismatch_and_bad_corpus_before_snapshot(tmp_path, monkeypatch):
    """§7: corpus/model/output errors are detected before any checkpoint load."""
    runtime = api()
    spec = load_model_spec("qwen35-4b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model_hf_id": "different", "domain": "agentic"}))
    monkeypatch.setattr(runtime, "read_corpus", lambda p: [{"ids": [1], "split": "fit"}])
    with pytest.raises(ValueError, match="model"):
        runtime.prepare_fit(manifest, spec, tmp_path / "qwen35-4b-agentic-regression.npz")
    monkeypatch.setattr(
        runtime, "read_corpus", lambda p: (_ for _ in ()).throw(ValueError("corrupt"))
    )
    with pytest.raises(ValueError, match="corrupt"):
        runtime.prepare_fit(manifest, spec, tmp_path / "qwen35-4b-agentic-regression.npz")


def test_output_refusal_is_exclusive_and_validates_every_layer(tmp_path):
    """§4: archive creation cannot overwrite either artifact or omit a layer."""
    artifacts = importlib.import_module("local_llm_lab.pipeline.lens_fitting.artifacts")
    out = tmp_path / "tiny-agentic-regression.npz"
    with pytest.raises(ValueError, match="layers"):
        artifacts.write_lens(
            out,
            {1: np.eye(2)},
            hidden_size=2,
            num_layers=3,
            metadata={},
            identity=_TOY,
        )
    assert not out.exists()
    out.with_suffix(".json").write_text("existing")
    with pytest.raises(FileExistsError):
        artifacts.write_lens(
            out,
            {1: np.eye(2), 2: np.eye(2)},
            hidden_size=2,
            num_layers=3,
            metadata={},
            identity=_TOY,
        )
    assert not out.exists()
    assert out.with_suffix(".json").read_text() == "existing"


def test_primary_lock_refusal_precedes_loader_import(tmp_path, monkeypatch):
    """§7: primary lock wins even in a worktree, before load_policy/Metal import."""
    runtime = api()
    from local_llm_lab import runlock

    primary, worktree = tmp_path / "primary", tmp_path / "linked"
    primary.mkdir()
    worktree.mkdir()
    lock = primary / runlock.LOCK_RELATIVE_PATH
    lock.parent.mkdir()
    lock.write_text('{"pid": 999999, "command": "other run"}')
    monkeypatch.setattr(runlock, "PROJECT_ROOT", worktree)
    monkeypatch.setattr(runtime, "primary_worktree", lambda: primary)
    monkeypatch.setattr(runtime, "_load_policy", lambda *a: pytest.fail("loader reached"))
    prepared = SimpleNamespace(spec=load_model_spec("qwen35-4b"), snapshot={})
    with pytest.raises(runlock.RunLockBusy):
        runtime.load_runtime(prepared)
    assert lock.exists()
    assert not (worktree / runlock.LOCK_RELATIVE_PATH).exists()


def test_runtime_loads_pinned_snapshot_preserves_hf_id_and_capture_strategy(tmp_path, monkeypatch):
    """§§4/7/13: exact local snapshot, primary lock, and none before capture load."""
    runtime = api()
    from local_llm_lab import runlock

    root = snapshot(tmp_path)
    identity = runtime.snapshot_identity(root, hf_id="example/tiny")
    spec = replace(load_model_spec("qwen35-4b"), hf_id="example/tiny", cache_strategy="history")
    primary = tmp_path / "primary"
    primary.mkdir()
    monkeypatch.setattr(runtime, "primary_worktree", lambda: primary)
    monkeypatch.setattr(runlock, "PROJECT_ROOT", tmp_path / "linked")
    # Override conftest's fixed temporary default with the production root formula,
    # still confined to this test's temporary primary checkout.
    monkeypatch.setattr(
        runlock, "default_lock_path", lambda: runlock.PROJECT_ROOT / runlock.LOCK_RELATIVE_PATH
    )
    from local_llm_lab import project

    ordinary_root = project.PROJECT_ROOT

    def loader(local_spec, adapter):
        assert runlock.read_lock(primary / runlock.LOCK_RELATIVE_PATH) is not None
        # Exercise the loader's second acquire, with no explicit path, while held.
        assert runlock.hold_model_run_lock() == primary / runlock.LOCK_RELATIVE_PATH
        assert local_spec.hf_id == str(root)
        assert local_spec.cache_strategy == "none"
        assert adapter is None
        assert ordinary_root == project.PROJECT_ROOT

        @dataclass
        class Resolution:
            spec: object
            snapshot_revision: str | None
            cache_strategy: str

        return "model", "tokenizer", "view", Resolution(local_spec, None, "none")

    monkeypatch.setattr(runtime, "_load_policy", loader)
    prepared = SimpleNamespace(spec=spec, snapshot=identity)
    loaded = runtime.load_runtime(prepared, capture=True)
    assert loaded.spec.hf_id == "example/tiny"
    assert loaded.spec.cache_strategy == "none"
    assert loaded.snapshot == identity
    assert loaded.resolved.spec.hf_id == "example/tiny"
    assert loaded.resolved.snapshot_revision == "a" * 40
    assert loaded.resolved.cache_strategy == "none"


@pytest.mark.parametrize(
    "arguments,exit_code",
    [
        (["--help"], 0),
        (
            [
                "--kind",
                "regression",
                "--model",
                "qwen35-4b",
                "--corpus",
                "/does/not/exist",
                "--out",
                "/does/not/exist/qwen35-4b-agentic-regression.npz",
            ],
            2,
        ),
        (["--adapter", "/adapter"], 2),
    ],
)
def test_cli_help_and_bad_input_do_not_import_mlx(arguments, exit_code):
    """§7: help and invalid input must never initialize the model runtime."""
    import sys

    from local_llm_lab.spawn import run

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "lens_fit.py"
    code = (
        "import runpy,sys; sys.argv="
        + repr([str(script), *arguments])
        + ";\ntry: runpy.run_path("
        + repr(str(script))
        + ",run_name='__main__')\n"
        f"except SystemExit as e: assert e.code == {exit_code}\n"
        "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)\n"
    )
    result = run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_resolve_uses_primary_cache_offline(tmp_path, monkeypatch):
    """§7: linked worktrees reuse the primary checkpoint cache without downloading."""
    runtime = api()
    import huggingface_hub

    primary = tmp_path / "primary"
    primary.mkdir()
    root = snapshot(tmp_path)
    monkeypatch.setattr(runtime, "primary_worktree", lambda: primary)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    observed = []

    def local_snapshot(hf_id, **kwargs):
        observed.append((hf_id, kwargs))
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", local_snapshot)
    spec = replace(load_model_spec("qwen35-4b"), hf_id="example/tiny")
    identity = runtime.resolve_snapshot(spec, revision="a" * 40)
    assert observed == [
        (
            "example/tiny",
            {
                "revision": "a" * 40,
                "local_files_only": True,
                "cache_dir": str(primary / ".cache/huggingface/hub"),
                "allow_patterns": list(runtime.SNAPSHOT_ALLOW_PATTERNS),
            },
        )
    ]
    assert identity["resolved_revision"] == "a" * 40


def test_complete_cli_sidecar_and_output_refusal_without_weights(tmp_path, monkeypatch, capsys):
    """§§3.2/4/12: public CLI publishes reproducible penalties and selection caveats."""
    import runpy

    runtime = api()
    regression = importlib.import_module("local_llm_lab.pipeline.lens_fitting.regression")
    spec = load_model_spec("qwen35-4b")
    out = tmp_path / "qwen35-4b-agentic-regression.npz"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_hf_id": spec.hf_id,
                "domain": "agentic",
                "sequences": {"sha256": "sequence-hash"},
            }
        )
    )
    rows = [{"ids": [1], "split": "fit"}, {"ids": [2], "split": "held"}]
    monkeypatch.setattr(runtime, "read_corpus", lambda path: rows)
    root = snapshot(tmp_path)
    identity = {"snapshot_path": str(root), "snapshot_sha256": "fixture", "hf_id": spec.hf_id}
    monkeypatch.setattr(runtime, "resolve_snapshot", lambda *a, **kw: identity)
    seen = []

    def fake_load(prepared):
        seen.append(prepared)
        return SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None),
            view=SimpleNamespace(hidden_size=2, num_layers=3),
            spec=spec,
            snapshot=identity,
            resolved=SimpleNamespace(cache_strategy="history"),
            lock_path=tmp_path / "primary-lock",
        )

    monkeypatch.setattr(runtime, "load_runtime", fake_load)
    policies = []

    def cache_policy():
        policies.append("configured")
        return dict(previous_limit_bytes=1024, limit_bytes=0)

    monkeypatch.setattr(runtime, "configure_allocator_cache", cache_policy)
    monkeypatch.setattr(
        regression,
        "fit_regression",
        lambda *a, **kw: regression.RegressionResult(
            {1: np.array([[1, 2], [3, 4]], dtype=np.float32), 2: np.eye(2, dtype=np.float32)},
            {
                "1": {
                    "alpha": 0.001,
                    "dbar": 2,
                    "n": 1,
                    "absolute_penalty": 0.002,
                    "lambda_per_position": 0.002,
                    "held_out_uncentered_r2": 0.9,
                }
            },
            {"fit": {"positions": 1, "sequences": 1}, "held": {"positions": 1, "sequences": 1}},
            0.1,
            0.01,
        ),
    )
    fake_fit = regression.fit_regression

    def fit_with_progress(*args, **kwargs):
        kwargs["progress"](
            {"event": "sequence", "counts": {"fit": {"positions": 5}, "held": {"positions": 2}}}
        )
        return fake_fit(*args, **kwargs)

    monkeypatch.setattr(regression, "fit_regression", fit_with_progress)
    monkeypatch.setattr(
        runtime, "resource_snapshot", lambda: dict(peak_memory_gib=1, working_set_share=0.1)
    )
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/lens_fit.py"))["main"]
    argv = [
        "--kind",
        "regression",
        "--model",
        spec.name,
        "--corpus",
        str(manifest),
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    progress_event = json.loads(capsys.readouterr().out.splitlines()[0])
    assert progress_event["tokens_done"] == 7
    assert progress_event["elapsed_s"] >= 0
    assert progress_event["peak_memory_gib"] == 1
    assert progress_event["working_set_share"] == 0.1
    metadata = json.loads(out.with_suffix(".json").read_bytes())
    assert metadata["lambda_grid"] == [0.001, 0.01, 0.1, 1, 10]
    # The lens records the lineage it was fitted on, not the artefact it was fitted from:
    # the artefact is where the weights were, and two conversions of one base share a lens (R60).
    assert metadata["model"]["base"] == spec.base
    assert metadata["layers"] == [1, 2]
    assert "share trajectories" in metadata["held_split_role"]
    assert "not measured" in metadata["generalisation_evaluation"]
    assert metadata["r2_definition"].startswith("uncentered")
    assert metadata["cache_strategy"] == "history"
    assert policies == ["configured"]
    assert metadata["allocator_cache"] == dict(previous_limit_bytes=1024, limit_bytes=0)
    assert metadata["corpus_manifest_sha256"] == runtime.file_sha256(manifest)
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main(argv)
    assert len(seen) == 1
    assert out.read_bytes() == before


@pytest.mark.parametrize("issue", ["nonfinite", "shape", "size"])
def test_invalid_artifact_does_not_publish(tmp_path, monkeypatch, issue):
    """§4: finite square float32 maps and the sub-1GB bound are mandatory."""
    artifacts = importlib.import_module("local_llm_lab.pipeline.lens_fitting.artifacts")
    maps = {1: np.eye(2)}
    if issue == "nonfinite":
        maps[1][0, 0] = float("nan")
    elif issue == "shape":
        maps[1] = np.ones((2, 3))
    else:
        monkeypatch.setattr(artifacts, "MAX_NPZ_BYTES", 1)
    out = tmp_path / "tiny-prose-regression.npz"
    with pytest.raises(ValueError):
        artifacts.write_lens(
            out, maps, hidden_size=2, num_layers=2, metadata={}, identity=_TOY
        )
    assert not out.exists()
    assert not out.with_suffix(".json").exists()


@pytest.mark.parametrize("phase", ["before", "during"])
def test_changed_snapshot_cannot_be_claimed_as_loaded_identity(tmp_path, monkeypatch, phase):
    """§4: refuse changing bytes before or during load rather than publishing stale hashes."""
    runtime = api()
    from local_llm_lab import runlock

    root = snapshot(tmp_path)
    identity = runtime.snapshot_identity(root, hf_id="example/tiny")
    spec = replace(load_model_spec("qwen35-4b"), hf_id="example/tiny")
    primary = tmp_path / "primary"
    primary.mkdir()
    monkeypatch.setattr(runtime, "primary_worktree", lambda: primary)
    monkeypatch.setattr(runlock, "PROJECT_ROOT", primary)
    reached = []

    def mutate():
        (root / "model-00001-of-00002.safetensors").write_bytes(b"mutation")

    def loader(*args):
        reached.append(True)
        mutate()
        return (None, None, None, None)

    monkeypatch.setattr(runtime, "_load_policy", loader)
    if phase == "before":
        mutate()
    with pytest.raises(ValueError, match="snapshot changed"):
        runtime.load_runtime(SimpleNamespace(spec=spec, snapshot=identity))
    assert bool(reached) == (phase == "during")


def test_primary_worktree_is_first_git_record(tmp_path, monkeypatch):
    """§7: the lock root is Git's first primary record, never the current linked tree."""
    runtime = api()
    primary = tmp_path / "primary with spaces"
    primary.mkdir()
    called = []

    def git(argv, **kwargs):
        called.append(argv)
        return SimpleNamespace(stdout=f"worktree {primary}\0HEAD deadbeef\0\0worktree /linked\0")

    monkeypatch.setattr(runtime, "run", git)
    assert runtime.primary_worktree() == primary
    assert called[0][-4:] == ["worktree", "list", "--porcelain", "-z"]


def test_offline_complete_model_snapshot_does_not_require_hub_docs(tmp_path, monkeypatch):
    """§4: absent README/.gitattributes must not block complete cached runtime assets."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import IncompleteSnapshotError

    runtime = api()
    root = snapshot(tmp_path)
    (root / "tokenizer.json").write_text("{}")
    repo_cache = root.parent.parent
    tree = repo_cache / "trees" / f"{root.name}.json"
    tree.parent.mkdir()
    names = [path.name for path in root.iterdir()] + ["README.md", ".gitattributes"]
    tree.write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": {name: {"size": 1, "blob_id": "b" * 40} for name in names},
            }
        )
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    # Exercise the installed Hub offline completeness check against real local cache files.
    with pytest.raises(IncompleteSnapshotError, match="README.md"):
        snapshot_download(
            "example/tiny", revision=root.name, cache_dir=tmp_path, local_files_only=True
        )
    identity = runtime.resolve_snapshot(
        replace(load_model_spec("qwen35-4b"), hf_id="example/tiny"), revision=root.name
    )
    assert identity["resolved_revision"] == root.name
    assert {row["name"] for row in identity["files"]} == set(names) - {
        "README.md",
        ".gitattributes",
    }
    assert not (root / "README.md").exists()
    assert not (root / ".gitattributes").exists()
    (root / "model-00001-of-00002.safetensors").unlink()
    with pytest.raises(IncompleteSnapshotError, match="model-00001"):
        runtime.resolve_snapshot(
            replace(load_model_spec("qwen35-4b"), hf_id="example/tiny"), revision=root.name
        )


def test_allocator_cache_policy_preserves_previous_limit():
    """§7/R47: fit policy bounds unused allocation bytes, not live buffer count."""
    seen = []

    def setter(value):
        seen.append(value)
        return 2048

    result = api().configure_allocator_cache(set_limit=setter)
    assert seen == [0]
    assert result["previous_limit_bytes"] == 2048
    assert result["limit_bytes"] == 0


def test_resource_snapshot_reports_device_share_without_native_import():
    """§7/R46: progress includes measured peak and fraction of device working set."""
    backend = SimpleNamespace(
        get_peak_memory=lambda: 2**30,
        device_info=lambda: {"max_recommended_working_set_size": 4 * 2**30},
    )
    assert api().resource_snapshot(array_api=backend) == dict(
        peak_memory_gib=1, working_set_share=0.25
    )
