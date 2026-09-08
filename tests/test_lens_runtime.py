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
    barrier = """import importlib.abc,sys
class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx','mlx_lm'}:
            raise RuntimeError('real MLX forbidden in CLI fixture')
sys.meta_path.insert(0, NoMLX())
"""
    result = run([sys.executable, "-c", barrier + code], capture_output=True, text=True)
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
    rows = [{"ids": [1] * 1024, "split": "fit"}, {"ids": [2] * 1024, "split": "held"}]
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
        assert kwargs["residual_source"] == "native"
        kwargs["progress"](
            {
                "event": "sequence",
                "counts": {
                    "fit": {"positions": 5, "input_positions": 50},
                    "held": {"positions": 2, "input_positions": 20},
                    "residual_source": "native",
                },
            }
        )
        return fake_fit(*args, **kwargs)

    monkeypatch.setattr(regression, "fit_regression", fit_with_progress)
    monkeypatch.setattr(
        runtime, "resource_snapshot", lambda: dict(peak_memory_gib=1, working_set_share=0.1)
    )
    from local_llm_lab.pipeline.lens_fitting import memory_policy

    monkeypatch.setattr(memory_policy, "runtime_fingerprint", lambda: {"fake": "runtime"})
    monkeypatch.setattr(memory_policy, "require_owned_window", lambda: None)
    monkeypatch.setattr(memory_policy, "device_working_set", lambda: 10 * 2**30)
    prepared = runtime.prepare_fit(manifest, spec, out)
    bound = memory_policy.binding(prepared, "native", "memory")
    report = tmp_path / "preflight.jsonl"
    initial = 2 * 2**30
    fixed = 2**30 + 128
    measurements = []
    for tokens in (256, 512, 1024):
        projected = (
            initial
            if len(measurements) < 2
            else memory_policy.project_peak(measurements, tokens, fixed_bytes=fixed)
        )
        measurements.append(
            {
                "event": "measured",
                "tokens": tokens,
                "peak_bytes": 2**30,
                "projected_peak_bytes": projected,
                "scored_positions_per_sequence": tokens,
            }
        )
    envelope = {
        "event": "full_fit_envelope",
        "dense_matrix_bytes": 16,
        "nonfinal_layers": 2,
        "active_floor_bytes": 2**30,
        "host_maps_and_serialization_bytes": 64,
        "solver_workspace_bytes": 256,
        "projected_peak_bytes": 2**30 + 320,
    }
    report.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {
                    "event": "begin",
                    "mode": "qualification",
                    "qualification_binding": bound,
                    "initial_bound_bytes": initial,
                    "lengths": [256, 512, 1024],
                },
                {
                    "event": "loaded",
                    "load_peak_bytes": 2**30,
                    "resident_bytes": 2**30,
                    "statistics_bytes": 128,
                },
                *measurements,
                envelope,
                {
                    "event": "measured_solve",
                    "peak_bytes": 2**30,
                    "projected_peak_bytes": 2**30 + 320,
                },
                {"event": "end", "status": "measured", "fit_started": False},
            ]
        )
    )
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/lens_fit.py"))["main"]
    argv = [
        "--memory-preflight",
        str(report),
        "--kind",
        "regression",
        "--model",
        spec.name,
        "--corpus",
        str(manifest),
        "--out",
        str(out),
        "--residual-source",
        "native",
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
    assert metadata["model"] == {"base": spec.base, "num_layers": 3}
    assert metadata["snapshot"] == identity
    assert metadata["model_name"] == spec.name
    assert metadata["residual_source"] == "native"
    from local_llm_lab.pipeline.live_lens.instruments import LENS_IDENTITY_KEY

    with np.load(out, allow_pickle=False) as archive:
        assert json.loads(bytes(archive[LENS_IDENTITY_KEY])) == metadata["model"]
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
        artifacts.write_lens(out, maps, hidden_size=2, num_layers=2, metadata={}, identity=_TOY)
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


@pytest.fixture
def numpy_regression_backend(monkeypatch):
    """Only tiny statistics tests use this substitute; no native module is imported."""
    import sys
    from types import ModuleType

    package, core = ModuleType("mlx"), ModuleType("mlx.core")
    for name in ("zeros", "sum", "float32", "array"):
        setattr(core, name, getattr(np, name))
    core.eval = lambda *args: None
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_masked_native_statistics_keep_full_context_and_report_scored_counts(
    numpy_regression_backend,
):
    from local_llm_lab.pipeline.lens_fitting.regression import accumulate

    calls, events = [], []

    def native(ids, layers):
        calls.append(list(ids))
        x = np.array([[[i, 1] for i in ids]], dtype=np.float32)
        return {1: x, 2: x * 3}

    view = SimpleNamespace(hidden_size=2, num_layers=2, native_residuals=native)
    rows = [dict(ids=[1, 100, 3], split=s, score_positions=[0, 2]) for s in ("fit", "held")]
    sums, counts = accumulate(view, rows, residual_source="native", progress=events.append)
    assert calls == [[1, 100, 3], [1, 100, 3]]
    np.testing.assert_array_equal(sums["fit"][1].xtx, [[10, 4], [4, 2]])
    np.testing.assert_array_equal(sums["fit"][1].xty, [[30, 12], [12, 6]])
    assert sums["fit"][1].n == 2
    assert counts["fit"] == dict(sequences=1, positions=2, scored_positions=2, input_positions=3)
    assert counts["residual_source"] == events[0]["counts"]["residual_source"] == "native"
    assert events[0]["counts"]["held"]["positions"] == 0


@pytest.mark.parametrize("positions", [[], [1, 0], [0, 0], [-1], [3], [True], [1.0], None])
def test_invalid_score_positions_refused_before_forward(numpy_regression_backend, positions):
    from local_llm_lab.pipeline.lens_fitting.regression import accumulate

    view = SimpleNamespace(
        hidden_size=2,
        num_layers=2,
        native_residuals=lambda *a: pytest.fail("forward reached for invalid score mask"),
    )
    with pytest.raises(ValueError, match="score_positions"):
        accumulate(
            view,
            [dict(ids=[1, 2, 3], split="fit", score_positions=positions)],
            residual_source="native",
        )


def test_unmasked_statistics_keep_all_positions(numpy_regression_backend):
    from local_llm_lab.pipeline.lens_fitting.regression import accumulate

    def residuals(ids, layers):
        x = np.array(ids, dtype=np.float32)[None, :, None]
        return {1: x, 2: 2 * x}

    view = SimpleNamespace(hidden_size=1, num_layers=2, residuals=residuals)
    sums, counts = accumulate(view, [dict(ids=[2, 3], split=s) for s in ("fit", "held")])
    assert sums["fit"][1].xtx.item() == 13
    assert counts["fit"] == dict(sequences=1, positions=2, scored_positions=2, input_positions=2)


def transcript_prepare_fixture(tmp_path, monkeypatch):
    runtime = api()
    spec = load_model_spec("gemma3-4b")
    root = snapshot(tmp_path)
    (root / "tokenizer.json").write_text('{"tokens": ["toy"]}')
    assets = [{"name": "tokenizer.json", "sha256": runtime.file_sha256(root / "tokenizer.json")}]
    identity = dict(snapshot_path=str(root), files=assets, hf_id=spec.hf_id)
    monkeypatch.setattr(runtime, "resolve_snapshot", lambda *a, **k: identity)
    monkeypatch.setattr(
        runtime, "read_corpus", lambda p: [dict(ids=[1], split=s) for s in ("fit", "held")]
    )
    manifest = dict(
        schema_version=2,
        domain="agentic",
        format="transcript-native-replay-v1",
        model_hf_id="source-bf16-artifact",
        model_identity=dict(base=spec.base, training=None, num_layers=3),
        tokenizer=dict(assets=assets),
        precision="bf16",
        acceptance={
            "status": "passed",
            "reasons": [],
            "repeated_run_rule": "declared",
            "concentration": {"fit": {}, "held": {}, "combined": {}},
        },
    )
    path = tmp_path / "transcript.json"
    return runtime, spec, path, manifest


def test_prepare_shared_precision_corpus_requires_identity_and_exact_tokenizer(
    tmp_path, monkeypatch
):
    runtime, spec, path, manifest = transcript_prepare_fixture(tmp_path, monkeypatch)
    path.write_text(json.dumps(manifest))
    prepared = runtime.prepare_fit(path, spec, tmp_path / "gemma3-4b-agentic-regression.npz")
    assert prepared.manifest["precision"] == "bf16"
    assert prepared.manifest["model_hf_id"] == "source-bf16-artifact"
    assert prepared.spec == spec


@pytest.mark.parametrize(
    "issue",
    [
        "missing_identity",
        "missing_training",
        "base",
        "training",
        "depth",
        "tokenizer",
        "extra_asset",
        "missing_assets",
    ],
)
def test_prepare_new_schema_refuses_identity_or_tokenizer_mismatch(tmp_path, monkeypatch, issue):
    runtime, spec, path, manifest = transcript_prepare_fixture(tmp_path, monkeypatch)
    if issue == "missing_identity":
        del manifest["model_identity"]
    elif issue == "missing_training":
        del manifest["model_identity"]["training"]
    elif issue == "base":
        manifest["model_identity"]["base"] = "another/model"
    elif issue == "training":
        manifest["model_identity"]["training"] = {"run": "fine-tuned"}
    elif issue == "depth":
        manifest["model_identity"]["num_layers"] = 4
    elif issue == "tokenizer":
        manifest["tokenizer"]["assets"] = [{"name": "tokenizer.json", "sha256": "0" * 64}]
    elif issue == "extra_asset":
        manifest["tokenizer"]["assets"] = manifest["tokenizer"]["assets"] + [
            {"name": "added_tokens.json", "sha256": "0" * 64}
        ]
    else:
        del manifest["tokenizer"]["assets"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity|tokenizer"):
        runtime.prepare_fit(path, spec, tmp_path / "gemma3-4b-agentic-regression.npz")


def test_legacy_corpus_cannot_establish_training_identity(tmp_path, monkeypatch):
    runtime = api()
    spec = replace(load_model_spec("qwen35-4b"), training={"run": "trained"})
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(dict(model_hf_id=spec.hf_id, domain="agentic")))
    monkeypatch.setattr(
        runtime, "read_corpus", lambda p: [dict(ids=[1], split=s) for s in ("fit", "held")]
    )
    monkeypatch.setattr(
        runtime, "resolve_snapshot", lambda *a, **k: pytest.fail("snapshot reached")
    )
    with pytest.raises(ValueError, match="training"):
        runtime.prepare_fit(path, spec, tmp_path / "qwen35-4b-agentic-regression.npz")


@pytest.mark.parametrize("source_name,accepted", [("gemma3-4b", True), ("gemma3-4b-bf16", False)])
def test_legacy_registry_alias_requires_same_artifact(tmp_path, monkeypatch, source_name, accepted):
    runtime = api()
    spec = load_model_spec("gemma3-4b")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(dict(model_hf_id=source_name, domain="agentic")))
    monkeypatch.setattr(
        runtime, "read_corpus", lambda p: [dict(ids=[1], split=s) for s in ("fit", "held")]
    )
    root = snapshot(tmp_path)
    monkeypatch.setattr(runtime, "resolve_snapshot", lambda *a, **k: dict(snapshot_path=str(root)))
    if accepted:
        assert (
            runtime.prepare_fit(path, spec, tmp_path / "gemma3-4b-agentic-regression.npz").spec
            == spec
        )
    else:
        with pytest.raises(ValueError, match="model"):
            runtime.prepare_fit(path, spec, tmp_path / "gemma3-4b-agentic-regression.npz")


def test_tokenizer_asset_hashes_exclude_precision_config_and_include_templates():
    snapshot = dict(
        files=[
            dict(name=name, sha256="a" * 64, bytes=2)
            for name in [
                "config.json",
                "model.safetensors",
                "chat_templates/tool.jinja",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]
        ]
    )
    assert [row["name"] for row in api().tokenizer_asset_hashes(snapshot)] == [
        "chat_templates/tool.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]


@pytest.mark.parametrize(
    "acceptance",
    [
        None,
        {},
        {"status": "passed"},
        {
            "status": "ruling_required",
            "reasons": ["concentration exceeds cap"],
            "concentration": {"fit": {}, "held": {}, "combined": {}},
            "repeated_run_rule": "declared",
        },
        {
            "status": "passed",
            "reasons": ["unresolved"],
            "concentration": {"fit": {}, "held": {}, "combined": {}},
            "repeated_run_rule": "declared",
        },
    ],
)
def test_transcript_acceptance_refused_before_snapshot(tmp_path, monkeypatch, acceptance):
    runtime, spec, path, manifest = transcript_prepare_fixture(tmp_path, monkeypatch)
    manifest["acceptance"] = acceptance
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        runtime, "resolve_snapshot", lambda *a, **k: pytest.fail("snapshot reached")
    )
    with pytest.raises(ValueError, match="acceptance|ruling"):
        runtime.prepare_fit(path, spec, tmp_path / "gemma3-4b-agentic-regression.npz")
