"""Exercise the native-source fit CLI on real NumPy arithmetic, without importing MLX."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def test_native_fit_cli_counts_and_artifact(tmp_path, monkeypatch, capsys):
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.lens_fitting import runtime

    mx = ModuleType("mlx.core")
    for name in ("zeros", "eye", "sum", "mean", "diag", "isfinite", "all"):
        setattr(mx, name, getattr(np, name))
    mx.float32 = np.float32
    mx.eval = lambda *args: None
    mx.cpu = None
    mx.linalg = SimpleNamespace(solve=lambda a, b, **kwargs: np.linalg.solve(a, b))
    mx.get_peak_memory = lambda: 0
    monkeypatch.setitem(sys.modules, "mlx", ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    class View:
        hidden_size, num_layers = 2, 3

        def residuals(self, *args):
            raise AssertionError("native request reached the hand-run loop")

        def native_residuals(self, ids, layers):
            x = np.array([[[i, 1] for i in ids]], dtype=np.float32)
            return {1: x, 2: x * 2, 3: x * 3}

    spec = load_model_spec("gemma3-4b")
    out = tmp_path / "lens.npz"
    rows = [{"ids": [1, 2, 3], "split": "fit"}, {"ids": [4, 5], "split": "held"}]
    prepared = SimpleNamespace(
        rows=rows,
        output=out,
        corpus_manifest_sha256="test",
        manifest={"domain": "prose", "sequences": {"sha256": "test"}},
    )
    loaded = SimpleNamespace(
        model=SimpleNamespace(eval=lambda: None),
        view=View(),
        spec=spec,
        snapshot={"snapshot_sha256": "frozen-snapshot"},
        lock_path=tmp_path / "lock",
        resolved=SimpleNamespace(cache_strategy="none"),
    )
    monkeypatch.setattr(runtime, "prepare_fit", lambda *a, **kw: prepared)
    monkeypatch.setattr(runtime, "load_runtime", lambda *a, **kw: loaded)
    monkeypatch.setattr(runtime, "configure_allocator_cache", lambda: {"limit_bytes": 0})
    monkeypatch.setattr(runtime, "resource_snapshot", lambda: {"peak_memory_gib": 0})
    file = Path(__file__).resolve().parents[1] / "scripts/lens_fit.py"
    module_spec = importlib.util.spec_from_file_location("fit_cli", file)
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    assert (
        cli.main(
            [
                "--kind",
                "regression",
                "--model",
                "gemma3-4b",
                "--corpus",
                "unused",
                "--out",
                str(out),
                "--residual-source",
                "native",
            ]
        )
        == 0
    )
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    sequences = [e for e in events if e["event"] == "sequence"]
    assert [e["tokens_done"] for e in sequences] == [3, 5]
    assert all(e["counts"]["residual_source"] == "native" for e in sequences)
    metadata = json.loads(out.with_suffix(".json").read_text())
    assert metadata["residual_source"] == metadata["counts"]["residual_source"] == "native"
    assert metadata["layers"] == [1, 2]
    assert metadata["checkpoint"]["snapshot_sha256"] == "frozen-snapshot"
    assert metadata["counts"]["held"]["positions"] == 2
