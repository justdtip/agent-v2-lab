"""Tiny NumPy fakes only: never import the native regression test module."""

import sys
import types
import weakref

import numpy as np
import pytest

from local_llm_lab.pipeline.lens_fitting import regression as reg


@pytest.fixture
def fake_mlx(monkeypatch):
    mx = types.ModuleType("mlx.core")
    for name in ("zeros", "array", "sum", "all", "isfinite", "mean", "diag", "eye", "float32"):
        setattr(mx, name, getattr(np, name))
    mx.eye = lambda n, **kw: np.eye(n, **({"dtype": np.float32} | kw))
    mx.cpu = "cpu"
    mx.linalg = types.SimpleNamespace(solve=lambda a, b, **kw: np.linalg.solve(a, b))
    mx.evaluations = []
    mx.eval = lambda *args: mx.evaluations.append(len(args))
    mx.get_peak_memory = lambda: 123
    mx.get_active_memory = lambda: 45
    mx.get_cache_memory = lambda: 0
    package = types.ModuleType("mlx")
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx


class View:
    hidden_size, num_layers = 2, 3

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def native_residuals(self, ids, layers):
        self.calls.append(tuple(ids))
        if self.fail and ids[0] == 9:
            raise RuntimeError("forward failure")
        x = np.array([[[i, 1] for i in ids]], dtype=np.float32)
        return {1: x, 2: x * 2, 3: x @ np.array([[2, 1], [-1, 3]], dtype=np.float32)}


ROWS = [
    {"ids": [1, 2, 3], "split": "fit", "score_positions": [1, 2]},
    {"ids": [9, 10], "split": "held", "score_positions": [1]},
    {"ids": [4, 5], "split": "fit"},
    {"ids": [11, 12], "split": "held"},
]


def test_spill_exact_stats_order_and_layer_loads(fake_mlx, tmp_path):
    expected, old_counts = reg.accumulate(View(), ROWS, residual_source="native")
    fake_mlx.evaluations.clear()
    view, events = View(), []
    sums, counts = reg.accumulate(
        view,
        iter(ROWS),
        residual_source="native",
        statistics_storage="split_spill",
        scratch_dir=tmp_path,
        memory_progress=events.append,
    )
    assert view.calls == [(1, 2, 3), (4, 5), (9, 10), (11, 12)]
    assert fake_mlx.evaluations == [3] * 8
    assert counts["fit"] == old_counts["fit"]
    assert counts["held"] == old_counts["held"]
    assert counts["batch_size"] == 1
    assert counts["statistics_sync"] == "per_layer"
    assert len(list(tmp_path.rglob("*.npy"))) == 12
    for split in ("fit", "held"):
        assert not isinstance(sums[split], dict)
        for layer in (1, 2):
            first, second = sums[split][layer], sums[split][layer]
            assert first is not second
            got = sums[split].pop(layer)
            for name in ("xtx", "xty", "yty"):
                np.testing.assert_array_equal(
                    getattr(got, name), getattr(expected[split][layer], name)
                )
            assert got.n == expected[split][layer].n
        assert len(sums[split]) == 0
    assert not list(tmp_path.rglob("*.npy"))
    assert all(event["active_memory_bytes"] == 45 for event in events)
    first = [event for event in events if event.get("sequence") == 1]
    assert [event["phase"] for event in first] == [
        "before_forward",
        "after_graph",
        "after_statistics_eval",
        "after_residual_release",
    ]
    assert all(event["input_positions"] == 3 and event["scored_positions"] == 2 for event in first)
    assert [event["split"] for event in events if event["phase"] == "split_allocated"] == [
        "fit",
        "held",
    ]


def test_fit_exact_maps_and_cleanup(fake_mlx, tmp_path):
    expected = reg.fit_regression(View(), ROWS, residual_source="native")
    actual = reg.fit_regression(
        View(),
        ROWS,
        residual_source="native",
        statistics_storage="split_spill",
        scratch_parent=tmp_path,
    )
    for layer in expected.maps:
        np.testing.assert_array_equal(actual.maps[layer], expected.maps[layer])
        assert (
            actual.per_layer[str(layer)]["candidates"]
            == expected.per_layer[str(layer)]["candidates"]
        )
    assert list(tmp_path.iterdir()) == []


def test_failure_cleans_owned_scratch(fake_mlx, tmp_path):
    sentinel = tmp_path / "keep"
    sentinel.write_text("unrelated")
    with pytest.raises(RuntimeError, match="forward failure"):
        reg.fit_regression(
            View(fail=True),
            ROWS,
            residual_source="native",
            statistics_storage="split_spill",
            scratch_parent=tmp_path,
        )
    assert list(tmp_path.iterdir()) == [sentinel]
    with pytest.raises(RuntimeError, match="forward failure"):
        reg.accumulate(
            View(fail=True),
            ROWS,
            residual_source="native",
            statistics_storage="split_spill",
            scratch_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    "kwargs", [{"statistics_storage": "unknown"}, {"statistics_storage": "split_spill"}]
)
def test_invalid_storage_before_forward(fake_mlx, kwargs):
    view = View()
    with pytest.raises(ValueError):
        reg.accumulate(view, ROWS, residual_source="native", **kwargs)
    assert not view.calls


def test_only_one_split_live_and_no_cached_layer(fake_mlx, monkeypatch, tmp_path):
    references = []
    original = reg.SufficientStats.zeros

    def zeros(cls, dimension):
        assert sum(ref() is not None for ref in references) < View.num_layers - 1
        result = original(dimension)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(reg.SufficientStats, "zeros", classmethod(zeros))
    sums, _ = reg.accumulate(
        View(),
        ROWS,
        residual_source="native",
        statistics_storage="split_spill",
        scratch_dir=tmp_path,
    )
    assert all(ref() is None for ref in references)
    loaded = sums["fit"][1]
    ref = weakref.ref(loaded)
    del loaded
    assert ref() is None


def test_solve_and_spill_failures_clean_scratch(fake_mlx, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    with monkeypatch.context() as patch:
        patch.setattr(reg, "solve_layer", fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            reg.fit_regression(
                View(),
                ROWS,
                residual_source="native",
                statistics_storage="split_spill",
                scratch_parent=tmp_path,
            )
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(reg.SpilledStats, "store", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        reg.accumulate(
            View(),
            ROWS,
            residual_source="native",
            statistics_storage="split_spill",
            scratch_dir=tmp_path,
        )
    assert not list(tmp_path.iterdir())
