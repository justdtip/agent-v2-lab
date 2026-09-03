from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from local_llm_lab.pipeline import jlens


class _View:
    """Small view seam: no model or checkpoint is involved in these contracts."""

    num_layers = 2

    def residuals(self, ids, layers):
        values = mx.array(ids).astype(mx.float32)[None, :, None]
        return {layer: values + float(layer) for layer in layers}

    def tail(self, layer):
        del layer
        return lambda value: (2.0 * value).astype(mx.float32)

    def final_norm(self, value):
        return value.astype(mx.float32)

    def unembed(self, value):
        return value.astype(mx.float32)


class _NonlinearView(_View):
    def tail(self, layer):
        del layer
        return lambda value: (value * value).astype(mx.float32)


@pytest.mark.parametrize("method", ["forward", "finite_difference"])
def test_jlens_averaging_and_linearity_for_view(method: str) -> None:
    view = _View()
    probe = mx.array([3.0], dtype=mx.float32)
    mapped, stats = jlens.jlens_map(view, 1, probe, [[1, 2], [4]], method=method)

    negative, _ = jlens.jlens_map(view, 1, -probe, [[1, 2], [4]], method=method)

    assert stats["method"] == method
    np.testing.assert_allclose(np.asarray(mapped + negative), 0.0, atol=1e-4)


@pytest.mark.parametrize("method", ["forward", "finite_difference"])
def test_jlens_map_matches_independent_nonlinear_context_average(method: str) -> None:
    view = _NonlinearView()
    probe = mx.array([2.0], dtype=mx.float32)
    mapped, stats = jlens.jlens_map(view, 1, probe, [[2], [4], []], method=method)

    # residuals are ids + layer, and d(x²)/dx = 2x: mean(2*3*2, 2*5*2) = 16.
    np.testing.assert_allclose(np.asarray(mapped), [16.0], atol=1e-3)
    assert mapped.dtype == mx.float32
    assert stats == {"used": 2, "skipped": 1, "method": method}


@pytest.mark.parametrize("method", ["forward", "finite_difference"])
def test_jvp_handles_zero_primal_and_zero_tangent(method: str) -> None:
    view = _NonlinearView()
    zero = mx.zeros((1, 2, 1), dtype=mx.float32)
    tangent = mx.ones((1, 2, 1), dtype=mx.float32)

    from_zero_primal = jlens.jacobian_vector_product(view, 1, zero, tangent, method=method)
    from_zero_tangent = jlens.jacobian_vector_product(view, 1, tangent, zero, method=method)

    np.testing.assert_allclose(np.asarray(from_zero_primal), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(from_zero_tangent), 0.0, atol=1e-5)


def test_residual_at_delegates_to_one_view_head_pass() -> None:
    class CountingView(_View):
        calls = 0

        def residuals(self, ids, layers):
            self.calls += 1
            return super().residuals(ids, layers)

    view = CountingView()
    result = jlens.residual_at(view, [1, 2, 3], 1)

    assert view.calls == 1
    assert result.dtype == mx.float32


def test_public_encode_and_distribution_use_public_view_seams() -> None:
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [len(text)]

    assert jlens.encode(Tokenizer(), "abc") == [3]
    distribution = jlens.distribution(_View(), mx.array([1.0], dtype=mx.float32))
    np.testing.assert_allclose(np.asarray(distribution), [1.0])


def test_jlens_cli_checks_gpu_before_cache_setup_or_model_load(monkeypatch) -> None:
    from local_llm_lab.pipeline import tasks
    from local_llm_lab.probes import guard

    task = SimpleNamespace(
        steps=[SimpleNamespace(action=SimpleNamespace(name="read_file", arguments={"path": "x"}))],
        files={"x": ""},
        task_id="fake",
    )
    monkeypatch.setattr(tasks, "make_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(jlens, "_replay_to_step", lambda *_args: ([], []))
    monkeypatch.setattr(jlens, "_unseen_path", lambda *_args: "other")
    calls = []
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: calls.append("guard"))
    from local_llm_lab import project

    monkeypatch.setattr(project, "configure_local_cache", lambda: calls.append("cache"))
    def fake_load(*_args, **_kwargs):
        calls.append("load")
        raise SystemExit(7)

    fake_mlx_lm = SimpleNamespace(load=fake_load)
    monkeypatch.setitem(__import__("sys").modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr("sys.argv", ["agent-v2-jlens"])

    with pytest.raises(SystemExit, match="7"):
        jlens.main()

    assert calls == ["guard", "cache", "load"]
