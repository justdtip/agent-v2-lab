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


@pytest.mark.parametrize("method", ["forward", "finite_difference"])
def test_jlens_averaging_and_linearity_for_view(method: str) -> None:
    view = _View()
    probe = mx.array([3.0], dtype=mx.float32)
    mapped, stats = jlens.jlens_map(view, 1, probe, [[1, 2], [4]], method=method)

    negative, _ = jlens.jlens_map(view, 1, -probe, [[1, 2], [4]], method=method)

    assert stats["method"] == method
    np.testing.assert_allclose(np.asarray(mapped + negative), 0.0, atol=1e-4)


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
    def fail_before_load(*_args):
        calls.append("guard")
        raise SystemExit(7)

    monkeypatch.setattr(guard, "require_idle_gpu", fail_before_load)
    monkeypatch.setattr("sys.argv", ["agent-v2-jlens"])

    with pytest.raises(SystemExit, match="7"):
        jlens.main()

    assert calls == ["guard"]
