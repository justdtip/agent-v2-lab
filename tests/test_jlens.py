import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from local_llm_lab.pipeline import jlens


def test_render_probe_prompt_forwards_the_selected_spec(monkeypatch) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import protocol

    selected = load_model_spec("qwen35-4b")
    seen = []
    monkeypatch.setattr(
        protocol,
        "build_prompt",
        lambda *_args, spec=None, **_kwargs: seen.append(spec) or "prompt",
    )

    assert jlens.render_probe_prompt(None, [], spec=selected) == "prompt"
    assert seen == [selected]


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


def test_probe_layers_threads_selected_method_and_emits_it(monkeypatch) -> None:
    view = _View()
    selected: list[str] = []

    def fake_map(_view, _layer, probe, _corpus, *, position, method):
        del position
        selected.append(method)
        return probe, {"used": 1, "skipped": 0, "method": method}

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [0]

        def decode(self, _ids):
            return "x"

    monkeypatch.setattr(jlens, "jlens_map", fake_map)
    records = jlens.probe_layers(
        view,
        Tokenizer(),
        [1, 2],
        [1],
        [[3]],
        {"candidate": "x"},
        method="finite_difference",
        k=1,
    )

    assert selected == ["finite_difference"]
    assert records[0]["jvp_method"] == "finite_difference"


def test_jlens_cli_checks_gpu_before_cache_setup_or_model_load(monkeypatch) -> None:
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard, policies

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

    monkeypatch.setattr(policies, "resolve_policy", lambda *_args: None)
    monkeypatch.setattr(evaluate, "load_policy", fake_load)
    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: pytest.fail("used the legacy direct MLX loader")
    )
    monkeypatch.setitem(__import__("sys").modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr("sys.argv", ["agent-v2-jlens", "--jvp-method", "finite_difference"])

    with pytest.raises(SystemExit, match="7"):
        jlens.main()

    assert calls == ["guard", "load"]


@pytest.mark.parametrize(
    ("layer_arguments", "expected"),
    [
        (
            [],
            {
                "source": "registry-default",
                "requested": ["0.167", "0.333", "0.5", "0.667", "0.833", "1.0"],
                "fractions": [0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
                "indices": [5, 11, 16, 21, 27, 32],
                "num_layers": 32,
            },
        ),
        (
            ["--layers", "1,0.5,1.0"],
            {
                "source": "cli",
                "requested": ["1", "0.5", "1.0"],
                "fractions": [0.03125, 0.5, 1.0],
                "indices": [1, 16, 32],
                "num_layers": 32,
            },
        ),
    ],
)
def test_jlens_main_uses_registry_policy_actual_depth_and_selection_metadata(
    monkeypatch, tmp_path, layer_arguments, expected
) -> None:
    """Catches hard-coded layers, direct MLX loading, or missing JSON provenance."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard, policies

    selected = SimpleNamespace(
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.167, 0.333, 0.5, 0.667, 0.833, 1.0),
    )
    model = object()

    class Tokenizer:
        bos_token = None

        def encode(self, _text, add_special_tokens=False):
            del add_special_tokens
            return [1]

    tokenizer = Tokenizer()
    seen = []
    task = SimpleNamespace(
        steps=[SimpleNamespace(action=SimpleNamespace(name="read_file", arguments={"path": "x"}))],
        files={"x": ""},
        task_id="fake",
    )
    monkeypatch.setattr(tasks, "make_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(jlens, "_replay_to_step", lambda *_args: ([], []))
    monkeypatch.setattr(jlens, "_unseen_path", lambda *_args: "other")
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(
        models,
        "load_model_spec",
        lambda model: seen.append(("load", model)) or selected,
    )
    monkeypatch.setattr(
        policies,
        "resolve_policy",
        lambda name, spec: seen.append(("policy", name, spec)) or None,
    )
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda name, adapter: seen.append(("model", name, adapter)) or (model, tokenizer),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "mlx_lm",
        SimpleNamespace(
            load=lambda *_args, **_kwargs: pytest.fail("used the legacy direct MLX loader")
        ),
    )
    monkeypatch.setattr(
        jlens.ArchitectureView,
        "from_model",
        lambda loaded: SimpleNamespace(num_layers=32) if loaded is model else None,
    )
    monkeypatch.setattr(
        jlens,
        "render_probe_prompt",
        lambda _tokenizer, _messages, *, spec: seen.append(("render", spec)) or "prompt",
    )
    monkeypatch.setattr(
        jlens,
        "probe_layers",
        lambda _view, _tokenizer, _token_ids, layers, *_args, **_kwargs: (
            seen.append(("probe", layers)) or []
        ),
    )
    monkeypatch.setattr(jlens, "_print_table", lambda *_args: None)
    output = tmp_path / "jlens.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-jlens",
            "--model",
            "qwen35-4b",
            "--corpus-size",
            "1",
            "--top-k",
            "1",
            "--output",
            str(output),
            *layer_arguments,
        ],
    )

    jlens.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["layers"] == expected["indices"]
    assert payload["layer_selection"] == expected
    assert ("policy", "base", selected) in seen
    assert ("model", "fake/hf", None) in seen
    assert ("render", selected) in seen
    assert ("probe", expected["indices"]) in seen


def test_jlens_rejects_malformed_layers_before_gpu_or_loader(monkeypatch) -> None:
    """Catches empty layer cells being discarded before model loading."""
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard

    task = SimpleNamespace(
        steps=[SimpleNamespace(action=SimpleNamespace(name="read_file", arguments={"path": "x"}))],
        files={"x": ""},
        task_id="fake",
    )
    monkeypatch.setattr(tasks, "make_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(jlens, "_replay_to_step", lambda *_args: ([], []))
    monkeypatch.setattr(jlens, "_unseen_path", lambda *_args: "other")
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_args: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_args: pytest.fail("reached the model loader")
    )
    monkeypatch.setattr("sys.argv", ["agent-v2-jlens", "--layers", "1,,2"])

    with pytest.raises(SystemExit) as raised:
        jlens.main()

    assert raised.value.code == 2


def test_jlens_rejects_depth_overflow_before_probe_execution(monkeypatch) -> None:
    """Catches an out-of-range layer being passed into the J-lens computation."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, tasks
    from local_llm_lab.probes import guard, policies

    selected = SimpleNamespace(hf_id="fake/hf", policies={}, probe_layer_fractions=(1.0,))
    model = object()
    task = SimpleNamespace(
        steps=[SimpleNamespace(action=SimpleNamespace(name="read_file", arguments={"path": "x"}))],
        files={"x": ""},
        task_id="fake",
    )
    tokenizer = SimpleNamespace(bos_token=None, encode=lambda *_args, **_kwargs: [1])
    monkeypatch.setattr(models, "load_model_spec", lambda _model: selected)
    monkeypatch.setattr(tasks, "make_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(jlens, "_replay_to_step", lambda *_args: ([], []))
    monkeypatch.setattr(jlens, "_unseen_path", lambda *_args: "other")
    monkeypatch.setattr(jlens, "render_probe_prompt", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda *_args: None)
    monkeypatch.setattr(evaluate, "load_policy", lambda *_args: (model, tokenizer))
    monkeypatch.setattr(
        jlens.ArchitectureView, "from_model", lambda _model: SimpleNamespace(num_layers=32)
    )
    monkeypatch.setattr(
        jlens, "probe_layers", lambda *_args, **_kwargs: pytest.fail("reached probe execution")
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "mlx_lm",
        SimpleNamespace(load=lambda *_args, **_kwargs: (model, tokenizer)),
    )
    monkeypatch.setattr("sys.argv", ["agent-v2-jlens", "--layers", "33"])

    with pytest.raises(SystemExit) as raised:
        jlens.main()

    assert raised.value.code == 2


def test_jlens_rejects_policy_and_adapter_alias_together(monkeypatch, tmp_path, capsys) -> None:
    """Catches ambiguous registry policy and deprecated directory alias selection."""
    monkeypatch.setattr(
        "sys.argv",
        ["agent-v2-jlens", "--policy", "base", "--adapter", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as raised:
        jlens.main()

    assert raised.value.code == 2
    assert "cannot be used together" in capsys.readouterr().err
