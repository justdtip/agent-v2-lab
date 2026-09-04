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

    def fake_map(
        _view, _layer, probe, _corpus, *, source_positions, readouts, method, capture_dtype
    ):
        del source_positions, capture_dtype
        selected.append(method)
        return (
            {name: probe for name in readouts},
            {"used": 1, "skipped": 0, "method": method, "window": {}, "source_positions": []},
        )

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [0]

        def decode(self, _ids):
            return "x"

    monkeypatch.setattr(jlens, "jlens_readouts", fake_map)
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


def test_jlens_cli_checks_gpu_before_cache_setup_or_model_load(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-jlens",
            "--jvp-method",
            "finite_difference",
            "--skip-preflight-check",
            "--log-dir",
            str(tmp_path / "run"),
        ],
    )

    with pytest.raises(SystemExit, match="7"):
        jlens.main()

    assert calls == ["guard", "load"]
    # R26(a): the log covers the whole run, so it exists even on the failing path.
    assert (tmp_path / "run" / "run.log").is_file()


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
        name="fake-model",
        hf_id="fake/hf",
        policies={},
        probe_layer_fractions=(0.167, 0.333, 0.5, 0.667, 0.833, 1.0),
        chat=SimpleNamespace(template_kwargs={}),
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
        lambda given, adapter: seen.append(("model", given.hf_id, adapter))
        or (
            model,
            tokenizer,
            SimpleNamespace(num_layers=32),
            SimpleNamespace(as_dict=lambda: {"resolved": "fake"}),
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "mlx_lm",
        SimpleNamespace(
            load=lambda *_args, **_kwargs: pytest.fail("used the legacy direct MLX loader")
        ),
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
            "--jvp-method",
            "finite_difference",
            "--skip-preflight-check",
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
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args: (model, tokenizer, SimpleNamespace(num_layers=32), object()),
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


# --------------------------------------------------------------- EXP-001 (issue #54) additions


class _BroadcastView(_View):
    """A tail that carries a source position forward, the way a recurrent state does.

    ``cumsum`` over the position axis makes the output at every position after the source
    depend on the source, so ``self``, ``future`` and ``all`` are genuinely different
    readouts rather than three names for one number.
    """

    def tail(self, layer):
        del layer
        return lambda value: mx.cumsum(value.astype(mx.float32), axis=1)


def test_jvp_method_defaults_to_the_method_the_preflight_recorded(tmp_path) -> None:
    """R18a/EXP-001 §3.2: the established method, not the literal ``forward``."""
    spec = SimpleNamespace(name="fake-model")
    (tmp_path / "fake-model.json").write_text(
        json.dumps({"jvp": {"finite": True, "layer": 4, "method": "finite_difference"}}),
        encoding="utf-8",
    )

    assert jlens.resolve_jvp_method(None, spec, output_root=tmp_path) == (
        "finite_difference",
        "preflight",
    )
    assert jlens.resolve_jvp_method("forward", spec, output_root=tmp_path) == ("forward", "cli")


def test_jvp_method_without_a_flag_or_a_preflight_record_fails_closed(tmp_path) -> None:
    """Neither source establishes a method: a named error, never a silent default."""
    spec = SimpleNamespace(name="fake-model")

    with pytest.raises(jlens.JvpMethodUnresolved) as raised:
        jlens.resolve_jvp_method(None, spec, output_root=tmp_path)

    assert "fake-model" in str(raised.value)


def test_future_readout_raises_rather_than_reading_a_structural_zero() -> None:
    """The Head of Interpretability's A1: an empty window raises, naming context and position."""
    probe = mx.array([3.0], dtype=mx.float32)

    with pytest.raises(jlens.EmptyFutureWindowError) as raised:
        jlens.jlens_readouts(
            _BroadcastView(),
            1,
            probe,
            [[1, 2, 3]],
            source_positions=jlens.resolve_source_positions("-1"),
        )

    message = str(raised.value)
    assert "context 0" in message and "position 2" in message


def test_self_future_and_all_are_three_distinct_readouts_of_one_jvp() -> None:
    """B3: one JVP per context yields self, future and all; ``all`` is their sum."""
    probe = mx.array([3.0], dtype=mx.float32)

    mapped, stats = jlens.jlens_readouts(
        _BroadcastView(),
        1,
        probe,
        [[1, 2, 3, 4]],
        source_positions=jlens.resolve_source_positions("0.25"),
    )

    # source at position 1 of a 4-token context: the cumulative tail carries the tangent to
    # positions 1, 2 and 3, so self = probe, future = 2 * probe, all = 3 * probe.
    np.testing.assert_allclose(np.asarray(mapped["self"]), [3.0], atol=1e-3)
    np.testing.assert_allclose(np.asarray(mapped["future"]), [6.0], atol=1e-3)
    np.testing.assert_allclose(np.asarray(mapped["all"]), [9.0], atol=1e-3)
    assert stats["window"]["median_future_window"] == 2


class _CharTokenizer:
    """One id per character: long enough corpus contexts without a real tokenizer."""

    bos_token = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % 64 for character in text]

    def decode(self, ids):
        return "".join(chr(64 + int(i) % 26) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        del add_generation_prompt, tokenize, kwargs
        return "\n".join(str(message["content"]) for message in messages)


class _CliView(_BroadcastView):
    num_layers = 4
    vocabulary = 64

    def layer_kind(self, index):
        return "linear_attention" if (index + 1) % 4 else "attention"

    def unembed(self, value):
        return (value.astype(mx.float32) * mx.ones((self.vocabulary,), dtype=mx.float32))


def _fake_preflight(tmp_path, spec):
    root = tmp_path / "preflight"
    root.mkdir(exist_ok=True)
    (root / f"{spec.name}.json").write_text(
        json.dumps(
            {
                "jvp": {"finite": True, "layer": 2, "method": "finite_difference"},
                "fp32_manual_vs_native": {"frobenius_relative": 0.004, "max_abs": 0.02},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_jlens_main_logs_one_event_per_layer_and_records_r34_and_r35(
    monkeypatch, tmp_path
) -> None:
    """R26(a)(g), R34, R35, R18a and the preflight-recorded JVP default, end to end."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, preflight, tasks
    from local_llm_lab.probes import guard, policies

    spec = models.load_model_spec("qwen35-4b")
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", _fake_preflight(tmp_path, spec))
    monkeypatch.setattr(preflight, "require_preflight", lambda *_args, **_kwargs: None)
    task = SimpleNamespace(
        steps=[SimpleNamespace(action=SimpleNamespace(name="read_file", arguments={"path": "x"}))],
        files={"x": ""},
        task_id="fake",
    )
    monkeypatch.setattr(tasks, "make_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(jlens, "_replay_to_step", lambda *_args: ([], []))
    monkeypatch.setattr(jlens, "_unseen_path", lambda *_args: "other")
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda *_args: None)
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args, **_kwargs: (object(), _CharTokenizer(), _CliView(), None),
    )
    monkeypatch.setattr(jlens, "render_probe_prompt", lambda *_args, **_kwargs: "note prefix")
    monkeypatch.setattr(jlens, "_print_table", lambda *_args: None)
    output = tmp_path / "run" / "jlens.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-jlens",
            "--model",
            "qwen35-4b",
            "--layers",
            "2,3",
            "--corpus-size",
            "2",
            "--corpus-length",
            "24",
            "--top-k",
            "1",
            "--skip-preflight-check",
            "--output",
            str(output),
        ],
    )

    jlens.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["jvp_method"] == "finite_difference"
    assert payload["jvp_method_source"] == "preflight"

    conformance = payload["conformance"]
    for key in (
        "layer_index_convention",
        "layers",
        "layer_kinds",
        "source_positions",
        "output_positions_read",
        "readouts",
        "primary_readout",
        "self_only_limiting_case",
        "corpus_size",
        "corpus_length",
        "window",
        "jvp_method",
        "jvp_method_source",
        "capture_dtype",
    ):
        assert key in conformance, key
    assert "after block L-1" in conformance["layer_index_convention"]
    assert conformance["window"]["median_future_window"] >= 1
    assert set(conformance["readouts"]) == set(jlens.READOUTS)

    comparability = payload["comparability"]
    for key in (
        "policy",
        "adapter",
        "derivative_method",
        "prompt_rendering",
        "estimator_variant",
        "layer_selection",
        "generator_version",
        "fp32_manual_vs_native",
    ):
        assert key in comparability, key
    assert comparability["prompt_rendering"]["template_kwargs"] == dict(
        spec.chat.template_kwargs
    )
    assert comparability["fp32_manual_vs_native"] == {
        "frobenius_relative": 0.004,
        "max_abs": 0.02,
    }

    run_log = output.parent / "run.log"
    events = output.parent / "events.jsonl"
    assert run_log.is_file() and events.is_file()
    records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    layer_events = [
        record for record in records if record.get("label") == "layer"
    ]
    assert [record["step"] for record in layer_events] == [1, 2]
    assert records[0]["kind"] == "start"
    assert records[-1]["kind"] == "end"


# --------------------------------------- EXP-001 §3.5 (issue #54): the kind-matched family


def _hybrid_args(**overrides):
    """A real ``TextModelArgs`` at toy width.

    R31: the seam this slice leans on is the library's hybrid period, so the configuration
    dataclass and ``DecoderLayer`` are the real ones; only the widths are shrunk, and no
    weights are loaded. Left alone, ``num_hidden_layers`` and ``full_attention_interval``
    carry the 4B's own values, which is what EXP-001 §3.5 pre-registers its layer list from.
    """
    from mlx_lm.models import qwen3_5

    tiny = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
    }
    return qwen3_5.TextModelArgs(**{**tiny, **overrides})


def _library_layer_kinds(args) -> dict[int, str]:
    """Probe-layer kinds from the library's own ``DecoderLayer.is_linear`` (R31).

    Probe layer ``L`` is the residual after block ``L - 1`` (R34), so block ``index`` writes
    probe layer ``index + 1``.
    """
    from mlx_lm.models import qwen3_5

    return {
        index + 1: (
            "linear_attention"
            if qwen3_5.DecoderLayer(args=args, layer_idx=index).is_linear
            else "attention"
        )
        for index in range(args.num_hidden_layers)
    }


def test_in_band_layers_drop_the_shallow_fraction_and_the_final_layer() -> None:
    """EXP-001 §2: fractions 1/3 to 5/6 decide; fraction 1/6 and the final layer are reported."""
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.probes.policies import resolve_layers

    depth = _hybrid_args().num_hidden_layers
    selection = resolve_layers(None, load_model_spec("qwen35-4b"), depth)

    band = jlens.in_band_layers(selection.indices, depth)

    assert band == tuple(
        index
        for index in selection.indices
        if round(depth / 3) <= index <= round(depth * 5 / 6) and index != depth
    )
    assert selection.indices[0] not in band
    assert depth not in band


def test_kind_matched_family_reproduces_the_4b_sweep_from_its_own_configuration() -> None:
    """EXP-001 §3.5 and issue #54: registry fractions plus ``full_attention_interval`` partners.

    The expectation is computed from the library's own layer kinds and the registry's own
    fractions, so this pins the *derivation*: the code must build the family the way §3.5 says,
    from whatever those two sources hold. It is by construction blind to a change in the sources
    themselves -- move the fractions and the expectation moves with them -- which is why the
    literal nine-layer list is pinned separately below (C4, issue #62).
    """
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.probes.policies import resolve_layers

    args = _hybrid_args()
    kinds = _library_layer_kinds(args)
    depth = args.num_hidden_layers
    selection = resolve_layers(None, load_model_spec("qwen35-4b"), depth)

    family = jlens.kind_matched_layer_family(
        selection.indices,
        num_layers=depth,
        period=args.full_attention_interval,
        kind_of=kinds.get,
    )

    attention = tuple(layer for layer, kind in kinds.items() if kind == "attention")
    expected_partners = tuple(
        sorted(
            {
                min(attention, key=lambda candidate: (abs(candidate - layer), candidate))
                for layer in jlens.in_band_layers(selection.indices, depth)
                if kinds[layer] != "attention"
            }
        )
    )
    assert family.partners == expected_partners
    assert family.layers == tuple(sorted(set(selection.indices) | set(expected_partners)))
    assert len(family.layers) == len(selection.indices) + len(expected_partners)
    assert all(1 <= layer <= depth for layer in family.layers)
    assert family.kinds == {layer: kinds[layer] for layer in family.layers}
    assert family.period == args.full_attention_interval

    # §3.5 structurally: an in-band layer written by a recurrent block is paired with an
    # attention-written layer nearer than one hybrid period; an in-band layer that is itself
    # an attention output takes no partner.
    for layer in family.in_band:
        assert family.roles[layer] == "primary"
        if kinds[layer] == "attention":
            assert layer not in family.pairs
        else:
            partner = family.pairs[layer]
            assert kinds[partner] == "attention"
            assert 0 < abs(partner - layer) < family.period
            assert family.roles[partner] in ("partner", "primary")
    for layer in family.partners:
        assert family.roles[layer] == "partner"
    assert set(family.primary_layers) == set(family.in_band) | set(family.partners)


def test_the_4b_default_family_is_the_pre_registered_nine_layer_list() -> None:
    """C4 (issue #62): pin EXP-001 §3.5's list as a literal, beside the derivation above.

    The derivation test computes its expectation from the two sources the code itself reads --
    the library's layer-kind rule and the registry's fractions -- which is the stronger test of
    the derivation but is blind to a change in those sources: move the fractions and the
    expectation moves with them. This one is blind to nothing, because the nine layers are
    written down. Together they fail on a change in either source and on a change in both that
    happens to agree, which is what §3.5 pre-registers.
    """
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.probes.policies import resolve_layers

    args = _hybrid_args()
    depth = args.num_hidden_layers
    selection = resolve_layers(None, load_model_spec("qwen35-4b"), depth)

    family = jlens.kind_matched_layer_family(
        selection.indices,
        num_layers=depth,
        period=args.full_attention_interval,
        kind_of=_library_layer_kinds(args).get,
    )

    assert family.layers == (5, 11, 12, 16, 20, 21, 27, 28, 32)
    assert family.partners == (12, 20, 28)
    assert family.pairs == {11: 12, 21: 20, 27: 28}
    assert family.in_band == (11, 16, 21, 27)


@pytest.mark.parametrize(("layer", "expected"), [(5, 4), (6, 4), (7, 8)])
def test_partner_is_the_nearest_opposite_kind_layer_and_the_lower_index_on_a_tie(
    layer: int, expected: int
) -> None:
    """Smaller depth difference wins; an exact tie takes the lower index."""
    family = jlens.kind_matched_layer_family((layer,), num_layers=8, period=4)

    assert family.pairs == {layer: expected}
    assert family.partners == (expected,)
    assert family.layers == tuple(sorted((layer, expected)))
    assert family.kinds[expected] == "attention"
    assert family.kinds[layer] == "linear_attention"


@pytest.mark.parametrize(
    ("period", "fragment"),
    [(None, "full_attention_interval"), (1, "every layer"), (9, "exceeds")],
)
def test_a_configuration_with_one_block_kind_keeps_the_selection_and_records_why(
    period: int | None, fragment: str
) -> None:
    """No opposite kind exists, so the sweep is the selection itself with the reason recorded.

    It returns rather than raises because EXP-001 §5 runs this same code on the dense 3B as
    the R35 comparator, where every layer is an attention output.
    """
    selected = (3, 5, 8)

    family = jlens.kind_matched_layer_family(selected, num_layers=8, period=period)

    assert family.layers == selected
    assert family.partners == ()
    assert family.pairs == {}
    assert family.derived is True
    assert fragment in family.reason


def test_explicit_layers_are_honoured_verbatim_and_take_no_partners() -> None:
    """An operator's own list is the list; the derivation is bypassed and marked as bypassed."""
    family = jlens.kind_matched_layer_family((7, 5), num_layers=8, period=4, derive=False)

    assert family.layers == (7, 5)
    assert family.partners == ()
    assert set(family.roles.values()) == {"explicit"}
    assert family.derived is False
    assert "verbatim" in family.reason


def test_hybrid_period_is_read_from_the_loaded_models_own_configuration() -> None:
    """The period is configuration, never a literal: it is read off a real ``TextModelArgs``."""
    args = _hybrid_args(num_hidden_layers=10, full_attention_interval=3)
    view = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(args=args)))

    period, source = jlens.hybrid_period(view)

    assert period == args.full_attention_interval
    assert source.endswith("full_attention_interval")
    assert jlens.hybrid_period(SimpleNamespace()) == (None, "unavailable")


class _HybridCliView(_CliView):
    """A deeper hybrid for the single-decision CLI, its period from a real ``TextModelArgs``."""

    num_layers = 10

    def __init__(self) -> None:
        self.args = _hybrid_args(num_hidden_layers=self.num_layers, full_attention_interval=3)
        self.model = SimpleNamespace(language_model=SimpleNamespace(args=self.args))
        self._kinds = _library_layer_kinds(self.args)

    def layer_kind(self, index):
        return self._kinds[index + 1]


def test_jlens_default_layers_take_the_kind_matched_family(monkeypatch, tmp_path) -> None:
    """EXP-001 §3.4 omits ``--layers``, so the spot check must sweep the derived family too."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, preflight, tasks
    from local_llm_lab.probes import guard, policies

    spec = models.load_model_spec("qwen35-4b")
    view = _HybridCliView()
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", _fake_preflight(tmp_path, spec))
    monkeypatch.setattr(preflight, "require_preflight", lambda *_args, **_kwargs: None)
    task = SimpleNamespace(
        steps=[SimpleNamespace(action=SimpleNamespace(name="read_file", arguments={"path": "x"}))],
        files={"x": ""},
        task_id="fake",
    )
    monkeypatch.setattr(tasks, "make_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(jlens, "_replay_to_step", lambda *_args: ([], []))
    monkeypatch.setattr(jlens, "_unseen_path", lambda *_args: "other")
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_args: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda *_args: None)
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_args, **_kwargs: (object(), _CharTokenizer(), view, None),
    )
    monkeypatch.setattr(jlens, "render_probe_prompt", lambda *_args, **_kwargs: "note prefix")
    monkeypatch.setattr(jlens, "_print_table", lambda *_args: None)
    output = tmp_path / "run" / "jlens.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-jlens",
            "--model",
            "qwen35-4b",
            "--corpus-size",
            "2",
            "--corpus-length",
            "24",
            "--top-k",
            "1",
            "--skip-preflight-check",
            "--output",
            str(output),
        ],
    )

    jlens.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    family = payload["layer_family"]
    assert payload["layer_selection"]["source"] == "registry-default"
    assert family["derived"] is True
    assert family["hybrid_period"] == view.args.full_attention_interval
    assert family["partners"]
    assert payload["layers"] == family["layers"]
    assert set(family["layers"]) > set(payload["layer_selection"]["indices"])
    assert payload["conformance"]["layer_family"] == family
    assert payload["comparability"]["layer_family"] == family
    for layer in family["partners"]:
        assert payload["layer_roles"][str(layer)] == "partner"
        assert payload["layer_kinds"][str(layer)] == "attention"
    assert [record["layer"] for record in payload["records"]] == family["layers"]
