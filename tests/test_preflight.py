from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec, ResolvedSpec
from local_llm_lab.pipeline.preflight import (
    _CALIBRATION_POINTS,
    _ENVELOPES,
    _TRAINING_HEADROOM_FRACTION,
    _estimated_peak_gib,
    _linear_attention_state_shape,
    _native_gate_passed,
    _residual_metrics,
    longest_row_tokens,
    require_preflight,
    run_preflight,
    run_residual_control,
)
from local_llm_lab.training.gated_delta_chunked import training_state_bytes


def _spec(*, memory_budget_gib: float = 1.0, train: dict[str, object] | None = None) -> ModelSpec:
    return ModelSpec(
        name="fake-model",
        hf_id="org/fake-model",
        family="fake",
        chat=ChatSpec("off", {"chat_template": "fake"}, "<eot>", ()),
        lora=LoraSpec("attention+mlp", 2, 4.0, 0.0),
        train={"batch_size": 2, "max_seq_length": 5} if train is None else train,
        cache_strategy="auto",
        cache_equivalence_verified={"date": "2026-09-04", "sha256": "a" * 64},
        probe_layer_fractions=(0.5,),
        memory_budget_gib=memory_budget_gib,
        policies={},
    )


# Reported by ``mx.device_info()`` on the M4 Pro this project runs on: 17.759765625 GiB, well
# under the 22 GiB the registry declares, which is the whole point of R32(b).
_DEVICE_WORKING_SET_BYTES = 19069665280
_DEVICE_WORKING_SET_GIB = _DEVICE_WORKING_SET_BYTES / 1024**3


def _device_info(working_set_bytes: int = _DEVICE_WORKING_SET_BYTES) -> dict[str, object]:
    return {
        "device_name": "fake-gpu",
        "max_recommended_working_set_size": working_set_bytes,
        "memory_size": working_set_bytes * 2,
    }


class _Tokenizer:
    def __init__(self) -> None:
        self.render_calls: list[dict[str, object]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return list(range(70))

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.render_calls.append({"messages": messages, **kwargs})
        return f"prompt-{len(self.render_calls)}"


class _CacheA:
    def is_trimmable(self) -> bool:
        return False


class _CacheB:
    def is_trimmable(self) -> bool:
        return False


class _TextModule:
    def __call__(self, ids: np.ndarray) -> np.ndarray:
        values = ids.astype(np.float32)[..., None] * np.ones((1, 1, 3), dtype=np.float32)
        return (values + 10.0) * 2.0


class _View:
    num_layers = 4
    hidden_size = 3
    vocab_size = 17
    tie_word_embeddings = True
    text_module = _TextModule()

    def layer_kind(self, index: int) -> str:
        return ("attention", "linear_attention", "attention", "linear_attention")[index]

    def make_cache(self) -> list[object]:
        return [_CacheA(), _CacheB(), _CacheA(), _CacheB()]

    def embed(self, ids: np.ndarray) -> np.ndarray:
        return ids.astype(np.float32)[..., None] * np.ones((1, 1, 3), dtype=np.float32)

    def masks(self, h: np.ndarray, cache) -> dict[str, object]:
        del h, cache
        return {"attention": None, "linear_attention": None}

    def run_block(self, index: int, h: np.ndarray, masks, cache) -> np.ndarray:
        del masks, cache
        return h + float(index + 1)

    def final_norm(self, h: np.ndarray) -> np.ndarray:
        return h * 2.0

    def diagnostic_native_final_residual(self, ids: np.ndarray) -> np.ndarray:
        return self.text_module(ids)

    def residuals(self, ids: np.ndarray, layers: list[int]) -> dict[int, np.ndarray]:
        assert layers == [2]
        return {2: self.embed(ids) + 3.0}

    def lora_targets(self, policy) -> tuple[str, ...]:
        assert policy == "attention+mlp"
        return ("layers.0.q_proj", "layers.3.down_proj")

    def lora_parameter_count(self, keys: tuple[str, ...], rank: int) -> int:
        assert keys == ("layers.0.q_proj", "layers.3.down_proj")
        assert rank == 2
        return 42


class _Recurrence:
    """The library's ``GatedDeltaNet`` exposes exactly these three ints (qwen3_5.py:88-92)."""

    num_v_heads = 4
    head_v_dim = 8
    head_k_dim = 8


class _HybridBlock(dict):
    """mlx's ``nn.Module`` subclasses ``dict``, so a block's members are its values."""

    def __init__(self, recurrence: object | None = None) -> None:
        super().__init__()
        self.is_linear = recurrence is not None
        if recurrence is not None:
            self["linear_attn"] = recurrence


class _HybridView(_View):
    """A view whose two linear-attention blocks expose the recurrence state shape."""

    blocks = [
        _HybridBlock(),
        _HybridBlock(_Recurrence()),
        _HybridBlock(),
        _HybridBlock(_Recurrence()),
    ]


class _ConfigTextModule(_TextModule):
    """A text module whose ``args`` carry the shape, as ``TextModel`` does (qwen3_5.py:281)."""

    args = SimpleNamespace(
        linear_num_value_heads=4, linear_value_head_dim=8, linear_key_head_dim=8
    )


class _ConfigShapeView(_View):
    """No recurrence module attributes: the shape has to come from the text module's config."""

    blocks = [_HybridBlock(), _HybridBlock(object()), _HybridBlock(), _HybridBlock(object())]
    text_module = _ConfigTextModule()


class _MismatchingTextModule(_TextModule):
    def __call__(self, ids: np.ndarray) -> np.ndarray:
        return super().__call__(ids) + 0.5


class _MismatchingView(_View):
    """Only the float32 manual loop deviates: a pure precision gap, not a structural defect."""

    text_module = _MismatchingTextModule()


class _NativeDivergentView(_View):
    """The native-dtype manual loop disagrees with the forward: a structural defect."""

    def diagnostic_native_final_residual(self, ids: np.ndarray) -> np.ndarray:
        return super().diagnostic_native_final_residual(ids) + 0.5


class _RecordingPreflightView(_View):
    """A full preflight view that records whether the loader's own instance did the work."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, ids: np.ndarray) -> np.ndarray:
        self.calls.append("embed")
        return super().embed(ids)

    def run_block(self, index: int, h: np.ndarray, masks, cache) -> np.ndarray:
        self.calls.append(f"run_block:{index}")
        return super().run_block(index, h, masks, cache)

    def final_norm(self, h: np.ndarray) -> np.ndarray:
        self.calls.append("final_norm")
        return super().final_norm(h)

    def diagnostic_native_final_residual(self, ids: np.ndarray) -> np.ndarray:
        self.calls.append("native_manual")
        return super().diagnostic_native_final_residual(ids)


class _ControlView(_View):
    def make_cache(self) -> list[object]:
        raise AssertionError("residual controls must not create a cache")

    def residuals(self, ids: np.ndarray, layers: list[int]) -> dict[int, np.ndarray]:
        del ids, layers
        raise AssertionError("residual controls must not capture JVP residuals")


class _RecordingControlView(_ControlView):
    """A control view that records which view instance the stage actually ran through."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, ids: np.ndarray) -> np.ndarray:
        self.calls.append("embed")
        return super().embed(ids)

    def masks(self, h: np.ndarray, cache) -> dict[str, object]:
        self.calls.append("masks")
        return super().masks(h, cache)

    def run_block(self, index: int, h: np.ndarray, masks, cache) -> np.ndarray:
        self.calls.append(f"run_block:{index}")
        return super().run_block(index, h, masks, cache)

    def final_norm(self, h: np.ndarray) -> np.ndarray:
        self.calls.append("final_norm")
        return super().final_norm(h)

    def diagnostic_native_final_residual(self, ids: np.ndarray) -> np.ndarray:
        self.calls.append("native_manual")
        return super().diagnostic_native_final_residual(ids)


class _MetricArray:
    def __init__(self, values: list[float], dtype: str) -> None:
        self.values = np.array(values, dtype=np.float64)
        self.dtype = dtype

    def __sub__(self, other: _MetricArray) -> _MetricArray:
        return _MetricArray((self.values - other.values).tolist(), self.dtype)

    def astype(self, dtype: str) -> _MetricArray:
        return _MetricArray(self.values.tolist(), dtype)


class _MetricApi:
    float32 = "float32"
    _info = {
        "bfloat16": SimpleNamespace(eps=2**-7, smallest_normal=2**-126),
        "float32": SimpleNamespace(eps=2**-23, smallest_normal=2**-126),
    }

    @staticmethod
    def abs(value: _MetricArray) -> _MetricArray:
        return _MetricArray(np.abs(value.values).tolist(), value.dtype)

    @staticmethod
    def max(value: _MetricArray) -> float:
        return float(np.max(value.values))

    @staticmethod
    def square(value: _MetricArray) -> _MetricArray:
        return _MetricArray((value.values**2).tolist(), value.dtype)

    @staticmethod
    def sum(value: _MetricArray) -> float:
        return float(np.sum(value.values))

    @staticmethod
    def sqrt(value: float) -> float:
        return float(np.sqrt(value))

    @classmethod
    def finfo(cls, dtype: str) -> SimpleNamespace:
        return cls._info[dtype]


class _Model:
    def parameters(self) -> dict[str, np.ndarray]:
        return {"weights": np.zeros(4, dtype=np.float32)}


def _resolved(spec: ModelSpec) -> ResolvedSpec:
    return ResolvedSpec(
        spec=spec,
        num_layers=4,
        hidden_size=3,
        vocab_size=17,
        tie_word_embeddings=True,
        layer_types=("attention", "linear_attention", "attention", "linear_attention"),
        lora_keys=("layers.0.q_proj", "layers.3.down_proj"),
        trainable_parameters=42,
        probe_layers=(2,),
        cache_strategy="snapshot",
        cache_strategy_reason="auto:equivalence_verified",
        snapshot_revision=None,
        jvp_method="untested",
    )


def _complete_artifact(
    *,
    passed: bool = True,
    residual_passed: bool = True,
    footprint: dict[str, object] | None = None,
) -> dict[str, object]:
    # The default footprint is a computed estimate that fits, which is what the training
    # consumer requires: every other case here then fails for the reason it names rather
    # than for a footprint that was never run.
    if footprint is None:
        footprint = {"passed": True, "refused": False, "skipped": False}
    return {
        "schema_version": 3,
        "model_name": "fake-model",
        "hf_id": "org/fake-model",
        "snapshot_revision": "current",
        "passed": passed,
        "memory": {"within_budget": True},
        "thinking_prompts": [
            {"mode": mode, "prompt": f"{mode}-prompt", "token_count": 1}
            for mode in ("unsupported", "off", "inference", "trained")
        ],
        "lora": {"keys": ["layers.0.q_proj"], "trainable_parameters": 1},
        "residual_equivalence": {"passed": residual_passed},
        "jvp": {"finite": True},
        "training_footprint": footprint,
    }


def test_run_preflight_writes_stable_complete_fake_report(tmp_path: Path) -> None:
    """A missing inspection or unstable report would break the preflight contract."""
    spec = _spec()
    tokenizers: list[_Tokenizer] = []
    calls: list[str] = []

    def loader(given, adapter, *, lazy: bool):
        assert (given, adapter, lazy) == (spec, None, True)
        calls.append(given.hf_id)
        tokenizer = _Tokenizer()
        tokenizers.append(tokenizer)
        return _Model(), tokenizer, _View(), _resolved(spec)

    def forward_jvp(view, layer, primal, tangent, *, method: str) -> np.ndarray:
        assert view.num_layers == 4
        assert layer == 2
        assert method == "forward"
        assert primal.dtype == np.float32
        assert tangent.dtype == np.float32
        return np.ones_like(primal)

    kwargs = {
        "loader": loader,
        "output_root": tmp_path,
        "spec_loader": lambda name: spec,
        "view_factory": lambda model: _View(),
        "resolver": lambda given, model, token: _resolved(given),
        "jvp": forward_jvp,
        "revision_reader": lambda given: "cached-revision",
        "array_api": np,
    }
    path = run_preflight(spec.name, **kwargs)
    first = path.read_bytes()
    second = run_preflight(spec.name, **kwargs).read_bytes()
    report = json.loads(first)

    assert path == tmp_path / "fake-model.json"
    assert first == second
    assert calls == [spec.hf_id, spec.hf_id]
    assert report["schema_version"] == 3
    assert report["model_name"] == spec.name
    assert report["hf_id"] == spec.hf_id
    assert report["snapshot_revision"] == "cached-revision"
    assert report["passed"] is True
    assert report["architecture"] == {
        "hidden_size": 3,
        "layer_counts": {"attention": 2, "linear_attention": 2},
        "num_layers": 4,
        "tie_word_embeddings": True,
        "vocab_size": 17,
    }
    assert report["cache"] == {
        "entry_types": ["_CacheA", "_CacheB", "_CacheA", "_CacheB"],
        "strategy": "snapshot",
        "strategy_reason": "auto:equivalence_verified",
    }
    exact_metrics = {
        "absolute_tolerance": 3 * np.finfo(np.float32).eps * 146.0,
        "frobenius_relative_error": 0.0,
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "reference_dtype": "float32",
        "reference_epsilon": np.finfo(np.float32).eps,
        "reference_scale": 146.0,
        "relative_tolerance": 3 * np.finfo(np.float32).eps,
        "rounding_steps": 9,
        "within_tolerance": True,
    }
    assert report["residual_equivalence"] == {
        "criterion": "native_dtype_rms_roundoff_and_frobenius_relative",
        "fp32_manual_vs_native": {
            **exact_metrics,
            "gates": False,
            "purpose": (
                "precision-gap measurement: the float32 capture path against the native "
                "forward, reported for the capture-dtype decision and never gated"
            ),
        },
        "frobenius_relative_tolerance": 1e-4,
        "gated_comparison": "native_manual_vs_native",
        "native_manual_vs_native": {
            **exact_metrics,
            "gates": True,
            "purpose": (
                "structural equivalence: the manual block loop in the model's native dtype "
                "against the native forward, expected exact or within both the derived "
                "rounding floor and the Frobenius-relative bound"
            ),
        },
        "passed": True,
        "token_count": 64,
    }
    assert report["jvp"] == {"finite": True, "layer": 2, "method": "forward"}
    assert report["lora"] == {
        "keys": ["layers.0.q_proj", "layers.3.down_proj"],
        "trainable_parameters": 42,
    }
    assert report["memory"] == {
        "activation_bytes": 480,
        "budget_gib": 1.0,
        "budget_source": "registry",
        "device_working_set_gib": None,
        "device_working_set_note": "no device info source available; the registry value stands",
        "parameter_bytes": 16,
        "registry_budget_gib": 1.0,
        "total_bytes": 496,
        "total_gib": 496 / 1024**3,
        "within_budget": True,
    }
    assert report["training_footprint"]["skipped"] is True
    assert report["training_footprint"]["passed"] is True
    assert [entry["mode"] for entry in report["thinking_prompts"]] == [
        "unsupported",
        "off",
        "inference",
        "trained",
    ]
    assert [entry["token_count"] for entry in report["thinking_prompts"]] == [70, 70, 70, 70]
    assert "enable_thinking" not in tokenizers[0].render_calls[0]
    assert tokenizers[0].render_calls[1]["enable_thinking"] is False
    assert tokenizers[0].render_calls[2]["enable_thinking"] is True
    assert tokenizers[0].render_calls[3]["enable_thinking"] is True


def test_run_preflight_uses_finite_difference_only_after_bad_forward_jvp(tmp_path: Path) -> None:
    """A non-finite forward derivative must select the finite-difference fallback."""
    spec = _spec()
    methods: list[str] = []

    def jvp(view, layer, primal, tangent, *, method: str) -> np.ndarray:
        del view, layer, tangent
        methods.append(method)
        return np.full_like(primal, np.inf) if method == "forward" else np.ones_like(primal)

    report = json.loads(
        run_preflight(
            spec.name,
            loader=lambda given, adapter, *, lazy: (
                _Model(),
                _Tokenizer(),
                _View(),
                _resolved(given),
            ),
            output_root=tmp_path,
            spec_loader=lambda name: spec,
            view_factory=lambda model: _View(),
            resolver=lambda given, model, token: _resolved(given),
            jvp=jvp,
            revision_reader=lambda given: "cached-revision",
            array_api=np,
        ).read_text(encoding="utf-8")
    )

    assert methods == ["forward", "finite_difference"]
    assert report["jvp"] == {"finite": True, "layer": 2, "method": "finite_difference"}


def test_nonfinite_jvp_writes_failed_evidence_then_exits_nonzero(tmp_path: Path) -> None:
    """A terminal JVP failure must remain inspectable after the preflight exits."""
    spec = _spec()
    methods: list[str] = []

    def jvp(view, layer, primal, tangent, *, method: str) -> np.ndarray:
        del view, layer, tangent
        methods.append(method)
        return np.full_like(primal, np.inf)

    with pytest.raises(SystemExit, match="preflight failed"):
        run_preflight(
            spec.name,
            loader=lambda given, adapter, *, lazy: (
                _Model(),
                _Tokenizer(),
                _View(),
                _resolved(given),
            ),
            output_root=tmp_path,
            spec_loader=lambda name: spec,
            view_factory=lambda model: _View(),
            resolver=lambda given, model, token: _resolved(given),
            jvp=jvp,
            revision_reader=lambda given: "cached-revision",
            array_api=np,
        )

    report = json.loads((tmp_path / "fake-model.json").read_text(encoding="utf-8"))
    assert methods == ["forward", "finite_difference"]
    assert report["passed"] is False
    assert report["jvp"] == {"finite": False, "layer": 2, "method": "finite_difference"}


def test_distributed_error_passes_the_derived_floor_but_fails_the_frobenius_gate() -> None:
    """A 0.2% distributed deviation hides inside the bfloat16 rounding budget; R18a's
    Frobenius-relative bound is the AND-condition that catches exactly that defect class."""
    actual = _MetricArray([100.2], "float32")
    bfloat_reference = _MetricArray([100.0], "bfloat16")
    float_reference = _MetricArray([100.0], "float32")

    bfloat_metrics = _residual_metrics(
        actual, bfloat_reference, num_layers=4, array_api=_MetricApi
    )
    float_metrics = _residual_metrics(actual, float_reference, num_layers=4, array_api=_MetricApi)
    exact_metrics = _residual_metrics(
        _MetricArray([100.0], "bfloat16"), bfloat_reference, num_layers=4, array_api=_MetricApi
    )

    assert bfloat_metrics["reference_dtype"] == "bfloat16"
    assert bfloat_metrics["reference_epsilon"] == 2**-7
    assert bfloat_metrics["reference_scale"] == 100.0
    assert bfloat_metrics["rounding_steps"] == 9
    assert bfloat_metrics["relative_tolerance"] == 3 * 2**-7
    assert bfloat_metrics["absolute_tolerance"] == 100.0 * 3 * 2**-7
    assert bfloat_metrics["within_tolerance"] is True
    assert bfloat_metrics["frobenius_relative_error"] == pytest.approx(0.002)
    assert _native_gate_passed(bfloat_metrics) is False
    assert float_metrics["reference_dtype"] == "float32"
    assert float_metrics["absolute_tolerance"] == 100.0 * 3 * 2**-23
    assert float_metrics["within_tolerance"] is False
    assert float_metrics["frobenius_relative_error"] == pytest.approx(0.002)
    assert _native_gate_passed(float_metrics) is False
    assert exact_metrics["frobenius_relative_error"] == 0.0
    assert _native_gate_passed(exact_metrics) is True
    assert bfloat_metrics["absolute_tolerance"] != 1e-5


def test_residual_metrics_uses_rms_roundoff_boundaries() -> None:
    """The layer-count budget must accept only errors inside its structural bound."""
    reference = _MetricArray([100.0], "bfloat16")
    relative_tolerance = 3 * 2**-7
    absolute_tolerance = 100.0 * relative_tolerance

    inside = _residual_metrics(
        _MetricArray([100.0 + absolute_tolerance * 0.999], "float32"),
        reference,
        num_layers=4,
        array_api=_MetricApi,
    )
    outside = _residual_metrics(
        _MetricArray([100.0 + absolute_tolerance * 1.001], "float32"),
        reference,
        num_layers=4,
        array_api=_MetricApi,
    )

    assert inside["within_tolerance"] is True
    assert outside["within_tolerance"] is False
    assert inside["rounding_steps"] == 9


@pytest.mark.parametrize("num_layers", [False, 0, -1, 1.0, "4"])
def test_residual_metrics_rejects_invalid_layer_counts(num_layers: object) -> None:
    """A tolerance cannot be computed from a non-positive or non-integral block count."""
    with pytest.raises(ValueError, match="num_layers"):
        _residual_metrics(
            _MetricArray([1.0], "float32"),
            _MetricArray([1.0], "float32"),
            num_layers=num_layers,
            array_api=_MetricApi,
        )


def test_run_residual_control_writes_only_the_three_residual_comparisons(tmp_path: Path) -> None:
    """A residual control must load once and avoid unrelated preflight inspection work."""
    spec = _spec()
    calls: list[tuple[str, bool]] = []
    output_path = tmp_path / "native-loop.json"

    path = run_residual_control(
        spec.name,
        output_path=output_path,
        loader=lambda given, adapter, *, lazy: (
            calls.append((given.hf_id, lazy))
            or (_Model(), _Tokenizer(), _ControlView(), _resolved(given))
        ),
        spec_loader=lambda name: spec,
        view_factory=lambda model: _ControlView(),
        revision_reader=lambda given: "cached-revision",
        array_api=np,
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert path == output_path
    assert calls == [(spec.hf_id, True)]
    assert report["schema_version"] == 3
    assert report["model_name"] == spec.name
    assert report["hf_id"] == spec.hf_id
    assert report["snapshot_revision"] == "cached-revision"
    assert report["token_identity"] == {"ids": list(range(64)), "token_count": 64}
    assert report["fp32_manual_vs_native"]["max_abs_error"] == 0.0
    assert report["native_manual_vs_native"]["max_abs_error"] == 0.0
    assert report["fp32_manual_vs_native"]["rounding_steps"] == 9
    assert report["native_manual_vs_native"]["rounding_steps"] == 9
    assert report["fp32_manual_vs_native"]["frobenius_relative_error"] == 0.0
    assert report["native_manual_vs_native"]["frobenius_relative_error"] == 0.0
    metric_keys = {
        "absolute_tolerance",
        "frobenius_relative_error",
        "max_abs_error",
        "max_relative_error",
        "reference_dtype",
        "reference_epsilon",
        "reference_scale",
        "relative_tolerance",
        "rounding_steps",
        "within_tolerance",
    }
    assert set(report["fp32_manual_vs_native"]) == metric_keys
    assert set(report["native_manual_vs_native"]) == metric_keys


def test_residual_control_without_a_view_factory_uses_the_loader_view(
    tmp_path: Path, monkeypatch
) -> None:
    """Omitting `view_factory` is the production configuration; a rebuilt view would diverge."""
    from local_llm_lab.arch import ArchitectureView

    spec = _spec()
    loaded_view = _RecordingControlView()
    output_path = tmp_path / "loader-view.json"
    monkeypatch.setattr(
        ArchitectureView,
        "from_model",
        lambda model: pytest.fail("the residual control rebuilt a second view"),
    )

    path = run_residual_control(
        spec.name,
        output_path=output_path,
        loader=lambda given, adapter, *, lazy: (
            _Model(),
            _Tokenizer(),
            loaded_view,
            _resolved(given),
        ),
        spec_loader=lambda name: spec,
        revision_reader=lambda given: "cached-revision",
        array_api=np,
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert loaded_view.calls == [
        "embed",
        "masks",
        "run_block:0",
        "run_block:1",
        "run_block:2",
        "run_block:3",
        "final_norm",
        "native_manual",
    ]
    assert report["fp32_manual_vs_native"]["max_abs_error"] == 0.0
    assert report["native_manual_vs_native"]["max_abs_error"] == 0.0


def test_failed_preflight_writes_evidence_then_exits_nonzero(tmp_path: Path) -> None:
    """A native-dtype structural divergence must fail the gate and preserve its evidence."""
    spec = _spec()

    with pytest.raises(SystemExit, match="preflight failed"):
        run_preflight(
            spec.name,
            loader=lambda given, adapter, *, lazy: (
                _Model(),
                _Tokenizer(),
                _View(),
                _resolved(given),
            ),
            output_root=tmp_path,
            spec_loader=lambda name: spec,
            view_factory=lambda model: _NativeDivergentView(),
            resolver=lambda given, model, token: _resolved(given),
            jvp=lambda *args, **kwargs: np.ones((1, 64, 3), dtype=np.float32),
            revision_reader=lambda given: "cached-revision",
            array_api=np,
        )

    report = json.loads((tmp_path / "fake-model.json").read_text(encoding="utf-8"))
    residual = report["residual_equivalence"]
    assert report["passed"] is False
    assert residual["passed"] is False
    assert residual["criterion"] == "native_dtype_rms_roundoff_and_frobenius_relative"
    assert residual["gated_comparison"] == "native_manual_vs_native"
    assert residual["native_manual_vs_native"]["within_tolerance"] is False
    assert residual["native_manual_vs_native"]["max_abs_error"] == 0.5
    assert residual["fp32_manual_vs_native"]["within_tolerance"] is True
    assert residual["fp32_manual_vs_native"]["max_abs_error"] == 0.0


def test_fp32_precision_gap_is_reported_but_never_gates(tmp_path: Path) -> None:
    """A pure float32-path deviation is R18a evidence for capture_dtype, not a gate failure."""
    spec = _spec()

    path = run_preflight(
        spec.name,
        loader=lambda given, adapter, *, lazy: (
            _Model(),
            _Tokenizer(),
            _View(),
            _resolved(given),
        ),
        output_root=tmp_path,
        spec_loader=lambda name: spec,
        view_factory=lambda model: _MismatchingView(),
        resolver=lambda given, model, token: _resolved(given),
        jvp=lambda *args, **kwargs: np.ones((1, 64, 3), dtype=np.float32),
        revision_reader=lambda given: "cached-revision",
        array_api=np,
    )

    residual = json.loads(path.read_text(encoding="utf-8"))["residual_equivalence"]
    assert residual["passed"] is True
    assert residual["fp32_manual_vs_native"]["within_tolerance"] is False
    assert residual["fp32_manual_vs_native"]["max_abs_error"] == 0.5
    assert residual["fp32_manual_vs_native"]["gates"] is False
    assert residual["native_manual_vs_native"]["within_tolerance"] is True
    assert residual["native_manual_vs_native"]["gates"] is True


def test_run_preflight_defaults_use_the_loader_view_and_resolved_spec(
    tmp_path: Path, monkeypatch
) -> None:
    """cli.py passes no view_factory or resolver; the loader's own view and spec must gate."""
    from local_llm_lab.arch import ArchitectureView

    spec = _spec()
    loaded_view = _RecordingPreflightView()
    resolved = dataclasses.replace(_resolved(spec), snapshot_revision="resolved-revision")
    monkeypatch.setattr(
        ArchitectureView,
        "from_model",
        lambda model: pytest.fail("the default preflight path rebuilt a second view"),
    )

    path = run_preflight(
        spec.name,
        loader=lambda given, adapter, *, lazy: (_Model(), _Tokenizer(), loaded_view, resolved),
        output_root=tmp_path,
        spec_loader=lambda name: spec,
        jvp=lambda *args, **kwargs: np.ones((1, 64, 3), dtype=np.float32),
        revision_reader=lambda given: pytest.fail("the loader's resolved revision was ignored"),
        array_api=np,
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert loaded_view.calls[:6] == ["embed"] + [f"run_block:{i}" for i in range(4)] + [
        "final_norm"
    ]
    assert "native_manual" in loaded_view.calls
    assert report["passed"] is True
    assert report["snapshot_revision"] == "resolved-revision"
    assert report["cache"]["strategy"] == "snapshot"
    assert report["cache"]["strategy_reason"] == "auto:equivalence_verified"
    assert report["lora"] == {
        "keys": ["layers.0.q_proj", "layers.3.down_proj"],
        "trainable_parameters": 42,
    }
    assert report["residual_equivalence"]["gated_comparison"] == "native_manual_vs_native"


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (None, "preflight artifact is missing"),
        ("not json", "preflight artifact is malformed"),
        ({"schema_version": 3, "model_name": "wrong"}, "model name"),
        ({"schema_version": 3, "model_name": "fake-model", "hf_id": "wrong"}, "hf_id"),
        (
            {"schema_version": 3, "model_name": "fake-model", "hf_id": "org/fake-model"},
            "snapshot revision",
        ),
        (
            {
                "model_name": "fake-model",
                "hf_id": "org/fake-model",
                "snapshot_revision": "current",
                "passed": True,
                "memory": {"within_budget": True},
            },
            "schema version",
        ),
        (
            {
                "schema_version": 3,
                "model_name": "fake-model",
                "hf_id": "org/fake-model",
                "snapshot_revision": "old",
                "passed": True,
                "memory": {"within_budget": True},
            },
            "snapshot revision",
        ),
        (
            {
                "schema_version": 3,
                "model_name": "fake-model",
                "hf_id": "org/fake-model",
                "snapshot_revision": "current",
                "passed": False,
                "memory": {"within_budget": True},
            },
            "training evidence",
        ),
        (
            {
                "schema_version": 3,
                "model_name": "fake-model",
                "hf_id": "org/fake-model",
                "snapshot_revision": "current",
                "passed": True,
                "memory": {"within_budget": False},
            },
            "training evidence",
        ),
    ],
)
def test_require_preflight_rejects_invalid_artifacts_before_actions(
    tmp_path: Path, artifact: dict[str, object] | str | None, message: str
) -> None:
    """Any stale or malformed evidence must stop before a model action can begin."""
    path = tmp_path / "fake-model.json"
    if isinstance(artifact, dict):
        path.write_text(json.dumps(artifact), encoding="utf-8")
    elif isinstance(artifact, str):
        path.write_text(artifact, encoding="utf-8")
    action = []

    with pytest.raises(SystemExit, match=message):
        require_preflight(
            _spec(),
            output_root=tmp_path,
            revision_reader=lambda spec: "current",
            action=lambda: action.append("loaded"),
        )
    assert action == []


def test_require_preflight_accepts_current_evidence_and_skip_avoids_reads(tmp_path: Path) -> None:
    """A valid artifact permits the caller, while skip bypasses artifact and cache inspection."""
    artifact = _complete_artifact()
    (tmp_path / "fake-model.json").write_text(json.dumps(artifact), encoding="utf-8")
    assert (
        require_preflight(_spec(), output_root=tmp_path, revision_reader=lambda spec: "current")
        == artifact
    )
    assert require_preflight(
        _spec(),
        skip=True,
        output_root=tmp_path,
        revision_reader=lambda spec: (_ for _ in ()).throw(AssertionError("cache read")),
    ) is None


def test_require_preflight_allows_training_evidence_without_view_equivalence(
    tmp_path: Path,
) -> None:
    """Training may use sufficient non-view evidence while the default view consumer fails."""
    artifact = _complete_artifact(passed=False, residual_passed=False)
    (tmp_path / "fake-model.json").write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(SystemExit, match="view evidence"):
        require_preflight(_spec(), output_root=tmp_path, revision_reader=lambda spec: "current")
    assert (
        require_preflight(
            _spec(),
            consumer="training",
            output_root=tmp_path,
            revision_reader=lambda spec: "current",
        )
        == artifact
    )


@pytest.mark.parametrize(
    ("consumer", "mutate"),
    [
        ("training", lambda artifact: artifact["memory"].update(within_budget=False)),
        ("training", lambda artifact: artifact["thinking_prompts"].pop()),
        ("training", lambda artifact: artifact["thinking_prompts"][0].update(prompt="")),
        ("training", lambda artifact: artifact["lora"].update(keys=[])),
        ("training", lambda artifact: artifact["lora"].update(trainable_parameters=0)),
        ("view", lambda artifact: artifact["residual_equivalence"].update(passed=False)),
        ("view", lambda artifact: artifact["jvp"].update(finite=False)),
        ("view", lambda artifact: artifact.update(passed=False)),
    ],
)
def test_require_preflight_rejects_malformed_consumer_evidence(
    tmp_path: Path, consumer: str, mutate
) -> None:
    """Each consumer must reject a missing prerequisite before it can invoke an action."""
    artifact = _complete_artifact()
    mutate(artifact)
    (tmp_path / "fake-model.json").write_text(json.dumps(artifact), encoding="utf-8")
    action: list[str] = []

    with pytest.raises(SystemExit, match="evidence"):
        require_preflight(
            _spec(),
            consumer=consumer,
            output_root=tmp_path,
            revision_reader=lambda spec: "current",
            action=lambda: action.append("loaded"),
        )
    assert action == []


def test_require_preflight_rejects_unknown_consumer_before_actions(tmp_path: Path) -> None:
    """An unknown consumer cannot accidentally inherit the training or view predicate."""
    action: list[str] = []
    with pytest.raises(SystemExit, match="unknown preflight consumer"):
        require_preflight(
            _spec(),
            consumer="unknown",  # type: ignore[arg-type]
            output_root=tmp_path,
            revision_reader=lambda spec: "current",
            action=lambda: action.append("loaded"),
        )
    assert action == []


def test_default_loader_is_the_one_shared_policy_loader(monkeypatch) -> None:
    """A second mlx_lm.load call site would let preflight and the stages disagree."""
    from local_llm_lab.pipeline import evaluate, preflight

    spec = _spec()
    loaded = (object(), object(), object(), _resolved(spec))
    calls: list[tuple[object, object, bool]] = []

    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda given, adapter, *, lazy: calls.append((given, adapter, lazy)) or loaded,
    )

    assert preflight._default_loader(spec, None, lazy=True) == loaded
    assert calls == [(spec, None, True)]


# ------------------------------------------------------------------ R32: budget and footprint


def _preflight(spec: ModelSpec, view: object, tmp_path: Path, **overrides: object):
    """Run one fake preflight; overrides carry only the R32 inputs a test is about."""
    kwargs: dict[str, object] = {
        "loader": lambda given, adapter, *, lazy: (
            _Model(),
            _Tokenizer(),
            view,
            _resolved(given),
        ),
        "output_root": tmp_path,
        "spec_loader": lambda name: spec,
        "jvp": lambda *args, **kwargs: np.ones((1, 64, 3), dtype=np.float32),
        "revision_reader": lambda given: "cached-revision",
        "array_api": np,
    }
    kwargs.update(overrides)
    return json.loads(run_preflight(spec.name, **kwargs).read_text(encoding="utf-8"))


def test_memory_budget_is_the_device_working_set_when_it_undercuts_the_registry(
    tmp_path: Path,
) -> None:
    """R32(b): the registry value is a cap, and this machine grants less than it declares."""
    spec = _spec(memory_budget_gib=22.0)

    report = _preflight(spec, _View(), tmp_path, device_info=lambda: _device_info())

    assert report["memory"]["registry_budget_gib"] == 22.0
    assert report["memory"]["device_working_set_gib"] == _DEVICE_WORKING_SET_GIB
    assert report["memory"]["budget_gib"] == _DEVICE_WORKING_SET_GIB
    assert report["memory"]["budget_source"] == "device"
    assert report["memory"]["within_budget"] is True


def test_memory_budget_keeps_the_registry_value_when_it_is_the_smaller(tmp_path: Path) -> None:
    """The minimum runs both ways: a registry cap under the device's grant still binds."""
    spec = _spec(memory_budget_gib=16.0)

    report = _preflight(spec, _View(), tmp_path, device_info=lambda: _device_info())

    assert report["memory"]["registry_budget_gib"] == 16.0
    assert report["memory"]["device_working_set_gib"] == _DEVICE_WORKING_SET_GIB
    assert report["memory"]["budget_gib"] == 16.0
    assert report["memory"]["budget_source"] == "registry"


def test_memory_budget_falls_back_to_the_registry_with_a_recorded_note(tmp_path: Path) -> None:
    """A runtime that reports no working set must not silently produce a budget of zero."""
    spec = _spec(memory_budget_gib=22.0)

    report = _preflight(spec, _View(), tmp_path, device_info=lambda: {"device_name": "fake-gpu"})

    assert report["memory"]["device_working_set_gib"] is None
    assert report["memory"]["budget_gib"] == 22.0
    assert report["memory"]["budget_source"] == "registry"
    assert "no device working set" in report["memory"]["device_working_set_note"]


def test_within_budget_is_judged_against_the_device_minimum_not_the_registry(
    tmp_path: Path,
) -> None:
    """The B4 defect exactly: 3.85 GiB passed a 22 GiB registry cap the device never granted."""
    spec = _spec(memory_budget_gib=22.0)

    with pytest.raises(SystemExit, match="preflight failed"):
        _preflight(spec, _View(), tmp_path, device_info=lambda: _device_info(400))

    report = json.loads((tmp_path / "fake-model.json").read_text(encoding="utf-8"))
    assert report["memory"]["budget_gib"] == 400 / 1024**3
    assert report["memory"]["within_budget"] is False
    assert report["passed"] is False


def test_training_footprint_is_the_calibrated_envelope_for_the_configured_form(
    tmp_path: Path,
) -> None:
    """R32(d): the peak is the measured envelope for the form the arm runs, evaluated here.

    The architecture evidence beside it - how many linear-attention layers there are, the
    recurrence state's shape, what gradient checkpointing bounds - is still introspected and
    still recorded; it just no longer computes the number the gate judges.
    """
    spec = _spec(
        memory_budget_gib=22.0,
        train={"batch_size": 1, "max_seq_length": 5, "grad_checkpoint": True},
    )

    report = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        device_info=lambda: _device_info(),
        max_row_tokens=997,
        gated_delta_chunk=64,
    )
    footprint = report["training_footprint"]

    assert footprint["skipped"] is False
    assert footprint["refused"] is False
    assert footprint["max_row_tokens"] == 997
    assert footprint["max_row_tokens_source"] == "max_row_tokens"
    assert footprint["linear_attention_layers"] == 2
    assert footprint["linear_attention_state_shape"] == {"heads_v": 4, "dim_v": 8, "dim_k": 8}
    assert footprint["linear_attention_state_shape_source"] == "recurrence module"
    assert footprint["retained_recurrence_layers"] == 1
    assert footprint["calibration_domain_departures"] == []
    assert footprint["estimates"]["chunked"]["estimated_train_peak_gib"] == _estimated_peak_gib(
        "chunked", 997
    )
    assert footprint["estimates"]["chunked"]["estimated_train_peak_bytes"] == round(
        _estimated_peak_gib("chunked", 997) * 1024**3
    )
    # Every measured chunked peak at this row length is under the envelope, and the envelope
    # is not the analytic sum's near-zero recurrence term.
    assert footprint["estimates"]["chunked"]["estimated_train_peak_gib"] >= max(
        point.peak_gib
        for point in _CALIBRATION_POINTS
        if point.mode == "chunked" and point.tokens == 997
    )
    assert footprint["gated_estimate"] == "chunked"
    assert footprint["estimates"]["chunked"]["gates"] is True
    assert footprint["estimates"]["unrolled"]["gates"] is False
    assert footprint["passed"] is True


def test_training_footprint_reads_the_state_shape_from_the_text_module_config(
    tmp_path: Path,
) -> None:
    """A block that hides the recurrence still declares the shape on the text module's args."""
    spec = _spec(memory_budget_gib=22.0)

    report = _preflight(
        spec,
        _ConfigShapeView(),
        tmp_path,
        device_info=lambda: _device_info(),
        max_row_tokens=100,
        gated_delta_chunk=10,
    )
    footprint = report["training_footprint"]

    assert footprint["linear_attention_state_shape"] == {"heads_v": 4, "dim_v": 8, "dim_k": 8}
    assert footprint["linear_attention_state_shape_source"] == "text module config"
    assert footprint["skipped"] is False


def test_a_failed_footprint_keeps_its_evidence_without_failing_the_command(
    tmp_path: Path,
) -> None:
    """An arm may not pass preflight and die at step one; the estimate must be inspectable.

    The failure belongs to the arm's training configuration, not to the model, so it is
    ``require_preflight(consumer="training")`` that refuses it.  ``report["passed"]`` carries
    model-level evidence only, and the command still exits zero with its artifact written.
    """
    spec = _spec(memory_budget_gib=1.0)

    # ``_preflight`` reads the path ``run_preflight`` returned, so getting a report back at
    # all is the command completing normally rather than raising ``SystemExit``.
    report = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
    )

    assert report == json.loads((tmp_path / "fake-model.json").read_text(encoding="utf-8"))
    assert report["memory"]["within_budget"] is True
    assert report["residual_equivalence"]["passed"] is True
    assert report["training_footprint"]["passed"] is False
    assert report["training_footprint"]["estimates"]["chunked"]["fits_with_headroom"] is False
    assert report["passed"] is True


def test_the_alternative_forms_are_recorded_beside_the_one_that_gates(
    tmp_path: Path,
) -> None:
    """The unrolled reality that OOMs at 997 tokens stays visible next to what will run."""
    spec = _spec(memory_budget_gib=22.0)

    report = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
    )
    footprint = report["training_footprint"]

    assert sorted(footprint["estimates"]) == ["chunked", "chunkwise", "floor", "unrolled"]
    assert footprint["estimates"]["unrolled"]["gates"] is False
    assert footprint["estimates"]["chunked"]["gates"] is True
    assert footprint["gated_estimate"] == "chunked"
    # The comparison the calibration lane was run to make: at run D's shorter rows the
    # unrolled loop costs more than the chunked form, and the chunkwise form costs least.
    assert (
        footprint["estimates"]["chunkwise"]["estimated_train_peak_gib"]
        < footprint["estimates"]["chunked"]["estimated_train_peak_gib"]
        < footprint["estimates"]["unrolled"]["estimated_train_peak_gib"]
    )
    assert footprint["passed"] is True
    assert report["passed"] is True


def test_unrolled_estimate_gates_when_no_chunk_is_configured(tmp_path: Path) -> None:
    """Without a chunk the library runs the unrolled loop, so that is the estimate that binds."""
    spec = _spec(memory_budget_gib=1.0)

    footprint = _preflight(spec, _HybridView(), tmp_path, max_row_tokens=997)[
        "training_footprint"
    ]
    assert footprint["chunk"] is None
    assert footprint["gated_estimate"] == "unrolled"
    assert footprint["estimates"]["unrolled"]["gates"] is True
    assert footprint["estimates"]["chunked"]["gates"] is False
    assert footprint["passed"] is False


def test_training_footprint_is_skipped_with_a_reason_without_a_row_count(
    tmp_path: Path,
) -> None:
    """The existing `preflight --model` command must keep working, and say what it skipped."""
    spec = _spec(memory_budget_gib=22.0)

    report = _preflight(spec, _HybridView(), tmp_path, device_info=lambda: _device_info())
    footprint = report["training_footprint"]

    assert footprint["skipped"] is True
    assert footprint["max_row_tokens"] is None
    assert footprint["max_row_tokens_source"] is None
    assert footprint["estimates"] == {}
    assert footprint["gated_estimate"] is None
    assert "--data" in footprint["skip_reason"]
    assert "--max-row-tokens" in footprint["skip_reason"]
    assert footprint["passed"] is True
    assert report["passed"] is True


def test_an_undiscoverable_state_shape_no_longer_blocks_the_gate(tmp_path: Path) -> None:
    """The envelope is fitted to measured peaks, so it never reads the recurrence's shape.

    The shape stays introspected and recorded as architecture evidence - with its reason when
    it cannot be found - but a gate that no longer needs it may not skip for want of it.
    """
    spec = _spec(memory_budget_gib=22.0)

    footprint = _preflight(spec, _View(), tmp_path, max_row_tokens=997, gated_delta_chunk=64)[
        "training_footprint"
    ]

    assert footprint["linear_attention_state_shape"] is None
    assert "neither" in footprint["linear_attention_state_shape_source"]
    assert footprint["skipped"] is False
    assert footprint["gated_estimate"] == "chunked"
    assert footprint["passed"] is True


def test_running_without_gradient_checkpointing_is_recorded_not_scaled_for(
    tmp_path: Path,
) -> None:
    """Every point was measured with checkpointing on, so its absence is named, never modelled.

    How many layers would retain their graph is still recorded, because it is what makes the
    departure serious; the envelope simply has no measurement to extrapolate from.
    """
    spec = _spec(
        memory_budget_gib=22.0,
        train={"batch_size": 1, "max_seq_length": 5, "grad_checkpoint": False},
    )

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
    )["training_footprint"]

    assert footprint["grad_checkpoint"] is False
    assert footprint["retained_recurrence_layers"] == 2
    assert footprint["calibration"]["grad_checkpoint"] is True
    assert len(footprint["calibration_domain_departures"]) == 1
    assert "checkpointing is off" in footprint["calibration_domain_departures"][0]
    # The estimate is unchanged: the fit is in the row length alone.
    assert footprint["estimates"]["chunked"]["estimated_train_peak_gib"] == _estimated_peak_gib(
        "chunked", 997
    )


def test_preflight_artifact_stays_json_safe_at_schema_version_three(tmp_path: Path) -> None:
    """A schema bump is only useful if every new field survives the round trip unchanged."""
    spec = _spec(memory_budget_gib=22.0)

    path = tmp_path / "fake-model.json"
    report = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        device_info=lambda: _device_info(),
        max_row_tokens=100,
        gated_delta_chunk=10,
    )

    assert report["schema_version"] == 3
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert json.loads(json.dumps(report)) == report


def test_a_failed_footprint_stops_training_and_leaves_the_view_consumer_alone(
    tmp_path: Path,
) -> None:
    """R32(d) joins R15 condition 1: training may not start on a failed footprint estimate.

    A probe is the other half of the same ruling.  It loads no optimiser, takes no gradient
    through a training step and never reaches ``max_seq_length``, so a training peak it will
    never allocate is not evidence against it.
    """
    artifact = _complete_artifact(footprint={"passed": False, "refused": False, "skipped": False})
    (tmp_path / "fake-model.json").write_text(json.dumps(artifact), encoding="utf-8")
    action: list[str] = []

    with pytest.raises(SystemExit, match="training footprint"):
        require_preflight(
            _spec(),
            consumer="training",
            output_root=tmp_path,
            revision_reader=lambda spec: "current",
            action=lambda: action.append("loaded"),
        )
    assert action == []
    assert (
        require_preflight(_spec(), output_root=tmp_path, revision_reader=lambda spec: "current")
        == artifact
    )


def test_a_skipped_footprint_serves_a_view_consumer_and_stops_training(tmp_path: Path) -> None:
    """The bare ``preflight --model`` artifact estimated no training peak at all.

    Its footprint block records ``passed`` so the standalone command keeps its meaning, but a
    training run must not clear the gate on an estimate that was never computed, and the
    rejection has to say which input was missing.
    """
    artifact = _complete_artifact(
        footprint={
            "passed": True,
            "refused": False,
            "skipped": True,
            "skip_reason": "no training row token count; pass --data <dir> or --max-row-tokens",
        }
    )
    (tmp_path / "fake-model.json").write_text(json.dumps(artifact), encoding="utf-8")
    action: list[str] = []

    assert (
        require_preflight(_spec(), output_root=tmp_path, revision_reader=lambda spec: "current")
        == artifact
    )
    with pytest.raises(SystemExit, match="row count"):
        require_preflight(
            _spec(),
            consumer="training",
            output_root=tmp_path,
            revision_reader=lambda spec: "current",
            action=lambda: action.append("loaded"),
        )
    assert action == []


@pytest.mark.parametrize(
    ("footprint", "expected"),
    [
        # A refused block returns before the line that clears ``skipped``, so a guard that read
        # ``skipped`` first would blame a missing row count for an unmeasured recurrence form.
        ({"passed": False, "refused": True, "skipped": True}, "recurrence form"),
        (None, "no training footprint block"),
    ],
)
def test_training_says_which_footprint_it_cannot_accept_and_view_reads_none_of_it(
    tmp_path: Path, footprint: dict[str, object] | None, expected: str
) -> None:
    """The reason must name the real defect, and the view consumer must not consult the block."""
    artifact = _complete_artifact()
    if footprint is None:
        del artifact["training_footprint"]
    else:
        artifact["training_footprint"] = footprint
    (tmp_path / "fake-model.json").write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(SystemExit, match=expected):
        require_preflight(
            _spec(),
            consumer="training",
            output_root=tmp_path,
            revision_reader=lambda spec: "current",
        )
    assert (
        require_preflight(_spec(), output_root=tmp_path, revision_reader=lambda spec: "current")
        == artifact
    )


# ------------------------------------------------- issue #51 lane 2: the calibrated envelope


def _calibration_point(mode: str, tokens: int, chunk: int | None = None):
    """The one table entry for a mode, row length and chunk, so no test restates a number."""
    matches = [
        point
        for point in _CALIBRATION_POINTS
        if point.mode == mode and point.tokens == tokens and point.chunk == chunk
    ]
    assert len(matches) == 1, f"expected exactly one {mode} point at {tokens} tokens"
    return matches[0]


def _point_id(point) -> str:
    return f"{point.mode}-{point.tokens}-{point.chunk}-{point.outcome}"


@pytest.mark.parametrize("point", _CALIBRATION_POINTS, ids=_point_id)
def test_the_envelope_sits_above_every_measured_calibration_point(point) -> None:
    """The decisive property: the estimate is never optimistic where a measurement exists.

    An ``ok`` point is an observed peak, so the envelope must sit at or above it.  An ``oom``
    point is a *lower bound* — the run died at that allocation and its true peak is unknown
    but larger — so it constrains the envelope from below and the estimate must clear it.
    """
    prediction = _estimated_peak_gib(point.mode, point.tokens)

    if point.outcome == "ok":
        assert prediction >= point.peak_gib
    else:
        assert prediction > point.peak_gib


@pytest.mark.parametrize("point", _CALIBRATION_POINTS, ids=_point_id)
def test_the_gate_reaches_the_measured_verdict_at_every_calibration_point(point) -> None:
    """The gate would have refused every run that died, and admitted the ones that lived.

    Judged against this machine's own working set, the same budget the memory block resolves.
    A point's verdict is what its *measured* peak decides under the same 10% headroom, so the
    estimate is being asked to agree with the measurement, not with a hand-picked threshold.
    """
    headroom = 1.0 + _TRAINING_HEADROOM_FRACTION
    predicted_fits = _estimated_peak_gib(point.mode, point.tokens) * headroom
    measured_fits = point.peak_gib * headroom

    assert (predicted_fits <= _DEVICE_WORKING_SET_GIB) is (
        measured_fits <= _DEVICE_WORKING_SET_GIB
    )
    if point.outcome == "oom":
        assert predicted_fits > _DEVICE_WORKING_SET_GIB


def test_the_estimate_does_not_swing_with_chunk_length_the_way_the_analytic_sum_does() -> None:
    """Defect 1: the analytic sum makes chunk dominate; measurement says it barely moves.

    ``training_state_bytes`` is left alone (it is still the analytic description of retained
    state); this test pins why it may not be the gate's estimate.
    """
    at_one_row = [
        _calibration_point("chunked", 997, chunk) for chunk in (32, 64, 128)
    ]
    analytic = [
        training_state_bytes(
            batch=1, heads_v=1, dim_v=1, dim_k=1, tokens=point.tokens, chunk=point.chunk
        )
        for point in at_one_row
    ]
    measured = [point.peak_gib for point in at_one_row]

    assert max(analytic) / min(analytic) > 2.0
    assert max(measured) / min(measured) < 1.1
    # The envelope reads only the row length, so every chunk gets the one estimate.
    assert _estimated_peak_gib("chunked", 997) >= max(measured)


def test_the_recorded_estimate_is_the_same_at_every_measured_chunk_length(
    tmp_path: Path,
) -> None:
    """The gate itself, not just the model: three chunks, one number."""
    spec = _spec(memory_budget_gib=22.0)

    peaks = {
        chunk: _preflight(
            spec,
            _HybridView(),
            tmp_path,
            max_row_tokens=997,
            gated_delta_chunk=chunk,
        )["training_footprint"]["estimates"]["chunked"]["estimated_train_peak_gib"]
        for chunk in (32, 64, 128)
    }

    assert len(set(peaks.values())) == 1


def test_the_unrolled_envelope_cannot_take_the_no_recurrence_floors_slope() -> None:
    """Why ``unrolled`` is fitted proportional to the row and not flat like ``chunkwise``.

    One retained state per token is the library loop's own shape, and the 997-token OOM is
    the evidence: a flat fit anchored at the single 495-token point predicts a peak the run
    demonstrably blew past.
    """
    measured = _calibration_point("unrolled", 495)
    died = _calibration_point("unrolled", 997)
    floor_slope = _ENVELOPES["floor"].slope_gib_per_token
    flat_anchor = measured.peak_gib - floor_slope * measured.tokens

    assert floor_slope * died.tokens + flat_anchor < died.peak_gib
    assert _estimated_peak_gib("unrolled", died.tokens) > died.peak_gib


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("chunkwise", "chunkwise"), ("checkpointed", "chunked"), (None, "chunked")],
)
def test_the_gating_estimate_is_the_mode_the_arm_configures(
    tmp_path: Path, configured: str | None, expected: str
) -> None:
    """Defect 2: the form the run will take is a configuration choice, not a chunk's presence."""
    spec = _spec(memory_budget_gib=22.0)

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
        gated_delta_mode=configured,
    )["training_footprint"]

    assert footprint["recurrence_mode"] == expected
    assert footprint["gated_estimate"] == expected
    assert footprint["estimates"][expected]["gates"] is True
    assert [name for name, entry in footprint["estimates"].items() if entry["gates"]] == [expected]


def test_without_a_configured_chunk_the_librarys_unrolled_loop_is_what_gates(
    tmp_path: Path,
) -> None:
    """Nothing is installed, so the reference loop runs whatever the mode field says."""
    spec = _spec(memory_budget_gib=22.0)

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=495,
        gated_delta_mode="chunkwise",
    )["training_footprint"]

    assert footprint["chunk"] is None
    assert footprint["recurrence_mode"] == "unrolled"
    assert footprint["estimates"]["unrolled"]["gates"] is True


def test_a_backbone_with_no_recurrence_gates_on_the_floor_alone(tmp_path: Path) -> None:
    """A dense backbone has no recurrence to charge, so the no-recurrence floor is the estimate."""

    class _AttentionOnlyView(_View):
        def layer_kind(self, index: int) -> str:
            del index
            return "attention"

    spec = _spec(memory_budget_gib=22.0)

    footprint = _preflight(
        spec,
        _AttentionOnlyView(),
        tmp_path,
        max_row_tokens=2874,
        gated_delta_chunk=64,
    )["training_footprint"]

    assert footprint["linear_attention_layers"] == 0
    assert footprint["recurrence_mode"] == "floor"
    assert footprint["estimates"]["floor"]["gates"] is True
    assert footprint["estimates"]["floor"]["estimated_train_peak_gib"] == _estimated_peak_gib(
        "floor", 2874
    )


def test_the_gate_refuses_a_mode_with_no_calibration_points(tmp_path: Path) -> None:
    """R32(d): a form nobody measured must be refused by name, never guessed at.

    The refusal is the arm's configured recurrence form, not the model, so it lands in the
    block for the training consumer to read and leaves the command's exit code alone.
    """
    spec = _spec(memory_budget_gib=22.0)

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
        gated_delta_mode="fused",
    )["training_footprint"]

    assert footprint["refused"] is True
    assert footprint["passed"] is False
    assert footprint["recurrence_mode"] is None
    assert footprint["gated_estimate"] is None
    assert footprint["estimates"] == {}
    assert "fused" in footprint["skip_reason"]
    assert "chunkwise" in footprint["skip_reason"]


def test_the_calibrated_headroom_boundary_fits_at_exactly_one_point_one(
    tmp_path: Path,
) -> None:
    """10% headroom, at the boundary: exactly 1.10x fits and the next float down does not."""
    peak = _estimated_peak_gib("chunked", 997)
    exactly = peak * (1.0 + _TRAINING_HEADROOM_FRACTION)

    fits = _preflight(
        _spec(memory_budget_gib=exactly),
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
    )["training_footprint"]
    over = _preflight(
        _spec(memory_budget_gib=math.nextafter(exactly, 0.0)),
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
    )["training_footprint"]

    assert fits["estimates"]["chunked"]["estimated_train_peak_gib"] == peak
    assert fits["estimates"]["chunked"]["fits_with_headroom"] is True
    assert fits["headroom_fraction"] == 0.10
    assert fits["passed"] is True
    assert over["estimates"]["chunked"]["fits_with_headroom"] is False
    assert over["passed"] is False


def test_the_artifact_records_the_coefficients_and_the_points_they_came_from(
    tmp_path: Path,
) -> None:
    """A reader must be able to re-derive the gated number from the artifact alone."""
    spec = _spec(memory_budget_gib=22.0)

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=1591,
        gated_delta_chunk=64,
    )["training_footprint"]
    chunked = footprint["estimates"]["chunked"]
    coefficients = chunked["coefficients"]

    assert footprint["recurrence_mode"] == "chunked"
    assert footprint["chunk"] == 64
    assert chunked["chunk"] == 64
    assert (
        coefficients["slope_gib_per_token"] * footprint["max_row_tokens"]
        + coefficients["intercept_gib"]
    ) == chunked["estimated_train_peak_gib"]
    assert coefficients["shape"] == "affine"
    assert [(point["tokens"], point["chunk"], point["outcome"]) for point in chunked["points"]] == [
        (997, 32, "ok"),
        (997, 64, "ok"),
        (997, 128, "ok"),
        (1591, 64, "ok"),
        (2085, 64, "oom"),
        (2874, 64, "oom"),
    ]
    assert [
        (point["tokens"], point["outcome"])
        for point in footprint["calibration"]["floor_points"]
    ] == [(997, "ok"), (1591, "ok"), (2085, "ok"), (2874, "ok")]
    assert footprint["calibration"]["batch_size"] == 1
    assert footprint["calibration"]["grad_checkpoint"] is True


def test_the_chunkwise_envelope_says_it_rests_on_a_single_observation(
    tmp_path: Path,
) -> None:
    """R32(d) honesty: one point is not a well-determined curve, and must not read as one."""
    spec = _spec(memory_budget_gib=22.0)

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=2874,
        gated_delta_chunk=64,
        gated_delta_mode="chunkwise",
    )["training_footprint"]
    chunkwise = footprint["estimates"]["chunkwise"]

    assert chunkwise["single_observation"] is True
    assert "single observation" in chunkwise["caveat"]
    assert str(_calibration_point("chunkwise", 2874, 64).tokens) in chunkwise["caveat"]
    assert "extrapolation" in chunkwise["caveat"]
    assert chunkwise["coefficients"]["shape"] == "flat"
    assert (
        chunkwise["coefficients"]["slope_gib_per_token"]
        == footprint["estimates"]["floor"]["coefficients"]["slope_gib_per_token"]
    )
    assert footprint["estimates"]["chunked"]["single_observation"] is False
    assert footprint["estimates"]["chunked"]["caveat"] is None


def test_the_artifact_names_where_the_arm_leaves_the_calibrated_conditions(
    tmp_path: Path,
) -> None:
    """Every point was measured at batch 1 with checkpointing on; a departure is not modelled."""
    spec = _spec(
        memory_budget_gib=22.0,
        train={"batch_size": 2, "max_seq_length": 5, "grad_checkpoint": False},
    )

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        max_row_tokens=997,
        gated_delta_chunk=64,
    )["training_footprint"]
    departures = " ".join(footprint["calibration_domain_departures"])

    assert "batch" in departures
    assert "checkpoint" in departures


class _WordTokenizer(_Tokenizer):
    """Counts tokens as whitespace-separated words so row lengths are hand-checkable."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def _write_rows(path: Path, rows: list[tuple[int, int]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"prompt": "p " * prompt, "completion": "c " * completion}) + "\n"
            for prompt, completion in rows
        ),
        encoding="utf-8",
    )


def test_longest_row_tokens_reads_both_splits_through_the_real_dataset(tmp_path: Path) -> None:
    """The gate's row length must be the trainer's own, so it reads the trainer's dataset."""
    _write_rows(tmp_path / "train.jsonl", [(3, 2), (10, 7)])
    _write_rows(tmp_path / "valid.jsonl", [(4, 4), (20, 9)])
    _write_rows(tmp_path / "test.jsonl", [(500, 500)])

    assert longest_row_tokens(tmp_path, _WordTokenizer(), max_seq_length=64) == 29


def test_longest_row_tokens_reports_the_length_the_trainer_sees_after_truncation(
    tmp_path: Path,
) -> None:
    """`max_seq_length` clamps a row before the trainer ever sees it, so the estimate uses it."""
    _write_rows(tmp_path / "train.jsonl", [(10, 100)])
    _write_rows(tmp_path / "valid.jsonl", [(2, 2)])

    assert longest_row_tokens(tmp_path, _WordTokenizer(), max_seq_length=32) == 32


def test_run_preflight_reads_the_longest_row_from_a_data_directory(tmp_path: Path) -> None:
    """`--data` must resolve the same count `--max-row-tokens` would be given by hand."""
    data = tmp_path / "data"
    data.mkdir()
    _write_rows(data / "train.jsonl", [(3, 2), (10, 7)])
    _write_rows(data / "valid.jsonl", [(4, 4)])
    spec = _spec(memory_budget_gib=22.0, train={"batch_size": 1, "max_seq_length": 64})

    footprint = _preflight(
        spec,
        _HybridView(),
        tmp_path,
        loader=lambda given, adapter, *, lazy: (
            _Model(),
            _WordTokenizer(),
            _HybridView(),
            _resolved(given),
        ),
        data_dir=data,
        gated_delta_chunk=64,
    )["training_footprint"]

    assert footprint["max_row_tokens"] == 17
    assert footprint["max_row_tokens_source"] == f"data:{data}"
    # The row count the dataset produced is the row count the envelope was evaluated at.
    assert footprint["estimates"]["chunked"]["estimated_train_peak_gib"] == _estimated_peak_gib(
        "chunked", 17
    )


def test_preflight_cli_threads_the_row_count_chunk_and_data_directory(monkeypatch) -> None:
    """The Deputy's regeneration command has to reach `run_preflight` unchanged."""
    from local_llm_lab.pipeline import cli

    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        cli, "run_preflight", lambda name, **kwargs: calls.append((name, kwargs))
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "agent-pipeline",
            "preflight",
            "--model",
            "qwen35-4b",
            "--data",
            "data/agent_v2b-qwen35-4b",
            "--max-row-tokens",
            "2257",
            "--gated-delta-chunk",
            "64",
            "--gated-delta-mode",
            "chunkwise",
        ],
    )

    cli.main()

    assert calls == [
        (
            "qwen35-4b",
            {
                "data_dir": Path("data/agent_v2b-qwen35-4b"),
                "max_row_tokens": 2257,
                "gated_delta_chunk": 64,
                "gated_delta_mode": "chunkwise",
            },
        )
    ]


def test_preflight_cli_keeps_working_with_only_a_model(monkeypatch) -> None:
    """R32 must not break `agent-pipeline preflight --model <name>`."""
    from local_llm_lab.pipeline import cli

    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        cli, "run_preflight", lambda name, **kwargs: calls.append((name, kwargs))
    )
    monkeypatch.setattr(cli.sys, "argv", ["agent-pipeline", "preflight", "--model", "qwen35-4b"])

    cli.main()

    assert calls == [
        (
            "qwen35-4b",
            {
                "data_dir": None,
                "max_row_tokens": None,
                "gated_delta_chunk": None,
                "gated_delta_mode": None,
            },
        )
    ]


def test_the_state_shape_is_read_off_the_librarys_own_decoder_layer() -> None:
    """R31: the introspection route is judged against mlx-lm's real module and its real args.

    A renamed attribute upstream would otherwise skip the footprint gate silently, which is
    the failure mode the gate exists to prevent.
    """
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
    )

    class _LibraryView:
        num_layers = 1
        text_module = SimpleNamespace(args=args)
        blocks = [DecoderLayer(args, 0)]

        def layer_kind(self, index: int) -> str:
            return "linear_attention" if self.blocks[index].is_linear else "attention"

    class _HiddenRecurrenceView(_LibraryView):
        blocks = [SimpleNamespace(is_linear=True)]

    expected = {"heads_v": 4, "dim_v": 8, "dim_k": 8}

    assert _linear_attention_state_shape(_LibraryView(), 1) == (expected, "recurrence module")
    assert _linear_attention_state_shape(_HiddenRecurrenceView(), 1) == (
        expected,
        "text module config",
    )


def test_module_candidates_reach_a_blocks_plain_attributes_as_well_as_its_members() -> None:
    """The dict-subclass trap again, at the second site the Chief named (R31).

    ``mlx.nn.Module`` is a ``dict`` subclass whose dict holds only registered parameters and
    submodules. ``_module_candidates`` promises "a decoder block and its direct members,
    whichever way the block stores them", but a Mapping-first branch that ``return``s cuts the
    attribute half off entirely: a recurrence configuration hung off the block as a plain
    dataclass -- exactly how ``is_linear`` and ``args`` are stored on a real ``DecoderLayer``
    -- is never offered to ``_state_shape_from``.

    On today's ``qwen3_5`` block the shape happens to be found anyway, because the recurrence
    module *is* a registered submodule. This pins the contract rather than the luck.
    """
    from collections.abc import Mapping

    import mlx.nn as nn

    from local_llm_lab.pipeline.preflight import _module_candidates

    recurrence = SimpleNamespace(num_v_heads=4, head_v_dim=8, head_k_dim=8)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(8)  # a registered submodule: it lives in the module's dict
            self.recurrence = recurrence  # a plain attribute: it does not

    block = _Block()

    assert isinstance(block, Mapping), "the premise: an nn.Module is a dict subclass"
    assert "recurrence" not in dict(block), "a plain attribute is not a registered member"
    candidates = list(_module_candidates(block))
    assert block in candidates
    assert any(candidate is block.norm for candidate in candidates), "registered members"
    assert recurrence in candidates, "and plain attributes, which the Mapping branch dropped"


def test_the_probe_precision_block_reads_the_artifact_the_preflight_actually_writes(
    tmp_path: Path,
) -> None:
    """R18a: reader and writer disagreed on the *level*, and the fake agreed with the reader.

    ``run_preflight`` nests the float32 deviation under ``residual_equivalence``; it writes
    nothing at the top level of this artifact. ``preflight_precision_block`` read the top
    level, so every probe artifact since 6f84217 recorded ``null`` for a number R18a requires
    in every one of them, and the existing coverage passed throughout because it fed a
    hand-made dict with the key where the reader hoped it was.

    The record here therefore comes out of ``run_preflight`` itself. Hand-shaping it would
    only re-record the assumption that failed. This test lives beside the writer's own fake
    harness for exactly that reason -- it must be the writer's shape, not a copy of it.
    """
    from local_llm_lab.probes.state_probe import preflight_precision_block

    spec = _spec()
    record = _preflight(spec, _View(), tmp_path)

    assert "fp32_manual_vs_native" not in record, "the writer does not use the top level"
    written = record["residual_equivalence"]["fp32_manual_vs_native"]
    assert written, "and it does write the block one level down"

    assert preflight_precision_block(spec, output_root=tmp_path) == written
