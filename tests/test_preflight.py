from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec, ResolvedSpec
from local_llm_lab.pipeline.preflight import (
    _residual_metrics,
    require_preflight,
    run_preflight,
    run_residual_control,
)


def _spec() -> ModelSpec:
    return ModelSpec(
        name="fake-model",
        hf_id="org/fake-model",
        family="fake",
        chat=ChatSpec("off", {"chat_template": "fake"}, "<eot>", ()),
        lora=LoraSpec("attention+mlp", 2, 4.0, 0.0),
        train={"batch_size": 2, "max_seq_length": 5},
        cache_strategy="auto",
        cache_equivalence_verified={"date": "2026-09-04", "sha256": "a" * 64},
        probe_layer_fractions=(0.5,),
        memory_budget_gib=1.0,
        policies={},
    )


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


class _MismatchingTextModule(_TextModule):
    def __call__(self, ids: np.ndarray) -> np.ndarray:
        return super().__call__(ids) + 0.5


class _MismatchingView(_View):
    text_module = _MismatchingTextModule()


class _ControlView(_View):
    def make_cache(self) -> list[object]:
        raise AssertionError("residual controls must not create a cache")

    def residuals(self, ids: np.ndarray, layers: list[int]) -> dict[int, np.ndarray]:
        del ids, layers
        raise AssertionError("residual controls must not capture JVP residuals")


class _MetricArray:
    def __init__(self, values: list[float], dtype: str) -> None:
        self.values = np.array(values, dtype=np.float64)
        self.dtype = dtype

    def __sub__(self, other: _MetricArray) -> _MetricArray:
        return _MetricArray((self.values - other.values).tolist(), self.dtype)


class _MetricApi:
    _info = {
        "bfloat16": SimpleNamespace(eps=2**-7, tiny=2**-126),
        "float32": SimpleNamespace(eps=2**-23, tiny=2**-126),
    }

    @staticmethod
    def abs(value: _MetricArray) -> _MetricArray:
        return _MetricArray(np.abs(value.values).tolist(), value.dtype)

    @staticmethod
    def max(value: _MetricArray) -> float:
        return float(np.max(value.values))

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


def _complete_artifact(*, passed: bool = True, residual_passed: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
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
    }


def test_run_preflight_writes_stable_complete_fake_report(tmp_path: Path) -> None:
    """A missing inspection or unstable report would break the preflight contract."""
    spec = _spec()
    tokenizers: list[_Tokenizer] = []
    calls: list[str] = []

    def loader(hf_id: str, *, lazy: bool) -> tuple[_Model, _Tokenizer]:
        assert (hf_id, lazy) == (spec.hf_id, True)
        calls.append(hf_id)
        tokenizer = _Tokenizer()
        tokenizers.append(tokenizer)
        return _Model(), tokenizer

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
    assert report["schema_version"] == 1
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
    assert report["residual_equivalence"] == {
        "absolute_tolerance": 2 * np.finfo(np.float32).eps * 146.0,
        "criterion": "exact_pre_control",
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "passed": True,
        "reference_dtype": "float32",
        "reference_epsilon": np.finfo(np.float32).eps,
        "reference_scale": 146.0,
        "relative_tolerance": 2 * np.finfo(np.float32).eps,
        "token_count": 64,
        "within_tolerance": True,
    }
    assert report["jvp"] == {"finite": True, "layer": 2, "method": "forward"}
    assert report["lora"] == {
        "keys": ["layers.0.q_proj", "layers.3.down_proj"],
        "trainable_parameters": 42,
    }
    assert report["memory"] == {
        "activation_bytes": 480,
        "budget_gib": 1.0,
        "parameter_bytes": 16,
        "total_bytes": 496,
        "total_gib": 496 / 1024**3,
        "within_budget": True,
    }
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
            loader=lambda hf_id, *, lazy: (_Model(), _Tokenizer()),
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
            loader=lambda hf_id, *, lazy: (_Model(), _Tokenizer()),
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


def test_residual_metrics_follow_the_reference_dtype_and_scale() -> None:
    """Changing the reference precision must change the derived error budget."""
    actual = _MetricArray([100.2], "float32")
    bfloat_reference = _MetricArray([100.0], "bfloat16")
    float_reference = _MetricArray([100.0], "float32")

    bfloat_metrics = _residual_metrics(actual, bfloat_reference, array_api=_MetricApi)
    float_metrics = _residual_metrics(actual, float_reference, array_api=_MetricApi)

    assert bfloat_metrics["reference_dtype"] == "bfloat16"
    assert bfloat_metrics["reference_epsilon"] == 2**-7
    assert bfloat_metrics["reference_scale"] == 100.0
    assert bfloat_metrics["relative_tolerance"] == 2**-6
    assert bfloat_metrics["absolute_tolerance"] == 100.0 * 2**-6
    assert bfloat_metrics["within_tolerance"] is True
    assert float_metrics["reference_dtype"] == "float32"
    assert float_metrics["absolute_tolerance"] == 100.0 * 2**-22
    assert float_metrics["within_tolerance"] is False
    assert bfloat_metrics["absolute_tolerance"] != 1e-5


def test_run_residual_control_writes_only_the_three_residual_comparisons(tmp_path: Path) -> None:
    """A residual control must load once and avoid unrelated preflight inspection work."""
    spec = _spec()
    calls: list[tuple[str, bool]] = []
    output_path = tmp_path / "native-loop.json"

    path = run_residual_control(
        spec.name,
        output_path=output_path,
        loader=lambda hf_id, *, lazy: (calls.append((hf_id, lazy)) or (_Model(), _Tokenizer())),
        spec_loader=lambda name: spec,
        view_factory=lambda model: _ControlView(),
        revision_reader=lambda given: "cached-revision",
        array_api=np,
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert path == output_path
    assert calls == [(spec.hf_id, True)]
    assert report["schema_version"] == 1
    assert report["model_name"] == spec.name
    assert report["hf_id"] == spec.hf_id
    assert report["snapshot_revision"] == "cached-revision"
    assert report["token_identity"] == {"ids": list(range(64)), "token_count": 64}
    assert report["fp32_manual_vs_native"]["max_abs_error"] == 0.0
    assert report["native_manual_vs_native"]["max_abs_error"] == 0.0


def test_failed_preflight_writes_evidence_then_exits_nonzero(tmp_path: Path) -> None:
    """A failed equivalence check must preserve its diagnostic artifact before exiting."""
    spec = _spec()

    with pytest.raises(SystemExit, match="preflight failed"):
        run_preflight(
            spec.name,
            loader=lambda hf_id, *, lazy: (_Model(), _Tokenizer()),
            output_root=tmp_path,
            spec_loader=lambda name: spec,
            view_factory=lambda model: _MismatchingView(),
            resolver=lambda given, model, token: _resolved(given),
            jvp=lambda *args, **kwargs: np.ones((1, 64, 3), dtype=np.float32),
            revision_reader=lambda given: "cached-revision",
            array_api=np,
        )

    report = json.loads((tmp_path / "fake-model.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["residual_equivalence"]["passed"] is False
    assert report["residual_equivalence"]["criterion"] == "exact_pre_control"
    assert "absolute_tolerance" in report["residual_equivalence"]


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (None, "preflight artifact is missing"),
        ("not json", "preflight artifact is malformed"),
        ({"schema_version": 1, "model_name": "wrong"}, "model name"),
        ({"schema_version": 1, "model_name": "fake-model", "hf_id": "wrong"}, "hf_id"),
        (
            {"schema_version": 1, "model_name": "fake-model", "hf_id": "org/fake-model"},
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
                "schema_version": 1,
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
                "schema_version": 1,
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
                "schema_version": 1,
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
