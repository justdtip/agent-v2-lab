"""Fake-boundary tests for the cache-equivalence research CLI."""

from __future__ import annotations

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from types import SimpleNamespace

import pytest

from local_llm_lab.arch import ArchitectureView
from local_llm_lab.project import PROJECT_ROOT


def _load_cache_equivalence_module():
    path = PROJECT_ROOT / "research" / "cache_equivalence.py"
    spec = spec_from_file_location("test_cache_equivalence_script", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeView:
    num_layers = 4
    hidden_size = 8
    vocab_size = 32
    tie_word_embeddings = False
    cache_trimmable = False

    def layer_kind(self, index: int) -> str:
        return ("linear_attention", "linear_attention", "linear_attention", "attention")[index]

    def lora_targets(self, policy) -> tuple[str, ...]:
        assert policy == "auto"
        return ("linear_attn.out_proj",)

    def lora_parameter_count(self, keys, rank: int) -> int:
        assert keys == ("linear_attn.out_proj",)
        assert rank == 16
        return 256


@pytest.mark.parametrize(
    "argv",
    [[], ["--strategy", "none"], ["--strategy", "auto"]],
)
def test_cli_rejects_unattestable_cache_strategy(argv) -> None:
    """An omitted, disabled, or auto-disabled cache cannot attest snapshot equivalence."""
    cache_equivalence = _load_cache_equivalence_module()

    with pytest.raises(SystemExit):
        cache_equivalence._parse_args(argv)


def test_main_resolves_requested_model_and_strategy_at_fake_boundaries(monkeypatch, capsys) -> None:
    """A hard-coded model or omitted runner metadata makes equivalence evidence unauditable."""
    from local_llm_lab.pipeline import evaluate, runner, tasks

    cache_equivalence = _load_cache_equivalence_module()

    fake_mx = types.ModuleType("mlx.core")
    fake_mx.random = SimpleNamespace(seed=lambda value: None)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    import mlx

    monkeypatch.setattr(mlx, "core", fake_mx, raising=False)

    model = SimpleNamespace(snapshot_revision="fake-model-revision")
    tokenizer = object()
    load_calls = []

    def fake_load_policy(given, adapter):
        load_calls.append((given.hf_id, adapter))
        return model, tokenizer, fake_view, given.resolve(model, tokenizer)

    fake_view = _FakeView()
    monkeypatch.setattr(evaluate, "load_policy", fake_load_policy)
    monkeypatch.setattr(evaluate, "make_sampler", lambda temperature: "greedy")
    monkeypatch.setattr(
        ArchitectureView,
        "from_model",
        classmethod(lambda cls, actual_model: fake_view),
    )
    fake_tasks = [
        SimpleNamespace(task_id=f"task-{index}", family=family)
        for index, family in enumerate(
            ("ledger_reconcile", "batch_update", "aggregate_report", "read", "search", "update")
        )
    ]
    monkeypatch.setattr(tasks, "make_tasks", lambda split, count: fake_tasks)
    run_calls = []

    def fake_run_task(actual_model, actual_tokenizer, task, **kwargs):
        run_calls.append((actual_model, actual_tokenizer, task, kwargs))
        return SimpleNamespace(
            steps=[{"action": {"name": "finish"}, "raw": "same"}],
            verdict={"success": True},
            success=True,
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    cache_equivalence.main(["--model", "qwen35-4b", "--strategy", "snapshot"])

    assert load_calls == [("mlx-community/Qwen3.5-4B-MLX-4bit", None)]
    assert (
        "ATTESTATION selected_model=qwen35-4b resolved_model=qwen35-4b "
        "hf_id=mlx-community/Qwen3.5-4B-MLX-4bit "
        "selected_strategy=snapshot resolved_strategy=snapshot reason=explicit:snapshot"
        in capsys.readouterr().out
    )
    assert len(run_calls) == 12
    for index, (actual_model, actual_tokenizer, _task, kwargs) in enumerate(run_calls):
        assert (actual_model, actual_tokenizer) == (model, tokenizer)
        assert kwargs["spec"].name == "qwen35-4b"
        assert kwargs["spec"].cache_strategy == "snapshot"
        assert kwargs["view"] is fake_view
        assert kwargs["resolved"].cache_strategy == "snapshot"
        assert kwargs["resolved"].cache_strategy_reason == "explicit:snapshot"
        assert kwargs["use_cache"] is bool(index % 2)


def test_fixed_history_requires_two_bitwise_gates_and_real_reuse():
    from test_live_lens_native import tiny_model

    module = _load_cache_equivalence_module()
    model = tiny_model()
    view = ArchitectureView.from_model(model)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cases = [dict(prompt_ids=prompt, continuation_ids=[10, 11])] * 2
    report = module.check_fixed_history(model, view, cases)
    assert report["gate_a"]["status"] == "passed"
    assert report["gate_b"]["status"] == "passed"
    assert report["accepted"] is True
    assert report["reused_tokens"] > 0
    assert report["checkpoint_certified"] is False

    no_reuse = module.check_fixed_history(model, view, cases[:1])
    assert no_reuse["gate_a"]["status"] == "inconclusive"
    assert no_reuse["accepted"] is False


def test_logit_check_rejects_one_bit_difference_even_with_identical_greedy_tokens():
    import mlx.core as mx
    import numpy as np

    module = _load_cache_equivalence_module()
    a = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    b = a.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.float32(2.0))
    result = module.compare_logits(mx.array(a), mx.array(b))
    assert result["bit_identical"] is False
    assert result["argmax_equal"] is True
    assert result["max_abs_error"] > 0
    assert result["winning_margin"] == [2.0, 2.0]


def test_fixed_history_rejects_a_new_schedule_that_changes_only_nonwinning_logits():
    import mlx.core as mx
    from test_live_lens_native import tiny_model

    module = _load_cache_equivalence_module()
    base = tiny_model()

    class ShapeSensitiveModel:
        def __getattr__(self, name):
            return getattr(base, name)

        def __call__(self, ids, *, cache):
            logits = base(ids, cache=cache)
            # Controlled shape dependence at a prefill boundary, passed into future state.
            if ids.shape[1] == 4:
                cache[0].state[1] = cache[0].state[1] + mx.array(0.001)
            return logits

    model = ShapeSensitiveModel()
    view = ArchitectureView.from_model(model)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cases = [dict(prompt_ids=prompt, continuation_ids=[10])] * 2
    report = module.check_fixed_history(model, view, cases, prefill_step_size=4)
    assert report["gate_a"]["status"] == "passed"
    assert report["gate_b"]["status"] == "failed"
    assert report["accepted"] is False


def test_fixed_history_cli_rejects_missing_report_and_legacy_strategy_before_load():
    module = _load_cache_equivalence_module()
    for argv in (
        ["--strategy", "history"],
        ["--strategy", "history", "--fixed-history", "fixture.json"],
        ["--strategy", "snapshot", "--fixed-history", "fixture.json", "--report", "out.json"],
    ):
        with pytest.raises(SystemExit):
            module._parse_args(argv)


def test_fixed_history_gate_detects_a_corrupted_snapshot(monkeypatch):
    from test_live_lens_native import tiny_model

    from local_llm_lab.pipeline import runner

    module = _load_cache_equivalence_module()
    original = runner.clone_prompt_cache

    def corrupt(cache):
        result = original(cache)
        result[0].state[1] = result[0].state[1] + 0.001
        return result

    monkeypatch.setattr(runner, "clone_prompt_cache", corrupt)
    model = tiny_model()
    cases = [dict(prompt_ids=[1, 2, 3, 4, 5], continuation_ids=[6])] * 2
    report = module.check_fixed_history(model, ArchitectureView.from_model(model), cases)
    assert report["gate_a"]["status"] == "failed"
    assert report["accepted"] is False


@pytest.mark.parametrize("precision", ["float32", "bfloat16", "4bit"])
def test_fixed_history_with_growing_conversation_and_native_prefill_cadence(precision):
    import mlx.core as mx
    import mlx.nn as nn
    from test_live_lens_native import tiny_model

    module = _load_cache_equivalence_module()
    model = tiny_model()
    if precision != "float32":
        model.set_dtype(mx.bfloat16)
    if precision == "4bit":
        nn.quantize(model, group_size=64, bits=4)
    prompt = [i % 24 for i in range(2050)]
    cases = [
        dict(prompt_ids=prompt, continuation_ids=[3, 4]),
        dict(prompt_ids=prompt + [3, 4, 5, 6], continuation_ids=[7, 8]),
    ]
    report = module.check_fixed_history(model, ArchitectureView.from_model(model), cases)
    assert report["accepted"] is True
    assert report["turns"][1]["reused_tokens"] == 2048
    assert report["turns"][1]["gate_a"]["state_bit_identical"]
    assert report["turns"][1]["gate_b"]["state_bit_identical"]


def test_fixed_cli_writes_reviewable_report_without_changing_registry(tmp_path, monkeypatch):
    import hashlib
    import json

    from test_live_lens_native import tiny_model

    from local_llm_lab.pipeline import evaluate

    module = _load_cache_equivalence_module()
    model = tiny_model()
    tokenizer = SimpleNamespace(snapshot_revision="tiny-random-weights")
    calls = []

    def load(spec, adapter):
        calls.append((spec.name, adapter))
        return model, tokenizer, ArchitectureView.from_model(model), spec.resolve(model, tokenizer)

    monkeypatch.setattr(evaluate, "load_policy", load)
    registry = PROJECT_ROOT / "configs/models/qwen35-4b.yaml"
    before = hashlib.sha256(registry.read_bytes()).hexdigest()
    corpus = tmp_path / "history.json"
    report = tmp_path / "report.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [
                    {
                        "id": "tiny-fixture",
                        "turns": [dict(prompt_ids=[1, 2, 3, 4, 5], continuation_ids=[6])] * 2,
                    },
                    {
                        "id": "no-reuse-control",
                        "turns": [dict(prompt_ids=[1, 2, 3], continuation_ids=[4])],
                    },
                ],
            }
        )
    )
    argv = [
        "--model",
        "qwen35-4b",
        "--strategy",
        "history",
        "--fixed-history",
        str(corpus),
        "--report",
        str(report),
    ]
    module.main(argv)
    result = json.loads(report.read_text())
    assert result["accepted"] is True
    assert result["gate_a"]["status"] == result["gate_b"]["status"] == "passed"
    assert result["registry_updated"] is False
    assert result["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert result["model"]["spec"]["cache_equivalence_verified"] is None
    assert result["source_hashes"]["src/local_llm_lab/forward.py"]
    assert result["runtime_sources"]["mlx_lm/models/gated_delta.py"]
    assert before == hashlib.sha256(registry.read_bytes()).hexdigest()
    assert calls == [("qwen35-4b", None)]
    with pytest.raises(SystemExit, match="already exists"):
        module.main(argv)
    assert len(calls) == 1  # refusal occurs before attempting another load


def test_fixed_cli_rejects_invalid_corpus_before_loading_weights(tmp_path, monkeypatch):
    import json

    from local_llm_lab.pipeline import evaluate

    module = _load_cache_equivalence_module()

    def forbidden(*args):
        pytest.fail("invalid fixed history must not reach the loader")

    monkeypatch.setattr(evaluate, "load_policy", forbidden)
    corpus = tmp_path / "bad.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [
                    {"id": "bad", "turns": [dict(prompt_ids=[True], continuation_ids=[])]}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="integer token"):
        module.main(
            [
                "--strategy",
                "history",
                "--fixed-history",
                str(corpus),
                "--report",
                str(tmp_path / "out.json"),
            ]
        )
