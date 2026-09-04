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


def test_main_resolves_requested_model_and_strategy_at_fake_boundaries(
    monkeypatch, capsys
) -> None:
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
