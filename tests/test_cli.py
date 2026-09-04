from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec
from local_llm_lab.pipeline import cli
from local_llm_lab.pipeline.cli import load_config, stage_data
from local_llm_lab.pipeline.data import SplitSpec


def test_preflight_stage_dispatches_from_registry_without_loading_run_config(monkeypatch) -> None:
    """The standalone preflight command must not require a YAML training recipe."""
    received = []
    monkeypatch.setattr(cli, "run_preflight", lambda name: received.append(name), raising=False)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("run config must stay unloaded")),
    )
    monkeypatch.setattr(sys, "argv", ["agent-pipeline", "preflight", "--model", "qwen35-4b"])

    cli.main()

    assert received == ["qwen35-4b"]


@pytest.mark.parametrize(
    ("stage", "arguments", "first_stage"),
    [
        ("train", [], "train"),
        ("select", [], "select"),
        ("eval", ["--base"], "eval"),
        ("all", [], "train"),
    ],
)
def test_model_loading_stages_require_preflight_once_before_loading(
    monkeypatch, tmp_path: Path, stage: str, arguments: list[str], first_stage: str
) -> None:
    """A missing guard before any model-loading stage would permit stale architecture evidence."""
    events: list[str] = []
    config = {
        "model": "registry-name",
        "output": tmp_path / "out",
        "data": tmp_path / "data",
        "eval": {"max_steps": 1, "max_tokens": 1, "split": "test", "limit": 1},
        "keep_last": 1,
        "seed": 1,
    }
    spec = _training_model_spec()
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(
        cli,
        "require_preflight",
        lambda received, *, skip: events.append(f"guard-{skip}") or {},
        raising=False,
    )
    monkeypatch.setattr(cli, "stage_data", lambda *args, **kwargs: events.append("data"))
    monkeypatch.setattr(cli, "stage_train", lambda *args, **kwargs: events.append("train"))
    monkeypatch.setattr(cli, "stage_select", lambda *args, **kwargs: events.append("select"))
    monkeypatch.setattr(cli, "stage_eval", lambda *args, **kwargs: events.append("eval"))
    monkeypatch.setattr(sys, "argv", ["agent-pipeline", stage, *arguments])

    cli.main()

    assert events.count("guard-False") == 1
    assert events.index("guard-False") < events.index(first_stage)


def test_skip_preflight_check_propagates_to_the_central_guard(monkeypatch, tmp_path: Path) -> None:
    """The explicit override must reach the one guard before training can load a model."""
    config = {
        "model": "registry-name",
        "output": tmp_path / "out",
        "data": tmp_path / "data",
        "eval": {"max_steps": 1, "max_tokens": 1, "split": "test", "limit": 1},
        "keep_last": 1,
        "seed": 1,
    }
    received = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "load_model_spec", lambda name: _training_model_spec())
    monkeypatch.setattr(
        cli, "require_preflight", lambda spec, *, skip: received.append(skip), raising=False
    )
    monkeypatch.setattr(cli, "stage_train", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys, "argv", ["agent-pipeline", "train", "--skip-preflight-check"]
    )

    cli.main()

    assert received == [True]


@pytest.mark.parametrize("name", ("agent_v2.yaml", "agent_v2b.yaml", "agent_v2c.yaml"))
def test_shipped_configs_define_only_the_complete_two_cell_selection_screen(name: str) -> None:
    """Catch a stale legacy split/limit selector in any shipped run configuration."""
    config = load_config(Path(__file__).parents[1] / "configs" / name)

    assert config["select"] == {
        "screen": [
            {"split": "valid", "difficulty": 1, "per_family": {"default": 1, "long": 3}},
            {"split": "valid2", "difficulty": 2, "per_family": {"default": 1, "long": 3}},
        ]
    }


def test_stage_data_normalizes_legacy_and_explicit_splits(monkeypatch, tmp_path) -> None:
    """Catch a data stage that passes raw YAML counts instead of SplitSpec values."""
    received = []
    def capture_dataset(*args, **kwargs):
        received.append((args, kwargs))
        return {"splits": {}}

    monkeypatch.setattr(cli, "write_dataset", capture_dataset)
    monkeypatch.setattr(cli, "write_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_model_spec", lambda name: object())
    base = {
        "model": "fake",
        "output": tmp_path / "out",
        "data": tmp_path / "data",
        "seed": 1,
        "keep_last": 2,
    }
    stage_data({**base, "tasks": {"train": 2, "valid": 3, "test": 4}}, [])
    legacy = received[-1][0][1]
    assert legacy == {
        "train": SplitSpec(2, role="train"),
        "valid": SplitSpec(3, role="valid"),
        "test": SplitSpec(4, role="test"),
    }
    explicit = {"train": {"count": 2, "difficulty": 0, "perturb": True, "role": "train"}}
    stage_data({**base, "splits": explicit}, [])
    assert received[-1][0][1] == {"train": SplitSpec(2, difficulty=0, perturb=True, role="train")}
    with pytest.raises(ValueError, match="exactly one"):
        stage_data({**base, "tasks": {"train": 1}, "splits": explicit}, [])
    with pytest.raises(ValueError, match="exactly one"):
        stage_data(base, [])


def test_stage_data_passes_the_exact_six_run_d_chunks_and_recovery_multipliers(
    monkeypatch, tmp_path
) -> None:
    """Stage the shipped recipe through the fake writer; no YAML dict may leak through."""
    received = []

    def capture_dataset(*args, **kwargs):
        received.append((args, kwargs))
        return {"splits": {}}

    monkeypatch.setattr(cli, "write_dataset", capture_dataset)
    monkeypatch.setattr(cli, "write_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_model_spec", lambda name: object())
    config = load_config(Path(__file__).parents[1] / "configs" / "agent_v2d.yaml")
    config["data"] = tmp_path / "data"
    config["output"] = tmp_path / "output"

    stage_data(config, [])

    assert received[-1][0][1] == {
        "train": SplitSpec(240, difficulty=0, perturb=True, role="train"),
        "train1": SplitSpec(120, difficulty=1, perturb=True, role="train"),
        "valid": SplitSpec(24, difficulty=1, perturb=False, role="valid"),
        "valid2": SplitSpec(24, difficulty=2, perturb=False, role="valid"),
        "test": SplitSpec(180, difficulty=2, perturb=False, role="test"),
        "test3": SplitSpec(60, difficulty=3, perturb=False, role="test"),
    }
    assert received[-1][1]["recovery_repeats"] == {
        "transient": 1,
        "wrong_path": 2,
        "unknown_tool": 2,
        "stale_path": 6,
        "failed_edit": 6,
    }


def test_run_d_configs_are_literal_pairwise_recipes() -> None:
    """Catch a data-recipe drift between D3/D4 or B/B4."""
    root = Path(__file__).parents[1] / "configs"
    d3 = load_config(root / "agent_v2d.yaml")
    d4 = load_config(root / "agent_v2d_qwen35_4b.yaml")
    b = load_config(root / "agent_v2b.yaml")
    b4 = load_config(root / "agent_v2b_qwen35_4b.yaml")
    for left, right in ((d3, d4), (b, b4)):
        for key in ("model", "output"):
            left.pop(key)
            right.pop(key)
        assert left == right
    assert d3["splits"] == {
        "train": {"count": 240, "difficulty": 0, "perturb": True, "role": "train"},
        "train1": {"count": 120, "difficulty": 1, "perturb": True, "role": "train"},
        "valid": {"count": 24, "difficulty": 1, "perturb": False, "role": "valid"},
        "valid2": {"count": 24, "difficulty": 2, "perturb": False, "role": "valid"},
        "test": {"count": 180, "difficulty": 2, "perturb": False, "role": "test"},
        "test3": {"count": 60, "difficulty": 3, "perturb": False, "role": "test"},
    }


def test_stage_select_writes_provenance_after_selection_json(monkeypatch, tmp_path: Path) -> None:
    """Selection provenance must capture the same durable payload without loading a model."""
    output = tmp_path / "output"
    output.mkdir()
    adapter = tmp_path / "step-10"
    adapter.mkdir()
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 2, "max_tokens": 3},
        "select": {
            "screen": [
                {"split": "valid", "difficulty": 1, "per_family": {"default": 1}}
            ]
        },
    }
    spec = object()
    calls = []
    model_lookups = []

    monkeypatch.setattr(cli, "checkpoint_dirs", lambda config: [(10, adapter)])
    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    def capture_model_spec(model):
        model_lookups.append(model)
        return spec

    monkeypatch.setattr(cli, "load_model_spec", capture_model_spec)
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda **kwargs: {
            "rate_counts": {
                "success": {"numerator": 1, "denominator": 1},
                "clean": {"numerator": 1, "denominator": 1},
                "valid_actions": {"numerator": 1, "denominator": 1},
            },
            "by_family": {"read": {"successes": 1, "tasks": 1}},
        },
    )

    def capture_provenance(run_dir, *, resolved, spec, extra):
        assert (output / "selection.json").is_file()
        calls.append((run_dir, resolved, spec, extra))

    monkeypatch.setattr(cli, "write_provenance", capture_provenance)

    assert cli.stage_select(config, limit=None, quiet=True) == output / "best-adapter"

    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert calls == [(output, None, spec, {"stage": "select", "selection": selection})]
    assert model_lookups == [config["model"]]


def _training_model_spec() -> ModelSpec:
    return ModelSpec(
        name="registry-name",
        hf_id="registry/hf-id",
        family="fake",
        chat=ChatSpec("unsupported", {}, "<eot>", ()),
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
    )


def _training_config(tmp_path: Path) -> dict[str, object]:
    config = cli.load_config(cli.DEFAULT_CONFIG)
    config.update({"model": "registry-name", "output": tmp_path / "run", "data": tmp_path / "data"})
    return config


def test_resolve_training_spec_uses_registry_hf_id_and_configured_key_policy(
    monkeypatch, tmp_path: Path
) -> None:
    """Target resolution uses the declared base identifier and effective frozen LoRA policy."""
    spec = _training_model_spec()
    model, tokenizer = object(), object()
    resolved = SimpleNamespace(spec=spec, num_layers=7, lora_keys=("hybrid.q",))
    loaded = []
    effective = []
    cleared = []

    def fake_resolve(self, received_model, received_tokenizer):
        effective.append((self, received_model, received_tokenizer))
        return resolved

    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(
        cli,
        "_load_training_base",
        lambda hf_id: (loaded.append(hf_id) or (model, tokenizer)),
    )
    monkeypatch.setattr(cli, "_clear_model_cache", lambda: cleared.append(True), raising=False)
    monkeypatch.setattr(ModelSpec, "resolve", fake_resolve)
    auto = _training_config(tmp_path)
    auto["train"].pop("lora_keys", None)

    assert cli._resolve_training_spec(auto) is resolved
    explicit = _training_config(tmp_path)
    explicit["train"]["lora_keys"] = ["hybrid.v", "hybrid.q"]
    explicit["train"].update({"rank": 3, "scale": 4.0, "dropout": 0.2})
    assert cli._resolve_training_spec(explicit) is resolved

    assert loaded == [spec.hf_id, spec.hf_id]
    assert cleared == [True, True]
    assert effective[0][0].lora == LoraSpec("auto", 16, 32.0, 0.0)
    assert effective[1][0].lora == LoraSpec(("hybrid.v", "hybrid.q"), 3, 4.0, 0.2)
    assert all(
        (received_model, received_tokenizer) == (model, tokenizer)
        for _, received_model, received_tokenizer in effective
    )


def test_resolve_training_spec_releases_temporary_objects_before_clearing_cache(
    monkeypatch, tmp_path: Path
) -> None:
    """The transient base model and tokenizer must not remain alive during cache clearing."""
    config = _training_config(tmp_path)
    spec = _training_model_spec()
    resolved = SimpleNamespace(spec=spec, num_layers=7, lora_keys=("hybrid.q",))
    events = []

    class Temporary:
        def __init__(self, name: str) -> None:
            self.name = name

        def __del__(self) -> None:
            events.append(f"released-{self.name}")

    def fake_load(hf_id: str):
        events.append(f"load-{hf_id}")
        return Temporary("model"), Temporary("tokenizer")

    def fake_resolve(self, model, tokenizer):
        events.append("resolved")
        return resolved

    def clear_cache() -> None:
        assert "released-model" in events
        assert "released-tokenizer" in events
        events.append("cleared")

    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(cli, "_load_training_base", fake_load)
    monkeypatch.setattr(cli, "_clear_model_cache", clear_cache)
    monkeypatch.setattr(ModelSpec, "resolve", fake_resolve)

    assert cli._resolve_training_spec(config) is resolved
    assert events[-1] == "cleared"


@pytest.mark.parametrize("failure", ("load", "resolve"))
def test_resolve_training_spec_clears_cache_on_load_and_resolution_failures(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    """Cache cleanup protects the entire temporary-model lifecycle, including both failures."""
    config = _training_config(tmp_path)
    spec = _training_model_spec()
    clears = []

    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    if failure == "load":
        monkeypatch.setattr(
            cli,
            "_load_training_base",
            lambda hf_id: (_ for _ in ()).throw(RuntimeError("load failed")),
        )
    else:
        monkeypatch.setattr(cli, "_load_training_base", lambda hf_id: (object(), object()))
        monkeypatch.setattr(
            ModelSpec,
            "resolve",
            lambda self, model, tokenizer: (_ for _ in ()).throw(RuntimeError("resolve failed")),
        )
    monkeypatch.setattr(cli, "_clear_model_cache", lambda: clears.append(True))

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        cli._resolve_training_spec(config)
    assert clears == [True]


def test_lora_config_uses_resolved_architecture_and_fresh_keys(tmp_path: Path) -> None:
    """mlx-lm config follows architecture-resolved targets rather than static config fields."""
    config = _training_config(tmp_path)
    resolved = SimpleNamespace(
        spec=SimpleNamespace(hf_id="registry/hybrid"),
        num_layers=7,
        lora_keys=("layers.0.hybrid.q_proj", "layers.6.hybrid.v_proj"),
    )
    weights = tmp_path / "0000100_adapters.safetensors"
    weights.write_bytes(b"")

    lora = cli.lora_config(config, resolved, iters=7, resume_from=Path("relative") / ".." / weights)
    another = cli.lora_config(config, resolved)

    assert lora["model"] == resolved.spec.hf_id
    assert lora["num_layers"] == resolved.num_layers
    assert lora["lora_parameters"]["keys"] == list(resolved.lora_keys)
    assert lora["lora_parameters"]["keys"] is not another["lora_parameters"]["keys"]
    assert lora["iters"] == 7
    assert lora["grad_checkpoint"] is config["train"]["grad_checkpoint"]
    assert lora["resume_adapter_file"] == str(weights.resolve())
    assert "resume_adapter_file" not in another
    assert another["iters"] == config["train"]["iters"]
    del config["train"]["grad_checkpoint"]
    assert cli.lora_config(config, resolved)["grad_checkpoint"] is True


def test_stage_train_clears_only_its_checkpoint_directory(tmp_path, monkeypatch) -> None:
    """Starting a training run removes stale checkpoints without deleting sibling outputs."""
    from local_llm_lab.pipeline import cli

    config = cli.load_config(cli.DEFAULT_CONFIG)
    output = tmp_path / "run"
    config.update({"output": output, "data": tmp_path / "data"})
    checkpoint = output / "checkpoints" / "step-1" / "adapters.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"stale")
    keep = output / "evals" / "prior.json"
    keep.parent.mkdir(parents=True)
    keep.write_text("preserve", encoding="utf-8")

    resolved = SimpleNamespace(
        spec=SimpleNamespace(hf_id="registry/hybrid"),
        num_layers=7,
        lora_keys=("layers.0.hybrid.q_proj",),
    )
    calls = []

    class FakeProcess:
        stdout: list[str] = []

        def __init__(self, code: int) -> None:
            self.code = code

        def wait(self) -> int:
            return self.code

    def fake_popen(*args, **kwargs):
        assert not (output / "checkpoints").exists()
        return FakeProcess(0)

    def capture_provenance(run_dir, *, resolved, spec, extra):
        calls.append((run_dir, resolved, spec, extra))

    monkeypatch.setattr(cli, "configure_local_cache", lambda: None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli, "_resolve_training_spec", lambda config: resolved, raising=False)
    monkeypatch.setattr(cli, "write_provenance", capture_provenance)

    cli.stage_train(config, iters=1)

    assert not (output / "checkpoints").exists()
    assert keep.read_text(encoding="utf-8") == "preserve"
    training_config = yaml.safe_load((output / "lora.yaml").read_text(encoding="utf-8"))
    assert training_config["num_layers"] == resolved.num_layers
    assert training_config["lora_parameters"]["keys"] == list(resolved.lora_keys)
    assert calls == [
        (
            output,
            resolved,
            resolved.spec,
            {"stage": "train", "training_config": training_config},
        )
    ]

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: FakeProcess(1))
    with pytest.raises(SystemExit, match="training failed"):
        cli.stage_train(config, iters=1)
    assert len(calls) == 1


def test_stage_eval_writes_one_ordered_provenance_record(monkeypatch, tmp_path: Path) -> None:
    """Evaluation provenance is written once after base and adapter policy summaries exist."""
    output = tmp_path / "output"
    adapter = output / "best-adapter"
    adapter.mkdir(parents=True)
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"split": "test", "limit": 180, "max_steps": 2, "max_tokens": 3},
    }
    spec = object()
    summaries = [{"label": "base"}, {"label": "best-adapter"}]
    evaluations = []
    calls = []
    lookups = []
    events = []

    def fake_evaluation(**kwargs):
        evaluations.append(kwargs)
        events.append(f"evaluated-{kwargs['label']}")
        return summaries[len(evaluations) - 1]

    def capture_provenance(run_dir, *, resolved, spec, extra) -> None:
        events.append("provenance")
        calls.append((run_dir, resolved, spec, deepcopy(extra)))

    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    monkeypatch.setattr(cli, "run_evaluation", fake_evaluation)
    monkeypatch.setattr(cli, "load_model_spec", lambda model: lookups.append(model) or spec)
    monkeypatch.setattr(cli, "write_provenance", capture_provenance)

    cli.stage_eval(
        config,
        adapter,
        base=True,
        split=None,
        limit=None,
        stress=False,
        quiet=True,
    )

    assert [evaluation["adapter"] for evaluation in evaluations] == [None, adapter]
    assert calls == [
        (output, None, spec, {"stage": "eval", "evaluations": summaries})
    ]
    assert lookups == [config["model"]]
    assert events == ["evaluated-base", "evaluated-best-adapter", "provenance"]


def test_stage_eval_second_policy_failure_writes_no_partial_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    """A later evaluation error must prevent the completed base result becoming provenance."""
    output = tmp_path / "output"
    adapter = output / "best-adapter"
    adapter.mkdir(parents=True)
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"split": "test", "limit": 180, "max_steps": 2, "max_tokens": 3},
    }
    evaluations = []
    calls = []

    def fake_evaluation(**kwargs):
        evaluations.append(kwargs["label"])
        if kwargs["label"] == "best-adapter":
            raise RuntimeError("second policy failed")
        return {"label": kwargs["label"]}

    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    monkeypatch.setattr(cli, "run_evaluation", fake_evaluation)
    monkeypatch.setattr(cli, "write_provenance", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(RuntimeError, match="second policy failed"):
        cli.stage_eval(
            config,
            adapter,
            base=True,
            split=None,
            limit=None,
            stress=False,
            quiet=True,
        )
    assert evaluations == ["base", "best-adapter"]
    assert calls == []


def test_stage_rollout_writes_one_post_run_provenance_record(monkeypatch, tmp_path: Path) -> None:
    """Rollout provenance captures the returned summary after the sampling call succeeds."""
    output = tmp_path / "output"
    adapter = output / "best-adapter"
    config = {
        "output": output,
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 2, "max_tokens": 3},
        "rollout": {"limit": 4, "samples": 2, "temperature": 0.7},
    }
    spec = object()
    summary = {"accepted": 3, "split": "iter1"}
    calls = []
    lookups = []

    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    monkeypatch.setattr(cli, "run_rollout", lambda **kwargs: summary)
    monkeypatch.setattr(cli, "load_model_spec", lambda model: lookups.append(model) or spec)
    monkeypatch.setattr(
        cli,
        "write_provenance",
        lambda run_dir, *, resolved, spec, extra: calls.append((run_dir, resolved, spec, extra)),
    )

    cli.stage_rollout(config, adapter, "iter1", limit=None, samples=None, quiet=True)

    assert calls == [(output, None, spec, {"stage": "rollout", "summary": summary})]
    assert lookups == [config["model"]]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], {"base": False, "stress": False, "limit": 180}),
        (["--base", "--stress"], {"base": True, "stress": True, "limit": 180}),
        (["--limit", "17"], {"base": False, "stress": False, "limit": 17}),
    ],
)
def test_main_all_wires_one_best_adapter_evaluation(
    monkeypatch, tmp_path: Path, arguments, expected
) -> None:
    """The all command evaluates the selected adapter once unless --base explicitly includes it."""
    config_path = tmp_path / "config.yaml"
    config = {"output": tmp_path / "output"}
    calls = []

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "_require_config_preflight", lambda config, *, skip: None)
    monkeypatch.setattr(cli, "stage_data", lambda config, extra: calls.append(("data", extra)))
    monkeypatch.setattr(cli, "stage_train", lambda config, iters: calls.append(("train", iters)))
    monkeypatch.setattr(
        cli,
        "stage_select",
        lambda config, limit, quiet: calls.append(("select", limit, quiet)),
    )
    monkeypatch.setattr(
        cli,
        "stage_eval",
        lambda config, adapter, **kwargs: calls.append(("eval", adapter, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(config_path), "all", *arguments])

    cli.main()

    assert calls == [
        ("data", []),
        ("train", None),
        ("select", None, False),
        (
            "eval",
            config["output"] / "best-adapter",
            {
                "base": expected["base"],
                "split": None,
                "limit": expected["limit"],
                "stress": expected["stress"],
                "quiet": False,
            },
        ),
    ]
