from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import re
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
    monkeypatch.setattr(
        cli,
        "run_preflight",
        # The optional R32(d) footprint arguments are the preflight lane's; what this test
        # pins is the model name and that no run config is read to find it.
        lambda name, **kwargs: received.append(name),
        raising=False,
    )
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
    monkeypatch.setattr(cli, "load_model_spec", lambda name: SimpleNamespace(hf_id="fake/hf"))
    monkeypatch.setattr(cli, "_load_data_tokenizer", lambda hf_id: object())
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


def test_stage_data_uses_registered_tokenizer_without_loading_model_weights(
    monkeypatch, tmp_path
) -> None:
    spec = SimpleNamespace(hf_id="registry/hf-id")
    tokenizer = object()
    datasets = []
    provenance = []
    config = {
        "model": "registry-name", "output": tmp_path / "out", "data": tmp_path / "data",
        "seed": 1, "keep_last": 2, "tasks": {"train": 1, "valid": 1, "test": 1},
    }
    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(cli, "_load_data_tokenizer", lambda hf_id: tokenizer, raising=False)
    monkeypatch.setattr(cli, "_load_training_base", lambda *_: pytest.fail("loaded weights"))
    monkeypatch.setattr(ModelSpec, "resolve", lambda *_: pytest.fail("resolved model"))
    monkeypatch.setattr(
        cli,
        "write_dataset",
        lambda *args, **kwargs: datasets.append((args, kwargs)) or {"splits": {}},
    )
    monkeypatch.setattr(
        cli, "write_provenance", lambda run_dir, *, resolved, spec, extra: provenance.append(spec)
    )

    cli.stage_data(config, [])

    assert datasets[0][1]["tokenizer"] is tokenizer
    assert datasets[0][1]["spec"] is spec
    assert provenance == [spec]


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
    monkeypatch.setattr(cli, "load_model_spec", lambda name: SimpleNamespace(hf_id="fake/hf"))
    monkeypatch.setattr(cli, "_load_data_tokenizer", lambda hf_id: object())
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


def _has_gated_delta_recurrence(model: str) -> bool:
    """True when mlx-lm's backbone for this registry model runs the gated-delta recurrence.

    The registry entry names the model family; mlx-lm ships one backbone module per family,
    and the hybrid ones are exactly those defining ``gated_delta_update``. A family mlx-lm
    ships no module for is dense on either reading: the 3B arm's ``qwen2_5`` runs mlx-lm's
    ``qwen2`` backbone, which has no recurrence.
    """
    family = cli.load_model_spec(model).family
    if importlib.util.find_spec(f"mlx_lm.models.{family}") is None:
        return False
    return hasattr(importlib.import_module(f"mlx_lm.models.{family}"), "gated_delta_update")


def _effective_batch(train: dict[str, object]) -> int:
    return int(train["batch_size"]) * int(train["grad_accumulation_steps"])


def test_run_d_configs_are_literal_pairwise_recipes() -> None:
    """Catch a data-recipe drift between D3/D4 or B/B4, and pin the R21 render shape.

    The arms remain literal pairwise recipes, with two exceptions ruled by R32. Both are
    memory settings for this machine rather than changes to what is trained:

    * ``batch_size`` and ``grad_accumulation_steps`` are compared on their product, the
      EFFECTIVE batch, not literally. The 4B arm runs batch 1 x accumulation 4 where the 3B
      arm runs 2 x 2: the same effective batch and the same optimiser steps, at half the
      per-step attention and MLP activations (R32 efficiency item 2).
    * ``train.gated_delta_chunk`` may appear on a hybrid arm only, and is not required to
      match. It is the token length of one checkpointed segment of the training-time
      gated-delta recurrence, whose outputs, final state and gradients are bit-exact against
      mlx-lm's reference loop at every chunk length, so it trades memory against recompute
      time and changes no number the run produces (R32(a), (c)).

    Every other train key, and every other block, is still compared literally.
    """
    root = Path(__file__).parents[1] / "configs"
    d3 = load_config(root / "agent_v2d.yaml")
    d4 = load_config(root / "agent_v2d_qwen35_4b.yaml")
    b = load_config(root / "agent_v2b.yaml")
    b4 = load_config(root / "agent_v2b_qwen35_4b.yaml")
    for base, cross in ((d3, d4), (b, b4)):
        models = {"base": base["model"], "cross": cross["model"]}
        for key in ("model", "output"):
            base.pop(key)
            cross.pop(key)
        # R15 condition 5: each arm records its own SPEC-003 §5 success criterion, so the
        # criteria block is per-arm by design and is not part of the data recipe under test.
        base.pop("criteria", None)
        cross.pop("criteria", None)
        # R21(c): the cross-model arm renders the base arm's rows into a NEW directory; it
        # must never point its data output at the base arm's irreplaceable dataset.
        assert cross.pop("source_rows") == base["data"]
        base_data = base.pop("data")
        cross_data = cross.pop("data")
        assert cross_data.name == f"{base_data.name}-qwen35-4b"
        assert cross_data != base_data
        base_train = base.pop("train")
        cross_train = cross.pop("train")
        assert _effective_batch(cross_train) == _effective_batch(base_train)
        for role, arm in (("base", base_train), ("cross", cross_train)):
            del arm["batch_size"], arm["grad_accumulation_steps"]
            if "gated_delta_chunk" not in arm:
                continue
            assert _has_gated_delta_recurrence(models[role]), models[role]
            chunk = arm.pop("gated_delta_chunk")
            arm.pop("gated_delta_mode", None)  # R32 stage 2: the form that chunk belongs to.
            assert isinstance(chunk, int) and chunk >= 1
        assert base_train == cross_train
        assert base == cross
    assert d3["splits"] == {
        "train": {"count": 240, "difficulty": 0, "perturb": True, "role": "train"},
        "train1": {"count": 120, "difficulty": 1, "perturb": True, "role": "train"},
        "valid": {"count": 24, "difficulty": 1, "perturb": False, "role": "valid"},
        "valid2": {"count": 24, "difficulty": 2, "perturb": False, "role": "valid"},
        "test": {"count": 180, "difficulty": 2, "perturb": False, "role": "test"},
        "test3": {"count": 60, "difficulty": 3, "perturb": False, "role": "test"},
    }


def test_stage_data_with_source_rows_delegates_to_render_and_never_generates(
    monkeypatch, tmp_path: Path
) -> None:
    """R21(c): a source_rows config re-renders existing rows; generation must be unreachable."""
    received = []
    config = {
        "model": "registry-name",
        "output": tmp_path / "out",
        "data": tmp_path / "data" / "rendered",
        "source_rows": tmp_path / "data" / "source",
        "seed": 1,
        "keep_last": 2,
        "tasks": {"train": 1, "valid": 1, "test": 1},
    }
    monkeypatch.setattr(
        cli, "stage_render", lambda **kwargs: received.append(kwargs), raising=False
    )
    monkeypatch.setattr(
        cli, "write_dataset", lambda *args, **kwargs: pytest.fail("write_dataset was called")
    )
    monkeypatch.setattr(
        cli,
        "_load_data_tokenizer",
        lambda hf_id: pytest.fail("delegation must not load a tokenizer twice"),
    )

    cli.stage_data(config, [], overwrite=True)

    assert received == [
        {
            "source": config["source_rows"],
            "output": config["data"],
            "model": "registry-name",
            "overwrite": True,
        }
    ]
    with pytest.raises(SystemExit, match="--extra"):
        cli.stage_data(config, [tmp_path / "extra"])


def test_render_command_with_explicit_flags_skips_the_run_config(
    monkeypatch, tmp_path: Path
) -> None:
    """The R21(b) form render --source/--output/--model must not require a YAML recipe."""
    received = []
    monkeypatch.setattr(
        cli, "stage_render", lambda **kwargs: received.append(kwargs), raising=False
    )
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("run config must stay unloaded")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-pipeline",
            "render",
            "--source",
            str(tmp_path / "src"),
            "--output",
            str(tmp_path / "dst"),
            "--model",
            "registry-name",
            "--force-overwrite",
        ],
    )

    cli.main()

    assert received == [
        {
            "source": (tmp_path / "src").resolve(),
            "output": (tmp_path / "dst").resolve(),
            "model": "registry-name",
            "overwrite": True,
        }
    ]


def test_render_command_falls_back_to_config_source_rows(monkeypatch, tmp_path: Path) -> None:
    """render --config alone reads source_rows/data/model from the run configuration."""
    received = []
    config = {
        "model": "registry-name",
        "data": tmp_path / "rendered",
        "source_rows": tmp_path / "source",
    }
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli, "stage_render", lambda **kwargs: received.append(kwargs), raising=False
    )
    monkeypatch.setattr(sys, "argv", ["agent-pipeline", "render"])

    cli.main()

    assert received == [
        {
            "source": config["source_rows"],
            "output": config["data"],
            "model": "registry-name",
            "overwrite": False,
        }
    ]

    monkeypatch.setattr(cli, "load_config", lambda path: {"model": "m", "data": tmp_path})
    with pytest.raises(SystemExit):
        cli.main()
    assert len(received) == 1


def test_stage_render_uses_registry_tokenizer_and_writes_render_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    """Rendering loads the registered tokenizer only, and its provenance names the stage."""
    spec = SimpleNamespace(hf_id="registry/hf-id")
    tokenizer = object()
    manifest = {"outputs": {"train": {"rows": 2, "sha256": "x"}}, "source": {"directory": "s"}}
    renders = []
    provenance = []
    tokenizer_loads = []

    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(
        cli, "_load_data_tokenizer", lambda hf_id: tokenizer_loads.append(hf_id) or tokenizer
    )
    monkeypatch.setattr(cli, "_load_training_base", lambda *_: pytest.fail("loaded weights"))
    monkeypatch.setattr(
        cli,
        "render_dataset",
        lambda *args, **kwargs: renders.append((args, kwargs)) or manifest,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "write_provenance",
        lambda run_dir, *, resolved, spec, extra: provenance.append(
            (run_dir, resolved, spec, extra)
        ),
    )

    cli.stage_render(
        source=tmp_path / "src", output=tmp_path / "dst", model="registry-name", overwrite=True
    )

    assert tokenizer_loads == ["registry/hf-id"]
    assert renders == [
        ((tmp_path / "src", tmp_path / "dst", tokenizer), {"spec": spec, "overwrite": True})
    ]
    assert provenance == [
        (
            tmp_path / "dst",
            None,
            spec,
            {"stage": "render", "dataset_manifest": manifest},
        )
    ]


@pytest.mark.parametrize(("arguments", "overwrite"), [([], False), (["--force-overwrite"], True)])
def test_data_command_propagates_force_overwrite_to_the_stage(
    monkeypatch, tmp_path: Path, arguments: list[str], overwrite: bool
) -> None:
    received = []
    monkeypatch.setattr(cli, "load_config", lambda path: {"output": tmp_path})
    monkeypatch.setattr(
        cli,
        "stage_data",
        lambda config, extra, *, overwrite: received.append((extra, overwrite)),
    )
    monkeypatch.setattr(sys, "argv", ["agent-pipeline", "data", *arguments])

    cli.main()

    assert received == [([], overwrite)]


def test_stage_rollout_refuses_a_manifest_holding_target(monkeypatch, tmp_path: Path) -> None:
    """The R21 guard covers every stage that writes under data/, not only the data stage."""
    from local_llm_lab.pipeline.data import DatasetWriteGuardError

    target = tmp_path / "data" / "rollouts" / "iter1"
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("{}\n", encoding="utf-8")
    config = {
        "output": tmp_path / "output",
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 2, "max_tokens": 3},
        "rollout": {"limit": 4, "samples": 2, "temperature": 0.7},
    }
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    monkeypatch.setattr(cli, "run_rollout", lambda **kwargs: pytest.fail("rollout ran"))

    with pytest.raises(DatasetWriteGuardError, match="manifest.json") as excinfo:
        cli.stage_rollout(config, None, "iter1", limit=None, samples=None, quiet=True)
    assert "--force-overwrite" not in str(excinfo.value), (
        "the refusal must not advertise a flag the rollout stage does not expose"
    )


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

    spec = _training_model_spec()
    resolved = SimpleNamespace(spec=spec, num_layers=7, lora_keys=("layers.0.hybrid.q_proj",))
    calls = []
    model, tokenizer = object(), object()

    def fake_trainer(args, received_model, train_set, valid_set, training_callback=None):
        assert not (output / "checkpoints").exists()

    def capture_provenance(run_dir, *, resolved, spec, extra):
        calls.append((run_dir, resolved, spec, extra))

    monkeypatch.setattr(cli, "_load_training_entry", lambda: (fake_trainer, {}))
    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(cli, "_load_training_base", lambda hf_id: (model, tokenizer))
    monkeypatch.setattr(ModelSpec, "resolve", lambda self, model, tokenizer: resolved)
    monkeypatch.setattr(
        cli, "load_rendered_splits", lambda *args, **kwargs: (object(), object())
    )
    monkeypatch.setattr(cli, "_clear_model_cache", lambda: None)
    monkeypatch.setattr(cli, "write_provenance", capture_provenance)

    cli.stage_train(config, iters=1)

    assert not (output / "checkpoints").exists()
    assert keep.read_text(encoding="utf-8") == "preserve"
    training_config = yaml.safe_load((output / "lora.yaml").read_text(encoding="utf-8"))
    assert training_config["num_layers"] == resolved.num_layers
    assert training_config["lora_parameters"]["keys"] == list(resolved.lora_keys)
    ((run_dir, received_resolved, received_spec, extra),) = calls
    assert (run_dir, received_resolved, received_spec) == (output, resolved, resolved.spec)
    assert extra["stage"] == "train" and extra["training_config"] == training_config
    # R26(f), issue #35: the final health record travels with provenance. This fake trainer
    # reports nothing and saves nothing, which is exactly what an incomplete run looks like.
    assert extra["health"]["verdict"] == "incomplete"
    assert extra["health_thresholds"] == extra["health"]["thresholds"]



def test_training_entry_validates_pinned_five_argument_runtime(monkeypatch) -> None:
    """Training must accept only R14's pinned trainer boundary before model loading."""
    calls = []

    def exact_train_model(args, model, train_set, valid_set, training_callback=None):
        return None

    package = SimpleNamespace(__version__="0.31.3")
    module = SimpleNamespace(train_model=exact_train_model, CONFIG_DEFAULTS={"batch_size": 1})

    def fake_import(name):
        calls.append(name)
        return package if name == "mlx_lm" else module

    monkeypatch.setattr(cli.importlib, "import_module", fake_import, raising=False)

    trainer, defaults = cli._load_training_entry()

    assert trainer is exact_train_model
    assert defaults == {"batch_size": 1}
    assert calls == ["mlx_lm", "mlx_lm.lora"]


def test_training_entry_rejects_adversarial_signature_before_model_load(monkeypatch) -> None:
    def wrong_train_model(
        *args, model=None, train_set=None, valid_set=None, training_callback=None
    ):
        return None

    package = SimpleNamespace(__version__="0.31.3")
    module = SimpleNamespace(train_model=wrong_train_model, CONFIG_DEFAULTS={})
    monkeypatch.setattr(
        cli.importlib, "import_module", lambda name: package if name == "mlx_lm" else module
    )
    monkeypatch.setattr(cli, "_load_training_base", lambda *_: pytest.fail("loaded model"))

    with pytest.raises(SystemExit, match="signature mismatch"):
        cli._load_training_entry()


def test_data_tokenizer_uses_pinned_utils_loader_in_order(monkeypatch) -> None:
    events = []
    package = SimpleNamespace(__version__="0.31.3")
    tokenizer = object()
    utils = SimpleNamespace(
        load_tokenizer=lambda hf_id: events.append(("tokenizer", hf_id)) or tokenizer
    )
    monkeypatch.setattr(cli, "configure_local_cache", lambda: events.append("cache"))
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: events.append(("import", name)) or (package if name == "mlx_lm" else utils),
    )

    assert cli._load_data_tokenizer("registry/hf") is tokenizer
    assert events == [
        "cache", ("import", "mlx_lm"), ("import", "mlx_lm.utils"), ("tokenizer", "registry/hf")
    ]


def test_stage_train_uses_rendered_splits_and_five_argument_trainer(
    monkeypatch, tmp_path: Path
) -> None:
    """The in-process trainer gets model and rendered train/valid sets, never a tokenizer."""
    config = _training_config(tmp_path)
    output = config["output"]
    stale = output / "checkpoints" / "stale"
    stale.mkdir(parents=True)
    (output / "evals").mkdir(parents=True)
    (output / "evals" / "keep").write_text("keep", encoding="utf-8")
    spec = _training_model_spec()
    model, tokenizer = object(), object()
    resolved = SimpleNamespace(spec=spec, num_layers=7, lora_keys=("hybrid.q",))
    train_set, valid_set = object(), object()
    requested_splits = []
    trainer_calls = []
    provenance = []

    def exact_train_model(
        args, received_model, received_train, received_valid, training_callback=None
    ):
        trainer_calls.append((vars(args).copy(), received_model, received_train, received_valid))
        training_callback.on_val_loss_report({"iteration": 0, "val_loss": 1.25})
        training_callback.on_train_loss_report(
            {"iteration": 1, "train_loss": 1.0, "trained_tokens": 64}
        )
        print("Iter 1: Val loss 1.25")

    monkeypatch.setattr(cli, "_load_training_entry", lambda: (exact_train_model, {"extra": 9}))
    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(cli, "_load_training_base", lambda hf_id: (model, tokenizer))
    monkeypatch.setattr(ModelSpec, "resolve", lambda self, model, tokenizer: resolved)
    monkeypatch.setattr(
        cli,
        "load_rendered_splits",
        lambda path, received_tokenizer, *, max_seq_length, splits: (
            requested_splits.append(tuple(splits)) or (train_set, valid_set)
        ),
        raising=False,
    )
    monkeypatch.setattr(cli, "_clear_model_cache", lambda: None)
    monkeypatch.setattr(
        cli,
        "write_provenance",
        lambda run_dir, *, resolved, spec, extra: provenance.append(extra),
    )

    cli.stage_train(config, iters=1)

    rendered = yaml.safe_load((output / "lora.yaml").read_text(encoding="utf-8"))
    assert trainer_calls == [(rendered, model, train_set, valid_set)]
    # lora["test"] is False, so the stage must not pay for a split it never trains on.
    assert requested_splits == [("train", "valid")]
    assert not stale.exists() and (output / "evals" / "keep").is_file()
    (extra,) = provenance
    assert extra["stage"] == "train" and extra["training_config"] == rendered
    # R26(f), issue #35: health.json's record is copied into provenance, thresholds included.
    assert extra["health_thresholds"] == extra["health"]["thresholds"]
    metrics = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
    assert {key: value for key, value in metrics.items() if key != "elapsed"} == {
        "step": 1,
        "train_loss": None,
        "val_loss": 1.25,
        "tokens": None,
    }
    assert isinstance(metrics["elapsed"], float) and metrics["elapsed"] >= 0


# --------------------------------------------------------- run logging and health (issue #35, R26)

_TRAINER_LINE = "Iter 1: Train loss 1.000, Tokens/sec 100.000, Peak mem 0.500 GB"
_ELAPSED = re.compile(r'"elapsed": [0-9.e+-]+')


def _report(iteration: int, loss: float) -> dict[str, object]:
    """One mlx-lm 0.31.3 train report, with the exact field names TrainingCallback delivers."""
    return {
        "iteration": iteration,
        "train_loss": loss,
        "learning_rate": 3.0e-5,
        "tokens_per_second": 100.0,
        "trained_tokens": 64 * iteration,
        "peak_memory": 0.5,
    }


def _patch_stage_train(monkeypatch, trainer, *, tokenizer=None, splits=None) -> SimpleNamespace:
    """Wire stage_train to fakes only: no trainer entry point, no model, no dataset on disk.

    ``splits`` replaces the loader with a fake returning it; pass ``keep`` to leave the real
    ``load_rendered_splits`` in place and read the rendered rows the test wrote to disk.
    """
    spec = _training_model_spec()
    model = object()
    tokenizer = object() if tokenizer is None else tokenizer
    resolved = SimpleNamespace(spec=spec, num_layers=7, lora_keys=("hybrid.q",))
    record = SimpleNamespace(provenance=[], loads=[], resolved=resolved, spec=spec)

    def fake_load_base(hf_id):
        record.loads.append(hf_id)
        return model, tokenizer

    def fake_provenance(run_dir, *, resolved, spec, extra):
        record.provenance.append(extra)
        (run_dir / "provenance.json").write_text(
            json.dumps(extra, indent=2, default=str), encoding="utf-8"
        )

    monkeypatch.setattr(cli, "_load_training_entry", lambda: (trainer, {}))
    monkeypatch.setattr(cli, "load_model_spec", lambda name: spec)
    monkeypatch.setattr(cli, "_load_training_base", fake_load_base)
    monkeypatch.setattr(ModelSpec, "resolve", lambda self, model, tokenizer: resolved)
    if splits != "keep":
        loaded = (["a", "b", "c"], ["d", "e"]) if splits is None else splits
        monkeypatch.setattr(
            cli,
            "load_rendered_splits",
            lambda path, received_tokenizer, **kwargs: loaded,
            raising=False,
        )
    monkeypatch.setattr(cli, "_clear_model_cache", lambda: None)
    monkeypatch.setattr(cli, "write_provenance", fake_provenance)
    return record


def _healthy_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
    """A complete run: every planned iteration reported and a final checkpoint written."""
    print(_TRAINER_LINE)
    training_callback.on_train_loss_report(_report(1, 1.0))
    training_callback.on_val_loss_report({"iteration": 1, "val_loss": 1.25})
    training_callback.on_train_loss_report(_report(2, 0.9))
    Path(args.adapter_path, "adapters.safetensors").write_bytes(b"")


def _short_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
    """A run that stops one iteration short of the plan but still saves a checkpoint."""
    training_callback.on_train_loss_report(_report(1, 1.0))
    Path(args.adapter_path, "adapters.safetensors").write_bytes(b"")


def _no_checkpoint_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
    """A run that reports every iteration and leaves no final adapter file behind."""
    training_callback.on_train_loss_report(_report(1, 1.0))
    training_callback.on_train_loss_report(_report(2, 0.9))


def _nan_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
    """mlx-lm does not catch callback exceptions, so the abort must end the loop here."""
    training_callback.on_train_loss_report(_report(1, float("nan")))
    raise AssertionError("the training loop must not continue past a non-finite loss")


def _spiking_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
    """Five flat reports fill the trailing window; the sixth is well past the spike factor."""
    for iteration in range(1, 6):
        training_callback.on_train_loss_report(_report(iteration, 1.0))
    training_callback.on_train_loss_report(_report(6, 10.0))
    Path(args.adapter_path, "adapters.safetensors").write_bytes(b"")


def test_stage_train_writes_run_log_events_and_health_beside_the_existing_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    """R26(a): a training run must leave a readable stream and a machine record of itself."""
    config = _training_config(tmp_path)
    output = config["output"]
    _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    for name in ("train.log", "metrics.jsonl", "run.log", "events.jsonl", "health.json"):
        assert (output / name).is_file(), name


def test_stage_train_keeps_the_metrics_stream_byte_identical(monkeypatch, tmp_path: Path) -> None:
    """_validation_losses reads metrics.jsonl, so its keys, order and step convention are frozen."""
    config = _training_config(tmp_path)
    output = config["output"]
    _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    text = (output / "metrics.jsonl").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert [_ELAPSED.sub('"elapsed": T', line) for line in text.splitlines()] == [
        '{"step": 1, "train_loss": 1.0, "val_loss": null, "tokens": 64, "elapsed": T}',
        '{"step": 2, "train_loss": null, "val_loss": 1.25, "tokens": null, "elapsed": T}',
        '{"step": 2, "train_loss": 0.9, "val_loss": null, "tokens": 128, "elapsed": T}',
    ]
    assert cli._validation_losses(output) == {2: 1.25}


def test_stage_train_keeps_train_log_verbatim_and_mirrors_it_into_run_log(
    monkeypatch, tmp_path: Path
) -> None:
    """train.log stays exactly the trainer's own stdout while run.log adds the pipeline stream."""
    config = _training_config(tmp_path)
    output = config["output"]
    _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    assert (output / "train.log").read_text(encoding="utf-8") == _TRAINER_LINE + "\n"
    lines = (output / "run.log").read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if _TRAINER_LINE in line]
    assert [line for line in lines if "train 1/2" in line], "no progress line for the train report"
    assert [line for line in lines if "val_loss" in line and "1.25" in line]
    assert [line for line in lines if "hybrid.q" in line], "resolved targets must be logged"


def test_stage_train_events_are_ordered_and_carry_the_run_name(
    monkeypatch, tmp_path: Path
) -> None:
    """R26(e): events.jsonl is the machine record a lift request cites."""
    config = _training_config(tmp_path)
    config["config_path"] = str(tmp_path / "arm.yaml")
    manifest = tmp_path / "data" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"rows": 1}', encoding="utf-8")
    output = config["output"]
    _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = [event["kind"] for event in events]
    assert {event["run"] for event in events} == {"train"}
    assert kinds[0] == "start" and kinds[-1] == "end"
    assert kinds.index("info") < kinds.index("progress") < kinds.index("metric")
    progress = next(event for event in events if event["kind"] == "progress")
    assert progress["fraction"] == 0.5 and progress["eta_seconds"] is not None
    identity = events[0]["fields"]
    assert identity["model"] == "registry-name" and identity["hf_id"] == "registry/hf-id"
    assert identity["config"] == str(tmp_path / "arm.yaml")
    assert identity["data_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", identity["git_commit"])
    assert events[-1]["fields"]["verdict"] == "healthy"


def test_stage_train_records_a_healthy_verdict_and_copies_it_into_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    """R26(f): the provenance copy of the health record must be the final one."""
    config = _training_config(tmp_path)
    output = config["output"]
    record = _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "healthy" and health["status"] == "ok"
    assert health["flags"] == []
    (extra,) = record.provenance
    assert extra["stage"] == "train"
    assert extra["health"]["verdict"] == health["verdict"]
    assert extra["health"]["thresholds"] == health["thresholds"]
    assert extra["health_thresholds"] == health["thresholds"]


# --------------------------------------------- the training-time backbone patches (R32, issue #50)


class _FlagModel:
    """The only thing the validation wrapper touches on a model: its training flag."""

    def __init__(self) -> None:
        self.training = True

    def train(self, mode: bool = True) -> "_FlagModel":
        self.training = bool(mode)
        return self

    def eval(self) -> "_FlagModel":
        return self.train(False)


def test_the_validation_wrapper_runs_evaluate_in_eval_mode_and_restores_the_flag() -> None:
    """R32 efficiency item 1: validation takes the inference kernel, training mode resumes."""
    trainer = importlib.import_module("mlx_lm.tuner.trainer")
    library = trainer.evaluate
    model = _FlagModel()
    seen = []

    def recording(*args, **kwargs):
        seen.append((kwargs["model"] if "model" in kwargs else args[0]).training)
        return 1.25

    trainer.evaluate = recording
    try:
        with cli._kernel_path_validation():
            assert trainer.evaluate is not recording
            # ``trainer.train`` calls evaluate by keyword (trainer.py:290); positional works too.
            assert trainer.evaluate(model=model, dataset=[]) == 1.25
            assert model.training is True
            assert trainer.evaluate(model, []) == 1.25
            assert model.training is True
        assert trainer.evaluate is recording
    finally:
        trainer.evaluate = library

    # The model was in evaluation mode for the pass itself, both call styles.
    assert seen == [False, False]
    assert model.training is True


def test_the_validation_wrapper_restores_the_flag_when_evaluate_raises() -> None:
    """A validation pass that dies must not leave the model in eval mode for the next step."""
    trainer = importlib.import_module("mlx_lm.tuner.trainer")
    library = trainer.evaluate
    model = _FlagModel()

    def exploding(**kwargs):
        assert kwargs["model"].training is False
        raise RuntimeError("validation blew up")

    trainer.evaluate = exploding
    try:
        with cli._kernel_path_validation():
            with pytest.raises(RuntimeError, match="validation blew up"):
                trainer.evaluate(model=model, dataset=[])
    finally:
        trainer.evaluate = library

    assert model.training is True
    assert trainer.evaluate is library


def test_the_chunkwise_backbone_names_the_installer_stage_two_has_not_landed(
    monkeypatch,
) -> None:
    """An arm configuring a form this tree cannot install must be told which one is missing.

    ``install_chunkwise_gated_delta`` lands with R32 stage 2, a separate slice.  Until then the
    attribute lookup would fail with a bare ``AttributeError`` at train start, hours into a
    queued run, saying nothing about which slice supplies it.
    """
    installed: list[int] = []

    @contextlib.contextmanager
    def install_chunked_gated_delta(chunk: int):
        installed.append(chunk)
        yield

    # The training package exactly as this commit ships it: the stage-1 installer alone.
    monkeypatch.setattr(
        "local_llm_lab.training",
        SimpleNamespace(install_chunked_gated_delta=install_chunked_gated_delta),
        raising=False,
    )

    with (
        pytest.raises(ValueError, match="install_chunkwise_gated_delta") as error,
        cli._training_backbone(64, mode=cli.GATED_DELTA_CHUNKWISE),
    ):
        raise AssertionError("the guard must refuse before the trainer call")
    assert "R32 stage 2" in str(error.value)
    # No silent fallback: an arm that asked for the chunkwise form must not train under the
    # chunked one, so nothing may have been installed on the way out.
    assert installed == []

    # The mode that IS in the tree still installs, so the guard cannot be satisfied by
    # breaking the working path.
    with cli._training_backbone(64, mode=cli.GATED_DELTA_CHECKPOINTED):
        pass
    assert installed == [64]


def test_stage_train_installs_the_chunked_recurrence_and_records_the_chunk(
    monkeypatch, tmp_path: Path
) -> None:
    """R32(a),(c): the arm's chunk installs the recurrence and is recorded where it is used."""
    training = importlib.import_module("local_llm_lab.training")
    trainer_module = importlib.import_module("mlx_lm.tuner.trainer")
    library_evaluate = trainer_module.evaluate
    config = _training_config(tmp_path)
    config["train"]["gated_delta_chunk"] = 64
    output = config["output"]
    events: list[object] = []

    @contextlib.contextmanager
    def fake_installer(chunk):
        events.append(("install", chunk))
        try:
            yield
        finally:
            events.append("restore")

    def recording_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
        events.append(("train", args.gated_delta_chunk, trainer_module.evaluate is not library_evaluate))
        _healthy_trainer(args, model, train_set, valid_set, training_callback)

    monkeypatch.setattr(training, "install_chunked_gated_delta", fake_installer)
    record = _patch_stage_train(monkeypatch, recording_trainer)

    cli.stage_train(config, iters=2)

    # Installed before the trainer runs, restored after it returns; validation wrapped too.
    assert events == [("install", 64), ("train", 64, True), "restore"]
    assert trainer_module.evaluate is library_evaluate
    lora = yaml.safe_load((output / "lora.yaml").read_text(encoding="utf-8"))
    assert lora["gated_delta_chunk"] == 64
    (extra,) = record.provenance
    assert extra["training_config"]["gated_delta_chunk"] == 64


def test_stage_train_installs_no_recurrence_patch_for_a_dense_arm(
    monkeypatch, tmp_path: Path
) -> None:
    """A backbone with no gated-delta recurrence must be left exactly as the library ships it."""
    training = importlib.import_module("local_llm_lab.training")
    config = _training_config(tmp_path)
    output = config["output"]
    assert "gated_delta_chunk" not in config["train"]

    def forbidden(chunk):
        pytest.fail(f"a dense arm must not install the chunked recurrence (chunk={chunk})")

    monkeypatch.setattr(training, "install_chunked_gated_delta", forbidden)
    record = _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    lora = yaml.safe_load((output / "lora.yaml").read_text(encoding="utf-8"))
    assert "gated_delta_chunk" not in lora
    (extra,) = record.provenance
    assert "gated_delta_chunk" not in extra["training_config"]


@contextlib.contextmanager
def _null_installer(chunk: int):
    """An installer that patches nothing: the recurrence form is not what these tests check."""
    yield


def test_stage_train_records_the_recurrence_a_chunkwise_run_actually_took(
    monkeypatch, tmp_path: Path
) -> None:
    """K6(a): the form the run TOOK must reach health.json, provenance and the run log.

    ``lora.gated_delta_mode`` records the form the arm *configured*. A run that fell back to
    stage 1's checkpointed loop has stage 1's memory and step time, so a record that says only
    ``chunkwise`` would send the next attempt into the same out-of-memory wall with nothing
    saying why. The counts are the correction, and a non-empty one is a warning.
    """
    training = importlib.import_module("local_llm_lab.training")
    config = _training_config(tmp_path)
    config["train"]["gated_delta_chunk"] = 64
    config["train"]["gated_delta_mode"] = cli.GATED_DELTA_CHUNKWISE
    output = config["output"]

    monkeypatch.setattr(training, "install_chunkwise_gated_delta", _null_installer)
    # The counts a fallen-back chunkwise run would leave behind, read through the same
    # function the backbone reads, so the test pins the wiring and not a private detail.
    monkeypatch.setattr(training, "fallback_counts", lambda: {"mask": 3})
    record = _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["gated_delta_fallbacks"] == {"mask": 3}
    (extra,) = record.provenance
    assert extra["training_config"]["gated_delta_mode"] == cli.GATED_DELTA_CHUNKWISE
    assert extra["health"]["gated_delta_fallbacks"] == {"mask": 3}

    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    warnings = [event for event in events if event["kind"] == "warning"]
    assert warnings, "a chunkwise arm that fell back must warn"
    # The warning names which fallback fired and how many times, not merely that one did.
    assert any(event["fields"] == {"fallback_mask": 3} for event in warnings), warnings
    assert "fallback_mask=3" in (output / "run.log").read_text(encoding="utf-8")


def test_stage_train_records_no_fallback_count_it_did_not_earn(
    monkeypatch, tmp_path: Path
) -> None:
    """K6(b), the stale-global trap: a checkpointed or dense arm may not report counts at all.

    ``fallback_counts()`` reads a module-level dict that only the chunkwise installer resets.
    On a checkpointed arm it therefore still holds whatever an earlier chunkwise run in the
    same process left there — a number that looks like evidence and is not. The guard is that
    the counts function is never called on those arms, so a stale value cannot be recorded
    even in principle, and the record says explicitly that it does not apply.
    """
    training = importlib.import_module("local_llm_lab.training")

    def forbidden() -> dict[str, int]:
        pytest.fail("the fallback counter was read on an arm that never installed it")

    monkeypatch.setattr(training, "fallback_counts", forbidden)
    monkeypatch.setattr(training, "install_chunked_gated_delta", _null_installer)

    checkpointed = _training_config(tmp_path / "checkpointed")
    checkpointed["train"]["gated_delta_chunk"] = 64
    _patch_stage_train(monkeypatch, _healthy_trainer)
    cli.stage_train(checkpointed, iters=2)

    dense = _training_config(tmp_path / "dense")
    assert "gated_delta_chunk" not in dense["train"]
    _patch_stage_train(monkeypatch, _healthy_trainer)
    cli.stage_train(dense, iters=2)

    for run, expected in (
        (checkpointed, "not applicable: checkpointed"),
        (dense, "not applicable: no recurrence installed"),
    ):
        health = json.loads((run["output"] / "health.json").read_text(encoding="utf-8"))
        assert health["gated_delta_fallbacks"] == expected, run["output"]


def test_stage_train_records_the_fallbacks_a_crashed_chunkwise_run_took(
    monkeypatch, tmp_path: Path
) -> None:
    """The record is written on the error path too, where the diagnosis is worth the most.

    ``health.json`` and the run log are both written from the stage's ``finally``; a run that
    died inside the trainer is exactly the run whose next attempt needs to know which
    recurrence it was running when it died, and it never reaches the success path at all.
    """
    training = importlib.import_module("local_llm_lab.training")
    config = _training_config(tmp_path)
    config["train"]["gated_delta_chunk"] = 64
    config["train"]["gated_delta_mode"] = cli.GATED_DELTA_CHUNKWISE
    output = config["output"]

    def exploding_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
        training_callback.on_train_loss_report(_report(1, 1.0))
        raise RuntimeError("out of memory")

    monkeypatch.setattr(training, "install_chunkwise_gated_delta", _null_installer)
    monkeypatch.setattr(training, "fallback_counts", lambda: {"vectorised_gating": 1})
    _patch_stage_train(monkeypatch, exploding_trainer)

    with pytest.raises(RuntimeError, match="out of memory"):
        cli.stage_train(config, iters=2)

    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["status"] == "error"
    assert health["gated_delta_fallbacks"] == {"vectorised_gating": 1}
    assert "vectorised_gating=1" in (output / "run.log").read_text(encoding="utf-8")


def test_stage_train_flags_a_run_that_stops_short_or_saves_no_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """R26(c): a run that ends early is not a healthy run even though nothing raised."""
    short = _training_config(tmp_path / "short")
    _patch_stage_train(monkeypatch, _short_trainer)
    cli.stage_train(short, iters=2)
    stopped = json.loads((short["output"] / "health.json").read_text(encoding="utf-8"))

    unsaved_config = _training_config(tmp_path / "unsaved")
    _patch_stage_train(monkeypatch, _no_checkpoint_trainer)
    cli.stage_train(unsaved_config, iters=2)
    unsaved = json.loads((unsaved_config["output"] / "health.json").read_text(encoding="utf-8"))

    assert stopped["verdict"] == "incomplete" and stopped["status"] == "ok"
    assert unsaved["verdict"] == "incomplete" and unsaved["status"] == "ok"
    assert "incomplete_run" in {flag["flag"] for flag in stopped["flags"]}
    assert "incomplete_run" in {flag["flag"] for flag in unsaved["flags"]}


class _WordTokenizer:
    """One token per whitespace-separated word: a row's length is readable in the source."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [1] * len(text.split())


def _write_rendered_splits(data_dir: Path, *, prompt_words: dict[str, int]) -> None:
    """Write one rendered row per split, the prompt sized in words the fake tokenizer counts."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, words in prompt_words.items():
        (data_dir / f"{split}.jsonl").write_text(
            json.dumps({"prompt": " ".join(["w"] * words), "completion": "c"}) + "\n",
            encoding="utf-8",
        )
    # The commit-point stamp `load_rendered_splits` now requires (ruling on #73). Only its
    # presence is read; the row lengths beside it are what these tests are about.
    (data_dir / "manifest.json").write_text("{}\n", encoding="utf-8")


def test_stage_train_loads_only_the_splits_it_trains_on(monkeypatch, tmp_path: Path) -> None:
    """An over-long TEST row must not stop a run that never trains on the test split.

    This is the B4 start failure: ``lora["test"]`` is False, yet the stage loaded and
    tokenized test.jsonl and died on row 37 before iteration one.
    """
    config = _training_config(tmp_path)
    config["train"]["max_seq_length"] = 4
    output = config["output"]
    _write_rendered_splits(
        config["data"], prompt_words={"train": 1, "valid": 1, "test": 9}
    )
    _patch_stage_train(monkeypatch, _healthy_trainer, tokenizer=_WordTokenizer(), splits="keep")

    cli.stage_train(config, iters=2)

    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "healthy" and health["status"] == "ok"
    # The splits line still reports all three names: the one that was not loaded reads None
    # in events.jsonl, and RunLog renders that as "-" on the console side.
    splits = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["message"] == "splits"
    ]
    assert [event["fields"] for event in splits] == [{"train": 1, "valid": 1, "test": None}]
    lines = (output / "run.log").read_text(encoding="utf-8").splitlines()
    ((splits_line,)) = [line for line in lines if "splits" in line]
    assert "train=1 valid=1 test=-" in splits_line


def test_stage_train_still_fails_closed_on_an_over_long_training_row(
    monkeypatch, tmp_path: Path
) -> None:
    """A row the trainer would actually read is still refused, with the same message."""
    config = _training_config(tmp_path)
    config["train"]["max_seq_length"] = 4
    _write_rendered_splits(
        config["data"], prompt_words={"train": 9, "valid": 1, "test": 1}
    )
    _patch_stage_train(monkeypatch, _healthy_trainer, tokenizer=_WordTokenizer(), splits="keep")

    with pytest.raises(ValueError, match="prompt leaves no room for a completion token"):
        cli.stage_train(config, iters=2)


def _exploding_loader(path, tokenizer, **kwargs):
    """The loader failure that started this: it raises before the trainer is ever reached."""
    raise ValueError("row 37: prompt leaves no room for a completion token")


def _raising_trainer(args, model, train_set, valid_set, training_callback=None) -> None:
    """A trainer that dies partway through, after one report and with no checkpoint saved."""
    training_callback.on_train_loss_report(_report(1, 1.0))
    raise RuntimeError("the training loop died")


def test_stage_train_records_an_incomplete_run_when_the_loader_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """R26(c): a run that never reached iteration one can never read healthy."""
    config = _training_config(tmp_path)
    output = config["output"]
    record = _patch_stage_train(monkeypatch, _healthy_trainer, splits="keep")
    monkeypatch.setattr(cli, "load_rendered_splits", _exploding_loader, raising=False)

    with pytest.raises(ValueError, match="row 37"):
        cli.stage_train(config, iters=2)

    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["status"] == "error" and health["verdict"] == "incomplete"
    assert health["iterations_done"] == 0 and health["final_checkpoint"] is False
    incomplete = [flag for flag in health["flags"] if flag["flag"] == "incomplete_run"]
    assert len(incomplete) == 1 and incomplete[0]["severity"] == "fatal"
    assert incomplete[0]["detail"] == {
        "iterations_done": 0,
        "iters_planned": 2,
        "final_checkpoint": False,
    }
    assert record.provenance == []


def test_stage_train_records_an_incomplete_run_when_the_trainer_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """The same rule must produce the same record whichever call raised."""
    config = _training_config(tmp_path)
    output = config["output"]
    _patch_stage_train(monkeypatch, _raising_trainer)

    with pytest.raises(RuntimeError, match="the training loop died"):
        cli.stage_train(config, iters=2)

    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["status"] == "error" and health["verdict"] == "incomplete"
    assert health["iterations_done"] == 1
    assert "incomplete_run" in {flag["flag"] for flag in health["flags"]}


def test_stage_train_aborts_on_a_non_finite_loss_without_writing_provenance(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """R26(c): an aborted run's adapters are never eligible, so no provenance is written."""
    config = _training_config(tmp_path)
    output = config["output"]
    record = _patch_stage_train(monkeypatch, _nan_trainer)

    with pytest.raises(SystemExit) as error:
        cli.stage_train(config, iters=2)

    captured = capsys.readouterr()
    assert [
        line
        for line in captured.err.splitlines()
        if "training aborted" in line and "non_finite_loss" in line
    ], "the fatal flag is raised, not returned, so the stage must report it itself"
    assert "non_finite_loss" in str(error.value)
    assert str(output / "health.json") in str(error.value)
    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "aborted" and health["status"] == "aborted"
    # The abort path also runs the finish rule, and non_finite_loss outranks what it records.
    assert "incomplete_run" in {flag["flag"] for flag in health["flags"]}
    assert record.provenance == []
    assert not (output / "provenance.json").exists()


def test_stage_train_warns_on_a_loss_spike_and_records_it(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A warning is worth nothing unless the operator sees it while the run is going."""
    config = _training_config(tmp_path)
    output = config["output"]
    _patch_stage_train(monkeypatch, _spiking_trainer)

    cli.stage_train(config, iters=6)

    captured = capsys.readouterr()
    warnings = [line for line in captured.err.splitlines() if "health flag" in line]
    assert [line for line in warnings if "loss_spike" in line]
    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "warnings"
    spikes = [flag for flag in health["flags"] if flag["flag"] == "loss_spike"]
    assert len(spikes) == 1 and spikes[0]["iteration"] == 6


def test_stage_train_health_thresholds_come_from_the_arm_config(
    monkeypatch, tmp_path: Path
) -> None:
    """Changing a threshold is a config change, visible in health.json and in provenance."""
    config = _training_config(tmp_path)
    config["train"]["health"] = {"loss_spike_factor": 3.0}
    output = config["output"]
    record = _patch_stage_train(monkeypatch, _healthy_trainer)

    cli.stage_train(config, iters=2)

    health = json.loads((output / "health.json").read_text(encoding="utf-8"))
    assert health["thresholds"]["loss_spike_factor"] == 3.0
    assert health["thresholds"]["memory_budget_gib"] == _training_model_spec().memory_budget_gib
    assert record.provenance[0]["health_thresholds"]["loss_spike_factor"] == 3.0


def test_stage_train_rejects_an_unknown_health_key_before_loading_the_model(
    monkeypatch, tmp_path: Path
) -> None:
    """A typo in train.health must cost nothing: no weights, no output directory churn."""
    config = _training_config(tmp_path)
    config["train"]["health"] = {"loss_spike_factor": 3.0, "loss_spike_facter": 3.0}
    record = _patch_stage_train(monkeypatch, _healthy_trainer)

    with pytest.raises(ValueError, match="loss_spike_facter"):
        cli.stage_train(config, iters=2)

    assert record.loads == [], "the base model must not load for a rejected config"


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
    """Rollout provenance and its guard-arming manifest land after the sampling succeeds."""
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

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
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
    manifest_path = tmp_path / "data" / "rollouts" / "iter1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "stage": "rollout",
        "generator_version": cli.GENERATOR_VERSION,
        "model": "fake-model",
        "split": "iter1",
        "seed": 17,
        "adapter": str(adapter),
        "summary": summary,
    }
    from local_llm_lab.pipeline.data import DatasetWriteGuardError

    with pytest.raises(DatasetWriteGuardError, match="manifest.json"):
        cli.stage_rollout(config, adapter, "iter1", limit=None, samples=None, quiet=True)


def test_branch_command_stamps_its_output_so_the_guard_is_live(
    monkeypatch, tmp_path: Path
) -> None:
    """Branch mining writes a manifest next to its summary; a rerun then refuses."""
    from local_llm_lab.pipeline.data import DatasetWriteGuardError

    config = {
        "output": tmp_path / "output",
        "model": "fake-model",
        "seed": 17,
        "keep_last": 2,
        "eval": {"max_steps": 2, "max_tokens": 3},
    }
    summary = {"pairs": 5}
    runs = []

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli.Transcript, "start_run", lambda path: None)
    monkeypatch.setattr(
        cli, "run_branch_mining", lambda **kwargs: runs.append(kwargs) or summary
    )
    monkeypatch.setattr(sys, "argv", ["agent-pipeline", "branch", "--split", "pref1"])

    cli.main()

    manifest_path = tmp_path / "data" / "preferences" / "pref1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "stage": "branch",
        "generator_version": cli.GENERATOR_VERSION,
        "model": "fake-model",
        "split": "pref1",
        "seed": 17,
        "adapter": str(config["output"] / "best-adapter"),
        "summary": summary,
    }
    assert len(runs) == 1
    with pytest.raises(DatasetWriteGuardError, match="manifest.json"):
        cli.main()
    assert len(runs) == 1, "the guard must fire before mining runs again"


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


def test_stage_train_ignores_a_stale_checkpoint_from_an_earlier_run(monkeypatch, tmp_path: Path) -> None:
    """Chief's condition on #37: a checkpoint older than this run's start is not this run's."""
    import os
    import time

    config = _training_config(tmp_path / "stale")
    adapters = config["output"] / "adapters"
    adapters.mkdir(parents=True)
    stale = adapters / "adapters.safetensors"
    stale.write_bytes(b"left by an earlier run")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    _patch_stage_train(monkeypatch, _no_checkpoint_trainer)

    cli.stage_train(config, iters=2)

    health = json.loads((config["output"] / "health.json").read_text(encoding="utf-8"))
    assert health["verdict"] == "incomplete"
    assert health["final_checkpoint"] is False
    assert "incomplete_run" in {flag["flag"] for flag in health["flags"]}
