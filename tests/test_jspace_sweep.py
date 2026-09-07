"""EXP-001 (issue #54): the installed J-space sweep CLI and the ``jsweep`` split.

Fakes only (R10): no checkpoint, no tokenizer from the Hub, no ``ArchitectureView`` over a
real model. The registry ``ModelSpec`` and ``protocol.build_prompt`` are the library-facing
seams and are driven for real (R31); weights and compute are the parts that are faked.
"""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from local_llm_lab.pipeline import tasks as task_module
from local_llm_lab.pipeline.cli import dataset_splits, load_config
from local_llm_lab.pipeline.tasks import FAMILIES, family_balanced_tasks, make_tasks

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"
_REFERENCE_CONFIG = _CONFIG_DIR / "agent_v2c.yaml"
# Mirrors tests/test_tasks.py: pinned so a new run config cannot join the repository without
# being swept by the retroactive disjointness check below.
_CONFIGS_WITH_SPLIT_TABLES = (
    "agent_v2.yaml",
    "agent_v2b.yaml",
    "agent_v2b_qwen35_4b.yaml",
    "agent_v2c.yaml",
    "agent_v2d.yaml",
    "agent_v2d_qwen35_4b.yaml",
    "agent_v2e.yaml",
    "agent_v2e_qwen35_4b.yaml",
)


def _config_paths():
    return sorted(_CONFIG_DIR.glob("*.yaml"))


def _split_table(config: dict) -> dict:
    return config.get("tasks") or config.get("splits") or {}


def _screen_splits(config: dict) -> list[dict]:
    screen = (config.get("select") or {}).get("screen") or []
    return [entry for entry in screen if entry.get("split") not in _split_table(config)]


def _config_tasks(config: dict) -> list:
    """Regenerate every task a config's splits contain, deterministically and without files."""
    seed = config["seed"]
    generated = []
    for name, spec in dataset_splits(config).items():
        kwargs = {}
        if spec.difficulty is not None:
            kwargs["difficulty"] = spec.difficulty
        if spec.perturb is not None:
            kwargs["perturb"] = spec.perturb
        generated.extend(make_tasks(name, spec.count, seed, **kwargs))
    for entry in _screen_splits(config):
        generated.extend(
            family_balanced_tasks(
                entry["split"],
                difficulty=entry["difficulty"],
                per_family=entry["per_family"],
                seed=seed,
            )
        )
    return generated


# --------------------------------------------------------------------- the installed entry


def test_jspace_sweep_is_an_installed_console_script() -> None:
    """EXP-001 §4: ``research/`` is outside the wheel, so the sweep must live in the package."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["agent-v2-jspace-sweep"] == "local_llm_lab.probes.jspace_sweep:main"
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/local_llm_lab"
    ]

    module = importlib.import_module("local_llm_lab.probes.jspace_sweep")
    assert callable(module.main)


def test_research_wrapper_delegates_to_the_installed_module() -> None:
    """The research script keeps a thin wrapper; the body moved into the package."""
    source = (_ROOT / "research" / "jspace_sweep.py").read_text(encoding="utf-8")

    assert "local_llm_lab.probes.jspace_sweep" in source
    assert "from mlx_lm import load" not in source


# ------------------------------------------------------------------------ the jsweep split


def test_jsweep_split_is_declared_with_an_explicit_difficulty_and_perturbation() -> None:
    """The sweep's split stops depending on ``make_tasks``' unknown-split defaults."""
    assert task_module.JSPACE_SPLITS == (("jsweep", 1, False),)
    assert task_module.JSPACE_SPLIT_NAMES == ("jsweep",)
    assert task_module.JSPACE_SPLIT_LIMIT == 720
    with pytest.raises(ValueError):
        task_module.jspace_split("train")

    seed = load_config(_REFERENCE_CONFIG)["seed"]
    count = len(FAMILIES) * 4
    assert task_module.make_jspace_tasks("jsweep", count, seed) == make_tasks(
        "jsweep", count, seed, perturb=False, difficulty=1
    )


def test_declaring_jsweeps_difficulty_changes_no_task_the_sweep_reads() -> None:
    """Additive: the ledger cohort is identical to what the 0/1 alternation already produced."""
    seed = load_config(_REFERENCE_CONFIG)["seed"]
    count = len(FAMILIES) * 4
    fallback = [
        task
        for task in make_tasks("jsweep", count, seed, perturb=False)
        if task.family == "ledger_reconcile"
    ]
    declared = [
        task
        for task in task_module.make_jspace_tasks("jsweep", count, seed)
        if task.family == "ledger_reconcile"
    ]

    assert fallback and fallback == declared


def test_jsweep_tasks_are_content_disjoint_from_every_configured_split() -> None:
    """R28 applied retroactively to a published split (N2): assert, and report a failure.

    ``jsweep`` is not a new name -- the recorded 3B sweep already used it -- so this is a
    retroactive disjointness check on a published result rather than a precaution on a fresh
    split. Per amendment N2 a failure here is reported, not silently relaxed: the assertion
    stays strict.
    """
    seed = load_config(_REFERENCE_CONFIG)["seed"]
    sweep = {}
    for task in task_module.make_jspace_tasks("jsweep", len(FAMILIES) * 10, seed):
        sweep.setdefault(task_module.task_fingerprint(task), []).append(task.task_id)

    checked = []
    for path in _config_paths():
        config = load_config(path)
        table = _split_table(config)
        if not table:
            continue
        assert "jsweep" not in table, f"{path.name}: training table declares the sweep split"
        assert config["seed"] == seed
        collisions = [
            (task.task_id, sweep[fingerprint])
            for task in _config_tasks(config)
            if (fingerprint := task_module.task_fingerprint(task)) in sweep
        ]
        assert not collisions, (
            f"{path.name}: jsweep rows duplicate training content: {collisions[:5]}"
        )
        checked.append(path.name)
    assert tuple(checked) == _CONFIGS_WITH_SPLIT_TABLES


# ------------------------------------------------------------------------------- the sweep


class _SweepView:
    """A tail that carries a source position forward; no model and no weights."""

    num_layers = 4
    vocabulary = 64

    def residuals(self, ids, layers):
        values = mx.array(ids).astype(mx.float32)[None, :, None]
        return {layer: values + float(layer) for layer in layers}

    def tail(self, layer):
        del layer
        return lambda value: mx.cumsum(value.astype(mx.float32), axis=1)

    def final_norm(self, value):
        return value.astype(mx.float32)

    def unembed(self, value):
        return value.astype(mx.float32) * mx.ones((self.vocabulary,), dtype=mx.float32)

    def layer_kind(self, index):
        return "linear_attention" if (index + 1) % 4 else "attention"


class _SweepTokenizer:
    """One id per character, and a chat template that records the kwargs it was given."""

    bos_token = None

    def __init__(self) -> None:
        self.template_calls: list[dict] = []

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % 64 for character in text]

    def decode(self, ids):
        return "".join(chr(65 + int(i) % 26) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        del tokenize
        self.template_calls.append(dict(kwargs))
        body = "\n".join(str(message["content"]) for message in messages)
        if not add_generation_prompt:
            return body
        from local_llm_lab.models import load_model_spec
        from local_llm_lab.pipeline.protocol import generation_suffix

        return body + "\n" + generation_suffix(load_model_spec("qwen35-4b"))


def _write_preflight(tmp_path, spec):
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


def _run_sweep_cli(monkeypatch, tmp_path, extra=(), *, layers=("--layers", "2,3"), view=None):
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, preflight
    from local_llm_lab.probes import guard, jspace_sweep, policies

    spec = models.load_model_spec("qwen35-4b")
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", _write_preflight(tmp_path, spec))
    monkeypatch.setattr(preflight, "require_preflight", lambda *_a, **_k: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_a: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda *_a: None)
    tokenizer = _SweepTokenizer()
    loaded = _SweepView() if view is None else view
    monkeypatch.setattr(
        evaluate,
        "load_policy",
        lambda *_a, **_k: (object(), tokenizer, loaded, None),
    )
    output = tmp_path / "sweep"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-jspace-sweep",
            "--model",
            "qwen35-4b",
            "--count",
            str(len(FAMILIES) * 10),
            *layers,
            "--corpus-size",
            "2",
            "--corpus-length",
            "24",
            "--skip-preflight-check",
            "--output",
            str(output),
            *extra,
        ],
    )
    jspace_sweep.main()
    payload = json.loads((output / "sweep.json").read_text(encoding="utf-8"))
    return payload, output, tokenizer, spec


def test_sweep_renders_every_prompt_through_build_prompt_with_the_registry_kwargs(
    monkeypatch, tmp_path
) -> None:
    """B2: a bare ``apply_chat_template`` would probe inside an open think block."""
    _payload, _output, tokenizer, spec = _run_sweep_cli(monkeypatch, tmp_path)

    assert tokenizer.template_calls
    assert all(call == dict(spec.chat.template_kwargs) for call in tokenizer.template_calls)


def test_sweep_artifact_records_r34_conformance_and_r35_comparability(
    monkeypatch, tmp_path
) -> None:
    payload, output, _tokenizer, spec = _run_sweep_cli(monkeypatch, tmp_path)

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
    assert conformance["jvp_method"] == "finite_difference"
    assert conformance["jvp_method_source"] == "preflight"

    comparability = payload["comparability"]
    for key in (
        "policy",
        "adapter",
        "derivative_method",
        "prompt_rendering",
        "estimator_variant",
        "layer_selection",
        "generator_version",
        "data_seed",
        "fp32_manual_vs_native",
    ):
        assert key in comparability, key
    assert comparability["prompt_rendering"]["template_kwargs"] == dict(spec.chat.template_kwargs)
    assert comparability["fp32_manual_vs_native"] == {
        "frobenius_relative": 0.004,
        "max_abs": 0.02,
    }
    assert set(payload["results"]) >= {f"jlens_L2_{name}" for name in ("self", "future", "all")}
    assert (output / "sweep.md").is_file()


def test_sweep_writes_a_run_log_with_one_event_per_probe_point(monkeypatch, tmp_path) -> None:
    """R26 (a) and (g): the run is never silent for more than one outer unit of work."""
    payload, output, _tokenizer, _spec = _run_sweep_cli(monkeypatch, tmp_path)

    assert (output / "run.log").is_file()
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    points = [event for event in events if event.get("label") == "probe point"]
    assert [event["step"] for event in points] == list(range(1, payload["cases"] + 1))
    assert events[0]["kind"] == "start"
    assert events[-1]["kind"] == "end"


def test_sweep_refuses_an_unresolvable_jvp_method(monkeypatch, tmp_path) -> None:
    """Fail closed: no flag and no preflight record is a named error, not ``forward``."""
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, preflight
    from local_llm_lab.probes import guard, jspace_sweep

    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", tmp_path / "empty")
    monkeypatch.setattr(
        guard, "require_idle_gpu", lambda *_a: pytest.fail("reached the GPU guard")
    )
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_a, **_k: pytest.fail("reached the model loader")
    )
    assert models.load_model_spec("qwen35-4b").name == "qwen35-4b"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-jspace-sweep",
            "--model",
            "qwen35-4b",
            "--output",
            str(tmp_path / "sweep"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        jspace_sweep.main()

    assert raised.value.code == 2


def test_sign_test_and_holm_are_exact_and_ordered() -> None:
    from local_llm_lab.probes import jspace_sweep

    assert jspace_sweep.sign_test(0, 0) == 1.0
    assert jspace_sweep.sign_test(5, 5) == pytest.approx(2 / 32)
    adjusted = jspace_sweep.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.08)
    assert adjusted["c"] == pytest.approx(0.5)


def test_selection_helpers_keep_the_recorded_sweeps_semantics() -> None:
    from local_llm_lab.probes import jspace_sweep

    assert jspace_sweep.suffix_of("lab/x/ledger/invoice-2-537.txt") == "537"
    assert jspace_sweep.note_prefix("Reading invoice-2-537.txt now", "invoice-2-537.txt") == (
        "Reading invoice-2-"
    )
    assert jspace_sweep.note_prefix("nothing here", "invoice-2-537.txt") is None
    stripped = jspace_sweep.strip_pending(
        [
            {"role": "assistant", "content": "note; pending: a, b\nrest"},
            {"role": "user", "content": "keep; pending: a"},
        ]
    )
    assert stripped[0]["content"] == "note.\nrest"
    assert stripped[1]["content"] == "keep; pending: a"


def test_case_selection_drops_leaked_and_degenerate_pairs(monkeypatch) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.probes import jspace_sweep

    spec = load_model_spec("qwen35-4b")
    tokenizer = _SweepTokenizer()
    seed = load_config(_REFERENCE_CONFIG)["seed"]
    tasks = [
        task
        for task in task_module.make_jspace_tasks("jsweep", len(FAMILIES) * 10, seed)
        if task.family == "ledger_reconcile"
    ]

    cases = jspace_sweep.select_cases(tasks, tokenizer, spec=spec, probe_step=3)

    assert cases
    for case in cases:
        assert case["true"] != case["false"]
        assert case["true"] not in case["prompt"]
        assert case["prompt"].endswith(case["prefix"])


def test_source_positions_and_corpus_length_reach_the_estimator(monkeypatch, tmp_path) -> None:
    payload, _output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, extra=("--source-positions", "0.5")
    )

    assert payload["conformance"]["source_positions"] == [
        {"token": "0.5", "kind": "fraction", "value": 0.5}
    ]
    assert payload["conformance"]["window"]["median_future_window"] >= 1
    assert payload["conformance"]["corpus_length"]["min_tokens"] >= 24


def test_empty_future_window_is_reported_rather_than_read_as_zero(monkeypatch, tmp_path) -> None:
    """A1: the sweep fails loudly when the source leaves no window, never silently zero."""
    from local_llm_lab.probes import jspace_sweep

    with pytest.raises(SystemExit) as raised:
        _run_sweep_cli(monkeypatch, tmp_path, extra=("--source-positions", "-1"))

    assert raised.value.code == 2
    assert jspace_sweep is not None


def test_the_comparability_block_records_the_row_window_and_the_no_rewindow_rule(
    monkeypatch, tmp_path
) -> None:
    """C3 (issue #62): ``keep_last`` alone does not let a reader reconstruct the context.

    The rows reach ``select_cases`` already windowed by ``build_rows``; ``build_prompt`` is
    then handed the row's own tool count precisely so that it re-windows nothing. Recording
    only the number passed to ``build_prompt`` hides the window the context actually carries,
    which is the coordinate R35 asks for.
    """
    from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST

    payload, _output, _tokenizer, _spec = _run_sweep_cli(monkeypatch, tmp_path)

    rendering = payload["comparability"]["prompt_rendering"]
    assert rendering["row_keep_last"] == DEFAULT_KEEP_LAST
    assert rendering["rewindowed"] is False
    assert "build_rows" in rendering["windowing_rule"]


# ------------------------------------- C1 (issue #61): one J-lens map per distinct prompt


class _ScoringSweepView(_SweepView):
    """``_SweepView`` with a token-dependent unembedding.

    The base fake gives every token the same logit, so its softmax is uniform and a cache that
    served the *wrong* prompt's maps would still produce identical numbers. Here each token has
    its own scale, so the identity assertion below can actually fail.
    """

    def unembed(self, value):
        return value.astype(mx.float32) * mx.arange(1, self.vocabulary + 1, dtype=mx.float32)


def _cache_probe_cases(count: int = 4) -> list[dict]:
    """Probe points in the shape ``select_cases`` returns: distinct prompts, distinct pairs."""
    return [
        {
            "task_id": f"ledger-{index}",
            "difficulty": index % 2,
            "prompt": f"note {'ab' * (index + 1)} Reading invoice-{index}-",
            "prefix": f"Reading invoice-{index}-",
            "true": f"t{index}",
            "false": f"f{index}",
            "true_token": index + 1,
            "false_token": index + count + 1,
        }
        for index in range(count)
    ]


def _sweep_counting_map_computations(monkeypatch, *, cache: bool):
    """Run ``run_sweep`` over the fakes with a spy on the J-lens map entry point."""
    from local_llm_lab.pipeline.jlens import (
        DEFAULT_SOURCE_POSITIONS,
        build_corpus,
        jlens_readouts,
        resolve_source_positions,
    )
    from local_llm_lab.probes import jspace_sweep

    computations: list[tuple[int, int]] = []

    def counted(view, layer, probe, corpus_ids, **kwargs):
        computations.append((layer, len(corpus_ids)))
        return jlens_readouts(view, layer, probe, corpus_ids, **kwargs)

    monkeypatch.setattr(jspace_sweep, "jlens_readouts", counted)
    tokenizer = _SweepTokenizer()
    payload = jspace_sweep.run_sweep(
        _ScoringSweepView(),
        tokenizer,
        _cache_probe_cases(),
        layers=[2, 3],
        corpus_ids=build_corpus(tokenizer, size=2, length=24),
        source_positions=resolve_source_positions(DEFAULT_SOURCE_POSITIONS),
        method="finite_difference",
        capture_dtype="float32",
        cache=cache,
    )
    return payload, len(computations)


def test_the_prompt_cache_halves_the_map_computations_and_changes_no_number(
    monkeypatch,
) -> None:
    """C1 (issue #61): the null for case *i* is the matched computation for case *i+1*.

    The rotation makes every prompt appear twice -- once as its own case's treatment, once as
    its predecessor's null -- so without a cache every J-lens map in the run is computed
    twice. Keyed on the prompt itself, not the case index: an index key would miss every hit
    (the same prompt sits at two indices) and a positional key would read another prompt's
    maps. Because the key is the prompt, the saving is exact and no number moves.
    """
    cases = _cache_probe_cases()
    uncached, uncached_maps = _sweep_counting_map_computations(monkeypatch, cache=False)
    cached, cached_maps = _sweep_counting_map_computations(monkeypatch, cache=True)

    assert cached == uncached
    assert uncached_maps == 2 * len(cases) * len(uncached["layers"])
    assert cached_maps == len(cases) * len(cached["layers"])
    assert uncached_maps == 2 * cached_maps


# ------------------------------ EXP-001 §3.5 (issue #54): the derived kind-matched family


class _HybridSweepView(_SweepView):
    """A deeper hybrid whose period comes from a real ``TextModelArgs`` (R31).

    Weights and compute stay fake; the configuration dataclass and ``DecoderLayer.is_linear``
    are the library's own, because the hybrid period is exactly the seam this slice reads.
    """

    num_layers = 10

    def __init__(self) -> None:
        from mlx_lm.models import qwen3_5

        self.args = qwen3_5.TextModelArgs(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            num_hidden_layers=self.num_layers,
            full_attention_interval=3,
        )
        self.model = SimpleNamespace(language_model=SimpleNamespace(args=self.args))

    def layer_kind(self, index):
        from mlx_lm.models import qwen3_5

        block = qwen3_5.DecoderLayer(args=self.args, layer_idx=index)
        return "linear_attention" if block.is_linear else "attention"


def test_default_layers_sweep_the_kind_matched_family_and_record_every_role(
    monkeypatch, tmp_path
) -> None:
    """Omitting ``--layers`` runs the registry fractions *plus* their kind-matched partners."""
    view = _HybridSweepView()
    payload, _output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, layers=(), view=view
    )

    family = payload["layer_family"]
    assert payload["layer_selection"]["source"] == "registry-default"
    assert family["derived"] is True
    assert family["hybrid_period"] == view.args.full_attention_interval
    assert "full_attention_interval" in family["hybrid_period_source"]
    assert family["partners"], "a hybrid default sweep must carry kind-matched partners"
    assert payload["layers"] == family["layers"]
    assert set(family["layers"]) > set(payload["layer_selection"]["indices"])
    for layer in family["partners"]:
        assert family["roles"][str(layer)] == "partner"
        assert family["kinds"][str(layer)] == "attention"
        assert payload["layer_kinds"][str(layer)] == "attention"
    for layer in family["in_band"]:
        assert family["roles"][str(layer)] == "primary"
    assert payload["conformance"]["layer_family"] == family
    assert payload["comparability"]["layer_family"] == family
    assert set(family["pairs"])


def test_holm_corrects_across_the_whole_family_including_the_partners(
    monkeypatch, tmp_path
) -> None:
    """EXP-001 §2 multiplicity: the family is every layer *scored* under one readout.

    "Scored", not "swept". The registry fractions include 1.0, and issue #68 excludes ``future``
    and ``all`` at the final layer -- no decoder block remains, so both would be structural
    zeros -- which makes those two families one member smaller than ``self``'s. That is the
    correction actually applied, and it must be built that way rather than adjusted afterwards.
    """
    payload, _output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, layers=(), view=_HybridSweepView()
    )

    family = payload["layer_family"]
    assert family["partners"]
    final = _HybridSweepView.num_layers
    assert final in family["layers"], "the premise: the default sweep reaches the final layer"
    for readout in ("self", "future", "all"):
        scored = [
            layer
            for layer in family["layers"]
            if readout not in ("future", "all") or layer != final
        ]
        corrected = sorted(
            key
            for key, value in payload["results"].items()
            if key.startswith("jlens_L")
            and key.endswith(f"_{readout}")
            and "matched_p_holm" in value
        )
        assert corrected == sorted(f"jlens_L{layer}_{readout}" for layer in scored)


def test_explicit_layers_bypass_the_derivation_and_are_marked_as_such(
    monkeypatch, tmp_path
) -> None:
    """``--layers`` is honoured verbatim; the ``cli`` source record is extended, not replaced."""
    payload, _output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, view=_HybridSweepView()
    )

    assert payload["layer_selection"]["source"] == "cli"
    assert payload["layers"] == [2, 3]
    family = payload["layer_family"]
    assert family["derived"] is False
    assert family["layers"] == [2, 3]
    assert family["partners"] == []
    assert set(family["roles"].values()) == {"explicit"}


def test_sweep_markdown_states_the_family_the_period_and_the_pairs(
    monkeypatch, tmp_path
) -> None:
    """R34: a reader of ``sweep.md`` can see which pairs the per-kind contrast rests on."""
    _payload, output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, layers=(), view=_HybridSweepView()
    )

    text = (output / "sweep.md").read_text(encoding="utf-8")

    assert "hybrid period" in text
    assert "full_attention_interval" in text
    assert "kind-matched pairs" in text
    assert "roles" in text


# ------------------ the sweep refuses a hybrid whose period was never found (issue #54 lane)


class _DenseSweepView(_SweepView):
    """One block kind only: EXP-001 §5's R35 dense comparator, which legitimately has none."""

    def layer_kind(self, index):
        del index
        return "attention"


def test_a_hybrid_whose_period_was_not_found_refuses_to_sweep(
    monkeypatch, tmp_path, capsys
) -> None:
    """Two GPU hours are too expensive to spend on a silently degraded layer family.

    ``_SweepView`` is a hybrid by its own ``layer_kind`` -- both block kinds are present --
    but exposes neither blocks nor a configuration, so no period is derivable and the derived
    family collapses to the bare registry fractions with no kind-matched partner. That is
    exactly the shape the 4B run took: it logged ``hybrid_period=-`` and swept on regardless.
    """
    with pytest.raises(SystemExit):
        _run_sweep_cli(monkeypatch, tmp_path, layers=(), view=_SweepView())

    message = capsys.readouterr().err
    assert "kind-matched partner" in message
    assert "--layers" in message, "the message must name the way past it"


def test_a_dense_backbone_still_sweeps_with_no_partners(monkeypatch, tmp_path) -> None:
    """One block kind is not a missed period: the R35 comparator must still run normally."""
    payload, _output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, layers=(), view=_DenseSweepView()
    )

    family = payload["layer_family"]
    assert family["partners"] == []
    assert set(family["kinds"].values()) == {"attention"}
    assert payload["results"], "a dense sweep produces results like any other"


# --------------- issue #68: the final layer carries no future or all readout to report


def test_the_sweep_computes_only_self_and_the_logit_lens_at_the_final_layer(
    monkeypatch, tmp_path
) -> None:
    """The cells are absent and the artifact says why, rather than reporting structural zeros."""
    final = _SweepView.num_layers
    payload, _output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, layers=("--layers", f"2,{final}")
    )

    results = payload["results"]
    assert f"jlens_L{final}_self" in results
    assert f"logit_lens_L{final}" in results
    assert f"jlens_L{final}_future" not in results
    assert f"jlens_L{final}_all" not in results
    # An interior layer keeps all three, so the exclusion is the final layer's alone.
    assert {f"jlens_L2_{name}" for name in ("self", "future", "all")} <= set(results)

    exclusion = payload["conformance"]["final_layer_readout_exclusion"]
    assert exclusion["layer"] == final
    assert exclusion["excluded"] == ["future", "all"]
    assert "no decoder block remains" in exclusion["reason"]


def test_the_holm_family_for_future_is_built_without_the_final_layer(
    monkeypatch, tmp_path
) -> None:
    """The family is every layer scored under one readout, so dropping a member changes it.

    The correction must be built from the layers actually scored, not adjusted afterwards: at
    the final layer ``future`` is never scored at all, so it was never a family member.
    """
    final = _SweepView.num_layers
    payload, _output, _tokenizer, _spec = _run_sweep_cli(
        monkeypatch, tmp_path, layers=("--layers", f"2,{final}")
    )

    swept = payload["layers"]
    corrected = {
        name: [
            key
            for key, value in payload["results"].items()
            if key.endswith(f"_{name}") and "matched_p_holm" in value
        ]
        for name in ("self", "future", "all")
    }

    assert len(corrected["self"]) == len(swept)
    assert len(corrected["future"]) == len(swept) - 1
    assert len(corrected["all"]) == len(swept) - 1
