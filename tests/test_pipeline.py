from __future__ import annotations

import re
import sys
import types

import mlx.core as mx
import mlx.nn as nn
import pytest

from local_llm_lab.agent_protocol import Action
from local_llm_lab.pipeline import jlens
from local_llm_lab.pipeline.data import build_rows
from local_llm_lab.pipeline.env import TRANSIENT_ERROR, Fault, Simulator, validate_arguments
from local_llm_lab.pipeline.evaluate import percentile, summarize
from local_llm_lab.pipeline.protocol import (
    SYSTEM_PROMPT,
    TOOL_FENCE_CLOSE,
    TOOL_FENCE_OPEN,
    Step,
    assistant_message,
    parse_turn,
    render_turn,
    turn_is_complete,
)
from local_llm_lab.pipeline.runner import (
    Trajectory,
    detect_loop,
    trajectory_rows,
)
from local_llm_lab.pipeline.tasks import FAMILIES, VARIANTS, make_tasks


def test_registered_models_include_all_three_backbones() -> None:
    from local_llm_lab.models import registered_models

    assert registered_models() == ["qwen25-coder-3b", "qwen35-4b", "qwen35-9b"]


def test_raw_legacy_hf_id_uses_qwen25_compatibility_defaults() -> None:
    from local_llm_lab.models import load_model_spec

    spec = load_model_spec("mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")

    assert spec.name == "qwen25-coder-3b"
    assert spec.chat.thinking == "unsupported"
    assert spec.cache_strategy == "trim"
    assert spec.lora.keys == "attention+mlp"


def test_qwen25_model_spec_preserves_full_training_and_lora_values() -> None:
    from local_llm_lab.models import load_model_spec

    spec = load_model_spec("qwen25-coder-3b")

    assert (spec.lora.keys, spec.lora.rank, spec.lora.scale, spec.lora.dropout) == (
        "attention+mlp",
        16,
        32.0,
        0.0,
    )
    assert spec.train == {
        "max_seq_length": 2688,
        "batch_size": 2,
        "grad_accumulation_steps": 2,
        "learning_rate": 3.0e-5,
        "grad_checkpoint": True,
    }


@pytest.mark.parametrize(
    ("name", "hf_id", "thinking", "cache_strategy", "lora_keys", "train"),
    [
        (
            "qwen25-coder-3b",
            "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
            "unsupported",
            "trim",
            "attention+mlp",
            {
                "max_seq_length": 2688,
                "batch_size": 2,
                "grad_accumulation_steps": 2,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
            },
        ),
        (
            "qwen35-4b",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            "off",
            "none",
            "auto",
            {
                "max_seq_length": 2688,
                "batch_size": 2,
                "grad_accumulation_steps": 2,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
            },
        ),
        (
            "qwen35-9b",
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            "off",
            "none",
            "auto",
            {
                "max_seq_length": 2688,
                "batch_size": 1,
                "grad_accumulation_steps": 4,
                "learning_rate": 3.0e-5,
                "grad_checkpoint": True,
            },
        ),
    ],
)
def test_model_spec_registry_values(
    name: str,
    hf_id: str,
    thinking: str,
    cache_strategy: str,
    lora_keys: str,
    train: dict[str, object],
) -> None:
    from local_llm_lab.models import load_model_spec

    spec = load_model_spec(name)

    assert spec.hf_id == hf_id
    assert spec.chat.thinking == thinking
    assert spec.cache_strategy == cache_strategy
    assert spec.lora.keys == lora_keys
    assert spec.train == train


def test_model_spec_unknown_hf_id_uses_conservative_defaults() -> None:
    from local_llm_lab.models import load_model_spec

    hf_id = "example/unknown-model-4bit"
    spec = load_model_spec(hf_id)

    assert spec.name == hf_id and spec.hf_id == hf_id and spec.family == "unknown"
    assert spec.chat.thinking == "unsupported" and spec.chat.template_kwargs == {}
    assert spec.cache_strategy == "none"
    assert spec.lora.keys == "attention+mlp"
    assert (spec.lora.rank, spec.lora.scale, spec.lora.dropout) == (16, 32.0, 0.0)


def test_model_spec_resolve_reads_architecture_and_tokenizer_metadata(monkeypatch) -> None:
    from local_llm_lab.models import load_model_spec

    class FakeView:
        num_layers = 4
        hidden_size = 12
        vocab_size = 321
        tie_word_embeddings = True
        cache_trimmable = True

        @classmethod
        def from_model(cls, model):
            assert model is fake_model
            return cls()

        def layer_kind(self, index: int) -> str:
            return "attention" if index % 2 == 0 else "linear_attention"

        def lora_targets(self, policy):
            assert policy == "attention+mlp"
            return ("layers.0.q_proj", "layers.1.down_proj")

        def lora_parameter_count(self, keys, rank):
            assert keys == ("layers.0.q_proj", "layers.1.down_proj")
            assert rank == 16
            return 456

    fake_arch = types.ModuleType("local_llm_lab.arch")
    fake_arch.ArchitectureView = FakeView
    fake_arch.LORA_POLICIES = frozenset({"auto", "attention+mlp", "all-linear"})
    monkeypatch.setitem(sys.modules, "local_llm_lab.arch", fake_arch)
    fake_model = object()
    tokenizer = types.SimpleNamespace(snapshot_revision="fake-snapshot-revision")

    resolved = load_model_spec("qwen25-coder-3b").resolve(fake_model, tokenizer)

    assert resolved.vocab_size == 321
    assert resolved.tie_word_embeddings is True
    assert resolved.trainable_parameters == 456
    assert resolved.snapshot_revision == "fake-snapshot-revision"
    assert resolved.layer_types == ("attention", "linear_attention", "attention", "linear_attention")


@pytest.mark.parametrize(
    ("source", "replacement", "match"),
    [
        ("thinking: unsupported", "thinking: enabled", "chat.thinking"),
        ("strategy: trim", "strategy: reuse", "cache.strategy"),
        ("layer_fractions: [0.5]", "layer_fractions: [true]", "probes.layer_fractions"),
        ("layer_fractions: [0.5]", "layer_fractions: [null]", "probes.layer_fractions"),
    ],
)
def test_model_spec_rejects_invalid_declared_policies_and_fractions(
    tmp_path, monkeypatch, source: str, replacement: str, match: str
) -> None:
    from local_llm_lab import models

    config = """\
name: invalid
hf_id: example/invalid
family: test
chat:
  thinking: unsupported
  template_kwargs: {}
  end_of_turn: <|im_end|>
  extra_stop_tokens: []
lora:
  keys: attention+mlp
  rank: 16
  scale: 32.0
  dropout: 0.0
train: {}
cache:
  strategy: trim
probes:
  layer_fractions: [0.5]
memory:
  budget_gib: 22
policies: {}
""".replace(source, replacement)
    (tmp_path / "invalid.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setattr(models, "_REGISTRY_DIR", tmp_path)

    with pytest.raises(ValueError, match=match):
        models.load_model_spec("invalid")


def test_every_split_verifies_and_supervises_only_good_steps() -> None:
    for split in ("train", "valid", "test", "iter1"):
        for task in make_tasks(split, 48):
            rows = build_rows(task)
            assert len(rows) == sum(step.supervise for step in task.steps), task.task_id
            assert all("tools" not in row for row in rows), "tool list lives in the system message"
            assert all(row["messages"][0]["content"] == SYSTEM_PROMPT for row in rows)
            for row in rows:
                target = row["messages"][-1]
                assert target["role"] == "assistant" and "tool_calls" not in target
                assert TOOL_FENCE_OPEN in target["content"], "target carries a fenced call"
                assert target["content"].endswith(TOOL_FENCE_CLOSE)
                turn = parse_turn(target["content"])
                assert turn.thought, "every target carries a note"


def test_training_split_covers_families_and_variants() -> None:
    tasks = make_tasks("train", 96)
    assert {task.family for task in tasks} == set(FAMILIES)
    assert {task.variant for task in tasks} == set(VARIANTS)
    assert all(task.variant == "clean" for task in make_tasks("test", 24))
    assert len(VARIANTS) == 6, "clean, wrong_path, transient, unknown_tool, stale_path, failed_edit"


def test_training_split_of_144_covers_all_six_variants_and_builds() -> None:
    tasks = make_tasks("train", 144)
    assert (
        {task.variant for task in tasks}
        == set(VARIANTS)
        == {
            "clean",
            "wrong_path",
            "transient",
            "unknown_tool",
            "stale_path",
            "failed_edit",
        }
    )
    for task in tasks:
        build_rows(task)


def test_splits_are_isolated_and_test_is_longer() -> None:
    train = make_tasks("train", 36)
    test = make_tasks("test", 36)
    assert {task.prompt for task in train}.isdisjoint(task.prompt for task in test)
    assert sum(task.horizon for task in test) > sum(task.horizon for task in train)


def test_recovery_variants_fail_then_recover() -> None:
    tasks = make_tasks("train", 96)
    wrong = next(task for task in tasks if task.variant == "wrong_path")
    unsupervised = [index for index, step in enumerate(wrong.steps) if not step.supervise]
    assert len(unsupervised) == 1
    transient = next(task for task in tasks if task.variant == "transient")
    assert (
        transient.faults
        and transient.steps[transient.faults[0].call_index + 1].action
        == transient.steps[transient.faults[0].call_index].action
    )
    bogus = next(task for task in tasks if task.variant == "unknown_tool")
    bad = next(step for step in bogus.steps if not step.supervise)
    assert bad.action.name not in {"replace_text", "read_file"}


def test_variant_applicability_probes_and_caches_each_requested_level(monkeypatch) -> None:
    """A level-specific recovery shape must not reuse the level-zero probe result."""
    from local_llm_lab.pipeline import tasks

    probes: list[int] = []

    def maker(split, index, level, rng):
        probes.append(level)
        steps = (_step("finish", "finish", answer="done"),)
        if level:
            steps = (
                _step("list", "list_files", directory="workspace"),
                _step("read", "read_file", path="workspace/report.txt"),
            )
        return tasks.Task("probe", "probe", "clean", "p", {}, steps, "", frozenset())

    def _step(thought, name, **arguments):
        return Step(thought, Action(name, arguments))

    monkeypatch.setitem(tasks._MAKERS, "probe", maker)
    tasks._APPLICABLE_VARIANTS_CACHE.pop(("probe", 0), None)
    tasks._APPLICABLE_VARIANTS_CACHE.pop(("probe", 1), None)

    assert "wrong_path" not in tasks.applicable_variants("probe", 0)
    assert "wrong_path" in tasks.applicable_variants("probe", 1)
    assert tasks.applicable_variants("probe", 1) == tasks.applicable_variants("probe", 1)
    assert probes == [0, 1]


def test_recovery_path_guesses_are_absent_and_notes_name_the_guess() -> None:
    """Changing a recovery guess to an existing path or a correct-path note must fail here."""
    recovery_tasks = [
        task
        for task in make_tasks("train", 144)
        if task.variant in {"wrong_path", "stale_path"}
    ]
    assert recovery_tasks
    for task in recovery_tasks:
        wrong = next(step for step in task.steps if not step.supervise)
        guessed = wrong.action.arguments["path"]
        assert guessed not in task.files
        assert guessed in wrong.thought


def test_wrong_path_constructor_rejects_an_impossible_guess() -> None:
    """Exhausting every path guess must not silently substitute a transient variant."""

    import random

    from local_llm_lab.pipeline import tasks

    directory = "workspace/only-existing"
    guarded = tasks.Task(
        "guarded",
        "list",
        "clean",
        "p",
        {f"{directory}/{name}": "present" for name in ("index.txt", "summary.md", "data.txt", "main.ini")},
        (
            Step("list", Action("list_files", {"directory": directory})),
            Step("finish", Action("finish", {"answer": "done"})),
        ),
        "done",
        frozenset(),
    )

    with pytest.raises(RuntimeError, match="guarded"):
        tasks._wrong_path(guarded, random.Random(0))


def test_stale_path_constructor_rejects_an_impossible_guess() -> None:
    """A task with no viable stale path must not silently substitute another variant."""

    import random

    from local_llm_lab.pipeline import tasks

    guarded = tasks.Task(
        "stale-guarded",
        "search",
        "clean",
        "p",
        {},
        (
            Step("search", Action("search_files", {"query": "missing"})),
            Step("finish", Action("finish", {"answer": "done"})),
        ),
        "done",
        frozenset(),
    )

    with pytest.raises(RuntimeError, match="stale-guarded"):
        tasks._stale_path(guarded, random.Random(0))


def test_hardened_update_prompts_do_not_leak_expected_answer() -> None:
    """Prompts must expose answer inputs and schemas without handing over the final answer."""
    import random

    from local_llm_lab.pipeline import tasks

    updates = [tasks._update("train", index, 0, random.Random(index)) for index in range(20)]
    batches = [tasks._batch_update("train", index, 1, random.Random(index)) for index in range(20)]
    for task in updates:
        _, _, target = task.expected_answer.partition("=")
        assert target in task.prompt
        assert "mode=<target>" in task.prompt
        assert task.expected_answer not in task.prompt
    for task in batches:
        _, _, count = task.expected_answer.partition("=")
        assert count in task.prompt
        assert "updated-and-verified=<count>" in task.prompt
        assert task.expected_answer not in task.prompt


def test_stale_path_variant_relists_before_recovering() -> None:
    """A guessed stale path fails, and the only supervised recovery is to re-list the
    directory (the hidden listing) before reading the exact path it reports."""
    tasks = make_tasks("train", 144)
    stale = [task for task in tasks if task.variant == "stale_path"]
    assert stale
    for task in stale:
        unsupervised = [index for index, step in enumerate(task.steps) if not step.supervise]
        assert len(unsupervised) == 1, task.task_id
        (k,) = unsupervised
        assert task.steps[k].action.name == "read_file", task.task_id

        simulator = Simulator.for_task(task, faults=())
        for step in task.steps[:k]:
            simulator.execute(step.action)
        observation = simulator.execute(task.steps[k].action)
        assert observation.startswith("ERROR: file not found"), (task.task_id, observation)

        listing_step = task.steps[k + 1]
        assert listing_step.supervise and listing_step.action.name == "list_files", task.task_id
        correct_step = task.steps[k + 2]
        assert correct_step.supervise and correct_step.action.name == "read_file", task.task_id
        correct_path = correct_step.action.arguments["path"]
        assert listing_step.action.arguments["directory"] == correct_path.rsplit("/", 1)[0], (
            task.task_id
        )
        assert correct_path != task.steps[k].action.arguments["path"], task.task_id


def test_failed_edit_variant_rereads_before_recovering() -> None:
    """A ``replace_text`` with the wrong ``old`` text fails, and the only supervised recovery
    is to re-read the file before retrying the original, correct replacement."""
    tasks = make_tasks("train", 144)
    failed = [task for task in tasks if task.variant == "failed_edit"]
    assert failed
    for task in failed:
        unsupervised = [index for index, step in enumerate(task.steps) if not step.supervise]
        assert len(unsupervised) == 1, task.task_id
        (k,) = unsupervised
        assert task.steps[k].action.name == "replace_text", task.task_id

        simulator = Simulator.for_task(task, faults=())
        for step in task.steps[:k]:
            simulator.execute(step.action)
        observation = simulator.execute(task.steps[k].action)
        assert observation == "ERROR: old text not found", (task.task_id, observation)

        path = task.steps[k].action.arguments["path"]
        reread_step = task.steps[k + 1]
        assert reread_step.supervise and reread_step.action.name == "read_file", task.task_id
        assert reread_step.action.arguments["path"] == path, task.task_id

        correct_step = task.steps[k + 2]
        assert correct_step.supervise and correct_step.action.name == "replace_text", task.task_id
        assert correct_step.action.arguments["path"] == path, task.task_id
        assert correct_step.action.arguments["old"] != task.steps[k].action.arguments["old"], (
            task.task_id
        )
        assert correct_step.action.arguments["new"] == task.steps[k].action.arguments["new"], (
            task.task_id
        )


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _long_horizon_tasks() -> list:
    return [*make_tasks("train", 96), *make_tasks("test", 24)]


def test_every_grounded_argument_is_derivable_from_its_context() -> None:
    """Every ``path``/``directory`` argument in a target must be composed from text already
    visible in its windowed context, not invented.

    The note and the tool call are generated in the same turn, so a path named only in the
    target's own note is one the model had to invent -- exactly the failure mode observed in a
    trained checkpoint: it wrote a confident recovery note naming a path that did not exist,
    then read it 20 times. For every grounded argument, both the basename and the directory
    half of the path must independently appear as substrings of the context that precedes the
    target (never the target itself).
    """
    for task in [*make_tasks("train", 240), *make_tasks("valid", 36), *make_tasks("test", 60)]:
        for row in build_rows(task):
            context = "\n".join(message["content"] for message in row["messages"][:-1])
            target = row["messages"][-1]["content"]
            turn = parse_turn(target)
            for key in ("path", "directory"):
                if key not in turn.action.arguments:
                    continue
                value = turn.action.arguments[key]
                basename = value.rsplit("/", 1)[-1]
                directory = value.rsplit("/", 1)[0]
                missing = [
                    half
                    for half, text in (("basename", basename), ("directory", directory))
                    if text not in context
                ]
                assert not missing, (
                    f"{task.task_id} step {row['metadata']['step']}: argument {key}={value!r} "
                    f"has its {' and '.join(missing)} missing from context; target: {target!r}"
                )


def test_cross_reference_notes_resolve_multiple_search_matches() -> None:
    for task in _long_horizon_tasks():
        if task.family != "cross_reference":
            continue
        simulator = Simulator.for_task(task, faults=())
        steps = [step for step in task.steps if step.supervise]
        searches = [i for i, step in enumerate(steps) if step.action.name == "search_files"]
        assert searches, task.task_id
        for hop, i in enumerate(searches):
            result = simulator.execute(steps[i].action)
            assert result.startswith("MATCHES: "), (task.task_id, result)
            matches = result.removeprefix("MATCHES: ").split(", ")
            read = next(step for step in steps[i + 1 :] if step.action.name == "read_file")
            target = read.action.arguments["path"]
            assert target in matches, f"{task.task_id} hop {hop}: reads {target} not in {matches}"
            if hop >= 1:
                assert "the file whose Lookup-Key equals it is the new one" in steps[i].thought, (
                    task.task_id
                )
            if len(matches) == 1:
                assert read.thought.endswith(f"Search matched {_name(target)}. Reading it."), (
                    task.task_id,
                    read.thought,
                )
                simulator.execute(read.action)
                continue
            assert "already read" in read.thought, f"{task.task_id} hop {hop}: {read.thought!r}"
            assert f"reading the new one: {_name(target)}." in read.thought, (
                task.task_id,
                read.thought,
            )
            for other in matches:
                if other != target:
                    assert _name(other) in read.thought, (task.task_id, other, read.thought)
                    already = [
                        c.arguments["path"] for c in simulator.calls if c.name == "read_file"
                    ]
                    assert other in already, f"{task.task_id}: {other} was not read before"
            simulator.execute(read.action)


def _parse_agg_bucket(label: str) -> list[int]:
    """``18, 14, 75 (full)`` -> ``[18, 14, 75]``; ``none`` -> ``[]``."""
    label = label.removesuffix(" (full)")
    return [] if label == "none" else [int(part) for part in label.split(", ")]


def _metric_value(task, step) -> int:
    content = task.files[step.action.arguments["path"]]
    match = re.search(r"value=(\d+)", content)
    assert match, (task.task_id, step.action.arguments["path"])
    return int(match.group(1))


def test_aggregate_report_reading_drops_no_value() -> None:
    """Every metric value must land in exactly one running bucket by the time reading
    finishes, never be invented by indexing into a remembered list.

    Each read note states the buckets as of *before* that read (mirroring
    ``ledger_reconcile``), so the last metric's own value is not yet in either bucket at the
    moment its note is written -- it is the value that read is about to fetch. Folding that
    one value in, by position, must reconstruct the full value list with nothing dropped and
    nothing duplicated; this is exactly the failure mode a trained checkpoint showed (losing
    one value while splitting the list in one shot).
    """
    for task in [*make_tasks("train", 144), *make_tasks("test", 60)]:
        if task.family != "aggregate_report":
            continue
        # Only supervised notes are real training decisions (a wrong_path/stale_path guess is
        # unsupervised noise); a transient retry duplicates the immediately preceding note
        # verbatim (same path), so adjacent duplicates by path are collapsed too.
        raw_notes = [
            step
            for step in task.steps
            if step.supervise and step.action.name == "read_file" and "first half:" in step.thought
        ]
        metric_notes = [
            step
            for i, step in enumerate(raw_notes)
            if i == 0 or step.action.arguments["path"] != raw_notes[i - 1].action.arguments["path"]
        ]
        assert metric_notes, task.task_id
        metric_count = len(metric_notes)
        split_at = metric_count // 2

        match = re.search(
            r"first half: (.*?); second half: (.*?)\. Reading", metric_notes[-1].thought
        )
        assert match, (task.task_id, metric_notes[-1].thought)
        first_half = _parse_agg_bucket(match.group(1))
        second_half = _parse_agg_bucket(match.group(2))
        last_value = _metric_value(task, metric_notes[-1])
        (first_half if metric_count - 1 < split_at else second_half).append(last_value)

        all_values = [_metric_value(task, step) for step in metric_notes]
        assert sorted(first_half + second_half) == sorted(all_values), task.task_id
        assert len(first_half) + len(second_half) == len(all_values), task.task_id


def _worker_managed_order(task) -> list[str]:
    """The managed worker basenames in manifest order, derived independently of the notes."""
    return [
        _name(path)
        for path in sorted(
            task.expected_files,
            key=lambda p: int(re.search(r"worker-(\d+)\.ini", p).group(1)),
        )
    ]


def test_batch_update_apply_phase_names_a_queue_head_then_transitions_to_verify() -> None:
    """Apply-phase notes name a queue head, never a position to index: exactly one worker per
    note, and the heads across the phase visit every managed worker once each in manifest
    order. A ``transient`` retry may legitimately repeat the immediately preceding head (it is
    the same call, retried unchanged), so only adjacent duplicates are collapsed before
    comparing. The verify phase must announce its own start as a stated fact rather than
    something to infer from the apply phase falling silent -- the second, independent bug this
    family was rewritten to fix (it never transitioned and kept re-applying to the last
    worker).
    """
    for task in [*make_tasks("train", 144), *make_tasks("valid", 36), *make_tasks("test", 60)]:
        if task.family != "batch_update":
            continue
        supervised = [step for step in task.steps if step.supervise]

        apply_steps = [step for step in supervised if step.action.name == "replace_text"]
        assert apply_steps, task.task_id
        heads = []
        for step in apply_steps:
            matches = re.findall(r"Next: (worker-\d+\.ini) mode=", step.thought)
            assert len(matches) == 1, (task.task_id, step.thought)
            heads.append(matches[0])
        deduped = [head for i, head in enumerate(heads) if i == 0 or head != heads[i - 1]]
        assert deduped == _worker_managed_order(task), task.task_id

        last_apply_index = max(
            i for i, step in enumerate(supervised) if step.action.name == "replace_text"
        )
        first_verify = supervised[last_apply_index + 1]
        assert first_verify.action.name == "read_file", task.task_id
        assert "phase verify begins" in first_verify.thought, (
            task.task_id,
            first_verify.thought,
        )


def test_conditional_update_note_before_replace_names_the_modified_service() -> None:
    """The note attached to the (sole) ``replace_text`` step -- written immediately before that
    call, in the same turn -- must already know the winner from the incrementally tracked
    running best, and that winner must be the service actually modified. This replaces a
    one-shot ``max()`` over a remembered list, the same class of bug that made a trained
    checkpoint throttle the wrong worker.
    """
    for task in [*make_tasks("train", 144), *make_tasks("valid", 36), *make_tasks("test", 60)]:
        if task.family != "conditional_update":
            continue
        supervised = [step for step in task.steps if step.supervise]
        replace_steps = [step for step in supervised if step.action.name == "replace_text"]
        assert len(replace_steps) == 1, task.task_id
        replace_step = replace_steps[0]
        match = re.search(r"highest so far: (service-\d+)=\d+ \(final\)", replace_step.thought)
        assert match, (task.task_id, replace_step.thought)
        modified = _name(replace_step.action.arguments["path"]).removesuffix(".ini")
        assert match.group(1) == modified, task.task_id


def test_simulator_verdict_reports_reasons_and_recovery() -> None:
    task = make_tasks("valid", 12)[4]  # update family, clean
    simulator = Simulator.for_task(task, faults=(Fault(call_index=0),))
    first = simulator.execute(task.steps[0].action)
    assert first.startswith("ERROR")
    for step in task.steps:
        simulator.execute(step.action)
    verdict = simulator.verdict()
    assert verdict.success and not verdict.clean and verdict.recovered_errors == 1
    bad = Simulator.for_task(task)
    unknown = bad.execute(Action("update_file", {}))
    assert unknown.startswith("ERROR: invalid call: unknown tool: update_file")
    assert "available tools are list_files, read_file" in unknown
    bad.execute(Action("finish", {"answer": "nope"}))
    bad_verdict = bad.verdict()
    reasons = bad_verdict.reasons
    assert "wrong answer" in reasons and any(r.startswith("file state wrong") for r in reasons)
    assert bad_verdict.calls == 2 and bad_verdict.schema_failures == 1
    assert bad_verdict.executable_calls == 1 and bad_verdict.errors == 1
    assert verdict.as_dict()["calls"] == len(task.steps) + 1


def test_verdict_rejects_unexpected_changes_and_keeps_raw_normalized_answer() -> None:
    """Removing initial-file comparison, successful-call tracking, or normalization breaks this."""
    simulator = Simulator(
        files={
            "workspace/expected.ini": "old",
            "workspace/extra.ini": "keep",
            "workspace/removed.ini": "remove",
        },
        expected_answer="updated",
        expected_files={"workspace/expected.ini": "new"},
        required_tools=frozenset({"read_file"}),
    )
    simulator.execute(Action("read_file", {"path": "workspace/missing.ini"}))
    simulator.execute(
        Action(
            "replace_text",
            {"path": "workspace/expected.ini", "old": "old", "new": "new"},
        )
    )
    simulator.execute(
        Action("replace_text", {"path": "workspace/extra.ini", "old": "keep", "new": "changed"})
    )
    simulator.files["workspace/added.ini"] = "added"
    del simulator.files["workspace/removed.ini"]
    simulator.execute(Action("finish", {"answer": "`updated.`"}))

    verdict = simulator.verdict()

    assert not verdict.success
    assert verdict.raw_answer == "`updated.`"
    assert verdict.answer == "updated"
    assert verdict.unexpected_files == (
        "workspace/added.ini",
        "workspace/extra.ini",
        "workspace/removed.ini",
    )
    assert "unexpected file change: extra.ini" in verdict.reasons
    assert "unexpected file change: added.ini" in verdict.reasons
    assert "unexpected file change: removed.ini" in verdict.reasons
    assert "required tools unused: read_file" in verdict.reasons
    assert verdict.as_dict()["raw_answer"] == "`updated.`"


def test_trajectory_rows_skip_errored_steps_except_injected_faults() -> None:
    task = make_tasks("valid", 12)[0]
    trajectory = Trajectory(task.task_id, task.family, task.variant, "t", task.prompt, faults=[0])
    trajectory.steps = [
        {
            "index": 0,
            "thought": "n0",
            "action": {"name": "read_file", "arguments": {"path": "p"}},
            "observation": "ERROR: temporary",
        },
        {
            "index": 1,
            "thought": "n1",
            "action": {"name": "read_file", "arguments": {"path": "zzz"}},
            "observation": "ERROR: file not found",
        },
        {
            "index": 2,
            "thought": "n2",
            "action": {"name": "finish", "arguments": {"answer": "a"}},
            "observation": "FINISHED",
        },
    ]
    rows = trajectory_rows(trajectory, task)
    assert [row["metadata"]["step"] for row in rows] == [0, 2]
    assert all("tools" not in row for row in rows)
    assert rows[0]["messages"][-1] == assistant_message("n0", Action("read_file", {"path": "p"}))


def _fake_trajectory(
    task_id: str,
    *,
    success: bool,
    reasons: tuple[str, ...] = (),
    errors: int = 0,
    calls: int = 4,
    schema_failures: int = 0,
    executable_calls: int | None = None,
    elapsed: float = 1.0,
    tokens: int = 0,
    **fields,
) -> Trajectory:
    """Trajectory with a verdict dict shaped like ``Verdict.as_dict``; ``fields`` set flags."""
    return Trajectory(
        task_id,
        "read",
        "clean",
        "t",
        "p",
        turns=calls,
        valid_turns=calls,
        elapsed_seconds=elapsed,
        generated_tokens=tokens,
        verdict={
            "success": success,
            "clean": success and errors == 0,
            "errors": errors,
            "recovered_errors": errors if success else 0,
            "reasons": list(reasons),
            "calls": calls,
            "schema_failures": schema_failures,
            "executable_calls": calls - errors if executable_calls is None else executable_calls,
        },
        **fields,
    )


def test_summarize_counts_recoveries_and_reasons() -> None:
    good = _fake_trajectory("a", success=True, errors=1, calls=2, tokens=40)
    bad = _fake_trajectory(
        "b", success=False, reasons=("no finish call",), calls=3, tokens=60, parse_error="x"
    )
    bad.valid_turns = 2
    summary = summarize([good, bad])
    assert summary["successes"] == 1 and summary["clean_successes"] == 0
    assert summary["recovered_errors"] == 1 and summary["failure_reasons"] == {"parse error": 1}
    assert summary["valid_action_rate"] == 0.8
    assert summary["schema_validity_rate"] == 1.0
    assert summary["executable_call_rate"] == 0.8, "4 of 5 calls ran (one recovered error)"
    assert summary["loop_failures"] == 0 and summary["exhausted"] == 0
    assert summary["generated_tokens"] == 100 and summary["tokens_per_success"] == 100.0
    assert summary["latency_p50_seconds"] == 1.0 and summary["latency_p95_seconds"] == 1.0


def test_summarize_attributes_loops_budget_and_rates() -> None:
    trajectories = [
        _fake_trajectory("ok", success=True, elapsed=1.0, tokens=100),
        _fake_trajectory(
            "loop",
            success=False,
            reasons=("no finish call",),
            elapsed=4.0,
            tokens=300,
            loop_detected=True,
            exhausted=True,
        ),
        _fake_trajectory(
            "budget", success=False, reasons=("no finish call",), elapsed=3.0, exhausted=True
        ),
        _fake_trajectory(
            "parse",
            success=False,
            reasons=("no finish call",),
            elapsed=2.0,
            parse_error="x",
            loop_detected=True,
        ),
        _fake_trajectory(
            "wrong",
            success=False,
            reasons=("wrong answer", "file state wrong: a.txt"),
            errors=2,
            schema_failures=2,
            elapsed=10.0,
        ),
        _fake_trajectory("loop-but-won", success=True, elapsed=1.5, tokens=50, loop_detected=True),
    ]
    summary = summarize(trajectories)
    assert summary["failure_reasons"] == {
        "repetition loop": 1,
        "step budget exhausted": 1,
        "parse error": 1,
        "wrong answer": 1,
    }
    # Counts are independent of attribution: the parse-error run also looped, but a
    # successful run that looped is never a loop failure.
    assert summary["loop_failures"] == 2
    assert summary["exhausted"] == 2
    assert summary["schema_validity_rate"] == round(1 - 2 / 24, 4)
    assert summary["executable_call_rate"] == round(22 / 24, 4)
    assert summary["generated_tokens"] == 450 and summary["tokens_per_success"] == 225.0
    assert summary["latency_p50_seconds"] == 2.0 and summary["latency_p95_seconds"] == 10.0
    empty = summarize([])
    assert empty["schema_validity_rate"] == 0.0 and empty["executable_call_rate"] == 0.0
    assert empty["tokens_per_success"] is None and empty["latency_p95_seconds"] == 0.0
    assert summarize([trajectories[1]])["tokens_per_success"] is None


def test_percentile_uses_nearest_rank() -> None:
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 5.0
    assert percentile(values, 0) == 1.0 and percentile(values, 100) == 5.0
    assert percentile(list(range(1, 11)), 50) == 5 and percentile(list(range(1, 11)), 95) == 10
    assert percentile([], 50) == 0.0
    assert values == [5.0, 1.0, 4.0, 2.0, 3.0], "input is not sorted in place"
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile(values, 101)


def test_validate_arguments_enforces_tool_schema() -> None:
    validate_arguments(Action("read_file", {"path": "a"}))
    validate_arguments(Action("replace_text", {"path": "a", "old": "x", "new": "y"}))
    with pytest.raises(ValueError, match=r"unexpected argument\(s\): file, mode; expected: path"):
        validate_arguments(Action("read_file", {"path": "a", "file": "b", "mode": "r"}))
    with pytest.raises(
        ValueError, match=r"missing required argument\(s\): old, new; expected: path, old, new"
    ):
        validate_arguments(Action("replace_text", {"path": "a"}))
    with pytest.raises(ValueError, match="path must be string, got integer"):
        validate_arguments(Action("read_file", {"path": 3}))
    with pytest.raises(
        ValueError,
        match="unknown tool: nope; available tools are list_files, read_file, search_files, calculate, replace_text, finish",
    ):
        validate_arguments(Action("nope", {}))
    custom = [
        {
            "type": "function",
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer"},
                        "x": {"type": "number"},
                        "flag": {"type": "boolean"},
                        "items": {"type": "array"},
                        "obj": {"type": "object"},
                    },
                    "required": ["n"],
                },
            },
        }
    ]
    validate_arguments(
        Action("t", {"n": 1, "x": 2.5, "flag": True, "items": [], "obj": {}}), custom
    )
    validate_arguments(Action("t", {"n": 1, "x": 2}), custom)  # an integer is a number
    with pytest.raises(ValueError, match="n must be integer, got boolean"):
        validate_arguments(Action("t", {"n": True}), custom)
    with pytest.raises(ValueError, match="x must be number, got boolean"):
        validate_arguments(Action("t", {"n": 1, "x": False}), custom)
    with pytest.raises(ValueError, match="n must be integer, got number"):
        validate_arguments(Action("t", {"n": 1.5}), custom)
    with pytest.raises(ValueError, match="flag must be boolean, got string"):
        validate_arguments(Action("t", {"n": 1, "flag": "yes"}), custom)


def test_simulator_validates_before_executing_and_counts_calls() -> None:
    task = make_tasks("valid", 12)[0]
    path, content = next(iter(task.files.items()))
    simulator = Simulator.for_task(task, faults=(Fault(call_index=0),))
    # Injected faults come first: they are the environment, not the policy's schema mistake.
    assert simulator.execute(Action("read_file", {"path": 3})) == TRANSIENT_ERROR
    assert simulator.errors == 1 and simulator.schema_failures == 0
    invalid = simulator.execute(Action("read_file", {"path": 3}))
    assert invalid.startswith("ERROR: invalid call:") and "path must be string" in invalid
    assert simulator.schema_failures == 1 and simulator.errors == 2
    assert simulator.executable_calls == 0
    assert simulator.execute(Action("read_file", {"path": path})) == content
    assert simulator.executable_calls == 1 and simulator.errors == 2
    assert simulator.execute(Action("read_file", {"path": "no/such"})) == (
        "ERROR: file not found: no/such"
    )
    assert simulator.executable_calls == 1 and simulator.errors == 3
    assert simulator.schema_failures == 1, "execution errors are not schema failures"
    assert simulator.execute(Action("read_file", {"path": ""})) == (
        "ERROR: path must be a non-empty string"
    )
    verdict = simulator.verdict().as_dict()
    assert verdict["calls"] == 5 and verdict["schema_failures"] == 1
    assert verdict["executable_calls"] == 1 and verdict["errors"] == 4


def _executed(index: int, name: str, arguments: dict, observation: str = "ok") -> dict:
    return {
        "index": index,
        "thought": "",
        "action": {"name": name, "arguments": arguments},
        "observation": observation,
        "raw": "",
    }


def test_detect_loop_flags_repetition_patterns() -> None:
    same = [_executed(i, "calculate", {"expression": "1+1"}, "RESULT: 2") for i in range(3)]
    assert detect_loop(same), "three identical calls"
    assert not detect_loop(same[:2]), "a transient retry is not a loop"
    alternating = [
        _executed(
            i, "read_file" if i % 2 else "search_files", {"path": "a"} if i % 2 else {"query": "q"}
        )
        for i in range(10)
    ]
    assert not detect_loop(alternating)
    ping_pong = [_executed(i, "read_file", {"path": "a" if i % 2 else "b"}) for i in range(10)]
    assert detect_loop(ping_pong), "eight same-shaped calls on one tool without an error"
    assert not detect_loop(ping_pong[:7])
    errors = [_executed(i, "read_file", {"path": f"p{i}"}, "ERROR: not found") for i in range(6)]
    assert detect_loop(errors), "six consecutive errors on the same tool"
    assert not detect_loop(errors[:5])
    assert not detect_loop(errors[:5] + [_executed(5, "read_file", {"path": "p"}, "text")])
    counting = [
        _executed(i, "calculate", {"expression": f"{i}+1"}, f"RESULT: {i + 1}") for i in range(8)
    ]
    assert detect_loop(counting), "same tool and argument keys eight times without an error"
    assert not detect_loop(counting[:7])
    assert not detect_loop(counting[:7] + [_executed(7, "finish", {"answer": "8"}, "FINISHED")])
    parse_error_step = {"index": 0, "raw": "", "parse_error": "x"}
    assert not detect_loop([parse_error_step, *same[:2]]), "parse-error steps carry no action"
    assert detect_loop([parse_error_step, *same])
    assert not detect_loop([])


def test_render_completion_matches_chat_template_layout() -> None:
    from local_llm_lab.pipeline.branch import render_completion

    action = Action("read_file", {"path": "a"})
    text = render_completion("note", action)
    assert text == render_turn("note", action) + "<|im_end|>"
    assert text == 'note\n```json\n{"name": "read_file", "arguments": {"path": "a"}}\n```<|im_end|>'
    assert render_completion("", action) == render_turn("", action) + "<|im_end|>"
    turn = parse_turn(text)
    assert turn.thought == "note" and turn.action == action
    assert turn_is_complete(text)


def _cells(line: str) -> list[str]:
    """Split a report row on its two-space column separator (cells may contain one space)."""
    import re

    return re.split(r" {2,}", line.strip())


def test_report_prints_dashes_for_summaries_without_new_metrics(tmp_path) -> None:
    import json

    from local_llm_lab.pipeline.report import load_summaries, render

    legacy = {
        "tasks": 2,
        "successes": 1,
        "success_rate": 0.5,
        "clean_successes": 1,
        "clean_rate": 0.5,
        "valid_action_rate": 1.0,
        "tool_errors": 0,
        "recovered_errors": 0,
        "mean_steps": 3.0,
        "by_family": {"read": {"successes": 1, "tasks": 2, "success_rate": 0.5}},
        "label": "old",
        "split": "test",
    }
    (tmp_path / "old.json").write_text(json.dumps({"summary": legacy}))
    table = render(load_summaries(tmp_path))
    cells = _cells(table.splitlines()[2])
    assert cells[:5] == ["old", "test", "1/2 (50%)", "50%", "100%"]
    assert cells[5] == "-" and cells[6] == "-", "schema and exec columns"
    assert cells[7:] == ["0", "3.0", "1/2"]


def test_lora_config_reflects_grad_checkpoint_and_resume(tmp_path) -> None:
    from pathlib import Path

    from local_llm_lab.pipeline.cli import DEFAULT_CONFIG, load_config, lora_config

    config = load_config(DEFAULT_CONFIG)
    assert isinstance(config["train"]["grad_checkpoint"], bool), "project config sets it explicitly"
    lora = lora_config(config)
    assert lora["grad_checkpoint"] is config["train"]["grad_checkpoint"]
    assert "resume_adapter_file" not in lora
    assert lora["iters"] == config["train"]["iters"]
    assert lora["adapter_path"] == str(Path(config["output"]) / "adapters")

    config["train"]["grad_checkpoint"] = True
    assert lora_config(config, iters=7)["grad_checkpoint"] is True
    assert lora_config(config, iters=7)["iters"] == 7
    del config["train"]["grad_checkpoint"]
    assert lora_config(config)["grad_checkpoint"] is True, "defaults to on when unset"

    weights = tmp_path / "0000100_adapters.safetensors"
    weights.write_bytes(b"")
    resumed = lora_config(config, resume_from=Path("relative") / ".." / weights)
    assert resumed["resume_adapter_file"] == str(weights.resolve())
    assert Path(resumed["resume_adapter_file"]).is_absolute()
    assert (
        lora_config(config)["lora_parameters"]["keys"]
        is not lora_config(config)["lora_parameters"]["keys"]
    ), "each call returns a fresh key list"


def test_checkpoint_dirs_replaces_stale_weight_copy(tmp_path) -> None:
    """A target adapter with different bytes must be refreshed from its checkpoint source."""
    from local_llm_lab.pipeline.cli import checkpoint_dirs

    output = tmp_path / "run"
    adapters = output / "adapters"
    adapters.mkdir(parents=True)
    (adapters / "adapter_config.json").write_text("{}", encoding="utf-8")
    source = adapters / "0000042_adapters.safetensors"
    source.write_bytes(b"fresh weights")
    stale = output / "checkpoints" / "step-42" / "adapters.safetensors"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale weights")

    found = checkpoint_dirs({"output": output})

    assert found == [(42, stale.parent)]
    assert stale.read_bytes() == b"fresh weights"


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

    class FakeProcess:
        stdout: list[str] = []

        def wait(self) -> int:
            return 0

    def fake_popen(*args, **kwargs):
        assert not (output / "checkpoints").exists()
        return FakeProcess()

    monkeypatch.setattr(cli, "configure_local_cache", lambda: None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)

    cli.stage_train(config, iters=1)

    assert not (output / "checkpoints").exists()
    assert keep.read_text(encoding="utf-8") == "preserve"


def _write_pairs(path, rows) -> None:
    import json

    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_pairs_round_trips_and_strips_metadata(tmp_path) -> None:
    from local_llm_lab.pipeline.branch import render_completion
    from local_llm_lab.pipeline.prefer import load_pairs

    prompt = (
        "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n"
    )
    chosen = render_completion("note", Action("read_file", {"path": "a"}))
    rejected = render_completion("note", Action("read_file", {"path": "zzz"}))
    rows = [
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {"task_id": "t", "family": "read", "step": 0},
        },
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": "garbage<|im_end|>",
            "metadata": {"step": 1},
        },
    ]
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, rows)
    loaded = load_pairs(path)
    assert loaded == [
        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
        {"prompt": prompt, "chosen": chosen, "rejected": "garbage<|im_end|>"},
    ]
    assert all(set(row) == {"prompt", "chosen", "rejected"} for row in loaded)
    assert loaded[0]["chosen"].endswith("<|im_end|>") and loaded[0]["prompt"].endswith(
        "assistant\n"
    )


def test_load_pairs_rejects_malformed_rows_with_line_numbers(tmp_path) -> None:
    from local_llm_lab.pipeline.prefer import load_pairs

    good = {"prompt": "p", "chosen": "c", "rejected": "r"}
    path = tmp_path / "pairs.jsonl"

    _write_pairs(path, [good, {"prompt": "p", "chosen": "", "rejected": "r"}])
    with pytest.raises(ValueError, match=r":2: 'chosen' must be a non-empty string"):
        load_pairs(path)

    _write_pairs(path, [good, good, {"prompt": "p", "rejected": "r"}])
    with pytest.raises(ValueError, match=r":3: 'chosen'"):
        load_pairs(path)

    _write_pairs(path, [{"prompt": 5, "chosen": "c", "rejected": "r"}])
    with pytest.raises(ValueError, match=r":1: 'prompt'"):
        load_pairs(path)

    path.write_text('{"prompt": "p", "chosen": "c", "rejected": "r"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        load_pairs(path)

    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: expected a JSON object"):
        load_pairs(path)


def test_load_pairs_rejects_identical_chosen_and_rejected(tmp_path) -> None:
    from local_llm_lab.pipeline.prefer import load_pairs

    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [{"prompt": "p", "chosen": "same", "rejected": "same"}])
    with pytest.raises(ValueError, match=r":1: chosen and rejected are identical"):
        load_pairs(path)


def test_load_pairs_max_chars_drops_long_rows_only(tmp_path) -> None:
    from local_llm_lab.pipeline.prefer import load_pairs

    short = {"prompt": "pp", "chosen": "cc", "rejected": "rr"}  # 4 chars each side
    long_chosen = {"prompt": "pp", "chosen": "c" * 10, "rejected": "rr"}
    long_rejected = {"prompt": "pp", "chosen": "cc", "rejected": "r" * 10}
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [short, long_chosen, long_rejected, short])
    assert len(load_pairs(path)) == 4
    assert load_pairs(path, max_chars=4) == [short, short]
    assert load_pairs(path, max_chars=12) == [short, long_chosen, long_rejected, short]
    assert load_pairs(path, max_chars=3) == []


def test_prefer_config_builds_dpo_config_with_frozen_reference(tmp_path) -> None:
    from mlx_tune.rl_trainers import DPOConfig

    from local_llm_lab.pipeline.prefer import prefer_config

    config = prefer_config(
        beta=0.25,
        learning_rate=3e-6,
        max_steps=42,
        batch_size=2,
        max_seq_length=1024,
        output_dir=tmp_path / "trainer",
        grad_accumulation=8,
    )
    assert isinstance(config, DPOConfig)
    assert config.precompute_ref_logprobs is True
    assert config.beta == 0.25
    assert config.learning_rate == 3e-6
    assert config.max_steps == 42
    assert config.per_device_train_batch_size == 2
    assert config.gradient_accumulation_steps == 8
    assert config.max_seq_length == 1024
    assert config.loss_type == "sigmoid"
    assert config.output_dir == str(tmp_path / "trainer")
    assert not (tmp_path / "trainer").exists(), "builder is pure"


def test_finalize_adapter_copies_weights_and_source_config_verbatim(tmp_path) -> None:
    from local_llm_lab.pipeline.prefer import finalize_adapter

    source = tmp_path / "best-adapter"
    source.mkdir()
    config_text = '{\n    "fine_tune_type": "lora",\n    "num_layers": 36,\n    "lora_parameters": {"rank": 16, "scale": 32.0, "keys": ["self_attn.q_proj"]}\n}'
    (source / "adapter_config.json").write_text(config_text, encoding="utf-8")
    (source / "adapters.safetensors").write_bytes(b"OLD-SFT-WEIGHTS")

    trainer_dir = tmp_path / "trainer" / "adapters"
    trainer_dir.mkdir(parents=True)
    saved = trainer_dir / "adapters.safetensors"
    saved.write_bytes(b"NEW-DPO-WEIGHTS")
    (trainer_dir / "adapter_config.json").write_text('{"wrong": true}', encoding="utf-8")

    target = tmp_path / "prefer" / "adapters"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("x", encoding="utf-8")
    (target / "old-dir").mkdir()

    result = finalize_adapter(saved, source, target)
    assert result == target
    assert sorted(p.name for p in target.iterdir()) == [
        "adapter_config.json",
        "adapters.safetensors",
    ]
    assert (target / "adapters.safetensors").read_bytes() == b"NEW-DPO-WEIGHTS"
    assert (target / "adapter_config.json").read_text(encoding="utf-8") == config_text
    assert (source / "adapters.safetensors").read_bytes() == b"OLD-SFT-WEIGHTS", "source untouched"
    assert saved.read_bytes() == b"NEW-DPO-WEIGHTS", "trainer output untouched"

    # In-place finalisation (target is the trainer directory) keeps the weights and fixes the config.
    finalize_adapter(saved, source, trainer_dir)
    assert sorted(p.name for p in trainer_dir.iterdir()) == [
        "adapter_config.json",
        "adapters.safetensors",
    ]
    assert (trainer_dir / "adapter_config.json").read_text(encoding="utf-8") == config_text

    with pytest.raises(FileNotFoundError, match="weights"):
        finalize_adapter(tmp_path / "missing.safetensors", source, tmp_path / "t2")
    with pytest.raises(FileNotFoundError, match="config"):
        finalize_adapter(saved, tmp_path / "no-such-adapter", tmp_path / "t3")


def test_prefer_stage_is_registered_with_defaults(monkeypatch, tmp_path) -> None:
    from local_llm_lab.pipeline import cli

    calls: list[dict] = []
    monkeypatch.setattr(cli, "run_prefer", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path: {
            "model": "m",
            "output": tmp_path / "run",
            "seed": 7,
            "prefer": {"max_steps": 9},
        },
    )
    monkeypatch.setattr(sys, "argv", ["agent-pipeline", "prefer", "--beta", "0.2"])
    cli.main()
    (kwargs,) = calls
    assert kwargs["model_name"] == "m" and kwargs["seed"] == 7
    assert kwargs["output"] == tmp_path / "run" / "prefer"
    assert kwargs["adapter"] == tmp_path / "run" / "best-adapter"
    assert (
        kwargs["pairs_path"].name == "pairs.jsonl" and kwargs["pairs_path"].parent.name == "pref1"
    )
    assert kwargs["beta"] == 0.2 and kwargs["learning_rate"] == 5e-6
    assert kwargs["max_steps"] == 9, "config block overrides the fallback"
    assert kwargs["batch_size"] == 1 and kwargs["grad_accumulation"] == 4
    assert kwargs["max_seq_length"] == 3072 and kwargs["max_chars"] is None


def test_summary_table_lists_new_rows(tmp_path) -> None:
    from local_llm_lab.pipeline.transcript import summary_table

    summary = summarize(
        [
            _fake_trajectory("a", success=True, elapsed=1.0, tokens=80),
            _fake_trajectory(
                "b",
                success=False,
                reasons=("no finish call",),
                schema_failures=1,
                errors=1,
                elapsed=3.0,
                loop_detected=True,
                exhausted=True,
            ),
        ]
    )
    table = summary_table(summary)
    for row in (
        "tasks",
        "success",
        "clean success",
        "valid action rate",
        "schema validity",
        "executable calls",
        "tool errors",
        "recovered errors",
        "loop failures",
        "budget exhausted",
        "tokens/success",
        "latency p50/p95",
        "mean steps",
    ):
        assert f"  {row}" in table, row
    assert "87.5%" in table and "80.0" in table and "1.0s / 3.0s" in table
    assert "repetition loop" in table
    none_won = summary_table(summarize([_fake_trajectory("c", success=False, reasons=("x",))]))
    assert "tokens/success" in none_won and " -" in none_won


def test_tool_errors_never_escape_the_harness() -> None:
    """Model-generated arguments are untrusted: every failure must come back as an
    observation, not an exception. A base model passing prose to the calculator raises
    SyntaxError, which is not a ValueError and once aborted a whole evaluation run."""
    task = make_tasks("valid", 12)[0]
    simulator = Simulator.for_task(task)
    for expression in ("Status: queued", "import os", "1 +", "2 ** 99999999", "[]"):
        result = simulator.execute(Action("calculate", {"expression": expression}))
        assert result.startswith("ERROR"), (expression, result)
    assert simulator.errors == 5
    assert simulator.execute(Action("calculate", {"expression": "2 + 2"})) == "RESULT: 4"


# --------------------------------------------------------------------------- jlens


class _JLensBlock(nn.Module):
    """A tiny stand-in transformer block: nonlinear but cheap, called as ``(x, mask, cache)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)

    def __call__(self, x, mask=None, cache=None):
        del mask, cache
        return x + 0.1 * mx.tanh(self.linear(x))


class _JLensInner(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = [_JLensBlock(dim) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim)


class _JLensModel(nn.Module):
    """A fake tiny model with the ``model.{embed_tokens,layers,norm}`` shape jlens expects."""

    def __init__(self, vocab: int = 40, dim: int = 16, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _JLensInner(vocab, dim, n_layers)


def _jlens_model(seed: int = 0) -> _JLensModel:
    mx.random.seed(seed)
    return _JLensModel()


class _JLensTokenizer:
    """Deterministic word-hash tokenizer: no real vocabulary, just stable ids for readouts."""

    def __init__(self, vocab_size: int = 40) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        words = text.split() or [text]
        return [sum(map(ord, word)) % self.vocab_size for word in words]

    def decode(self, ids: list[int]) -> str:
        return "".join(f"<{token_id}>" for token_id in ids)


class _MappedTokenizer:
    """A tokenizer with an explicit string -> ids table, for token_evidence rank assertions."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.mapping[text])

    def decode(self, ids: list[int]) -> str:
        return "".join(f"<{token_id}>" for token_id in ids)


def test_residual_at_returns_float32_and_matches_embedding_at_layer_zero() -> None:
    model = _jlens_model()
    ids = [1, 5, 3, 7, 2]
    residual = jlens.residual_at(model, ids, 2)
    assert residual.shape == (1, len(ids), 16)
    assert residual.dtype == mx.float32
    embedded = model.model.embed_tokens(mx.array(ids)[None, :]).astype(mx.float32)
    assert bool(mx.allclose(jlens.residual_at(model, ids, 0), embedded).item())


def test_jacobian_vector_product_matches_finite_difference() -> None:
    model = _jlens_model()
    ids = [1, 5, 3, 7, 2]
    primal = jlens.residual_at(model, ids, 1)
    mx.random.seed(7)
    tangent = mx.random.normal(primal.shape)
    tangent = tangent / mx.sqrt(mx.sum(tangent * tangent))
    computed = jlens.jacobian_vector_product(model, 1, primal, tangent)

    def fn(x):
        h = x
        for layer in model.model.layers[1:]:
            h = layer(h, None, None)
        return model.model.norm(h)

    eps = 1e-3
    finite_diff = (fn(primal + eps * tangent) - fn(primal)) / eps
    relative_error = mx.sqrt(mx.sum((finite_diff - computed) ** 2) / mx.sum(computed**2))
    assert float(relative_error.item()) < 1e-2


def test_jlens_map_is_approximately_linear() -> None:
    model = _jlens_model()
    layer = 2
    probe = jlens.residual_at(model, [4, 4, 8, 2], layer)[0, -1]
    corpus_ids = [[2, 4, 6], [1, 1, 3, 5]]
    once = jlens.jlens_map(model, layer, probe, corpus_ids)
    twice = jlens.jlens_map(model, layer, 2 * probe, corpus_ids)
    assert bool(mx.allclose(twice, 2 * once, atol=1e-3, rtol=1e-3).item())


def test_readout_is_sorted_and_a_valid_distribution() -> None:
    model = _jlens_model()
    tokenizer = _JLensTokenizer()
    vector = jlens.residual_at(model, [1, 2, 3], 2)[0, -1]
    top = jlens.readout(model, vector, tokenizer, k=10)
    assert len(top) == 10
    probabilities = [probability for _, probability, _ in top]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    assert sum(probabilities) <= 1.0 + 1e-6


def test_logit_lens_is_the_same_machinery_without_a_jacobian() -> None:
    model = _jlens_model()
    tokenizer = _JLensTokenizer()
    vector = jlens.residual_at(model, [1, 2, 3], 2)[0, -1]
    assert jlens.logit_lens(model, vector, tokenizer, k=5) == jlens.readout(
        model, vector, tokenizer, k=5
    )


def test_token_evidence_finds_rank_one_on_a_one_hot_distribution() -> None:
    tokenizer = _MappedTokenizer({"needle": [3]})
    distribution = mx.zeros((6,))
    distribution[3] = 1.0
    evidence = jlens.token_evidence(distribution, tokenizer, {"needle": "needle"})
    assert evidence["needle"]["best_rank"] == 1
    assert evidence["needle"]["max_probability"] == 1.0
    assert evidence["needle"]["total_probability"] == 1.0


def test_token_evidence_reports_best_rank_for_a_known_ordering() -> None:
    tokenizer = _MappedTokenizer({"low": [4], "high": [0], "multi": [0, 4]})
    distribution = mx.array([0.5, 0.3, 0.1, 0.06, 0.03, 0.01])
    evidence = jlens.token_evidence(
        distribution, tokenizer, {"low": "low", "high": "high", "multi": "multi"}
    )
    assert evidence["high"]["best_rank"] == 1
    assert evidence["low"]["best_rank"] == 5
    assert evidence["multi"]["best_rank"] == 1
    assert evidence["multi"]["total_probability"] == pytest.approx(0.53)


def test_probe_layers_returns_one_record_per_layer_with_both_lenses() -> None:
    model = _jlens_model()
    tokenizer = _JLensTokenizer()
    ids = [1, 5, 3, 7, 2]
    corpus_ids = [[2, 4, 6], [1, 1, 3, 5]]
    candidates = {"a": "hello", "b": "world"}
    records = jlens.probe_layers(model, tokenizer, ids, [0, 1, 3], corpus_ids, candidates, k=5)
    assert [record["layer"] for record in records] == [0, 1, 3]
    for record in records:
        assert len(record["jlens_top_k"]) == 5
        assert len(record["logit_lens_top_k"]) == 5
        assert set(record["jlens_evidence"]) == {"a", "b"}
        assert set(record["logit_lens_evidence"]) == {"a", "b"}
