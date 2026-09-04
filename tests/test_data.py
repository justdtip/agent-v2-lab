from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict

import pytest

from local_llm_lab.agent_protocol import Action
from local_llm_lab.models import ChatSpec, LoraSpec, ModelSpec, load_model_spec
from local_llm_lab.pipeline import data as data_module
from local_llm_lab.pipeline.data import (
    PROTECTED_DATASETS,
    DatasetRenderError,
    DatasetWriteGuardError,
    ProtectedDatasetError,
    SplitSpec,
    build_rows,
    guard_dataset_write,
    read_jsonl,
    render_dataset,
    render_rows,
    write_dataset,
    write_jsonl,
)
from local_llm_lab.pipeline.protocol import assistant_message, generation_suffix
from local_llm_lab.pipeline.tasks import GENERATOR_VERSION, make_tasks
from local_llm_lab.project import PROJECT_ROOT
from local_llm_lab.tuner_data import RenderedRowsDataset


def _legacy_spec() -> ModelSpec:
    return ModelSpec(
        name="legacy",
        hf_id="legacy",
        family="legacy",
        chat=ChatSpec("unsupported", {}, "<eot>", ()),
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
        cache_equivalence_verified=None,
    )


def _thinking_spec(mode: str) -> ModelSpec:
    return ModelSpec(
        name=mode,
        hf_id=mode,
        family=mode,
        chat=ChatSpec(mode, {"enable_thinking": True}, "<eot>", ()),  # type: ignore[arg-type]
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
        cache_equivalence_verified=None,
    )


class _LegacyTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs) -> str:
        assert not tokenize and not kwargs
        rendered = "".join(f"{message['role']}:{message['content']}\n" for message in messages)
        return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


class _ThinkingTokenizer(_LegacyTokenizer):
    def __init__(self) -> None:
        self.template_kwargs: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs) -> str:
        self.template_kwargs.append(kwargs)
        rendered = "".join(f"{message['role']}:{message['content']}\n" for message in messages)
        if not add_generation_prompt:
            return rendered
        suffix = "<|im_start|>assistant\n"
        return rendered + (
            suffix + "<think>\n\n</think>\n\n"
            if kwargs["enable_thinking"] is False
            else suffix
        )


def _target_spec() -> ModelSpec:
    """A second registry-shaped spec whose rendering is visibly different from ``_legacy_spec``."""
    return ModelSpec(
        name="target",
        hf_id="target",
        family="target",
        chat=ChatSpec("unsupported", {}, "<eot2>", ()),
        lora=LoraSpec("attention+mlp", 1, 1.0, 0.0),
        train={},
        cache_strategy="none",
        probe_layer_fractions=(1.0,),
        memory_budget_gib=1.0,
        policies={},
        cache_equivalence_verified=None,
    )


class _TargetTokenizer(_LegacyTokenizer):
    """A chat template unlike the legacy one, so a re-render is detectable in the prompt.

    The generation prompt still ends with the protocol's generation suffix, which
    ``build_prompt`` asserts for every non-compatibility specification.
    """

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs) -> str:
        assert not tokenize and not kwargs
        rendered = "".join(f"[{message['role']}]{message['content']}\n" for message in messages)
        return rendered + (generation_suffix(_target_spec()) if add_generation_prompt else "")


class _Qwen25TemplateTokenizer:
    """Byte-level fake for the registered 3B Qwen2.5 chat-template contract."""

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs) -> str:
        assert not tokenize and not kwargs
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def _legacy_qwen25_messages(messages: list[dict[str, object]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )


def test_render_rows_adds_canonical_fields_without_mutating_messages() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant_message("note", Action("finish", {"answer": "done"})),
        ],
        "metadata": {"task_id": "fake"},
    }
    original = copy.deepcopy(row)

    rendered = render_rows([row], _LegacyTokenizer(), spec=_legacy_spec())

    assert row == original
    assert rendered[0]["messages"] == original["messages"]
    assert rendered[0]["metadata"] == {"task_id": "fake"}
    assert rendered[0]["prompt"] == "system:rules\nuser:task\n<|im_start|>assistant\n"
    assert rendered[0]["completion"] == (
        'note\n```json\n{"name": "finish", "arguments": {"answer": "done"}}\n```<eot>\n'
    )


def test_write_jsonl_atomically_replaces_payloads_at_or_above_one_mebibyte(
    monkeypatch, tmp_path
) -> None:
    destination = tmp_path / "rows.jsonl"
    replacements: list[tuple[str, str]] = []
    original_replace = os.replace

    def record_replace(source: str, target: str) -> None:
        replacements.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", record_replace)
    digest = write_jsonl(destination, [{"payload": "x" * (1024 * 1024)}])

    assert len(replacements) == 1
    assert replacements[0][1] == str(destination)
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_write_jsonl_atomically_replaces_sub_mebibyte_array_payloads(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "rows.jsonl"
    replacements: list[tuple[str, str]] = []
    original_replace = os.replace

    def record_replace(source: str, target: str) -> None:
        replacements.append((source, target))
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", record_replace)

    write_jsonl(destination, [{"messages": [{"role": "user", "content": "small"}]}])

    assert len(replacements) == 1
    assert replacements[0][1] == str(destination)


def test_write_dataset_emits_rendered_rows_when_given_a_tokenizer_and_spec(tmp_path) -> None:
    write_dataset(
        tmp_path,
        {"train": 12},
        tokenizer=_LegacyTokenizer(),
        spec=_legacy_spec(),
    )

    row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"messages", "metadata", "prompt", "completion"} <= set(row)


def test_write_dataset_records_the_effective_inference_training_renderer(tmp_path) -> None:
    manifest = write_dataset(
        tmp_path,
        {"train": 12},
        tokenizer=_ThinkingTokenizer(),
        spec=_thinking_spec("inference"),
    )

    assert manifest["rendering"]["thinking"] == "off"
    assert manifest["rendering"]["template_kwargs"] == {"enable_thinking": False}


def test_write_dataset_records_the_complete_unresolved_spec_without_resolving(
    monkeypatch, tmp_path
) -> None:
    spec = _thinking_spec("inference")

    def forbid_resolve(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("write_dataset must not resolve a model specification")

    monkeypatch.setattr(ModelSpec, "resolve", forbid_resolve)

    manifest = write_dataset(
        tmp_path,
        {"train": 12},
        tokenizer=_ThinkingTokenizer(),
        spec=spec,
    )

    assert manifest["model"] == asdict(spec)
    assert manifest["rendering"]["thinking"] == "off"


def test_current_generator_qwen25_messages_migrate_to_identical_rendered_tokens() -> None:
    assert GENERATOR_VERSION == 4
    tokenizer = _Qwen25TemplateTokenizer()
    spec = load_model_spec("qwen25-coder-3b")
    row = build_rows(make_tasks("train", 12, seed=20260902)[0])[0]
    legacy_render = _legacy_qwen25_messages(row["messages"])
    legacy_tokens = list(legacy_render.encode("utf-8"))

    rendered = render_rows([row], tokenizer, spec=spec)[0]
    dataset = RenderedRowsDataset(
        [rendered], tokenizer, max_seq_length=len(legacy_tokens) + 1
    )
    tokens, offset = dataset[0]

    assert rendered["prompt"] + rendered["completion"] == legacy_render
    assert tokens == legacy_tokens
    assert offset == len(rendered["prompt"].encode("utf-8"))


def test_chat_rows_migrate_to_identical_rendered_tokens_under_the_template() -> None:
    """The verbatim chat fallback reproduces the legacy template render, like expert rows do."""
    tokenizer = _Qwen25TemplateTokenizer()
    spec = load_model_spec("qwen25-coder-3b")
    row = {
        "messages": [
            {"role": "user", "content": "please summarise the log"},
            {"role": "assistant", "content": "A plain reply with no tool call."},
        ],
        "metadata": {"source": "pre-expansion-policy-replay"},
    }
    legacy_render = _legacy_qwen25_messages(row["messages"])
    legacy_tokens = list(legacy_render.encode("utf-8"))

    rendered = render_rows([row], tokenizer, spec=spec)[0]
    dataset = RenderedRowsDataset(
        [rendered], tokenizer, max_seq_length=len(legacy_tokens) + 1
    )
    tokens, offset = dataset[0]

    assert rendered["prompt"] + rendered["completion"] == legacy_render
    assert tokens == legacy_tokens
    assert offset == len(rendered["prompt"].encode("utf-8"))


def test_render_rows_disables_inference_thinking_and_preserves_trained_reasoning() -> None:
    action = Action("finish", {"answer": "done"})
    thought = (
        "<think>reason</think>\n\nnote\n```json\n"
        '{"name": "finish", "arguments": {"answer": "done"}}\n```'
    )
    row = {
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": thought},
        ],
        "metadata": {},
    }
    inference_tokenizer = _ThinkingTokenizer()

    inference = render_rows([row], inference_tokenizer, spec=_thinking_spec("inference"))[0]
    trained = render_rows([row], _ThinkingTokenizer(), spec=_thinking_spec("trained"))[0]

    assert inference_tokenizer.template_kwargs == [{"enable_thinking": False}]
    assert "<think>reason</think>" not in inference["completion"]
    assert trained["completion"] == (
        "<think>reason</think>\n\n"
        + assistant_message("note", action)["content"]
        + "<eot>\n"
    )


def test_recovery_rows_are_marked_and_oversampled_in_train_only(tmp_path) -> None:
    task = next(t for t in make_tasks("train", 144) if t.variant == "failed_edit")
    rows = build_rows(task)
    marked = [row for row in rows if row["metadata"]["recovery"]]
    assert len(marked) == 1, "exactly the step after the deliberate failure is a recovery"

    manifest = write_dataset(tmp_path, {"train": 24, "valid": 12, "test": 12}, recovery_repeats=5)
    train = manifest["splits"]["train"]
    assert train["recovery_rows_after_repeats"] == train["recovery_targets"] * 5
    for split in ("valid", "test"):
        held = manifest["splits"][split]
        assert held["recovery_rows_after_repeats"] == held["recovery_targets"]


def test_split_specs_aggregate_named_chunks_by_role_deterministically(tmp_path) -> None:
    """Catch lost difficulty/perturb flags or accidental cross-role concatenation."""
    splits = {
        "train": SplitSpec(2, difficulty=0, perturb=True, role="train"),
        "train1": SplitSpec(2, difficulty=1, perturb=True, role="train"),
        "valid": SplitSpec(2, difficulty=1, perturb=False, role="valid"),
        "valid2": SplitSpec(2, difficulty=2, perturb=False, role="valid"),
        "test": SplitSpec(2, difficulty=2, perturb=False, role="test"),
        "test3": SplitSpec(2, difficulty=3, perturb=False, role="test"),
    }
    manifest = write_dataset(tmp_path, splits, recovery_repeats={"wrong_path": 2})

    split_metadata = {
        (name, info["difficulty"], info["perturb"], info["role"])
        for name, info in manifest["splits"].items()
    }
    assert split_metadata == {
        ("train", 0, True, "train"),
        ("train1", 1, True, "train"),
        ("valid", 1, False, "valid"),
        ("valid2", 2, False, "valid"),
        ("test", 2, False, "test"),
        ("test3", 3, False, "test"),
    }
    roles = {
        "train": {"train", "train1"},
        "valid": {"valid", "valid2"},
        "test": {"test", "test3"},
    }
    for role, names in roles.items():
        task_ids = {
            row["metadata"]["task_id"].split("-", 1)[0]
            for row in read_jsonl(tmp_path / f"{role}.jsonl")
            if row["metadata"]["source"] == "expert"
        }
        assert task_ids <= names
        assert task_ids == names
        assert manifest["outputs"][role]["rows"] == len(read_jsonl(tmp_path / f"{role}.jsonl"))


def _logical_hash(rows: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_split_specs_preserve_chunk_boundaries_effects_hashes_and_recovery_repeats(
    tmp_path,
) -> None:
    """Independently prove the six logical chunks concatenate in declaration order."""
    splits = {
        "train": SplitSpec(240, difficulty=0, perturb=True, role="train"),
        "train1": SplitSpec(120, difficulty=1, perturb=True, role="train"),
        "valid": SplitSpec(24, difficulty=1, perturb=False, role="valid"),
        "valid2": SplitSpec(24, difficulty=2, perturb=False, role="valid"),
        "test": SplitSpec(180, difficulty=2, perturb=False, role="test"),
        "test3": SplitSpec(60, difficulty=3, perturb=False, role="test"),
    }
    repeats = {
        "transient": 1,
        "wrong_path": 2,
        "unknown_tool": 2,
        "stale_path": 6,
        "failed_edit": 6,
    }
    first = write_dataset(tmp_path / "one", splits, recovery_repeats=repeats)
    second = write_dataset(tmp_path / "two", splits, recovery_repeats=repeats)

    for role, chunk_names in {
        "train": ("train", "train1"),
        "valid": ("valid", "valid2"),
        "test": ("test", "test3"),
    }.items():
        first_bytes = (tmp_path / "one" / f"{role}.jsonl").read_bytes()
        assert first_bytes == (tmp_path / "two" / f"{role}.jsonl").read_bytes()
        rows = read_jsonl(tmp_path / "one" / f"{role}.jsonl")
        offset = 0
        for name in chunk_names:
            info = first["splits"][name]
            chunk = rows[offset : offset + info["rows"]]
            offset += info["rows"]
            spec = splits[name]
            expected_tasks = make_tasks(
                name,
                spec.count,
                difficulty=spec.difficulty,
                perturb=spec.perturb,
            )
            expected_rows = [row for task in expected_tasks for row in build_rows(task)]
            recovery_by_variant = Counter(
                row["metadata"]["variant"]
                for row in expected_rows
                if row["metadata"]["recovery"]
            )
            recovery_multipliers = repeats if spec.role == "train" else {}
            expected_repeated_recovery = Counter(
                {
                    variant: count * recovery_multipliers.get(variant, 1)
                    for variant, count in recovery_by_variant.items()
                }
            )
            assert len(chunk) == info["rows"]
            assert info["count"] == spec.count
            assert info["tasks"] == spec.count
            assert {task.task_id for task in expected_tasks} == {
                row["metadata"]["task_id"] for row in chunk
            }
            assert {row["metadata"]["task_id"].split("-", 1)[0] for row in chunk} == {name}
            assert _logical_hash(chunk) == info["sha256"]
            assert {task.difficulty for task in expected_tasks} == {spec.difficulty}
            assert {row["metadata"]["difficulty"] for row in chunk} == {spec.difficulty}
            assert {row["metadata"]["perturb"] for row in chunk} == {spec.perturb}
            if info["perturb"]:
                assert {row["metadata"]["variant"] for row in chunk} != {"clean"}
            assert info["variants"] == dict(
                sorted(Counter(task.variant for task in expected_tasks).items())
            )
            assert info["families"] == dict(
                sorted(Counter(task.family for task in expected_tasks).items())
            )
            horizons = [task.horizon for task in expected_tasks]
            assert info["min_horizon"] == min(horizons)
            assert info["max_horizon"] == max(horizons)
            assert info["mean_horizon"] == round(sum(horizons) / len(horizons), 2)
            assert info["recovery_targets"] == sum(recovery_by_variant.values())
            assert info["recovery_rows_after_repeats"] == sum(expected_repeated_recovery.values())
            assert Counter(
                row["metadata"]["variant"]
                for row in chunk
                if row["metadata"]["recovery"]
            ) == expected_repeated_recovery
        assert offset == len(rows)
        assert hashlib.sha256(first_bytes).hexdigest() == first["outputs"][role]["sha256"]

    for name in ("train", "train1"):
        info = first["splits"][name]
        assert info["recovery_targets"] > 0
        assert info["recovery_rows_after_repeats"] > info["recovery_targets"]
    assert first == second
    assert (tmp_path / "one" / "manifest.json").read_bytes() == (
        tmp_path / "two" / "manifest.json"
    ).read_bytes()


def test_split_specs_add_role_replay_once_and_reject_invalid_values(tmp_path) -> None:
    """Catch replay duplication on secondary chunks and malformed split declarations."""
    chat = tmp_path / "chat"
    chat.mkdir()
    for role in ("train", "valid", "test"):
        write_jsonl(chat / f"{role}.jsonl", [{"metadata": {"source": "chat", "role": role}}])
    extra = tmp_path / "extra"
    extra.mkdir()
    write_jsonl(extra / "train.jsonl", [{"metadata": {"source": "extra"}}])
    splits = {
        "train": SplitSpec(1, role="train"), "train1": SplitSpec(1, role="train"),
        "valid": SplitSpec(1, role="valid"), "valid2": SplitSpec(1, role="valid"),
        "test": SplitSpec(1, role="test"), "test3": SplitSpec(1, role="test"),
    }
    manifest = write_dataset(tmp_path / "out", splits, chat_dir=chat, extra_dirs=[extra])
    assert {
        role: manifest["outputs"][role]["chat_rows"]
        for role in ("train", "valid", "test")
    } == {"train": 1, "valid": 1, "test": 1}
    assert manifest["outputs"]["train"]["extra_rows"] == 1
    with pytest.raises(ValueError):
        SplitSpec(0)
    with pytest.raises(ValueError):
        SplitSpec(1, difficulty=-1)
    with pytest.raises(ValueError):
        SplitSpec(1, perturb="yes")  # type: ignore[arg-type]


def test_write_dataset_refuses_an_existing_manifest_unless_forced_and_records_it(tmp_path) -> None:
    """R21(a): the write boundary itself refuses to clobber a dataset that has a manifest."""
    target = tmp_path / "out"
    first = write_dataset(target, {"train": 1, "valid": 1, "test": 1})
    assert "force_overwrite" not in first

    with pytest.raises(DatasetWriteGuardError, match="--force-overwrite"):
        write_dataset(target, {"train": 1, "valid": 1, "test": 1})

    replaced = write_dataset(target, {"train": 1, "valid": 1, "test": 1}, overwrite=True)
    assert replaced["force_overwrite"] is True
    written = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert written["force_overwrite"] is True


def test_protected_datasets_refuse_writes_even_with_the_override(monkeypatch, tmp_path) -> None:
    """The four irreplaceable directories have no override path through any writer."""
    monkeypatch.setattr(data_module, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "safe-src"
    write_dataset(source, {"train": 1, "valid": 1, "test": 1})
    for name in sorted(PROTECTED_DATASETS):
        target = tmp_path / name
        with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
            write_dataset(target, {"train": 1}, overwrite=True)
        with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
            render_dataset(source, target, _TargetTokenizer(), spec=_target_spec(), overwrite=True)
        assert not target.exists(), "the refusal must fire before anything is written"


def test_protected_datasets_guard_the_real_repository_paths() -> None:
    """Pin the R21 protected list and prove the guard rejects the real directories."""
    assert PROTECTED_DATASETS == {
        "data/agent_v2",
        "data/agent_v2b",
        "data/agent_v2c",
        "data/chat_replay",
    }
    for name in sorted(PROTECTED_DATASETS):
        with pytest.raises(ProtectedDatasetError):
            guard_dataset_write(PROJECT_ROOT / name, overwrite=True)
        with pytest.raises(ProtectedDatasetError):
            guard_dataset_write(PROJECT_ROOT / name / "any" / "subpath", overwrite=True)


def test_protected_guard_refuses_subpaths_of_protected_datasets(monkeypatch, tmp_path) -> None:
    """Any child of a protected directory refuses even when forced: pollution is impossible.

    The parent walk uses the same file-identity comparison, so a mis-cased parent on a
    case-insensitive filesystem is caught too. Temp protected-root fixture only.
    """
    monkeypatch.setattr(data_module, "PROJECT_ROOT", tmp_path)
    real = tmp_path / "data" / "agent_v2b"
    real.mkdir(parents=True)

    child = real / "nested" / "deeper"
    with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
        guard_dataset_write(child, overwrite=True)
    with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
        write_dataset(child, {"train": 1}, overwrite=True)

    miscased_parent = tmp_path / "data" / "AGENT_V2B"
    if miscased_parent.exists():
        # Case-insensitive filesystem: the mis-cased parent names the protected directory.
        with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
            guard_dataset_write(miscased_parent / "sub", overwrite=True)
    assert list(real.iterdir()) == [], "no refusal may leave anything behind in the dataset"


def test_protected_guard_refuses_aliased_paths_by_file_identity(monkeypatch, tmp_path) -> None:
    """A mis-cased or symlinked alias of a protected directory refuses even when forced.

    ``resolve()`` preserves the case the caller typed, so on a case-insensitive filesystem a
    mis-cased path names the protected directory while comparing unequal as a path; the
    guard must match by on-disk identity. Exercised against a temp protected root only —
    never the real data/.
    """
    monkeypatch.setattr(data_module, "PROJECT_ROOT", tmp_path)
    real = tmp_path / "data" / "agent_v2b"
    real.mkdir(parents=True)

    miscased = tmp_path / "data" / "AGENT_V2B"
    if miscased.exists():
        # Case-insensitive filesystem: the mis-cased alias names the same directory.
        with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
            guard_dataset_write(miscased, overwrite=True)
        with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
            write_dataset(miscased, {"train": 1}, overwrite=True)
    else:
        # Case-sensitive filesystem: a genuinely different directory stays writable.
        assert guard_dataset_write(miscased) is False

    link = tmp_path / "aliased-by-symlink"
    link.symlink_to(real)
    with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
        guard_dataset_write(link, overwrite=True)
    with pytest.raises(ProtectedDatasetError, match="irreplaceable"):
        write_dataset(link, {"train": 1}, overwrite=True)
    assert list(real.iterdir()) == [], "no refusal may leave anything behind in the dataset"


def test_render_dataset_carries_task_content_verbatim_and_rerenders_only(
    monkeypatch, tmp_path
) -> None:
    """R21(b): rendering changes prompt/completion only; rows and generator version carry over."""
    source = tmp_path / "src"
    write_dataset(
        source,
        {"train": 1, "valid": 1, "test": 1},
        tokenizer=_LegacyTokenizer(),
        spec=_legacy_spec(),
    )
    manifest_path = source / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest["generator_version"] = 2  # simulate a source from an older generator
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        data_module, "make_tasks", lambda *args, **kwargs: pytest.fail("render regenerated tasks")
    )

    output = tmp_path / "dst"
    manifest = render_dataset(source, output, _TargetTokenizer(), spec=_target_spec())

    assert manifest["generator_version"] == 2
    assert manifest["generator_version"] != GENERATOR_VERSION
    assert manifest["seed"] == source_manifest["seed"]
    assert manifest["source"]["directory"] == str(source.resolve())
    assert (
        manifest["source"]["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert manifest["model"] == asdict(_target_spec())
    assert manifest["rendering"]["generation_suffix"] == generation_suffix(_target_spec())
    assert "force_overwrite" not in manifest
    for role in ("train", "valid", "test"):
        source_bytes = (source / f"{role}.jsonl").read_bytes()
        assert manifest["source"]["sha256"][role] == hashlib.sha256(source_bytes).hexdigest()
        source_rows = read_jsonl(source / f"{role}.jsonl")
        rendered_rows = read_jsonl(output / f"{role}.jsonl")
        assert len(rendered_rows) == len(source_rows) == manifest["outputs"][role]["rows"]
        for original, rendered in zip(source_rows, rendered_rows):
            payloads = []
            for row in (original, rendered):
                stripped = {
                    key: value
                    for key, value in row.items()
                    if key not in {"prompt", "completion"}
                }
                payloads.append(
                    json.dumps(stripped, ensure_ascii=False, separators=(",", ":"))
                )
            assert payloads[0] == payloads[1], "task content must carry over byte-identical"
            assert rendered["prompt"].startswith("[system]")
            assert rendered["prompt"].endswith(generation_suffix(_target_spec()))
            assert rendered["prompt"] != original["prompt"]
            assert rendered["completion"].endswith("<eot2>\n")
        assert (
            manifest["outputs"][role]["sha256"]
            == hashlib.sha256((output / f"{role}.jsonl").read_bytes()).hexdigest()
        )


def test_render_dataset_guards_output_and_requires_a_complete_source(tmp_path) -> None:
    """The render output obeys the same guard, and a partial source fails before any write."""
    source = tmp_path / "src"
    write_dataset(source, {"train": 1, "valid": 1, "test": 1})
    output = tmp_path / "dst"
    render_dataset(source, output, _TargetTokenizer(), spec=_target_spec())

    with pytest.raises(DatasetWriteGuardError, match="manifest.json"):
        render_dataset(source, output, _TargetTokenizer(), spec=_target_spec())
    forced = render_dataset(
        source, output, _TargetTokenizer(), spec=_target_spec(), overwrite=True
    )
    assert forced["force_overwrite"] is True
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["force_overwrite"] is True

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    write_jsonl(incomplete / "train.jsonl", [{"messages": [], "metadata": {}}])
    fresh = tmp_path / "fresh"
    with pytest.raises(DatasetRenderError, match="valid.jsonl"):
        render_dataset(incomplete, fresh, _TargetTokenizer(), spec=_target_spec())
    assert not fresh.exists()
    with pytest.raises(DatasetRenderError, match="not a directory"):
        render_dataset(tmp_path / "absent", fresh, _TargetTokenizer(), spec=_target_spec())


def test_render_dataset_without_a_source_manifest_records_unknown_provenance(tmp_path) -> None:
    """A pre-versioning source renders, carrying explicit nulls instead of current values."""
    source = tmp_path / "src"
    row = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant_message("note", Action("finish", {"answer": "done"})),
        ],
        "metadata": {"task_id": "fake", "source": "expert"},
    }
    for role in ("train", "valid", "test"):
        write_jsonl(source / f"{role}.jsonl", [row])

    manifest = render_dataset(source, tmp_path / "dst", _TargetTokenizer(), spec=_target_spec())

    assert manifest["generator_version"] is None
    assert "manifest_sha256" not in manifest["source"]
    assert manifest["outputs"]["train"]["rows"] == 1


def test_render_rows_verbatim_chat_fallback_is_an_explicit_allowlist() -> None:
    """Only allowlisted replay sources render verbatim; anything else fails loudly.

    In a controlled arm, silent is the failure mode to fear: a rollout or expert row that
    somehow lost its tool call must raise, never be rendered as chat.
    """
    assert data_module.CHAT_COMPLETION_SOURCES == {"pre-expansion-policy-replay"}
    chat_row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "plain reply with no tool call"},
        ],
        "metadata": {"source": "pre-expansion-policy-replay"},
    }

    rendered = render_rows([chat_row], _LegacyTokenizer(), spec=_legacy_spec())[0]

    assert rendered["completion"] == "plain reply with no tool call<eot>\n"
    assert rendered["messages"] == chat_row["messages"]

    for source in ("expert", "rollout", "chat", None):
        broken = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "no call here either"},
            ],
            "metadata": {"source": source, "task_id": "train-read-0000-clean"},
        }
        with pytest.raises(ValueError, match="parseable tool call"):
            render_rows([broken], _LegacyTokenizer(), spec=_legacy_spec())
    with pytest.raises(ValueError, match="parseable tool call"):
        render_rows(
            [{"messages": chat_row["messages"]}], _LegacyTokenizer(), spec=_legacy_spec()
        )
