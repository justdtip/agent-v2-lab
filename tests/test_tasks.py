from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_llm_lab.pipeline import tasks as task_module
from local_llm_lab.pipeline.cli import load_config, stage_data
from local_llm_lab.pipeline.data import write_dataset
from local_llm_lab.pipeline.tasks import (
    FAMILIES,
    GENERATOR_VERSION,
    LONG_HORIZON_FAMILIES,
    family_balanced_tasks,
    make_tasks,
    task_from_id,
)

_REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "agent_v2c.yaml"
_REPIN_MESSAGE = (
    "Intentional generator row change: bump GENERATOR_VERSION and re-pin both the "
    "generator-only and configured-replay hash oracles."
)


def test_data_stage_writes_generator_version_provenance(tmp_path) -> None:
    """The data stage must record its generator without resolving or loading model weights."""
    config = {
        "model": "qwen25-coder-3b",
        "output": tmp_path / "output",
        "data": tmp_path / "data",
        "seed": 20260902,
        "keep_last": 2,
        "tasks": {"train": 1, "valid": 1, "test": 1},
        "chat_replay": None,
        "chat_repeats": 1,
        "recovery_repeats": 1,
    }

    stage_data(config, [])

    provenance_path = config["output"] / "provenance.json"
    assert provenance_path.is_file()
    assert json.loads(provenance_path.read_text(encoding="utf-8"))["generator_version"] == (
        GENERATOR_VERSION
    )


def test_dataset_manifest_records_generator_version(tmp_path) -> None:
    """Every generated manifest, returned and persisted, must identify its row generator."""
    manifest = write_dataset(tmp_path, {"train": 1, "valid": 1, "test": 1})

    assert manifest["generator_version"] == GENERATOR_VERSION
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written["generator_version"] == GENERATOR_VERSION


def test_reference_generator_hashes_are_pinned_by_version(tmp_path) -> None:
    """A task/expert-row change must bump the generator version and both hash oracles."""
    config = load_config(_REFERENCE_CONFIG)
    manifest = write_dataset(
        tmp_path,
        config["tasks"],
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=None,
        recovery_repeats=config["recovery_repeats"],
    )
    expected = {
        2: {
            "train": "e7fa63ef9a2fe70b3563a23fa5421b4a6b11d11971802d2c4b6bce5a0a3c5d58",
            "valid": "816543d1dee8299b2dbf514e93a2de480b69f84d6e8c53bda955c20c79f6213f",
            "test": "10042da5d9a4789f9a28fc6c0c9a8ea8efc59d090688f1e167c6de8d1de66877",
        },
        3: {
            "train": "6a2875aff061e7dcdde0b8a394fc3115ff5c6f2b4dbcb8f399db5af373c6970e",
            "valid": "6df3a890d09e989c83baaf6078a27d56ec9da035807c8820e83b44c09a8259e3",
            "test": "925f5b282885e4c52f2a20280372804a17dfc9c2f1d8039478de98211eee1303",
        },
        4: {
            "train": "99176c0b378d85502e738f23b8d174dd8321cb48a51e10c3a9bcd7fd9db3f035",
            "valid": "41ec07d6f122a123ee7780885d59ba2bc77ca5ce836e896ed8ddd2861f680deb",
            "test": "8b8aeaf28460798e0863ab0e5cb217da25b8661b7b3a802d21182fcd36a294a7",
        },
    }

    assert manifest["generator_version"] == GENERATOR_VERSION
    assert {
        split: info["sha256"] for split, info in manifest["splits"].items()
    } == expected.get(GENERATOR_VERSION), _REPIN_MESSAGE


def test_reference_generator_hashes_include_configured_replay(tmp_path) -> None:
    """The shipped replay mix stays pinned when its protected input directory is available."""
    config = load_config(_REFERENCE_CONFIG)
    replay_dir = config["chat_replay"]
    if not replay_dir.is_dir():
        pytest.skip(f"configured protected replay directory is absent: {replay_dir}")
    manifest = write_dataset(
        tmp_path,
        config["tasks"],
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=config["chat_replay"],
        chat_repeats=config["chat_repeats"],
        recovery_repeats=config["recovery_repeats"],
    )
    expected = {
        2: {
            "train": "ad660e83cd89958dcee9fba2ab1e53115d1fb079813ea694cd4b0e89530b3a79",
            "valid": "d6dbc53573744751d74565a0de6ca5c6d381cba6b488ff6410194bf9b0d4e6d8",
            "test": "fc69b03fef8f423ee85a174214ad955fe3f4d324554217510b92f53435841ce3",
        },
        3: {
            "train": "92f40d1868438553f306be192b09cace7b3d5efc8cff4a55c84fc3a96827ddf2",
            "valid": "6e2617d5159e42fc7d26077e332a83e9f27fc0e755d50140d6336c75116c44b7",
            "test": "f79d6fc673674719cc17664278a15ee8ba9fcb6de8f9418d9226b99b5d24de59",
        },
        4: {
            "train": "67d49cddc00c95fafed9a0166b431021d405348354d029cb7941006d31117eec",
            "valid": "27ec5981da0864f094de8e764a28140ec063443a1a20c89e4f7a40f489d6c6a2",
            "test": "67e4ead6a0a899de82395cc8c3101fdd81d5b9b853328b8d3d53b2bde908f0a6",
        },
    }

    assert manifest["generator_version"] == GENERATOR_VERSION
    assert {
        split: info["sha256"] for split, info in manifest["splits"].items()
    } == expected.get(GENERATOR_VERSION), _REPIN_MESSAGE


def test_selection_tasks_preserve_difficulty_and_exact_reconstruction() -> None:
    """Catch wrong default mapping, ignored overrides, and non-exact reconstruction."""
    assert [task.difficulty for task in make_tasks("valid2", 4)] == [0, 1, 0, 1]
    overridden = make_tasks("valid2", 48, difficulty=2)
    assert {task.difficulty for task in overridden} == {2}
    for task in overridden:
        assert task_from_id(task.task_id, 20260902, 2) == task


def test_family_balanced_selection_is_clean_deterministic_and_split_isolated() -> None:
    """Catch incorrect quotas, recovery leakage, nondeterminism, and split collisions."""
    quotas = {"default": 1, "long": 3}
    cells = [
        family_balanced_tasks("valid", difficulty=1, per_family=quotas),
        family_balanced_tasks("valid2", difficulty=2, per_family=quotas),
    ]
    for tasks, split, level in zip(cells, ("valid", "valid2"), (1, 2), strict=True):
        assert len(tasks) == 24
        assert tasks == family_balanced_tasks(split, difficulty=level, per_family=quotas)
        assert len({task.task_id for task in tasks}) == len(tasks)
        assert {task.variant for task in tasks} == {"clean"}
        assert {task.difficulty for task in tasks} == {level}
        counts = {family: sum(task.family == family for task in tasks) for family in FAMILIES}
        assert counts == {
            family: 3 if family in LONG_HORIZON_FAMILIES else 1 for family in FAMILIES
        }

    split_tasks = {
        "train": make_tasks("train", 240),
        "valid": cells[0],
        "valid2": cells[1],
        "test": make_tasks("test", 180),
    }
    for left_name, left in split_tasks.items():
        for right_name, right in split_tasks.items():
            if left_name >= right_name:
                continue
            assert not {task.prompt for task in left} & {task.prompt for task in right}
            assert not {
                path for task in left for path in task.files
            } & {path for task in right for path in task.files}


@pytest.mark.parametrize("level", (0, 1, 2, 3))
def test_run_d_long_family_notes_preserve_full_ground_truth_state(level: int) -> None:
    """Catch compressed Run C notes or a level that stops extending a long horizon."""
    tasks = make_tasks("run-d", len(FAMILIES), difficulty=level, perturb=False)
    by_family = {task.family: task for task in tasks}

    for family in LONG_HORIZON_FAMILIES:
        if level:
            previous = make_tasks("run-d", len(FAMILIES), difficulty=level - 1, perturb=False)
            assert by_family[family].horizon > previous[FAMILIES.index(family)].horizon
        else:
            assert by_family[family].horizon > 0

    aggregate = by_family["aggregate_report"]
    values = [
        int(path_content.rsplit("value=", 1)[1])
        for path_content in aggregate.files.values()
        if "value=" in path_content
    ]
    split = len(values) // 2
    for step in aggregate.steps:
        if step.action.name in {"read_file", "calculate"}:
            assert f"split after {split} of {len(values)}" in step.thought
            assert "values so far:" in step.thought
            assert "(full)" not in step.thought.casefold()

    conditional = by_family["conditional_update"]
    for step in conditional.steps:
        is_service_step = "policy.txt" not in step.action.arguments.get("path", "")
        if step.action.name in {"read_file", "replace_text"} and is_service_step:
            assert "loads so far:" in step.thought

    batch = by_family["batch_update"]
    assert any("Inspected 1 of" in step.thought for step in batch.steps)
    assert any(
        "Applied 0 of" in step.thought and "Next: worker-0.ini mode=" in step.thought
        for step in batch.steps
    )


def test_run_d_renderer_is_canonical_and_recovery_notes_name_the_bad_path() -> None:
    """Catch a copied template or a recovery note/call path mismatch."""
    for variant in ("wrong_path", "stale_path"):
        task = next(
            task for task in make_tasks("train", 144, difficulty=1) if task.variant == variant
        )
        for index, step in enumerate(task.steps):
            assert task_module.render_expert_note(task, index) == step.thought
            if not step.supervise and step.action.name == "read_file":
                assert step.action.arguments["path"] in step.thought
    with pytest.raises(IndexError):
        task_module.render_expert_note(task, len(task.steps))


@pytest.mark.parametrize("level", range(4))
def test_run_d_conditional_and_recovery_notes_preserve_family_state(level: int) -> None:
    """Catch omitted empty load state or generic recovery notes that erase task state."""
    tasks = make_tasks("state", 144, difficulty=level)
    conditional = next(task for task in tasks if task.family == "conditional_update")
    for step in conditional.steps:
        if step.action.name in {"read_file", "list_files", "replace_text", "finish"}:
            assert "loads so far:" in step.thought

    for task in tasks:
        if task.variant not in {"wrong_path", "stale_path"}:
            continue
        for index, step in enumerate(task.steps):
            if step.supervise or step.action.name != "read_file":
                continue
            guessed = step.action.arguments["path"]
            assert guessed in step.thought
            recovery = task.steps[index + 1]
            assert recovery.supervise
            if task.family == "cross_reference":
                assert "Hop " in step.thought and "current key" in step.thought
            if task.family == "aggregate_report":
                assert "values so far:" in step.thought and "split after" in step.thought
            if task.family == "conditional_update":
                assert "loads so far:" in step.thought
            if task.variant == "stale_path":
                assert recovery.action.name == "list_files"
                if task.family == "cross_reference":
                    assert "Hop " in recovery.thought and "current key" in recovery.thought
                if task.family == "aggregate_report":
                    assert (
                        "values so far:" in recovery.thought
                        and "split after" in recovery.thought
                    )
                if task.family == "conditional_update":
                    assert "loads so far:" in recovery.thought
