from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        }
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
        }
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
