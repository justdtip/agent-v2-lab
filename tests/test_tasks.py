from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline import tasks as task_module
from local_llm_lab.pipeline.cli import (
    _load_data_tokenizer,
    dataset_splits,
    load_config,
    stage_data,
)
from local_llm_lab.pipeline.data import write_dataset
from local_llm_lab.pipeline.tasks import (
    FAMILIES,
    GENERATOR_VERSION,
    LONG_HORIZON_FAMILIES,
    family_balanced_tasks,
    make_tasks,
    replay_task_from_id,
    task_from_id,
)

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
_REFERENCE_CONFIG = _CONFIG_ROOT / "agent_v2c.yaml"
# R5: a determinism oracle belongs to every arm config a run's data is generated from, not
# only to the historical reference. Run C is the reference the generator versions were
# defined against; run D (SPEC-003 §3) is pinned from its first generation, so its rows can
# never drift under a later generator change without the pin going red.
_PINNED_CONFIGS = ("agent_v2c.yaml", "agent_v2d.yaml")
_REPIN_MESSAGE = (
    "Intentional generator row change: bump GENERATOR_VERSION and re-pin both the "
    "generator-only and configured-replay hash oracles for every config in _PINNED_CONFIGS."
)


def _pinned_digests(manifest: dict) -> dict[str, dict[str, str]]:
    """The two hash surfaces a pin must cover: every logical split, and every written file.

    A role file is the concatenation of its logical chunks in declaration order, so where a
    config declares more splits than roles (run D writes six splits into three role files)
    the chunk hashes alone do not determine the file bytes; the concatenation does.
    """
    return {
        "splits": {name: info["sha256"] for name, info in manifest["splits"].items()},
        "outputs": {role: info["sha256"] for role, info in manifest["outputs"].items()},
    }


def _first_read_for_each_path(steps):
    """Keep the first supervised read per path when a recovery re-reads it."""
    unique = []
    seen: set[str] = set()
    for step in steps:
        path = step.action.arguments["path"]
        if path not in seen:
            unique.append(step)
            seen.add(path)
    return unique


def _manifest_targets(task):
    """Parse batch order and replacements from the manifest, not its worker actions."""
    manifest_path = next(path for path in task.files if path.endswith("/manifest.txt"))
    targets = []
    for line in task.files[manifest_path].splitlines():
        match = re.fullmatch(r"(.+)\|mode=([a-z]+)->mode=([a-z]+)", line)
        assert match, f"malformed manifest entry: {line!r}"
        targets.append(match.groups())
    assert targets
    return manifest_path, targets


def _cross_reference_pairs(task):
    """Pair each supervised search with its read and the reads that preceded it."""
    pairs = []
    prior_reads: set[str] = set()
    hop = 0
    last_key = None
    for index, step in enumerate(task.steps):
        if step.supervise and step.action.name == "search_files":
            key = step.action.arguments["query"]
            hop += key != last_key
            last_key = key
            following = next(
                later
                for later in task.steps[index + 1 :]
                if later.supervise and later.action.name == "read_file"
            )
            pairs.append((hop, key, step, following, prior_reads.copy()))
        if step.supervise and step.action.name == "read_file":
            prior_reads.add(step.action.arguments["path"])
    return pairs


def _assert_cross_reference_pair(task, hop, key, search, following, prior_reads) -> None:
    assert f"Hop {hop}:" in search.thought
    assert f"current key {key}" in search.thought
    assert f"Hop {hop}:" in following.thought
    assert f"current key {key}" in following.thought
    matches = {path for path, content in task.files.items() if key in content}
    target = following.action.arguments["path"]
    assert target in matches
    if len(matches) <= 1:
        return
    competitors = matches - {target}
    assert target not in prior_reads
    assert competitors <= prior_reads
    assert "new one" in following.thought
    assert target.rsplit("/", 1)[-1] in following.thought
    assert "already read" in following.thought
    for competitor in competitors:
        assert competitor.rsplit("/", 1)[-1] in following.thought


def _aggregate_metric_state(task, metric_paths, values, split) -> None:
    seen: list[int] = []
    last_path = None
    for step in task.steps:
        if not step.supervise or step.action.name != "read_file":
            continue
        path = step.action.arguments["path"]
        if path not in metric_paths or path == last_path:
            continue
        assert f"values so far: {', '.join(map(str, seen)) or 'none'}" in step.thought
        assert f"split after {split} of {len(values)}" in step.thought
        assert "first half:" not in step.thought.casefold()
        assert "(full)" not in step.thought.casefold()
        seen.append(int(task.files[path].rsplit("value=", 1)[1]))
        last_path = path
    assert seen == values


def _canonical_calculations(supervised):
    calculations = []
    for step in supervised:
        if step.action.name != "calculate":
            continue
        if (
            calculations
            and step.action.arguments["expression"]
            == calculations[-1].action.arguments["expression"]
        ):
            continue
        calculations.append(step)
    return calculations


def _aggregate_completion_state(task, values, split) -> None:
    total = sum(values)
    complete_state = f"values so far: {', '.join(map(str, values))}"
    split_state = f"split after {split} of {len(values)}"
    supervised = [step for step in task.steps if step.supervise]
    calculations = _canonical_calculations(supervised)
    first_total, second_total = sum(values[:split]), sum(values[split:])
    expected_expressions = [
        " + ".join(map(str, values[:split])),
        " + ".join(map(str, values[split:])),
        f"{first_total} + {second_total}",
    ]
    assert [step.action.arguments["expression"] for step in calculations] == expected_expressions
    calculated_total = sum(
        map(int, calculations[-1].action.arguments["expression"].split(" + "))
    )
    assert calculated_total == total

    calculation_start = supervised.index(calculations[0])
    for step in supervised[calculation_start:]:
        assert complete_state in step.thought
        assert split_state in step.thought

    report_path = next(
        path for path, content in task.files.items() if content == "grand_total=PENDING"
    )
    replace_index = next(
        index for index, step in enumerate(supervised) if step.action.name == "replace_text"
    )
    inspection = [
        step
        for step in supervised[supervised.index(calculations[-1]) + 1 : replace_index]
        if step.action.name == "read_file"
    ]
    assert inspection
    assert all(step.action.arguments["path"] == report_path for step in inspection)
    replacement = supervised[replace_index]
    assert replacement.action.arguments == {
        "path": report_path,
        "old": "grand_total=PENDING",
        "new": f"grand_total={total}",
    }
    verification = [
        step for step in supervised[replace_index + 1 :] if step.action.name == "read_file"
    ]
    assert verification
    assert all(step.action.arguments["path"] == report_path for step in verification)
    finish = supervised[-1]
    assert finish.action.name == "finish"
    assert finish.action.arguments["answer"] == f"grand_total={total}"
    for step in (*inspection, replacement, finish):
        assert re.search(rf"(?<!\d){total}(?!\d)", step.thought)


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


# Generator-only pins: no replay directory mixed in, so these hold on any checkout.
_GENERATOR_ONLY_PINS: dict[str, dict[int, dict[str, dict[str, str]]]] = {
    "agent_v2c.yaml": {
        2: {
            "splits": {
                "train": "e7fa63ef9a2fe70b3563a23fa5421b4a6b11d11971802d2c4b6bce5a0a3c5d58",
                "valid": "816543d1dee8299b2dbf514e93a2de480b69f84d6e8c53bda955c20c79f6213f",
                "test": "10042da5d9a4789f9a28fc6c0c9a8ea8efc59d090688f1e167c6de8d1de66877",
            },
            "outputs": {
                "train": "e7fa63ef9a2fe70b3563a23fa5421b4a6b11d11971802d2c4b6bce5a0a3c5d58",
                "valid": "816543d1dee8299b2dbf514e93a2de480b69f84d6e8c53bda955c20c79f6213f",
                "test": "10042da5d9a4789f9a28fc6c0c9a8ea8efc59d090688f1e167c6de8d1de66877",
            },
        },
        3: {
            "splits": {
                "train": "6a2875aff061e7dcdde0b8a394fc3115ff5c6f2b4dbcb8f399db5af373c6970e",
                "valid": "6df3a890d09e989c83baaf6078a27d56ec9da035807c8820e83b44c09a8259e3",
                "test": "925f5b282885e4c52f2a20280372804a17dfc9c2f1d8039478de98211eee1303",
            },
            "outputs": {
                "train": "6a2875aff061e7dcdde0b8a394fc3115ff5c6f2b4dbcb8f399db5af373c6970e",
                "valid": "6df3a890d09e989c83baaf6078a27d56ec9da035807c8820e83b44c09a8259e3",
                "test": "925f5b282885e4c52f2a20280372804a17dfc9c2f1d8039478de98211eee1303",
            },
        },
        4: {
            "splits": {
                "train": "99176c0b378d85502e738f23b8d174dd8321cb48a51e10c3a9bcd7fd9db3f035",
                "valid": "41ec07d6f122a123ee7780885d59ba2bc77ca5ce836e896ed8ddd2861f680deb",
                "test": "8b8aeaf28460798e0863ab0e5cb217da25b8661b7b3a802d21182fcd36a294a7",
            },
            "outputs": {
                "train": "99176c0b378d85502e738f23b8d174dd8321cb48a51e10c3a9bcd7fd9db3f035",
                "valid": "41ec07d6f122a123ee7780885d59ba2bc77ca5ce836e896ed8ddd2861f680deb",
                "test": "8b8aeaf28460798e0863ab0e5cb217da25b8661b7b3a802d21182fcd36a294a7",
            },
        },
    },
    # Run D (SPEC-003 §3) pinned at its first generation, 2026-09-05. Six logical splits
    # concatenate into three role files, so both surfaces are pinned.
    "agent_v2d.yaml": {
        4: {
            "splits": {
                "train": "f0c78560f797dd9aad5ee6301f26be37c0cde27787897c345fcf07a455e5ae9f",
                "train1": "4719e257391b79353ef8dabdd8214b76eb3f03b282a73ab7cd260762ddc8684c",
                "valid": "da4c1cf6bc745e758ab95fa8847c0f0ce3cbed63595cc918e8864042245b8b0d",
                "valid2": "b6c882ab48bf36aca5f7f4af140573e8e14d7e0264f3da952b0a6dcf9308d77b",
                "test": "21419637e0f7511a91d98c0d42d7eb2c5c91a7b2ac3cc3136429d2676870e138",
                "test3": "05346c2f2adc282ee0fd16a9e32d360b8a07587919411f8441eee65c07f1910b",
            },
            "outputs": {
                "train": "15610b8a826324e23429326dff5d0b92e6f9d0007aa3891a0e3e78911858c7cf",
                "valid": "93541214837237cdd699421a4b96a3e01c55dbf7d2eefc8918fec21dc939d0d5",
                "test": "3f45fc086d150ecf96bcf0ac149fc499825a33e93bbe29b9740f693adc261466",
            },
        },
    },
}

# Pins for the shipped replay mix, which needs the protected replay directory on disk.
_CONFIGURED_REPLAY_PINS: dict[str, dict[int, dict[str, dict[str, str]]]] = {
    "agent_v2c.yaml": {
        2: {
            "splits": {
                "train": "ad660e83cd89958dcee9fba2ab1e53115d1fb079813ea694cd4b0e89530b3a79",
                "valid": "d6dbc53573744751d74565a0de6ca5c6d381cba6b488ff6410194bf9b0d4e6d8",
                "test": "fc69b03fef8f423ee85a174214ad955fe3f4d324554217510b92f53435841ce3",
            },
            "outputs": {
                "train": "ad660e83cd89958dcee9fba2ab1e53115d1fb079813ea694cd4b0e89530b3a79",
                "valid": "d6dbc53573744751d74565a0de6ca5c6d381cba6b488ff6410194bf9b0d4e6d8",
                "test": "fc69b03fef8f423ee85a174214ad955fe3f4d324554217510b92f53435841ce3",
            },
        },
        3: {
            "splits": {
                "train": "92f40d1868438553f306be192b09cace7b3d5efc8cff4a55c84fc3a96827ddf2",
                "valid": "6e2617d5159e42fc7d26077e332a83e9f27fc0e755d50140d6336c75116c44b7",
                "test": "f79d6fc673674719cc17664278a15ee8ba9fcb6de8f9418d9226b99b5d24de59",
            },
            "outputs": {
                "train": "92f40d1868438553f306be192b09cace7b3d5efc8cff4a55c84fc3a96827ddf2",
                "valid": "6e2617d5159e42fc7d26077e332a83e9f27fc0e755d50140d6336c75116c44b7",
                "test": "f79d6fc673674719cc17664278a15ee8ba9fcb6de8f9418d9226b99b5d24de59",
            },
        },
        4: {
            "splits": {
                "train": "67d49cddc00c95fafed9a0166b431021d405348354d029cb7941006d31117eec",
                "valid": "27ec5981da0864f094de8e764a28140ec063443a1a20c89e4f7a40f489d6c6a2",
                "test": "67e4ead6a0a899de82395cc8c3101fdd81d5b9b853328b8d3d53b2bde908f0a6",
            },
            "outputs": {
                "train": "67d49cddc00c95fafed9a0166b431021d405348354d029cb7941006d31117eec",
                "valid": "27ec5981da0864f094de8e764a28140ec063443a1a20c89e4f7a40f489d6c6a2",
                "test": "67e4ead6a0a899de82395cc8c3101fdd81d5b9b853328b8d3d53b2bde908f0a6",
            },
        },
    },
    # The mix actually written to data/agent_v2d, pinned at its first generation 2026-09-05:
    # 240/48/60 chat-replay rows join the train/valid/test role files.
    "agent_v2d.yaml": {
        4: {
            "splits": {
                "train": "6026b7715ad962e2d2b1f589867db3b3179b9ed79062f45c6b5501d4a6594556",
                "train1": "4719e257391b79353ef8dabdd8214b76eb3f03b282a73ab7cd260762ddc8684c",
                "valid": "9c48ec5c50adde5248a0f043a38ef7879f5563b3a749112b92a02ea008fffbd2",
                "valid2": "b6c882ab48bf36aca5f7f4af140573e8e14d7e0264f3da952b0a6dcf9308d77b",
                "test": "68de80f31c8d1308e99dbd719db7acb228976fd0db9b89aa6d8c5217e8ae5f50",
                "test3": "05346c2f2adc282ee0fd16a9e32d360b8a07587919411f8441eee65c07f1910b",
            },
            "outputs": {
                "train": "7eb3f2bb3d89a5f88a62fbcb287abb349268644a692dc98ee3603f415d82e444",
                "valid": "8b2f4b5f2baa168797646cce7fef085a215e38af85777abe98373854608bacb6",
                "test": "eac6150f93ee65aea34aba735a926f3058e5ef719f596d144d3aab2c4567de88",
            },
        },
    },
}


@pytest.mark.parametrize("config_name", _PINNED_CONFIGS)
def test_reference_generator_hashes_are_pinned_by_version(config_name, tmp_path) -> None:
    """A task/expert-row change must bump the generator version and both hash oracles."""
    config = load_config(_CONFIG_ROOT / config_name)
    manifest = write_dataset(
        tmp_path,
        dataset_splits(config),
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=None,
        recovery_repeats=config["recovery_repeats"],
    )

    assert manifest["generator_version"] == GENERATOR_VERSION
    assert _pinned_digests(manifest) == _GENERATOR_ONLY_PINS.get(config_name, {}).get(
        GENERATOR_VERSION
    ), _REPIN_MESSAGE


@pytest.mark.parametrize("config_name", _PINNED_CONFIGS)
def test_reference_generator_hashes_include_configured_replay(config_name, tmp_path) -> None:
    """The shipped replay mix stays pinned when its protected input directory is available."""
    config = load_config(_CONFIG_ROOT / config_name)
    replay_dir = config["chat_replay"]
    if not replay_dir.is_dir():
        pytest.skip(f"configured protected replay directory is absent: {replay_dir}")
    manifest = write_dataset(
        tmp_path,
        dataset_splits(config),
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=config["chat_replay"],
        chat_repeats=config["chat_repeats"],
        recovery_repeats=config["recovery_repeats"],
    )

    assert manifest["generator_version"] == GENERATOR_VERSION
    assert _pinned_digests(manifest) == _CONFIGURED_REPLAY_PINS.get(config_name, {}).get(
        GENERATOR_VERSION
    ), _REPIN_MESSAGE


def test_run_d_dataset_on_disk_regenerates_byte_for_byte(tmp_path) -> None:
    """SPEC-003 §3: the shipped run D dataset must be reproducible from its config alone.

    The R5 oracles above pin the generator's rows; this pins the artifact a training arm
    actually reads, rendering included. R31: the rendering seam is driven with the model's
    real tokenizer (weights are never loaded), because the boundary-merge behaviour a fake
    tokenizer would stub is exactly what decides the bytes.
    """
    config = load_config(_CONFIG_ROOT / "agent_v2d.yaml")
    committed_path = config["data"] / "manifest.json"
    if not committed_path.is_file():
        pytest.skip(f"run D dataset is not present on this checkout: {config['data']}")
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    spec = load_model_spec(config["model"])
    tokenizer = _load_data_tokenizer(spec.hf_id)

    manifest = write_dataset(
        tmp_path,
        dataset_splits(config),
        seed=config["seed"],
        keep_last=config["keep_last"],
        chat_dir=config.get("chat_replay"),
        chat_repeats=config.get("chat_repeats", 1),
        recovery_repeats=config["recovery_repeats"],
        tokenizer=tokenizer,
        spec=spec,
    )

    assert manifest["generator_version"] == committed["generator_version"]
    assert _pinned_digests(manifest) == _pinned_digests(committed), _REPIN_MESSAGE
    for role in ("train", "valid", "test"):
        regenerated = hashlib.sha256((tmp_path / f"{role}.jsonl").read_bytes()).hexdigest()
        assert regenerated == committed["outputs"][role]["sha256"], role


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


def test_run_d_cross_reference_actions_follow_current_keys_and_read_new_matches() -> None:
    """Search/read choices derive from keys and files, not retired prose templates."""
    for level in range(4):
        for task in make_tasks("run-d-cross", 144, difficulty=level):
            if task.family != "cross_reference":
                continue
            for pair in _cross_reference_pairs(task):
                _assert_cross_reference_pair(task, *pair)


def test_run_d_aggregate_notes_carry_action_derived_values_split_and_total() -> None:
    """Metric state is append-only and the final reported total is the file-derived sum."""
    for level in range(4):
        for task in make_tasks("run-d-aggregate", 144, difficulty=level):
            if task.family != "aggregate_report":
                continue
            metric_paths = [path for path, content in task.files.items() if "value=" in content]
            metric_paths.sort()
            values = [int(task.files[path].rsplit("value=", 1)[1]) for path in metric_paths]
            split = len(values) // 2
            _aggregate_metric_state(task, metric_paths, values, split)
            _aggregate_completion_state(task, values, split)


def test_run_d_batch_notes_follow_action_derived_worker_order_and_pending_queue() -> None:
    """Inspection, apply, and verify progress is grounded in worker files and actions."""
    for level in range(4):
        for task in make_tasks("run-d-batch", 144, difficulty=level):
            if task.family != "batch_update":
                continue
            manifest_path, targets = _manifest_targets(task)
            workers = [path for path, _, _ in targets]
            names = [path.rsplit("/", 1)[-1] for path in workers]
            supervised = [step for step in task.steps if step.supervise]
            assert supervised[0].action.name == "read_file"
            assert supervised[0].action.arguments["path"] == manifest_path
            replace_at = next(
                i for i, step in enumerate(supervised) if step.action.name == "replace_text"
            )
            inspect = _first_read_for_each_path([
                step
                for step in supervised[:replace_at]
                if step.action.name == "read_file" and step.action.arguments["path"] in workers
            ])
            assert [step.action.arguments["path"] for step in inspect] == workers
            for index, (step, target) in enumerate(zip(inspect, targets, strict=True)):
                path, old, new = target
                assert step.action.arguments["path"] == path
                assert f"Inspected {index} of {len(workers)}" in step.thought
                assert f"Next: {path.rsplit('/', 1)[-1]} mode={old} -> mode={new}" in step.thought
                assert f"pending: {', '.join(names[index:])}." in step.thought

            apply = [
                step for step in supervised if step.action.name == "replace_text"
            ]
            apply = [
                step for i, step in enumerate(apply)
                if i == 0 or step.action.arguments["path"] != apply[i - 1].action.arguments["path"]
            ]
            assert len(apply) == len(targets)
            for index, (step, target) in enumerate(zip(apply, targets, strict=True)):
                path, old, new = target
                assert step.action.arguments == {
                    "path": path,
                    "old": f"mode={old}",
                    "new": f"mode={new}",
                }
                assert f"Applied {index} of {len(workers)}" in step.thought
                assert f"Next: {names[index]} mode={old} -> mode={new}" in step.thought
                assert f"pending: {', '.join(names[index:])}." in step.thought

            verify = _first_read_for_each_path([
                step
                for step in supervised[supervised.index(apply[-1]) + 1 :]
                if step.action.name == "read_file" and step.action.arguments["path"] in workers
            ])
            assert len(verify) == len(targets)
            for index, (step, target) in enumerate(zip(verify, targets, strict=True)):
                path, _old, new = target
                assert step.action.arguments["path"] == path
                assert (
                    f"Applied {len(workers)} of {len(workers)}; verified {index} of {len(workers)}"
                    in step.thought
                )
                assert f"Next: {path.rsplit('/', 1)[-1]}, expect mode={new}" in step.thought
                assert f"pending: {', '.join(names[index:])}." in step.thought
            assert task.steps[-1].action.name == "finish"
            assert (
                f"Applied {len(workers)} of {len(workers)}; "
                f"verified {len(workers)} of {len(workers)}; "
                "pending: none."
            ) in task.steps[-1].thought


def test_run_d_conditional_notes_accumulate_loads_and_modify_the_true_maximum() -> None:
    """The modified service is the file-derived running maximum, without a retired marker."""
    for level in range(4):
        for task in make_tasks("run-d-conditional", 144, difficulty=level):
            if task.family != "conditional_update":
                continue
            services = sorted(
                (path for path, content in task.files.items() if "name=service-" in content),
                key=lambda path: int(re.search(r"service-(\d+)\.ini$", path).group(1)),
            )
            loads = {
                path: int(re.search(r"load=(\d+)", task.files[path]).group(1)) for path in services
            }
            replace_at = next(
                i for i, step in enumerate(task.steps) if step.action.name == "replace_text"
            )
            reads = _first_read_for_each_path([
                step
                for step in task.steps[:replace_at]
                if step.supervise
                and step.action.name == "read_file"
                and step.action.arguments["path"] in services
            ])
            assert [step.action.arguments["path"] for step in reads] == services
            seen: list[str] = []
            for step in reads:
                assert f"loads so far: {', '.join(seen) or 'none'}" in step.thought
                path = step.action.arguments["path"]
                service = path.rsplit("/", 1)[-1].removesuffix(".ini")
                seen.append(f"{service}={loads[path]}")
            replace = next(step for step in task.steps if step.action.name == "replace_text")
            expected = max(services, key=loads.__getitem__)
            service = expected.rsplit("/", 1)[-1].removesuffix(".ini")
            assert replace.action.arguments["path"] == expected
            assert f"loads so far: {', '.join(seen)}" in replace.thought
            assert f"highest so far: {service}={loads[expected]}" in replace.thought
            assert "(final)" not in replace.thought


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


def _assert_conditional_state_notes(task) -> None:
    for step in task.steps:
        if step.action.name in {"read_file", "list_files", "replace_text", "finish"}:
            assert "loads so far:" in step.thought


def _family_state_fragments(family):
    return {
        "cross_reference": ("Hop ", "current key"),
        "aggregate_report": ("values so far:", "split after"),
        "conditional_update": ("loads so far:",),
    }.get(family, ())


def _assert_recovery_state(task, index, step) -> None:
    guessed = step.action.arguments["path"]
    assert guessed in step.thought
    recovery = task.steps[index + 1]
    assert recovery.supervise
    fragments = _family_state_fragments(task.family)
    for fragment in fragments:
        assert fragment in step.thought
    if task.variant != "stale_path":
        return
    assert recovery.action.name == "list_files"
    for fragment in fragments:
        assert fragment in recovery.thought


def _assert_family_recoveries(task) -> None:
    for index, step in enumerate(task.steps):
        if step.supervise or step.action.name != "read_file":
            continue
        _assert_recovery_state(task, index, step)


@pytest.mark.parametrize("level", range(4))
def test_run_d_conditional_and_recovery_notes_preserve_family_state(level: int) -> None:
    """Catch omitted empty load state or generic recovery notes that erase task state."""
    tasks = make_tasks("state", 144, difficulty=level)
    conditional = next(task for task in tasks if task.family == "conditional_update")
    _assert_conditional_state_notes(conditional)

    for task in tasks:
        if task.variant not in {"wrong_path", "stale_path"}:
            continue
        _assert_family_recoveries(task)


@pytest.mark.parametrize(
    ("generator_version", "historical_finish"),
    [
        (1, "The file shows Owner: owner-7158. Task complete."),
        (2, "The file shows Owner: owner-7158. Task complete."),
        (3, "Owner: owner-7158; pending: none."),
        (4, "Owner: owner-7158; pending: none."),
    ],
)
def test_replay_task_from_id_keeps_current_structure_and_selects_historical_notes(
    generator_version: int, historical_finish: str
) -> None:
    """R12 replays the named revision's thought template without changing task structure."""
    task_id = "test-read-0000-clean"
    current = task_from_id(task_id, 20260902, 2)
    replayed = replay_task_from_id(task_id, 20260902, generator_version, 2)

    assert [step.action for step in replayed.steps] == [step.action for step in current.steps]
    assert replayed.prompt == current.prompt
    assert replayed.steps[-1].thought == historical_finish
    assert task_from_id(task_id, 20260902, 2) == current


@pytest.mark.parametrize("generator_version", [True, 0, 5, "3"])
def test_replay_task_from_id_rejects_invalid_historical_generator_versions(
    generator_version: object,
) -> None:
    with pytest.raises(ValueError, match="generator_version"):
        replay_task_from_id("test-read-0000-clean", 20260902, generator_version, 2)  # type: ignore[arg-type]


def test_normal_generation_has_no_historical_generator_switch() -> None:
    with pytest.raises(TypeError):
        make_tasks("test", 1, generator_version=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("family", "v2_fragment", "v3_fragment"),
    [
        ("ledger_reconcile", "Task complete.", "pending: none."),
        ("cross_reference", "Task complete.", "pending: none."),
        ("conditional_update", "Task complete.", "loads so far:"),
        ("batch_update", "Task complete.", "Applied"),
        ("aggregate_report", "Task complete.", "values so far:"),
    ],
)
def test_replay_selects_each_historical_long_family_note_profile(
    family: str, v2_fragment: str, v3_fragment: str
) -> None:
    task = next(
        item
        for item in make_tasks("train", 144, difficulty=1)
        if item.family == family and item.variant == "clean"
    )
    v2 = replay_task_from_id(task.task_id, 20260902, 2, 1)
    v3 = replay_task_from_id(task.task_id, 20260902, 3, 1)

    assert v2.steps[-1].thought != v3.steps[-1].thought
    assert v2_fragment in v2.steps[-1].thought
    assert v3_fragment in v3.steps[-1].thought


@pytest.mark.parametrize(
    ("variant", "v1_prefix", "v2_prefix", "v4_suffix", "difference_index"),
    [
        ("wrong_path", "Search matched", "Trying guessed path", "Search matched", 0),
        ("stale_path", "Invoices read:", "Trying stale guessed path", "Invoices read:", 0),
        ("failed_edit", "File contains", "File contains", "File contains", 1),
    ],
)
def test_replay_preserves_historical_recovery_injected_note_profiles(
    variant: str, v1_prefix: str, v2_prefix: str, v4_suffix: str, difference_index: int
) -> None:
    task = next(
        item
        for item in make_tasks("train", 144, difficulty=1)
        if item.variant == variant
    )
    v1 = replay_task_from_id(task.task_id, 20260902, 1, 1)
    v2 = replay_task_from_id(task.task_id, 20260902, 2, 1)
    v3 = replay_task_from_id(task.task_id, 20260902, 3, 1)
    v4 = replay_task_from_id(task.task_id, 20260902, 4, 1)
    injected = next(index for index, step in enumerate(v4.steps) if not step.supervise)

    assert v1.steps[injected].thought.startswith(v1_prefix)
    assert v2.steps[injected].thought.startswith(v2_prefix)
    assert v2.steps[injected].thought == v3.steps[injected].thought
    assert v4_suffix in v4.steps[injected].thought
    assert (
        v4.steps[injected + difference_index].thought
        != v3.steps[injected + difference_index].thought
    )
    assert v1.steps[-1].thought != v4.steps[-1].thought
    if variant == "wrong_path":
        assert v2.steps[injected + 1].thought.startswith(
            "That path does not exist; use the exact path"
        )
    elif variant == "stale_path":
        assert v2.steps[injected + 1].thought.endswith("instead of guessing.")
        assert v2.steps[injected + 2].thought.startswith("The listing gives the exact name. ")
    else:
        assert v2.steps[injected + 1].thought.endswith("instead of retrying the same edit.")
        assert v2.steps[injected + 2].thought.startswith(
            "The file's current text confirms the exact string to replace. "
        )


@pytest.mark.parametrize(
    "variant", ["wrong_path", "transient", "unknown_tool", "stale_path", "failed_edit"]
)
def test_replay_rewrites_every_historical_recovery_step(variant: str) -> None:
    task = next(item for item in make_tasks("train", 144, difficulty=1) if item.variant == variant)
    v2 = replay_task_from_id(task.task_id, 20260902, 2, 1)
    v3 = replay_task_from_id(task.task_id, 20260902, 3, 1)
    assert len(v2.steps) == len(task.steps) == len(v3.steps)
    assert [step.action for step in v2.steps] == [step.action for step in task.steps]
    if variant == "transient":
        retry = v2.faults[0].call_index + 1
        assert v2.steps[retry].thought.startswith("The tool reported a transient failure;")
    elif variant == "unknown_tool":
        bad = next(i for i, step in enumerate(v2.steps) if not step.supervise)
        assert v2.steps[bad + 1].thought.startswith(
            f"{v2.steps[bad].action.name} is not an available tool;"
        )
    elif variant == "failed_edit":
        bad = next(i for i, step in enumerate(v2.steps) if not step.supervise)
        assert v2.steps[bad].thought == v2.steps[bad + 2].thought.removeprefix(
            "The file's current text confirms the exact string to replace. "
        )


# ----------------------------------------------------------- SPEC-004 §2 / R28: P2 splits


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
# Every config that declares a generator split table. Pinned so a new run config cannot join
# the repository without being swept by the disjointness test below.
_CONFIGS_WITH_SPLIT_TABLES = (
    "agent_v2.yaml",
    "agent_v2b.yaml",
    "agent_v2b_qwen35_4b.yaml",
    "agent_v2c.yaml",
    "agent_v2d.yaml",
    "agent_v2d_qwen35_4b.yaml",
)


def _config_paths():
    return sorted(_CONFIG_DIR.glob("*.yaml"))


def _split_table(config: dict) -> dict:
    """The raw ``tasks:``/``splits:`` block of a config, or ``{}`` when it declares neither."""
    return config.get("tasks") or config.get("splits") or {}


def _generated_splits(config: dict) -> list[tuple[str, int, int | None, bool | None]]:
    """``(name, count, difficulty, perturb)`` for every split a config's table declares.

    ``dataset_splits`` normalises both supported schemas; the legacy ``tasks:`` schema leaves
    difficulty and perturbation to ``make_tasks``' own split-name defaults, which is exactly how
    those datasets were generated.
    """
    specs = dataset_splits(config)
    return [(name, spec.count, spec.difficulty, spec.perturb) for name, spec in specs.items()]


def _screen_splits(config: dict) -> list[dict]:
    """Selection-screen splits, which are generated but do not appear in the split table."""
    screen = (config.get("select") or {}).get("screen") or []
    return [entry for entry in screen if entry.get("split") not in _split_table(config)]


def _source_config_for(config: dict):
    """Resolve a ``source_rows`` config to the config that generated those rows."""
    source = config.get("source_rows")
    if source is None:
        return None
    for path in _config_paths():
        other = load_config(path)
        if other.get("data") == source:
            return path, other
    raise AssertionError(f"no config generates {source}")


def _config_tasks(config: dict) -> list:
    """Regenerate every task a config's splits contain, deterministically and without files."""
    seed = config["seed"]
    generated = []
    for name, count, level, perturb in _generated_splits(config):
        kwargs = {}
        if level is not None:
            kwargs["difficulty"] = level
        if perturb is not None:
            kwargs["perturb"] = perturb
        generated.extend(make_tasks(name, count, seed, **kwargs))
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


def _rename_split(task, new: str):
    """A byte-copy of a task minted under ``new``: the split tokens rewritten, nothing else."""
    old = task_module.split_of_task_id(task.task_id)

    def substitute(text: str) -> str:
        text = re.sub(
            rf"(?<![0-9A-Za-z_])(workspace|lab)/{re.escape(old)}/", rf"\1/{new}/", text
        )
        return re.sub(
            rf"(?<![0-9A-Za-z_])(KEY|REF)-{re.escape(old.upper())}-",
            rf"\1-{new.upper()}-",
            text,
        )

    return dataclasses.replace(
        task,
        task_id=new + task.task_id[len(old) :],
        prompt=substitute(task.prompt),
        files={substitute(path): substitute(body) for path, body in task.files.items()},
        expected_answer=substitute(task.expected_answer),
    )


def _p2_tasks(seed: int, limit: int) -> list:
    return [
        task
        for name in task_module.P2_SPLIT_NAMES
        for task in task_module.make_p2_tasks(name, limit, seed)
    ]


def test_p2_split_plan_matches_the_spec() -> None:
    """SPEC-004 §2: p2-d0/1/2, explicit difficulty, perturbed at 0 and 1, clean at 2."""
    assert task_module.P2_SPLITS == (("p2-d0", 0, True), ("p2-d1", 1, True), ("p2-d2", 2, False))
    assert task_module.P2_SPLIT_NAMES == ("p2-d0", "p2-d1", "p2-d2")
    assert task_module.P2_SPLIT_LIMIT == 120
    with pytest.raises(ValueError):
        task_module.p2_split("train")


def test_make_p2_tasks_applies_the_planned_difficulty_and_perturbation() -> None:
    """The helper, not the caller, carries the split's difficulty and perturb flags."""
    seed = load_config(_REFERENCE_CONFIG)["seed"]
    for name, level, perturb in task_module.P2_SPLITS:
        made = task_module.make_p2_tasks(name, len(FAMILIES) * 2, seed)
        assert len(made) == len(FAMILIES) * 2
        assert {task.difficulty for task in made} == {level}
        assert all(task.task_id.startswith(f"{name}-") for task in made)
        variants = {task.variant for task in made}
        assert (variants != {"clean"}) is perturb
        assert made == make_tasks(
            name, len(FAMILIES) * 2, seed, perturb=perturb, difficulty=level
        )


def test_task_fingerprint_ignores_the_split_but_not_the_content() -> None:
    """R28's normalisation: identity under renaming, sensitivity to everything else."""
    seed = load_config(_REFERENCE_CONFIG)["seed"]
    train = make_tasks("train", len(FAMILIES), seed)
    for task in train:
        assert task_module.task_fingerprint(task) == task_module.task_fingerprint(task)
        for name in task_module.P2_SPLIT_NAMES:
            renamed = _rename_split(task, name)
            assert task_module.split_of_task_id(renamed.task_id) == name
            assert task_module.task_fingerprint(renamed) == task_module.task_fingerprint(task), (
                f"{task.task_id}: fingerprint changed under a pure split rename"
            )
    assert len({task_module.task_fingerprint(task) for task in train}) == len(train)


def test_task_fingerprint_leaves_no_split_token_in_its_payload() -> None:
    """The normalisation is complete: no fingerprinted field still names its split."""
    seed = load_config(_REFERENCE_CONFIG)["seed"]
    for name in ("train", "valid", "test", *task_module.P2_SPLIT_NAMES):
        made = (
            task_module.make_p2_tasks(name, len(FAMILIES), seed)
            if name in task_module.P2_SPLIT_NAMES
            else make_tasks(name, len(FAMILIES), seed)
        )
        for task in made:
            fields = [
                task_module._normalise_split_tokens(task.prompt, name),
                task_module._normalise_split_tokens(task.expected_answer, name),
                *(
                    task_module._normalise_split_tokens(text, name)
                    for path, body in task.files.items()
                    for text in (path, body)
                ),
            ]
            residual = [text for text in fields if name in text or name.upper() in text]
            assert not residual, f"{task.task_id}: split token survives normalisation: {residual}"


def test_no_config_declares_a_p2_split() -> None:
    """R28 (a): no P2 split name appears in any config's ``tasks:``/``splits:`` block."""
    tabled = []
    for path in _config_paths():
        config = load_config(path)
        table = _split_table(config)
        if not table:
            continue
        tabled.append(path.name)
        overlap = set(table) & set(task_module.P2_SPLIT_NAMES)
        assert not overlap, f"{path.name}: training table declares P2 split(s) {sorted(overlap)}"
        screened = {entry["split"] for entry in _screen_splits(config)}
        assert not screened & set(task_module.P2_SPLIT_NAMES), f"{path.name}: screen uses a P2 split"
    assert tuple(tabled) == _CONFIGS_WITH_SPLIT_TABLES


def test_p2_tasks_are_content_disjoint_from_every_configured_split() -> None:
    """R28 (b): no P2 task's content fingerprint appears in any config's generated rows.

    Configs whose data directory is absent are regenerated through ``make_tasks`` from their
    split table, so the sweep never skips; configs that render from ``source_rows`` are checked
    against the table of the config that generated those rows as well as their own.
    """
    seed = load_config(_REFERENCE_CONFIG)["seed"]
    p2 = {}
    for task in _p2_tasks(seed, task_module.P2_SPLIT_LIMIT):
        p2.setdefault(task_module.task_fingerprint(task), []).append(task.task_id)
    assert len(p2) == len(task_module.P2_SPLIT_NAMES) * task_module.P2_SPLIT_LIMIT

    checked = []
    for path in _config_paths():
        config = load_config(path)
        if not _split_table(config):
            continue
        resolved = _source_config_for(config)
        if resolved is not None:
            source_path, source_config = resolved
            assert _split_table(config) == _split_table(source_config), (
                f"{path.name}: renders {source_path.name}'s rows but declares a different table"
            )
        assert config["seed"] == seed
        rows = _config_tasks(config)
        assert rows
        collisions = [
            (task.task_id, p2[fingerprint])
            for task in rows
            if (fingerprint := task_module.task_fingerprint(task)) in p2
        ]
        assert not collisions, f"{path.name}: P2 rows duplicate training content: {collisions[:5]}"
        checked.append(path.name)
    assert tuple(checked) == _CONFIGS_WITH_SPLIT_TABLES
