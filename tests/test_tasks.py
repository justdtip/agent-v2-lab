from __future__ import annotations

import json
import re
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
    replay_task_from_id,
    task_from_id,
)

_REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "agent_v2c.yaml"
_REPIN_MESSAGE = (
    "Intentional generator row change: bump GENERATOR_VERSION and re-pin both the "
    "generator-only and configured-replay hash oracles."
)


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
