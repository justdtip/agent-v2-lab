from __future__ import annotations

import json

from local_llm_lab.pipeline.runner import Trajectory


def test_trajectory_as_dict_json_round_trips_through_constructor() -> None:
    trajectory = Trajectory(
        task_id="test-ledger-0001-clean",
        family="ledger",
        variant="clean",
        label="unit",
        prompt="test prompt",
        steps=[{"thinking": None, "action": "read_file"}],
    )

    record = json.loads(json.dumps(trajectory.as_dict()))
    assert Trajectory(**record) == trajectory


def test_trajectory_accepts_write_report_fields_with_backward_compatible_defaults() -> None:
    legacy = Trajectory(
        task_id="test-ledger-0001-clean",
        family="ledger",
        variant="clean",
        label="unit",
        prompt="test prompt",
    )
    record = legacy.as_dict() | {"difficulty": 2, "integrity": {"clean": True}}

    restored = Trajectory(**json.loads(json.dumps(record)))

    assert restored.difficulty == 2
    assert restored.integrity == {"clean": True}
    assert Trajectory(**legacy.as_dict()).difficulty == -1
    assert Trajectory(**legacy.as_dict()).integrity == {}
