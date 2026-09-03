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
