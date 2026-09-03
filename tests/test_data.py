from __future__ import annotations

from local_llm_lab.pipeline.data import build_rows, write_dataset
from local_llm_lab.pipeline.tasks import make_tasks


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
