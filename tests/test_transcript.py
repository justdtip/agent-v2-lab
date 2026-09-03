from __future__ import annotations

import json

from local_llm_lab.pipeline.tasks import make_tasks
from local_llm_lab.pipeline.transcript import iter_task_records


def test_transcript_keeps_multiple_task_records_in_one_run(tmp_path) -> None:
    """Truncating JSONL on each task start drops all but the final task's transcript."""
    from local_llm_lab.pipeline.transcript import Transcript

    first_task, second_task = make_tasks("valid", 2)
    first = Transcript(stream=None, directory=tmp_path)
    first.start(first_task, "run")
    first.finish(
        {
            "success": True,
            "errors": 0,
            "answer": first_task.expected_answer,
            "expected_answer": first_task.expected_answer,
            "reasons": [],
        },
        0.1,
    )
    second = Transcript(stream=None, directory=tmp_path)
    second.start(second_task, "run")
    second.finish(
        {
            "success": False,
            "errors": 0,
            "answer": second_task.expected_answer,
            "expected_answer": second_task.expected_answer,
            "reasons": ["unexpected file change: extra.ini"],
        },
        0.1,
    )
    jsonl = tmp_path / "transcripts.jsonl"
    records = [
        __import__("json").loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    run_id = records[0]["run_id"]
    assert [record["task_id"] for record in records[1:]] == [
        first_task.task_id,
        second_task.task_id,
    ]
    assert [record["run_id"] for record in records[1:]] == [run_id, run_id]
    markdown = (tmp_path / f"run-{second_task.task_id}.md").read_text(encoding="utf-8")
    assert "**FAIL** (unexpected file change: extra.ini)" in markdown


def test_iter_task_records_skips_headers_blank_lines_and_metadata(tmp_path) -> None:
    """Transcript consumers must see task rows only, never the run header or metadata."""
    path = tmp_path / "transcripts.jsonl"
    first = {"task_id": "task-1", "verdict": {"success": True}}
    second = {"task_id": "task-2", "verdict": {"success": False}}
    path.write_text(
        "\n".join(
            (
                json.dumps({"run_id": "run-1"}),
                json.dumps(first),
                "",
                json.dumps({"summary": {"tasks": 2}}),
                json.dumps(second),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert list(iter_task_records(path)) == [first, second]


def test_transcript_explicit_run_boundary_replaces_prior_run(tmp_path) -> None:
    """A new explicit run must truncate old task records and allocate a new run id."""
    from local_llm_lab.pipeline.transcript import Transcript

    first_task, second_task = make_tasks("valid", 2)
    Transcript.start_run(tmp_path)
    first = Transcript(stream=None, directory=tmp_path)
    first.start(first_task, "run")
    first.finish(
        {"success": True, "errors": 0, "answer": "a", "expected_answer": "a", "reasons": []}, 0.1
    )
    Transcript.start_run(tmp_path)
    second = Transcript(stream=None, directory=tmp_path)
    second.start(second_task, "run")
    second.finish(
        {"success": True, "errors": 0, "answer": "b", "expected_answer": "b", "reasons": []}, 0.1
    )
    records = [
        __import__("json").loads(line)
        for line in (tmp_path / "transcripts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[1]["task_id"] == second_task.task_id
    assert records[1]["run_id"] == records[0]["run_id"]
