from __future__ import annotations

import json
from io import StringIO

from local_llm_lab.agent_protocol import Action
from local_llm_lab.pipeline.tasks import make_tasks
from local_llm_lab.pipeline.transcript import Transcript, iter_task_records


def test_transcript_keeps_multiple_task_records_in_one_run(tmp_path) -> None:
    """Truncating JSONL on each task start drops all but the final task's transcript."""
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
    assert set(records[0]) == {"run_id"}
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


def test_transcript_records_full_thinking_and_displays_collapsed_summary(tmp_path) -> None:
    """Dropping thinking or printing its full body would hide or flood run diagnostics."""
    task = make_tasks("valid", 1)[0]
    stream = StringIO()
    transcript = Transcript(stream=stream, directory=tmp_path, color=False)
    transcript.start(task, "run")

    thinking = "\n  first reason  \nintermediate detail\nlast conclusion\n"
    transcript.step(
        1,
        "write the result",
        Action(name="finish", arguments={"answer": "done"}),
        "finished",
        thinking=thinking,
        think_tokens=7,
    )

    assert transcript.record["steps"][0]["thinking"] == thinking
    assert transcript.record["steps"][0]["think_tokens"] == 7
    assert "think  first reason … last conclusion" in stream.getvalue()
    assert "intermediate detail" not in stream.getvalue()
