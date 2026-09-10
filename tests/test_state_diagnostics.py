"""The ten diagnostics on hand-written transcripts: fixed functions, never a model's reading."""

from __future__ import annotations

import pytest

from local_llm_lab.pipeline.state_programme import diagnostics as dx

TARGET = "workspace/fx/0000/notes/summary-42.md"
DIRECTORY = "workspace/fx/0000/notes"
LISTED_WITH = f"FILES: {DIRECTORY}/scratch.txt, {TARGET}"
LISTED_WITHOUT = f"FILES: {DIRECTORY}/scratch.txt"


def _row(i, name, observation, **arguments):
    action = {"name": name, "arguments": arguments}
    return {"index": i, "action": action, "observation": observation}


def _ctx(steps):
    return dx.Context.from_trajectory(steps, target=TARGET, directory=DIRECTORY)


def test_all_nine_single_context_diagnostics_are_registered_and_m_is_ten() -> None:
    assert list(dx.DIAGNOSTICS) == [f"D{i}" for i in range(1, 10)]
    assert dx.RELATION == "D10" and dx.M == 10


def test_the_reading_transcript_scores_read_and_report_exists() -> None:
    c = _ctx([
        _row(1, "list_files", LISTED_WITH, directory=DIRECTORY),
        _row(2, "read_file", "milestone-1\nbody", path=TARGET),
        _row(3, "finish", "", answer="milestone-1"),
    ])
    s = dx.score(c)
    assert s["D1"] == 1.0 and s["D2"] == 0.0 and s["D3"] == 0.0
    assert s["D4"] == 1.0 and s["D5"] == 0.0
    assert s["D6"] == 0.0, "the answer names the label, not the path"
    assert s["D7"] is None and s["D8"] is None and s["D9"] is None, "no contradiction occurred"


def test_the_absent_transcript_scores_no_read_and_report_absent() -> None:
    c = _ctx([
        _row(1, "list_files", LISTED_WITHOUT, directory=DIRECTORY),
        _row(2, "finish", "", answer="no summary"),
    ])
    s = dx.score(c)
    assert s["D1"] == 0.0 and s["D4"] == 0.0 and s["D5"] == 1.0


def test_search_and_relist_are_distinguished_at_the_first_decision() -> None:
    searched = _ctx([_row(1, "list_files", LISTED_WITHOUT, directory=DIRECTORY),
                     _row(2, "search_files", "MATCHES: (none)", query="summary")])
    relisted = _ctx([_row(1, "list_files", LISTED_WITHOUT, directory=DIRECTORY),
                     _row(2, "list_files", LISTED_WITHOUT, directory=DIRECTORY + "/")])
    assert dx.score(searched)["D2"] == 1.0 and dx.score(searched)["D3"] == 0.0
    assert dx.score(relisted)["D3"] == 1.0 and dx.score(relisted)["D2"] == 0.0


def test_a_transcript_that_never_reaches_a_position_scores_none_not_zero() -> None:
    nothing = _ctx([_row(1, "read_file", "ERROR: file not found: x", path="x")])
    s = dx.score(nothing)
    assert all(v is None for v in s.values()), s
    no_finish = _ctx([_row(1, "list_files", LISTED_WITH, directory=DIRECTORY)])
    assert dx.score(no_finish)["D1"] is None and dx.score(no_finish)["D4"] is None


def test_the_dynamic_diagnostics_read_the_decision_after_the_contradiction() -> None:
    # Told absent, then a truthful relisting shows it present, then the model reads it.
    c = _ctx([
        _row(1, "list_files", LISTED_WITHOUT, directory=DIRECTORY),
        _row(2, "list_files", LISTED_WITH, directory=DIRECTORY),
        _row(3, "read_file", "milestone-1", path=TARGET),
        _row(4, "finish", "", answer="milestone-1"),
    ])
    s = dx.score(c)
    assert c.state_observation_index() == 0 and c.contradicting_observation_index() == 1
    assert s["D1"] == 0.0, "the first decision was to relist, not read"
    assert s["D7"] == 1.0 and s["D8"] == 1.0 and s["D9"] == 0.0


def test_a_failed_read_after_a_present_listing_is_a_contradiction() -> None:
    c = _ctx([
        _row(1, "list_files", LISTED_WITH, directory=DIRECTORY),
        _row(2, "read_file", f"ERROR: file not found: {TARGET}", path=TARGET),
        _row(3, "search_files", "MATCHES: (none)", query="summary"),
        _row(4, "finish", "", answer="no summary"),
    ])
    s = dx.score(c)
    assert c.contradicting_observation_index() == 1
    assert s["D9"] == 1.0 and s["D7"] == 0.0 and s["D8"] == 0.0


def test_d6_reads_the_path_or_its_basename_in_the_answer() -> None:
    by_path = _ctx([_row(1, "list_files", LISTED_WITH, directory=DIRECTORY),
                    _row(2, "finish", "", answer=f"The file {TARGET} exists")])
    by_name = _ctx([_row(1, "list_files", LISTED_WITH, directory=DIRECTORY),
                    _row(2, "finish", "", answer="see summary-42.md")])
    assert dx.score(by_path)["D6"] == 1.0 and dx.score(by_name)["D6"] == 1.0


def test_d10_relation_requires_each_file_to_be_acted_on_by_its_own_state() -> None:
    present = _ctx([_row(1, "list_files", LISTED_WITH, directory=DIRECTORY),
                    _row(2, "read_file", "x", path=TARGET)])
    absent = _ctx([_row(1, "list_files", LISTED_WITHOUT, directory=DIRECTORY),
                   _row(2, "finish", "", answer="no summary")])
    confused = _ctx([_row(1, "list_files", LISTED_WITHOUT, directory=DIRECTORY),
                     _row(2, "read_file", f"ERROR: file not found: {TARGET}", path=TARGET)])
    assert dx.d10_relation(present, absent) == 1.0
    assert dx.d10_relation(present, confused) == 0.0
    assert dx.d10_relation(present, _ctx([])) is None


@pytest.mark.parametrize("name", list(dx.DIAGNOSTICS))
def test_every_diagnostic_is_a_pure_function_of_the_transcript(name) -> None:
    steps = [_row(1, "list_files", LISTED_WITH, directory=DIRECTORY),
             _row(2, "read_file", "m", path=TARGET), _row(3, "finish", "", answer="m")]
    fn = dx.DIAGNOSTICS[name]
    assert fn(_ctx(steps)) == fn(_ctx(steps))
