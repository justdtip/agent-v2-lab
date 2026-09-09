"""The acceptance kit: the golden harness, the measured band, and the gate reporting.

The harness is exercised against synthetic records rather than the stage-two corpus, so this
file runs anywhere and does not depend on a scratchpad path. The corpus itself is checked by
running the kit against it, which is a separate act with its own output.

The property under test throughout is that a check which has not really run cannot report
success: an unimplemented gate is never a pass, a band cannot be measured from nothing, and a
projection cannot exist without the measurement it rests on.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_ACCEPTANCE = Path(__file__).resolve().parents[1] / "research" / "acceptance"
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _path in (_ACCEPTANCE, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import acceptance_gates as gates  # noqa: E402
import golden_trajectories as golden  # noqa: E402
import readout_tolerance as tolerance  # noqa: E402


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_record(path: Path, events: list[dict]) -> None:
    """Write a hash-chained record the way the capture writer does."""
    previous = "0" * 64
    with path.open("w", encoding="utf-8") as handle:
        for sequence, event in enumerate(events):
            payload = {"event": event, "previous": previous, "sequence": sequence}
            digest = hashlib.sha256(_encoded(payload)).hexdigest()
            handle.write(_encoded(payload | {"sha256": digest}).decode() + "\n")
            previous = digest


def _episode_events(label: str, prompt_ids: list[int], emitted: list[int]) -> list[dict]:
    """One turn: a prefill forward, then one single-token forward per emission."""
    events: list[dict] = [
        {
            "kind": "manifest",
            "schema_version": 1,
            "provenance": {"episode": {"label": label, "kind": "agentic"}},
        },
        {"kind": "begin_turn", "turn": 0, "prompt_ids": prompt_ids, "context": {}},
    ]
    sequence = list(prompt_ids) + list(emitted)
    # The forward at offset p predicts position p + 1, which is the convention the whole
    # harness rests on and the one this fixture must reproduce faithfully to be a fixture.
    for index, token in enumerate(emitted):
        offset = len(prompt_ids) - 1 + index
        events.append(
            {
                "kind": "forward",
                "turn": 0,
                "offset": offset,
                "input_ids": sequence[: offset + 1],
                "argmax": [[token]],
                "logits_sha256": f"digest-{offset}",
                "logits_shape": [1, 1, 32],
            }
        )
        events.append({"kind": "reading", "turn": 0, "position": offset, "top": {"34": [token]}})
        events.append(
            {
                "kind": "emitted",
                "turn": 0,
                "position": len(prompt_ids) + index,
                "token_id": token,
                "span": "note",
            }
        )
    events.append(
        {
            "kind": "end_turn",
            "turn": 0,
            "status": "complete",
            "emitted_count": len(emitted),
            "forwarded_count": len(sequence),
        }
    )
    events.append({"kind": "end_record", "status": "complete"})
    return events


@pytest.fixture
def records(tmp_path: Path) -> Path:
    directory = tmp_path / "stage2"
    directory.mkdir()
    _write_record(directory / "one.jsonl", _episode_events("one", [5, 6, 7], [11, 12, 13, 14]))
    _write_record(directory / "two.jsonl", _episode_events("two", [1, 2], [21, 22]))
    return directory


def test_the_harness_reads_a_record_and_its_own_oracle_holds(records: Path) -> None:
    episodes = golden.load_episodes(records)
    assert [episode.label for episode in episodes] == ["one", "two"]
    assert [episode.emission_count for episode in episodes] == [4, 2]
    for episode in episodes:
        consistency = golden.record_consistency(episode)
        assert consistency.passed
        assert consistency.disagreements == 0 and consistency.missing_forwards == 0


def test_a_broken_position_convention_is_caught_by_the_oracle(tmp_path: Path) -> None:
    """The error class the bookkeeping check exists for: forwards off by one."""
    directory = tmp_path / "broken"
    directory.mkdir()
    events = _episode_events("shifted", [5, 6, 7], [11, 12, 13])
    for event in events:
        if event["kind"] == "forward":
            event["offset"] += 1
    _write_record(directory / "shifted.jsonl", events)
    consistency = golden.record_consistency(golden.load_episodes(directory)[0])
    assert not consistency.passed
    assert consistency.missing_forwards > 0
    assert consistency.detail, "a failure must say where, not only that"


def test_a_tampered_record_does_not_load(records: Path) -> None:
    path = records / "one.jsonl"
    lines = path.read_text().splitlines()
    payload = json.loads(lines[1])
    payload["event"]["prompt_ids"] = [9, 9, 9]
    lines[1] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="hash chain"):
        golden.load_episode(path)


def test_reproduction_reports_the_first_divergence_with_both_sides(records: Path) -> None:
    episode = golden.load_episodes(records)[0]
    honest = golden.recorded_generator(episode)
    assert golden.reproduce(episode, honest).reproduced

    def wrong_at_two(prompt_ids, *, max_tokens):
        generated = honest(prompt_ids, max_tokens=max_tokens)
        tokens = list(generated.token_ids)
        tokens[2] = 999
        return golden.GeneratedTurn(token_ids=tuple(tokens), final_top=generated.final_top)

    result = golden.reproduce(episode, wrong_at_two)
    divergence = result.first_divergence
    assert not result.reproduced
    assert divergence.index == 2
    assert divergence.expected_token == 13 and divergence.actual_token == 999
    assert divergence.position == len(episode.turns[0].prompt_ids) + 2
    assert divergence.expected_final_top == (13,), "the recorded layer-34 top travels with it"


def test_readout_agreement_says_when_bit_identity_was_not_assessed(records: Path) -> None:
    episode = golden.load_episodes(records)[0]
    agreement = golden.readout_agreement(episode, golden.recorded_generator(episode))
    assert agreement.compared == 4 and agreement.rate == 1.0
    assert agreement.digests_compared == 0
    assert "not assessed" in agreement.digest_note, (
        "a generator that supplies no digests must not produce a bit-identity number"
    )


def test_a_band_is_measured_and_carries_what_it_was_measured_on() -> None:
    deltas = [0.1 * index for index in range(1, 1001)]
    band = tolerance.measure_band(deltas, basis="cpu fp32 against mlx 4-bit, 15 episodes")
    assert band.n == 1000
    assert band.value == pytest.approx(99.9), (
        "the 0.999 quantile of a thousand samples is the 999th of them, not the largest; a "
        "band that silently equalled the maximum would be a chosen number wearing a quantile"
    )
    assert band.maximum == pytest.approx(100.0)
    assert band.median == pytest.approx(50.05)
    assert band.measured
    assert "15 episodes" in band.describe()


def test_a_band_cannot_be_conjured_from_nothing() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        tolerance.measure_band([], basis="anything")
    with pytest.raises(ValueError, match="must carry the measurement"):
        tolerance.measure_band([1.0], basis="   ")
    with pytest.raises(ValueError, match="NaN"):
        tolerance.measure_band([1.0, float("nan")], basis="a sample with a hole in it")


def test_a_projection_carries_its_basis_and_refuses_to_exist_without_one() -> None:
    measured = tolerance.measure_band([1.0, 2.0, 3.0], basis="cpu fp32 against mlx 4-bit")
    projected = tolerance.project_band(
        measured, factor=4.0, reason="bf16 has 8 fewer mantissa bits"
    )
    assert not projected.measured
    assert projected.value == pytest.approx(measured.value * 4.0)
    assert projected.projected_from is measured
    described = projected.describe()
    assert "projected from" in described and "cpu fp32 against mlx 4-bit" in described, (
        "a projection without its basis cannot be learned from, only failed"
    )
    with pytest.raises(ValueError, match="why its factor"):
        tolerance.project_band(measured, factor=2.0, reason="")
    with pytest.raises(ValueError, match="positive"):
        tolerance.project_band(measured, factor=0.0, reason="a reason")


def test_a_band_check_reports_the_worst_case_not_only_the_verdict() -> None:
    band = tolerance.measure_band([1.0, 1.0, 1.0], basis="a flat sample")
    inside = tolerance.check_band(band, [0.5, 0.9, 1.0])
    assert inside.passed and inside.worst == pytest.approx(1.0)
    outside = tolerance.check_band(band, [0.5, 7.25])
    assert not outside.passed and outside.exceeded == 1
    assert "7.25" in outside.describe()


def test_an_unimplemented_gate_is_never_a_pass(records: Path) -> None:
    exit_code = gates.main(["--records", str(records), "--keep-going"])
    assert exit_code == 1, "the kit is not green while gates remain unrunnable"
    for gate in gates.GATES:
        assert gate.owner in {"WS-A", "WS-B", "WS-D"}


def test_the_kit_reports_gate_five_unavailable_rather_than_passing(records: Path) -> None:
    arguments = gates.argparse.Namespace(records=records, model=None, keep_going=True)
    result = gates.gate_5_golden_trajectories(arguments)
    assert result.status == gates.UNAVAILABLE, (
        "records that read cleanly are not a reproduction, and reporting them as one would "
        "make the kit's green run meaningless"
    )
    assert any("bookkeeping" in note for note in result.notes)


def test_the_kit_fails_gate_five_when_the_record_disagrees_with_itself(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    events = _episode_events("bad", [5, 6], [11, 12])
    for event in events:
        if event["kind"] == "forward":
            event["argmax"] = [[404]]
    _write_record(directory / "bad.jsonl", events)
    arguments = gates.argparse.Namespace(records=directory, model=None, keep_going=True)
    result = gates.gate_5_golden_trajectories(arguments)
    assert result.status == gates.FAIL and result.blocking
    assert "disagreement" in result.saw


def test_the_kit_refuses_to_load_a_model_without_a_box_window(records: Path, monkeypatch) -> None:
    from local_llm_lab import runlock

    monkeypatch.setattr(runlock, "read_window", lambda *args, **kwargs: None)
    arguments = gates.argparse.Namespace(records=records, model="gemma3-4b", keep_going=True)
    with pytest.raises(SystemExit, match="no box window"):
        gates._load_backend(arguments)
