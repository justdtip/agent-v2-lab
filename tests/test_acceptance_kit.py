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
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ACCEPTANCE = Path(__file__).resolve().parents[1] / "research" / "acceptance"
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _path in (_ACCEPTANCE, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import acceptance_gates as gates  # noqa: E402
import golden_trajectories as golden  # noqa: E402
import readout_tolerance as readout  # noqa: E402
import tolerance  # noqa: E402


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


#: Deliberately not 34. Gemma 3 4B has 34 layers and the GPU is expected to bring other
#: models, so a fixture at 34 would pass against a reader that had the depth hardcoded.
FIXTURE_LAYERS = list(range(1, 13))
FINAL_LAYER = max(FIXTURE_LAYERS)


def _episode_events(label: str, prompt_ids: list[int], emitted: list[int]) -> list[dict]:
    """One turn: a prefill forward, then one single-token forward per emission."""
    events: list[dict] = [
        {
            "kind": "manifest",
            "schema_version": 1,
            "provenance": {"episode": {"label": label, "kind": "agentic"}},
        },
        {
            "kind": "begin_turn",
            "turn": 0,
            "prompt_ids": prompt_ids,
            "context": {},
            "layers": FIXTURE_LAYERS,
        },
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
        events.append(
            {
                "kind": "reading",
                "turn": 0,
                "position": offset,
                "top": {str(FINAL_LAYER): [token]},
            }
        )
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
    band = readout.measure_band(deltas, basis="cpu fp32 against mlx 4-bit, 15 episodes")
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
        readout.measure_band([], basis="anything")
    with pytest.raises(ValueError, match="must carry the measurement"):
        readout.measure_band([1.0], basis="   ")
    with pytest.raises(ValueError, match="NaN"):
        readout.measure_band([1.0, float("nan")], basis="a sample with a hole in it")


def test_a_projection_carries_its_basis_and_refuses_to_exist_without_one() -> None:
    measured = readout.measure_band([1.0, 2.0, 3.0], basis="cpu fp32 against mlx 4-bit")
    projected = readout.project_band(measured, factor=4.0, reason="bf16 has 8 fewer mantissa bits")
    assert not projected.measured
    assert projected.value == pytest.approx(measured.value * 4.0)
    assert projected.projected_from is measured
    described = projected.describe()
    assert "projected from" in described and "cpu fp32 against mlx 4-bit" in described, (
        "a projection without its basis cannot be learned from, only failed"
    )
    with pytest.raises(ValueError, match="why its factor"):
        readout.project_band(measured, factor=2.0, reason="")
    with pytest.raises(ValueError, match="positive"):
        readout.project_band(measured, factor=0.0, reason="a reason")


def test_a_band_check_reports_the_worst_case_not_only_the_verdict() -> None:
    band = readout.measure_band([1.0, 1.0, 1.0], basis="a flat sample")
    inside = readout.check_band(band, [0.5, 0.9, 1.0])
    assert inside.passed and inside.worst == pytest.approx(1.0)
    outside = readout.check_band(band, [0.5, 7.25])
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


# --- G-2(b): the tolerance half -------------------------------------------------------------


def _with_confidence(events: list[dict], probability: float) -> list[dict]:
    """Add the layer-34 horizon-1 rank rows the P >= 0.99 rule reads."""
    extra = []
    for event in events:
        if event["kind"] == "emitted":
            extra.append(
                {
                    "kind": "rank",
                    "turn": event["turn"],
                    "layer": FINAL_LAYER,
                    "horizon": 1,
                    "position": event["position"] - 1,
                    "token_id": event["token_id"],
                    "rank": 1,
                    "probability": probability,
                }
            )
    return events[:-1] + extra + events[-1:]


def test_teacher_forcing_keeps_every_position_an_independent_comparison(tmp_path: Path) -> None:
    directory = tmp_path / "tf"
    directory.mkdir()
    events = _with_confidence(_episode_events("tf", [5, 6], [11, 12, 13, 14]), 0.5)
    _write_record(directory / "tf.jsonl", events)
    episode = golden.load_episodes(directory)[0]

    produced = {
        (0, position): token for position, token in zip([2, 3, 4, 5], [11, 12, 13, 14], strict=True)
    }
    perfect = tolerance.teacher_forced_agreement(episode, produced)
    assert perfect.compared == 4 and perfect.rate == 1.0 and perfect.passed

    produced[(0, 3)] = 999
    one_flip = tolerance.teacher_forced_agreement(episode, produced)
    assert one_flip.compared == 4, (
        "a flip at one position does not stop the other three being compared; that is what "
        "teacher forcing buys and free running does not"
    )
    assert one_flip.agreed == 3 and len(one_flip.flips) == 1


def test_a_flip_at_high_recorded_confidence_fails_the_run(tmp_path: Path) -> None:
    directory = tmp_path / "hard"
    directory.mkdir()
    _write_record(
        directory / "hard.jsonl",
        _with_confidence(_episode_events("hard", [5, 6], [11, 12]), 0.999998),
    )
    episode = golden.load_episodes(directory)[0]

    clean = tolerance.teacher_forced_agreement(episode, {(0, 2): 11, (0, 3): 12})
    assert clean.passed and not tolerance.confidence_violations(clean)

    flipped = tolerance.teacher_forced_agreement(episode, {(0, 2): 11, (0, 3): 404})
    violations = tolerance.confidence_violations(flipped)
    assert len(violations) == 1 and violations[0].hard
    assert not flipped.passed, (
        "quantisation does not move an argmax at P = 0.999998; a flip there is a mask, "
        "position, entry or norm defect and fails outright"
    )
    assert "HARD" in violations[0].describe()


def test_a_flip_at_low_recorded_confidence_is_reported_and_not_gated(tmp_path: Path) -> None:
    directory = tmp_path / "soft"
    directory.mkdir()
    _write_record(
        directory / "soft.jsonl",
        _with_confidence(_episode_events("soft", [5, 6], [11, 12]), 0.51),
    )
    episode = golden.load_episodes(directory)[0]
    flipped = tolerance.teacher_forced_agreement(episode, {(0, 2): 11, (0, 3): 404})
    assert len(flipped.flips) == 1 and not flipped.hard_flips
    assert flipped.passed, "a near-tie can flip on precision alone and is not a defect"


def test_a_position_with_no_recorded_probability_is_counted_not_assumed(records: Path) -> None:
    """The fixture records carry no rank rows, so the hard rule cannot apply to them."""
    episode = golden.load_episodes(records)[0]
    report = tolerance.teacher_forced_agreement(episode, {(0, 3): 999})
    assert report.compared == 1 and report.unrecorded_probability == 1
    assert not report.hard_flips, "an unrecorded probability is never treated as a high one"
    assert "could not be applied" in report.describe()


def test_the_divergence_profile_reports_a_floor(records: Path) -> None:
    class _Result:
        def __init__(self, index):
            self.first_divergence = None if index is None else SimpleNamespace(index=index)

    profile = tolerance.divergence_indices(
        [_Result(None), _Result(120), _Result(3), _Result(64)], floor=16
    )
    assert profile.reproduced == 1
    assert profile.below_floor == [3]
    assert not profile.passed, "an episode that parts company at token 3 did not drift there"
    assert profile.percentile(0.5) == 64
    assert "floor of 16" in profile.describe()


def test_top_k_jaccard_uses_ids_because_that_is_all_the_record_has(records: Path) -> None:
    episode = golden.load_episodes(records)[0]
    # Position 3 is the first emission, token 11, and its reading sits at position 2.
    identical = tolerance.top_k_jaccard(episode, {(0, 3): [11]}, k=5)
    assert identical.mean == 1.0
    disjoint = tolerance.top_k_jaccard(episode, {(0, 3): [777]}, k=5)
    assert disjoint.mean == 0.0
    assert "top-5 Jaccard" in identical.describe()


def test_a_divergence_refuses_a_truncated_distribution() -> None:
    """The records cannot supply a layer-34 distribution, only a top-k slice and one probability."""
    with pytest.raises(ValueError, match="not a distribution"):
        tolerance.symmetric_kl([0.6, 0.3], [0.5, 0.4])
    with pytest.raises(ValueError, match="differ in support"):
        tolerance.symmetric_kl([0.5, 0.5], [1.0])
    same = tolerance.symmetric_kl([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])
    assert same == pytest.approx(0.0)
    apart = tolerance.symmetric_kl([1.0, 0.0], [0.0, 1.0])
    assert apart == pytest.approx(math.log(2))


def _fake_forward(rows_by_sequence: dict[tuple[int, ...], list]):
    def forward(sequence):
        return rows_by_sequence[tuple(sequence)]

    return forward


def test_the_runner_reads_each_deciding_position_with_the_recorded_prefix(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    _write_record(
        directory / "run.jsonl",
        _with_confidence(_episode_events("run", [5, 6], [11, 12, 13]), 0.999998),
    )
    episode = golden.load_episodes(directory)[0]

    # One row per input position. Position p's row is the prediction for p + 1, so the rows at
    # 1, 2 and 3 must name 11, 12 and 13. Position 0 and the last row are never read.
    sequence = (5, 6, 11, 12, 13)
    rows = [(999, (999,)), (11, (11,)), (12, (12,)), (13, (13,)), (0, (0,))]
    report = tolerance.run_tolerance(episode, _fake_forward({sequence: rows}))

    assert report.agreement.compared == 3 and report.agreement.rate == 1.0
    assert report.passed
    assert report.jaccard.mean == 1.0
    assert report.divergence is None
    assert "not measured" in report.describe()

    # Jaccard is over sets, so producing the right token plus extras is not a perfect score.
    # The fixture records a single id per position; a two-id top halves the overlap. That is
    # the statistic working, and it is worth pinning so nobody later "fixes" it into a
    # top-1 agreement rate wearing a Jaccard's name.
    wider = [(999, (999,)), (11, (11, 7)), (12, (12, 7)), (13, (13, 7)), (0, (0,))]
    padded = tolerance.run_tolerance(episode, _fake_forward({sequence: wider}))
    assert padded.agreement.rate == 1.0
    assert padded.jaccard.mean == 0.5


def test_the_runner_refuses_a_row_count_that_would_shift_the_join(tmp_path: Path) -> None:
    """An off-by-one here shifts every comparison and still yields a plausible rate."""
    directory = tmp_path / "shift"
    directory.mkdir()
    _write_record(directory / "shift.jsonl", _episode_events("shift", [5, 6], [11, 12]))
    episode = golden.load_episodes(directory)[0]
    with pytest.raises(ValueError, match="shifted and still look plausible"):
        tolerance.run_tolerance(episode, _fake_forward({(5, 6, 11, 12): [(0, (0,))] * 3}))


def test_the_runner_fails_on_a_confident_flip_and_survives_an_unconfident_one(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "flip"
    directory.mkdir()
    _write_record(
        directory / "hard.jsonl",
        _with_confidence(_episode_events("hard", [5, 6], [11, 12]), 0.999998),
    )
    _write_record(
        directory / "soft.jsonl",
        _with_confidence(_episode_events("soft", [5, 6], [11, 12]), 0.55),
    )
    hard, soft = golden.load_episodes(directory)

    rows = [(0, (0,)), (11, (11,)), (404, (404,)), (0, (0,))]
    forward = _fake_forward({(5, 6, 11, 12): rows})

    assert not tolerance.run_tolerance(hard, forward).passed
    assert tolerance.run_tolerance(soft, forward).passed, (
        "a near-tie can flip on precision alone; only a confident flip is a defect"
    )


def test_the_runner_gates_on_the_divergence_floor_too(tmp_path: Path) -> None:
    directory = tmp_path / "floor"
    directory.mkdir()
    _write_record(
        directory / "floor.jsonl",
        _with_confidence(_episode_events("floor", [5, 6], [11, 12]), 0.5),
    )
    episode = golden.load_episodes(directory)[0]
    rows = [(0, (0,)), (11, (11,)), (12, (12,)), (0, (0,))]
    forward = _fake_forward({(5, 6, 11, 12): rows})

    class _Result:
        def __init__(self, index):
            self.first_divergence = None if index is None else SimpleNamespace(index=index)

    early = tolerance.run_tolerance(episode, forward, free_running=[_Result(2)], floor=16)
    assert not early.passed, "an episode that parts company at token 2 did not drift there"
    late = tolerance.run_tolerance(episode, forward, free_running=[_Result(120)], floor=16)
    assert late.passed


def test_an_interrupted_run_leaves_what_completed_on_disk(records: Path, tmp_path: Path) -> None:
    """The defect that cost ten minutes of box time and recovered nothing.

    The runner accumulated every result and wrote once at the end, so an interrupt partway
    through lost all of it. On a laptop that is ten wasted minutes; on a rented device it is a
    paid hour with nothing to show for where it failed.
    """
    import tolerance_baseline as runner

    episodes = golden.load_episodes(records)
    assert len(episodes) == 2
    per_episode = tmp_path / "rows.jsonl"

    calls = {"n": 0}

    def forward(sequence):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt("stopped partway, as a run on a shared box is")
        return [(0, (0,))] * len(sequence)

    with pytest.raises(KeyboardInterrupt):
        runner.run_episodes(episodes, forward, per_episode=per_episode)

    rows = [json.loads(line) for line in per_episode.read_text().splitlines()]
    assert len(rows) == 1, "the episode that finished is on disk; the one that did not is not"
    assert rows[0]["label"] == episodes[0].label
    assert set(rows[0]) >= {"compared", "agreed", "hard_flips", "confident_positions", "seconds"}


def test_no_result_file_is_written_when_none_was_asked_for(records: Path) -> None:
    import tolerance_baseline as runner

    episodes = golden.load_episodes(records)[:1]
    reports = runner.run_episodes(
        episodes, lambda sequence: [(0, (0,))] * len(sequence), per_episode=None
    )
    assert len(reports) == 1
