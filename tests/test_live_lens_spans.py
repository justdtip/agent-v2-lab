"""The span join's gate, exercised on synthetic rows so it runs beside a model run.

No MLX and no model: every case here builds capture rows by hand, which is also the only way to
test the failures that matter. Real records are joined by ``spans.join_record``, and they are all
correct -- so what a real record cannot show is what happens when one is *wrong*, which is the
only question a gate is asked.

The negative controls are the point of the file. A gate that cannot fail is not a gate, and every
wrong join this module guards against returns a full, plausible table rather than raising -- so
each control below corrupts one row in one way and asserts that the join refuses the whole record.
"""

from __future__ import annotations

import pytest

from local_llm_lab.pipeline.live_lens.spans import (
    ScoredToken,
    SpanJoinError,
    attach_spans,
    join_spans,
)

LAYERS = (11, 34)
HORIZONS = (1, 4)
CAPACITY = 16


def _turn(
    turn: int,
    *,
    prompt_length: int,
    tokens: list[int],
    spans: list[str] | None,
    layers: tuple[int, ...] = LAYERS,
    horizons: tuple[int, ...] = HORIZONS,
    capacity: int = CAPACITY,
    lookahead: int = 0,
    status: str = "complete",
) -> list[dict]:
    """One turn's rows in the order ``CaptureSession`` writes them.

    The order is reproduced rather than approximated, because the module replays it: a ``reading``
    row per captured position in capture order, then per emitted token an ``emitted`` row followed
    immediately by its rank rows, one per (horizon, layer) whose source is still buffered. Prompt
    positions are captured during prefill, which is why the first tokens of a turn are scored
    against sources that lie in the prompt.

    ``lookahead`` forwards that many positions past the last emitted token, as a generator that
    reads one step ahead does; those captures produce no rank rows and are the turn's censoring.
    """
    rows: list[dict] = [
        {
            "kind": "begin_turn",
            "turn": turn,
            "prompt_ids": list(range(1000, 1000 + prompt_length)),
            "layers": list(layers),
            "rank_horizons": list(horizons),
            "distribution_capacity": capacity,
        }
    ]
    buffered: list[int] = []
    captured = 0

    def capture(position: int) -> None:
        nonlocal captured
        rows.append({"kind": "reading", "turn": turn, "position": position, "top": {}})
        buffered.append(position)
        del buffered[: max(0, len(buffered) - capacity)]
        captured += 1

    for position in range(prompt_length):
        capture(position)
    for index, token in enumerate(tokens):
        position = prompt_length + index
        emitted = {"kind": "emitted", "turn": turn, "position": position, "token_id": token}
        if spans is not None:
            emitted["span"] = spans[index]
        rows.append(emitted)
        for horizon in horizons:
            source = position - horizon
            if source not in buffered:
                continue
            for layer in layers:
                rows.append(
                    {
                        "kind": "rank",
                        "turn": turn,
                        "position": source,
                        "layer": layer,
                        "horizon": horizon,
                        "token_id": token,
                        "rank": 1 + (layer % 7) + horizon,
                        "probability": 0.5,
                    }
                )
        if index + 1 < len(tokens) or lookahead:
            capture(position)
    for step in range(1, lookahead):
        capture(prompt_length + len(tokens) - 1 + step)
    rows.append(
        {
            "kind": "end_turn",
            "turn": turn,
            "emitted_count": len(tokens),
            # One reading row per capture, which is what `end_turn.forwarded_count` counts.
            "forwarded_count": captured,
            "status": status,
        }
    )
    return rows


def _record(*turns: list[dict]) -> list[dict]:
    rows: list[dict] = [{"kind": "manifest", "schema_version": 1, "provenance": {}}]
    for turn in turns:
        rows.extend(turn)
    rows.append({"kind": "end_record", "status": "complete"})
    return rows


SPANS = ["note", "note", "call_skeleton", "call_argument", "call_argument", "call_skeleton"]
TOKENS = [70, 71, 72, 73, 74, 75]


def _simple(**overrides) -> list[dict]:
    return _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS, **overrides))


def test_the_scored_token_is_the_one_h_steps_after_the_row_s_own_position():
    """The relation itself: rank(turn, P, h) scores emitted(turn, P + h), never emitted(turn, P).

    Both fields are called ``position`` and both index ``prompt_ids + generated``, so this is the
    off-by-horizon trap in its simplest form. The assertion is on the row's coordinates rather than
    on a count, because a count is satisfied by the wrong join too.
    """
    rows = attach_spans(_simple())

    for row in rows:
        assert row.position == row.source_position + row.horizon
        assert row.token_id == TOKENS[row.generated_index]
        assert row.span == SPANS[row.generated_index]
    assert {(row.horizon, row.source_position) for row in rows if row.generated_index == 4} == {
        (1, 12),
        (4, 9),
    }


def test_the_count_is_emitted_times_horizons_times_layers_and_the_shortfall_is_itemised():
    """The naive product holds here, and the census says so rather than the assertion assuming it.

    Nine prompt tokens is more than the greatest horizon, so every source exists and the product
    is exact. The interesting figure is ``prompt_sourced_rows``: the first token of a turn is
    scored against the tail of the prompt, and dropping those rows would shrink h=4 by four rows
    per turn and h=1 by one, making a horizon comparison run on unequal n.
    """
    join = join_spans(_simple())

    assert join.rank_rows == len(TOKENS) * len(HORIZONS) * len(LAYERS) == 24
    assert join.expected_rank_rows == join.nominal_rank_rows == join.rank_rows
    assert join.absences == ()
    assert join.prompt_sourced_rows == (1 + 4) * len(LAYERS)
    assert join.emitted_rows == len(TOKENS)
    assert join.layers == LAYERS and join.horizons == HORIZONS
    assert join.spans_present is True


def test_a_disagreeing_token_id_aborts_and_names_the_row():
    """The negative control the whole module exists for.

    One rank row is left scoring a token the emitted row does not carry, which is what any wrong
    offset produces on most of its rows. It must abort, on that row, naming both ids -- not warn,
    not drop the row, not report a match rate.
    """
    rows = _simple()
    corrupted = next(r for r in rows if r["kind"] == "rank" and r["horizon"] == 4)
    corrupted["token_id"] = 999

    with pytest.raises(SpanJoinError, match="token id disagreement") as error:
        attach_spans(rows)
    message = str(error.value)
    assert "turn 0" in message and "horizon 4" in message
    assert "999" in message and str(TOKENS[corrupted["position"] + 4 - 9]) in message


def test_every_row_is_checked_rather_than_a_sample():
    """A single corrupted row anywhere in the record aborts, whichever row it is.

    Sampling would pass this a good fraction of the time, and the join it would pass is the one
    that put a wrong claim into a published record on 2026-09-08.
    """
    baseline = _simple()
    indices = [i for i, row in enumerate(baseline) if row["kind"] == "rank"]
    assert len(indices) == 24

    for index in indices:
        rows = _simple()
        rows[index] = dict(rows[index], token_id=4242)
        with pytest.raises(SpanJoinError, match="token id disagreement"):
            attach_spans(rows)


def test_a_missing_rank_row_aborts_rather_than_shrinking_a_facet():
    """Row loss is the failure that looks like data. It must not be absorbed into the count.

    A dropped row leaves a partial layer block, which is exactly what an eviction or a filtered
    read produces, and the result is a facet with fewer rows in one cell and no sign of it.
    """
    rows = _simple()
    dropped = next(i for i, row in enumerate(rows) if row["kind"] == "rank")
    del rows[dropped]

    with pytest.raises(SpanJoinError, match="layers"):
        attach_spans(rows)


def test_a_missing_whole_block_aborts_against_the_replayed_expectation():
    """Removing an entire layer block cannot hide behind the block's own completeness check."""
    rows = [
        row
        for row in _simple()
        if not (row["kind"] == "rank" and row["position"] == 12 and row["horizon"] == 1)
    ]

    with pytest.raises(SpanJoinError, match="predicted rank blocks are absent"):
        attach_spans(rows)


def test_rows_written_under_the_off_by_horizon_convention_are_refused():
    """The primary trap, simulated at the writer rather than at the reader.

    Every rank row here carries the scored token's own position instead of the source residual's,
    which is the convention a reader is most likely to assume and the one that makes
    ``rank.position == emitted.position`` look right. The whole table still looks well formed --
    same rows, same tokens, same layer blocks -- and the join must refuse it rather than return a
    facet that is h positions early.
    """
    rows = [
        dict(row, position=row["position"] + row["horizon"]) if row["kind"] == "rank" else row
        for row in _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    ]

    with pytest.raises(SpanJoinError) as error:
        attach_spans(rows)
    assert "does not predict" in str(error.value) or "absent" in str(error.value)


def test_a_rank_row_the_replay_does_not_predict_aborts():
    """The other direction: a row from somewhere else, at a source no capture ever held."""
    rows = _simple()
    stray = dict(next(r for r in rows if r["kind"] == "rank"), position=3, horizon=1, token_id=70)
    rows.insert(-2, stray)

    with pytest.raises(SpanJoinError, match="does not predict"):
        attach_spans(rows)


def test_an_evicted_source_aborts_instead_of_passing_as_censoring():
    """Buffer eviction is silent row loss, and it is not the same thing as a censored future.

    With a capacity of 2 and a horizon of 4, the source of every h=4 score has been pushed out of
    the buffer by the time its token is emitted, so those rows do not exist. Nothing in the rows
    themselves distinguishes that from ordinary end-of-turn censoring -- both are just absent rows
    -- which is why the expectation is replayed from the capture grid, and why this case is
    refused by name instead of quietly shrinking the h=4 population.
    """
    rows = _record(
        _turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS, capacity=2, horizons=(1, 4))
    )

    with pytest.raises(SpanJoinError, match="evicted"):
        attach_spans(rows)


def test_a_horizon_reaching_past_the_start_of_the_turn_is_allowed_and_itemised():
    """The one legitimate absence: the source never existed, so no row was lost.

    A two-token prompt with a horizon of 4 means the first two generated tokens have no source at
    all. Those blocks are absent, the count says so by name, and the ones that do exist are still
    checked exactly.
    """
    join = join_spans(
        _record(_turn(0, prompt_length=2, tokens=TOKENS, spans=SPANS, horizons=(1, 4)))
    )

    assert dict(join.absences) == {"before_turn": 2 * len(LAYERS)}
    assert join.nominal_rank_rows == len(TOKENS) * 2 * len(LAYERS) == 24
    assert join.expected_rank_rows == 24 - 2 * len(LAYERS) == 20
    assert join.rank_rows == 20
    assert len(join.rows) == 20


def test_censored_futures_are_counted_in_rows_so_they_balance_what_they_are_read_against():
    """The tail of a turn produces sources whose scored token is never emitted.

    They are counted for the reader and are absent from the expectation for a structural reason,
    not an exception: the expectation is built from the emitted side, so a source with no future
    predicts nothing. ``lookahead`` here is the generator's un-emitted read-ahead step.

    **The unit is rows, and it was pairs until an attack tried to check the argument the module's
    own docstring makes.** That argument is a reconciliation — source-side censoring is balanced by
    prompt-sourced rows — and the two figures were being reported in different units, differing by
    the layer count and never balancing. A figure whose only purpose is to let a reader verify a
    reconciliation has to be in the reconciliation's units.
    """
    join = join_spans(
        _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS, lookahead=1))
    )

    assert join.censored_futures == sum(HORIZONS) * len(LAYERS)
    assert join.censored_futures == join.prompt_sourced_rows, (
        "the reconciliation the docstring rests the count assertion on, now checkable"
    )
    assert join.rank_rows == join.expected_rank_rows == join.nominal_rank_rows == 24


def test_positions_are_per_turn_and_a_flat_key_would_cross_two_turns():
    """Two turns re-encoding from zero use the same integers for different tokens.

    This is the shape of the 2026-09-08 error. The join must keep the rows apart, so the same
    position in the two turns carries the two turns' own tokens and the two turns' own spans.
    """
    second = [t + 100 for t in TOKENS]
    join = join_spans(
        _record(
            _turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS),
            _turn(1, prompt_length=9, tokens=second, spans=["chat_prose"] * len(second)),
        )
    )

    assert join.turns == (0, 1)
    assert join.rank_rows == 48
    by_turn = join.facet("turn")
    assert {row.token_id for row in by_turn[(0,)]} == set(TOKENS)
    assert {row.token_id for row in by_turn[(1,)]} == set(second)
    assert {row.span for row in by_turn[(1,)]} == {"chat_prose"}
    shared = {row.source_position for row in by_turn[(0,)]} & {
        row.source_position for row in by_turn[(1,)]
    }
    assert shared, "the two turns must share position integers or this proves nothing"


def test_rows_from_two_turns_interleaved_abort():
    """A merged or reordered file cannot be joined, and says so rather than keying anyway."""
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    rows.insert(-2, dict(rows[5], turn=1))

    with pytest.raises(SpanJoinError, match="interleaved"):
        attach_spans(rows)


def test_a_duplicate_rank_row_aborts_instead_of_replacing_its_twin():
    """Duplicate keys are what a dictionary-building join hides best."""
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    rows.insert(-2, dict(next(r for r in rows if r["kind"] == "rank")))

    with pytest.raises(SpanJoinError, match="two rank rows share"):
        attach_spans(rows)


def test_a_repeated_horizon_aborts_because_it_would_double_the_rows():
    """``records.py`` does not de-duplicate horizons, unlike layers, so a repeat is possible.

    It would write every row twice and inflate every per-horizon count, and the duplicate rows
    would be identical, so nothing downstream could tell them apart.
    """
    rows = _record(
        _turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS, horizons=(1, 1, 4))
    )

    with pytest.raises(SpanJoinError, match="repeats a horizon"):
        attach_spans(rows)


def test_an_emitted_row_outside_the_prompt_plus_generated_coordinate_aborts():
    """Emitted positions run contiguously from ``len(prompt_ids)``; anything else is a coordinate
    error, and it is the coordinate the join keys on."""
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    emitted = next(r for r in rows if r["kind"] == "emitted")
    emitted["position"] = 0

    with pytest.raises(SpanJoinError, match="coordinate"):
        attach_spans(rows)


def test_the_ledger_s_emitted_count_must_match_the_rows_present():
    """``end_turn`` carries the turn's own count, so a lost emitted row is caught by the record."""
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    next(r for r in rows if r["kind"] == "end_turn")["emitted_count"] = 99

    with pytest.raises(SpanJoinError, match="emitted_count"):
        attach_spans(rows)


def test_a_record_without_spans_is_refused_by_default_and_joinable_on_request():
    """Records written before 2026-09-08 carry no ``span`` field, and the facet is not recoverable.

    Defaulting to a bare join would let a span-faceted table be built out of a missing field. The
    refusal names the escape, and taking it puts ``None`` in every span and False in
    ``spans_present``, so a grouped result cannot pass itself off as a segmentation.
    """
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=None))

    with pytest.raises(SpanJoinError, match="predates write-time span labelling"):
        attach_spans(rows)

    join = join_spans(rows, require_spans=False)
    assert join.rank_rows == 24
    assert join.spans_present is False
    assert {row.span for row in join.rows} == {None}


def test_an_aborted_turn_is_left_out_and_named():
    """Its rows are real but the turn stopped for a reason nobody has read.

    A partial turn's last emitted row may be missing the rank rows that were about to follow it,
    which is row-for-row indistinguishable from the loss this module refuses. Excluding it is
    reported in the census rather than done quietly.
    """
    join = join_spans(
        _record(
            _turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS),
            _turn(1, prompt_length=9, tokens=TOKENS, spans=SPANS, status="aborted"),
        )
    )

    assert join.turns == (0,)
    assert join.excluded_turns == (1,)
    assert join.rank_rows == 24


def test_an_incomplete_record_is_refused():
    """A fragment's final turn may be missing rank rows with nothing in the file to say so."""
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    rows[-1] = {"kind": "end_record", "status": "aborted"}

    with pytest.raises(SpanJoinError, match="fragment"):
        attach_spans(rows)


def test_a_turn_with_no_end_turn_row_aborts():
    """Whether it finished is unknown, and the answer decides whether its tail is censored."""
    rows = [row for row in _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
            if row["kind"] != "end_turn"]

    with pytest.raises(SpanJoinError, match="no end_turn"):
        attach_spans(rows)


def test_a_zero_rank_aborts_because_the_convention_would_be_off_by_one():
    """Competition ranks start at 1; a 0 means the rows were written by something else."""
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    next(r for r in rows if r["kind"] == "rank")["rank"] = 0

    with pytest.raises(SpanJoinError, match="competition"):
        attach_spans(rows)


def test_capture_positions_must_increase_within_a_turn():
    """``FutureRanks.capture`` refuses a repeat, so a record holding one was not written by it."""
    rows = _record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS))
    index = next(i for i, row in enumerate(rows) if row["kind"] == "reading")
    rows.insert(index + 1, dict(rows[index]))

    with pytest.raises(SpanJoinError, match="does not exceed"):
        attach_spans(rows)


def test_the_facet_groups_by_the_scored_token_s_span():
    """What the module is for: ranks by layer, horizon and span, over the right tokens.

    The counts here are the check. Each span's cell holds that span's token count times the layers
    times the horizons, which only comes out right if every score landed on the token whose label
    it carries.
    """
    join = join_spans(_simple())
    grouped = join.facet("span", "horizon", "layer")

    assert grouped[("call_argument", 1, 34)] and len(grouped[("call_argument", 1, 34)]) == 2
    assert len(grouped[("note", 4, 11)]) == 2
    assert sum(len(v) for k, v in grouped.items() if k[0] == "call_skeleton") == 2 * 2 * 2
    assert all(isinstance(row, ScoredToken) for rows in grouped.values() for row in rows)


def test_a_missing_reading_row_is_caught_by_the_turn_s_own_forwarded_count():
    """The defect two independent attacks reached, and the anchor that closes it.

    `_expected` replays the buffer from `reading` rows, so before this check those rows were both
    the evidence and the standard: whatever survived in the file defined what was supposed to be
    there, and a deletion simply moved the expectation to match. On the real record, removing the
    single reading at a turn's last position — 24 rows of 70,358 — passed every assertion and moved
    a reported figure by 23%; removing a turn's head readings passed while returning horizons on
    unequal n, which is the exact harm the module aborts on when it is called eviction.

    `end_turn.forwarded_count` is written by the session and is already checked by
    `lens_fitting.replay`. Checking it here anchors the expectation to something outside the rows
    it is replayed from, which the emitted side has always had.
    """
    events = list(_record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS, lookahead=1)))
    assert join_spans(events).rank_rows == 24, "the intact record joins"

    readings = [i for i, event in enumerate(events) if event.get("kind") == "reading"]
    for dropped, where in ((readings[-1], "the censored tail"), (readings[0], "the head")):
        thinned = [event for i, event in enumerate(events) if i != dropped]
        with pytest.raises(SpanJoinError, match="forwarded_count"):
            join_spans(thinned)


def test_the_ledger_check_is_not_satisfied_by_the_ledger_agreeing_with_itself():
    """A negative control on the anchor: corrupt the ledger and the check must still fire.

    If `forwarded_count` were compared against something derived from the same rows it counts, it
    would agree by construction and add nothing. It is compared against the reading rows, so a
    wrong ledger and a wrong grid are both caught, from either side.
    """
    events = list(_record(_turn(0, prompt_length=9, tokens=TOKENS, spans=SPANS, lookahead=1)))
    for event in events:
        if event.get("kind") == "end_turn":
            event["forwarded_count"] = 1
    with pytest.raises(SpanJoinError, match="forwarded_count"):
        join_spans(events)
