"""Attach the scored token's recorded span label to every rank row, behind a gate that aborts.

WHAT THE JOIN IS
================

The pre-registered comparison is *by span*: a foreknowledge rank, at layer L and horizon h, read
separately over note tokens, call-skeleton tokens and call-argument tokens. The segmentation is
recorded on ``emitted`` rows -- ``SpanLabeller`` writes it with the token, so the facet is a
property of the data rather than of an analysis run after the answer is visible. The *numbers*,
however, live on ``rank`` rows. The facet on a rank row is therefore **derived**, and the
derivation is a join between two row kinds whose ``position`` fields share one coordinate system
and name two different things:

===================  =========================================================================
``emitted.position`` the token slot itself
``rank.position``    the residual slot that was *read*, which predicts the slot h steps later
===================  =========================================================================

``FutureRanks.observe`` (``records.py``) writes ``source = position - horizon`` into the row,
where ``position`` is the just-emitted token's absolute index into ``prompt_ids + generated``.
The scored token is at ``rank.position + rank.horizon`` and never at ``rank.position``, so the
relation is::

    (turn, rank.position + rank.horizon)  ->  (turn, emitted.position)

many-to-one, with ``len(layers) * len(horizons)`` rank rows per emitted row, and those rows carry
``len(horizons)`` *different* position values. ``rank.token_id`` is the id of the scored token, so
``emitted[(turn, P + h)].token_id == rank.token_id`` is an exact, model-free predicate on the join
itself -- not a proxy for it. That predicate is what makes a gate possible here at all.

WHY THE GATE IS A MECHANISM AND NOT A STYLE
===========================================

Every plausible wrong join still returns a full, plausible table.

* Joining ``rank.position == emitted.position`` is semantically defensible on the face of it --
  both fields are called ``position`` and both index the same sequence -- and it agrees on the
  token id for 1.34% of rows (376 of 28,152 measured on ``agentic-d2-update-0028``) purely by
  token frequency. It attaches every score to the token h positions too early. For horizons 4 and
  8 the source is frequently a prompt token, which has no span at all.
* Dropping ``turn`` from the key returns a table too. Positions restart their meaning every turn:
  each turn re-encodes its own prompt of a different length, so the same integer denotes different
  tokens in different turns.
* Grouping rank rows by ``(turn, position)`` alone silently mixes three scored tokens: one source
  P is reused by the emissions at P+1, P+4 and P+8.

None of these raise. They produce a number, and the number is publishable, and it is wrong.

**This already happened.** On 2026-09-08 a join across these two coordinate systems put a wrong
claim into a published record (corrected in commit 97ac5d9): the record said the correct ``' "'``
opener reached a top-10 list at 2 of 24 forks, when it reaches one at all 24 and sits directly
behind the winner. The cause was a flat position dictionary that ignored ``turn``, so it silently
answered about the last turn that happened to use that integer. The reading was fine; the join was
not. The correction had to re-check every rank-based number in the record.

So the assertions below **run before any faceted row can be returned, and abort rather than warn**.
A warning is read by nobody in a pipeline that prints a table at the end, and a table is what a
wrong join produces.

WHAT IS ASSERTED
================

#. Structure. Rows sit inside an open turn; ``(turn, position)`` is unique for ``emitted`` and
   ``reading``; ``(turn, position, layer, horizon)`` is unique for ``rank``; capture positions
   increase strictly within a turn (``FutureRanks.capture``'s own precondition); emitted positions
   are contiguous from ``len(prompt_ids)`` and count out to ``end_turn.emitted_count``; horizons
   are not duplicated (``records.py`` does not de-duplicate them, and a repeat would double rows).
#. Token agreement on **every** joined row, not a sample, aborting on the first disagreement with
   the turn, the source position, the horizon and both token ids named.
#. Exact counts in both directions, against an expectation **reconstructed from the record's own
   rows** rather than assumed: the ``reading`` rows are one-per-capture and in capture order, so
   replaying them through ``FutureRanks``' fixed-capacity buffer says precisely which
   ``(emitted token, horizon)`` blocks must exist. Every predicted block must be present with its
   full layer set, and no rank row may exist that the replay does not predict.
#. The shortfall against the naive product ``|emitted| * |horizons| * |layers|`` is *itemised*,
   never absorbed. A missing block is allowed only where the horizon reaches back past the start of
   the turn's forwards, where nothing was lost because the source never existed. A block missing
   because its source was **evicted** from the capacity-16 buffer, or because the reading grid has
   a hole, aborts: that is silent row loss, it removes whole (horizon, layer) blocks, and it shrinks
   the horizons by unequal amounts so a horizon comparison would then run on unequal n.

Futures censored at the end of a turn need no exception, and are not granted one. The expectation
is built from the emitted side, so a source whose future never arrived simply predicts nothing;
the census is reported (``censored_futures``) rather than used to loosen a check.

WHAT IS DELIBERATELY NOT DONE HERE
==================================

No model, no MLX, no tokenizer: this module reads recorded rows only, which is what lets it run
beside a training run that holds the box lock. ``join_record`` reaches ``read_record`` through a
function-local import for the same reason -- ``session.py`` imports the architecture view, and
that maps Metal.

Rank rows are not reconstructed from ``reading.top``. The two disagree at shallow layers, and the
``rank`` field is a competition rank (1 + strictly-greater count) while ``reading.top`` uses a
stable token-id tie-break, so ``rank <= k`` and "in the top k" are different questions.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Reasons a predicted ``(emitted token, horizon)`` block can be absent without the record being
#: wrong. **Two**, and the docstring said one while the set held two — corrected after an attack
#: found the discrepancy. ``before_turn``: the horizon reaches back past the first position this
#: turn forwarded, so the source state was never captured and no row was lost. ``no_captures``: the
#: turn forwarded nothing at all, which is vacuous rather than lossy. Everything else is loss with
#: a cause, and both of these are now anchored by the ``forwarded_count`` check above, without
#: which "the source was never captured" and "the reading row is missing from the file" are
#: indistinguishable.
_SOURCE_BEFORE_TURN = "before_turn"
_SOURCE_NO_CAPTURES = "no_captures"
_SOURCE_EVICTED = "evicted"
_SOURCE_GAP = "gap"

_ALLOWED_ABSENCES = frozenset({_SOURCE_BEFORE_TURN, _SOURCE_NO_CAPTURES})

#: How many offending keys an abort message lists before it stops. Enough to see the shape of the
#: failure (one horizon? one turn? the whole file?) without printing 28,000 lines at a person.
_NAMED_IN_ERRORS = 8


class SpanJoinError(ValueError):
    """The join could not be proved correct, so no faceted row is returned.

    Raised rather than logged on purpose. Every failure this class reports is a failure that would
    otherwise yield a complete-looking table of ranks by span, which is exactly the artefact a
    reader trusts and cannot audit by looking at it.
    """


@dataclass(frozen=True)
class ScoredToken:
    """One rank row, resolved onto the token it scores and carrying that token's span.

    Both positions are kept, and that is not redundancy. ``source_position`` is the row's own
    ``position`` field -- the residual that was read -- and ``position`` is the token that residual
    was scored against. Anything downstream that groups by "where the model was" wants the first
    and anything that groups by "what it was asked about" wants the second, and a single field
    called ``position`` is how the two get confused.
    """

    turn: int
    #: ``rank.position``: the residual slot read, ``position - horizon``.
    source_position: int
    #: ``emitted.position``: the scored token, ``source_position + horizon``.
    position: int
    #: ``position - len(prompt_ids)`` for this turn: the index within the generated region. Always
    #: >= 0, because the scored token is always a generated token. ``source_position`` may well be
    #: negative in this coordinate, and legitimately so.
    generated_index: int
    layer: int
    horizon: int
    token_id: int
    rank: int
    probability: float
    #: The scored token's recorded label, or ``None`` for a record written before write-time span
    #: labelling, read only with ``require_spans=False``.
    span: str | None
    #: True when the residual read lies in this turn's prompt. Normal and required for the first
    #: h generated tokens of every turn; filtering these away shrinks the horizons unequally.
    source_in_prompt: bool


@dataclass(frozen=True)
class SpanJoin:
    """The joined rows plus the census that proves the join, kept together on purpose.

    A caller that prints ``rows`` without ever looking at the census still gets a checked table --
    the checks ran before this object existed. The census is here so that a record's shape (how
    many rank rows, how many prompt-sourced, how many futures censored, which turns were left out)
    can be reported next to the numbers instead of being reconstructed later by somebody arguing
    about whether the counts were right.
    """

    rows: tuple[ScoredToken, ...]
    turns: tuple[int, ...]
    #: Turns whose ``end_turn.status`` is not ``complete``. Their already-written rows are left out
    #: of ``rows`` entirely: an aborted turn's rank rows are real, but the turn stopped for a
    #: reason nobody has read, and a partial turn is not evidence about a generation.
    excluded_turns: tuple[int, ...]
    layers: tuple[int, ...]
    horizons: tuple[int, ...]
    emitted_rows: int
    rank_rows: int
    #: ``|emitted| * |horizons| * |layers|`` -- the count that holds whenever every horizon's
    #: source exists, which is the condition this pipeline satisfies (prompt longer than the
    #: greatest horizon, single-token decode, capacity above the greatest horizon).
    nominal_rank_rows: int
    #: What the buffer replay predicts. Equal to ``nominal_rank_rows`` minus the itemised absences.
    expected_rank_rows: int
    #: ``(reason, count)`` for every predicted-but-absent ``(emitted, horizon)`` block, in rows.
    absences: tuple[tuple[str, int], ...] = ()
    #: Rank rows whose read residual lies in the prompt: the early-turn foreknowledge signal.
    prompt_sourced_rows: int = 0
    #: ``(source, horizon)`` pairs whose scored token was never emitted before the turn ended.
    #: Reported, not subtracted from anything: the expectation is built from the emitted side.
    censored_futures: int = 0
    #: False when the record predates write-time span labels, in which case every ``span`` is
    #: None, and False for an empty join, where no span was seen and none was checked.
    spans_present: bool = True
    per_turn: tuple[Mapping[str, Any], ...] = field(default=(), repr=False)

    def facet(self, *fields: str) -> dict[tuple, list[ScoredToken]]:
        """Group the joined rows by the named row attributes, e.g. ``facet("layer", "horizon", "span")``."""
        grouped: dict[tuple, list[ScoredToken]] = {}
        for row in self.rows:
            grouped.setdefault(tuple(getattr(row, name) for name in fields), []).append(row)
        return grouped


class _Turn:
    """One turn's rows, in the order they were written, plus what ``begin_turn`` declared."""

    def __init__(self, event: Mapping[str, Any]):
        self.turn = _integer(event, "turn", "begin_turn")
        self.prompt_length = len(event.get("prompt_ids") or ())
        self.layers = tuple(event.get("layers") or ())
        self.horizons = tuple(event.get("rank_horizons") or ())
        self.capacity = event.get("distribution_capacity")
        self.stream: list[tuple[str, Mapping[str, Any]]] = []
        self.readings: dict[int, Mapping[str, Any]] = {}
        self.emitted: dict[int, Mapping[str, Any]] = {}
        #: ``(source, horizon) -> {layer: row}``. Rank rows arrive in whole layer-blocks
        #: (``session.py`` refuses a forward that did not capture every requested layer), so the
        #: block is the unit the count assertions are stated over and a partial one is a defect.
        self.blocks: dict[tuple[int, int], dict[int, Mapping[str, Any]]] = {}
        self.rank_rows = 0
        self.end: Mapping[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return self.end is not None and self.end.get("status") == "complete"


def _integer(event: Mapping[str, Any], name: str, kind: str) -> int:
    value = event.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpanJoinError(f"{kind} row needs an integer {name!r}, and carries {value!r}")
    return value


def _describe(keys: Iterable[tuple]) -> str:
    listed = list(keys)
    shown = ", ".join(repr(key) for key in listed[:_NAMED_IN_ERRORS])
    if len(listed) > _NAMED_IN_ERRORS:
        shown += f", and {len(listed) - _NAMED_IN_ERRORS} more"
    return shown


def _collect(events: Sequence[Mapping[str, Any]]) -> list[_Turn]:
    """Split the event stream into turns, checking the stream's own structure as it goes.

    Order matters and is checked, not assumed: the buffer replay in ``_expected`` depends on
    ``reading`` rows arriving in capture order and on each ``emitted`` row arriving after every
    capture that could feed it, which is how ``CaptureSession`` writes them. A reader that sorted
    the file first, or merged two files, would break that and get no warning from a set-based check.
    """
    turns: list[_Turn] = []
    seen: set[int] = set()
    open_turn: _Turn | None = None
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or "kind" not in event:
            raise SpanJoinError(f"row {index} is not a capture event: {event!r}")
        kind = event["kind"]
        if kind == "manifest":
            continue
        if kind == "end_record":
            if event.get("status") != "complete":
                raise SpanJoinError(
                    "the record's end_record is not 'complete', so the file is a fragment and "
                    "its final turn's rank rows may be missing with nothing to say so"
                )
            continue
        if kind == "begin_turn":
            if open_turn is not None:
                raise SpanJoinError(
                    f"turn {open_turn.turn} never ended before turn {event.get('turn')!r} began"
                )
            open_turn = _Turn(event)
            if open_turn.turn in seen:
                raise SpanJoinError(f"turn {open_turn.turn} begins twice in one record")
            seen.add(open_turn.turn)
            turns.append(open_turn)
            continue
        if open_turn is None:
            raise SpanJoinError(f"a {kind!r} row at index {index} sits outside any turn")
        if _integer(event, "turn", kind) != open_turn.turn:
            raise SpanJoinError(
                f"a {kind!r} row claims turn {event['turn']} inside turn {open_turn.turn}; "
                "rows from two turns are interleaved and no position key is safe"
            )
        if kind == "end_turn":
            open_turn.end = event
            open_turn = None
            continue
        if kind in ("reading", "emitted", "rank"):
            open_turn.stream.append((kind, event))
    if open_turn is not None:
        raise SpanJoinError(
            f"turn {open_turn.turn} has no end_turn row, so whether it finished is unknown"
        )
    return turns


def _declaration(turn: _Turn) -> None:
    """Check what ``begin_turn`` declares before any row is keyed against it.

    First, because these three fields *are* the join's multiplicity and its buffer discipline, and
    because a repeated horizon shows up downstream as a duplicate-row error, which describes the
    symptom and not the cause. Layers are sorted and set-deduplicated by ``CaptureSession``;
    horizons are taken as given by ``records.py``, which is exactly why the repeat is possible.
    """
    if not turn.horizons or len(set(turn.horizons)) != len(turn.horizons):
        raise SpanJoinError(
            f"turn {turn.turn}: rank_horizons {turn.horizons} is empty or repeats a horizon. "
            "records.py takes the tuple as given, so a repeat writes duplicate rows and inflates "
            "every count that is stated per horizon"
        )
    if not turn.layers or len(set(turn.layers)) != len(turn.layers):
        raise SpanJoinError(
            f"turn {turn.turn}: begin_turn declares layers {turn.layers}, which is empty or "
            "repeats one. The per-block layer count is the join's own multiplicity"
        )
    if not isinstance(turn.capacity, int) or isinstance(turn.capacity, bool) or turn.capacity < 1:
        raise SpanJoinError(
            f"turn {turn.turn}: begin_turn carries distribution_capacity {turn.capacity!r}. "
            "Without it the buffer cannot be replayed and row loss cannot be distinguished "
            "from censoring"
        )


def _index(turn: _Turn) -> None:
    """Key each of a turn's rows, refusing duplicates and coordinate violations.

    Duplicate keys are the failure that a dictionary-building join hides best: the second row
    overwrites the first and the count still looks plausible.
    """
    last_capture: int | None = None
    generated = 0
    for kind, event in turn.stream:
        position = _integer(event, "position", kind)
        if kind == "reading":
            if last_capture is not None and position <= last_capture:
                raise SpanJoinError(
                    f"turn {turn.turn}: capture position {position} does not exceed the previous "
                    f"{last_capture}. FutureRanks.capture refuses this, so these rows were not "
                    "written by one turn's forward partition"
                )
            last_capture = position
            turn.readings[position] = event
        elif kind == "emitted":
            expected = turn.prompt_length + generated
            if position != expected:
                raise SpanJoinError(
                    f"turn {turn.turn}: the {generated}th emitted token is at position {position}, "
                    f"but prompt_ids + generated puts it at {expected}. The emitted rows are not "
                    "in the coordinate the rank rows are joined in"
                )
            _integer(event, "token_id", kind)
            turn.emitted[position] = event
            generated += 1
        else:
            horizon = _integer(event, "horizon", kind)
            layer = _integer(event, "layer", kind)
            block = turn.blocks.setdefault((position, horizon), {})
            if layer in block:
                raise SpanJoinError(
                    f"turn {turn.turn}: two rank rows share (position, horizon, layer) "
                    f"{(position, horizon, layer)}. One would silently replace the other in any "
                    "keyed join"
                )
            block[layer] = event
            turn.rank_rows += 1
    counted = turn.end.get("emitted_count") if turn.end else None
    if counted != generated:
        raise SpanJoinError(
            f"turn {turn.turn}: end_turn records emitted_count {counted!r} against {generated} "
            "emitted rows. The turn's own ledger disagrees with the file"
        )
    # The same question asked of the *capture* grid, which is the half this module previously took
    # on trust. `_expected` replays the buffer from `reading` rows, so the reading rows were both
    # the evidence and the standard: whatever survived in the file defined what was supposed to be
    # there, and a missing reading row simply moved the expectation to match. Two adversarial
    # attacks reached the same defect through it. Deleting the single reading at each turn's last
    # position — 24 rows of 70,358 — passed every check and moved a reported figure by 23%, and
    # deleting a turn's head readings passed while returning horizons on unequal n, which is the
    # exact harm this module aborts on when it is called eviction.
    #
    # `end_turn.forwarded_count` is the ledger for that grid, written by the session and already
    # checked by `lens_fitting.replay`. Checking it here anchors the expectation to something
    # outside the rows it is replayed from, which is what the emitted-side check has always had.
    forwarded = turn.end.get("forwarded_count") if turn.end else None
    if forwarded != len(turn.readings):
        raise SpanJoinError(
            f"turn {turn.turn}: end_turn records forwarded_count {forwarded!r} against "
            f"{len(turn.readings)} reading rows. The capture grid this join's expectation is "
            "replayed from is incomplete, so every count derived from it would be self-consistent "
            "and wrong"
        )


def _expected(turn: _Turn) -> tuple[dict[tuple[int, int], int], dict[str, int], int]:
    """Replay ``FutureRanks``' buffer over this turn's rows to say which blocks must exist.

    This is the reason the expectation needs no assumption about configuration. ``reading`` rows
    are written one per ``FutureRanks.capture`` call, in the same loop and the same order
    (``session.py``), so replaying them through an ``OrderedDict`` of ``distribution_capacity``
    entries reproduces exactly which source states were still held when each token was emitted.
    Censoring at the turn boundary, sources that lie in the prompt, and eviction under a long
    lookahead all fall out of the replay instead of being guessed at.

    Returns the predicted ``(source, horizon) -> emitted position`` blocks, an itemised census of
    the blocks that cannot exist, and the count of futures censored at the turn's end.
    """
    live: OrderedDict[int, None] = OrderedDict()
    captured: set[int] = set()
    blocks: dict[tuple[int, int], int] = {}
    absences: dict[str, int] = {}
    for kind, event in turn.stream:
        if kind == "reading":
            position = event["position"]
            live[position] = None
            captured.add(position)
            while len(live) > turn.capacity:
                live.popitem(last=False)
        elif kind == "emitted":
            position = event["position"]
            for horizon in turn.horizons:
                source = position - horizon
                if source in live:
                    blocks[(source, horizon)] = position
                else:
                    reason = _absence(source, captured)
                    absences[reason] = absences.get(reason, 0) + 1
    last = max(turn.emitted, default=None)
    censored = 0
    if last is not None:
        # Counted in ROWS, not in (source, horizon) pairs, because the reconciliation this figure
        # exists to let a reader check is stated in rows: source-side censoring is balanced by
        # prompt-sourced rows. Reported as pairs it differed from its counterpart by a factor of
        # the layer count and the two did not balance, which an attack found by trying to check
        # the argument the docstring makes.
        censored = sum(
            len(turn.layers) for p in captured for h in turn.horizons if p + h > last
        )
    return blocks, absences, censored


def _absence(source: int, captured: set[int]) -> str:
    """Name why a source state was not available, because the reasons are not interchangeable."""
    if not captured:
        return _SOURCE_NO_CAPTURES
    if source < min(captured):
        return _SOURCE_BEFORE_TURN
    if source in captured:
        return _SOURCE_EVICTED
    return _SOURCE_GAP


def join_spans(
    events: Sequence[Mapping[str, Any]], *, require_spans: bool = True
) -> SpanJoin:
    """Join every rank row to the token it scores and carry that token's span across.

    ``events`` is a sequence of capture events -- the inner ``event`` objects, not the hash-chained
    envelopes; ``read_record`` already unwraps them. The sequence must be in file order.

    ``require_spans`` guards a real distinction rather than a preference. Records written before
    write-time span labelling (2026-09-08) carry no ``span`` field on their emitted rows, and for
    those the facet this module exists to produce simply does not exist. The default refuses them,
    because silently faceting on a missing field is how a table of "note vs argument" gets built
    out of one bucket. Passing ``require_spans=False`` says, explicitly and in the caller's own
    code, that the join is wanted without the facet; every returned ``span`` is then ``None`` and
    ``SpanJoin.spans_present`` is False, so a grouped result cannot pretend otherwise.

    Raises ``SpanJoinError`` before returning anything if any assertion fails. There is no partial
    result and no warning path: a half-verified join is the thing this module exists to prevent.
    """
    turns = _collect(events)
    # Only complete turns are indexed, and only complete turns are joined. An aborted turn's rows
    # are already written and are real, but the turn stopped for a reason nobody has read, and its
    # last emitted row may be missing the rank rows that were about to follow it -- which is
    # indistinguishable, row by row, from the loss this module exists to catch.
    usable = [turn for turn in turns if turn.complete]
    excluded = tuple(turn.turn for turn in turns if not turn.complete)
    for turn in usable:
        _declaration(turn)
        _index(turn)

    layers: set[int] = set()
    horizons: set[int] = set()
    rows: list[ScoredToken] = []
    per_turn: list[Mapping[str, Any]] = []
    nominal = expected_rows = rank_rows = emitted_rows = prompt_sourced = censored_total = 0
    absences: dict[str, int] = {}
    spans_present = True

    for turn in usable:
        layer_set = tuple(turn.layers)
        layers.update(layer_set)
        horizons.update(turn.horizons)
        blocks, turn_absences, censored = _expected(turn)

        # Both directions, named separately, because they fail for opposite reasons: a missing
        # block is row loss and an unpredicted one is a coordinate or turn-scoping error.
        found = set(turn.blocks)
        missing = sorted(set(blocks) - found)
        if missing:
            raise SpanJoinError(
                f"turn {turn.turn}: {len(missing)} predicted rank blocks are absent from the "
                f"record: (source, horizon) {_describe(missing)}. The buffer replay says the "
                "source state was held when the token was emitted, so these rows should exist"
            )
        unpredicted = sorted(found - set(blocks))
        if unpredicted:
            raise SpanJoinError(
                f"turn {turn.turn}: {len(unpredicted)} rank blocks exist that the buffer replay "
                f"does not predict: (source, horizon) {_describe(unpredicted)}. Either the rows "
                "are not from this turn or the capture grid in the record is not the one that "
                "produced them"
            )
        for reason, count in turn_absences.items():
            if reason not in _ALLOWED_ABSENCES:
                raise SpanJoinError(
                    f"turn {turn.turn}: {count} rank blocks are missing because the source state "
                    f"was {reason!r}, not because the horizon reached back past the start of the "
                    "turn. Eviction from the capacity-"
                    f"{turn.capacity} buffer and holes in the capture grid both drop whole "
                    "(horizon, layer) blocks with nothing in the record to distinguish them from "
                    "ordinary censoring, and they shrink the horizons by unequal amounts"
                )
            absences[reason] = absences.get(reason, 0) + count * len(layer_set)

        for (source, horizon), position in sorted(blocks.items()):
            target = turn.emitted[position]
            span = target.get("span")
            if span is None:
                if require_spans:
                    raise SpanJoinError(
                        f"turn {turn.turn}: the emitted row at position {position} carries no "
                        "'span'. This record predates write-time span labelling, so the facet is "
                        "not recorded anywhere and cannot be recovered from the rank rows. Pass "
                        "require_spans=False to take the join without the facet"
                    )
                spans_present = False
            block = turn.blocks[(source, horizon)]
            if set(block) != set(layer_set):
                raise SpanJoinError(
                    f"turn {turn.turn}: the block at source {source}, horizon {horizon} carries "
                    f"layers {sorted(block)} against the declared {sorted(layer_set)}. Blocks "
                    "arrive whole or not at all, so a partial one is a lost or a foreign row"
                )
            for layer in layer_set:
                row = block[layer]
                # The predicate the whole gate rests on, on every row and never on a sample: the
                # rank row names the token it scored, and the emitted row names the token at that
                # slot. Under any wrong offset these disagree almost everywhere, and under the
                # off-by-horizon join they still agree on a few percent of rows by frequency alone.
                if row["token_id"] != target["token_id"]:
                    raise SpanJoinError(
                        f"token id disagreement at turn {turn.turn}, source position {source}, "
                        f"horizon {horizon}, layer {layer}: the rank row scored token "
                        f"{row['token_id']} and the emitted row at position {position} carries "
                        f"token {target['token_id']}. The join is wrong, or the rows are not from "
                        "one capture; either way no faceted number from this record is safe"
                    )
                rank = _integer(row, "rank", "rank")
                probability = row.get("probability")
                if (
                    rank < 1
                    or not isinstance(probability, int | float)
                    or not 0 <= probability <= 1
                ):
                    raise SpanJoinError(
                        f"turn {turn.turn}, source {source}, horizon {horizon}, layer {layer}: "
                        f"rank {rank!r} and probability {probability!r}. Ranks are competition "
                        "ranks and start at 1; a 0 means a different convention and every "
                        "comparison against reading.top would be off by one"
                    )
                rows.append(
                    ScoredToken(
                        turn=turn.turn,
                        source_position=source,
                        position=position,
                        generated_index=position - turn.prompt_length,
                        layer=layer,
                        horizon=horizon,
                        token_id=row["token_id"],
                        rank=rank,
                        probability=float(probability),
                        span=span,
                        source_in_prompt=source < turn.prompt_length,
                    )
                )
                prompt_sourced += source < turn.prompt_length

        turn_nominal = len(turn.emitted) * len(turn.horizons) * len(layer_set)
        turn_expected = len(blocks) * len(layer_set)
        nominal += turn_nominal
        expected_rows += turn_expected
        rank_rows += turn.rank_rows
        emitted_rows += len(turn.emitted)
        censored_total += censored
        if turn.rank_rows != turn_expected:
            raise SpanJoinError(
                f"turn {turn.turn} holds {turn.rank_rows} rank rows against the {turn_expected} "
                "the buffer replay predicts. Every block matched, so this is a duplicate or a "
                "stray row that the block keys did not separate"
            )
        per_turn.append(
            {
                "turn": turn.turn,
                "prompt_length": turn.prompt_length,
                "emitted": len(turn.emitted),
                "readings": len(turn.readings),
                "rank_rows": turn.rank_rows,
                "nominal_rank_rows": turn_nominal,
                "expected_rank_rows": turn_expected,
                "censored_futures": censored,
            }
        )

    # The shortfall against the naive product is itemised, not absorbed: every row the product
    # promises and the record does not hold is accounted for by name above.
    if expected_rows + sum(absences.values()) != nominal:
        raise SpanJoinError(
            f"the expected {expected_rows} rank rows and the itemised absences "
            f"{sorted(absences.items())} do not add up to the product {nominal}. The census does "
            "not account for the difference, so the count assertion would be a guess"
        )
    if len(rows) != rank_rows or len(rows) != expected_rows:
        raise SpanJoinError(
            f"{len(rows)} joined rows against {rank_rows} rank rows and {expected_rows} expected. "
            "Some rank row was neither joined nor named"
        )
    return SpanJoin(
        rows=tuple(rows),
        turns=tuple(turn.turn for turn in usable),
        excluded_turns=excluded,
        layers=tuple(sorted(layers)),
        horizons=tuple(sorted(horizons)),
        emitted_rows=emitted_rows,
        rank_rows=rank_rows,
        nominal_rank_rows=nominal,
        expected_rank_rows=expected_rows,
        absences=tuple(sorted(absences.items())),
        prompt_sourced_rows=prompt_sourced,
        censored_futures=censored_total,
        spans_present=spans_present and bool(rows),
        per_turn=tuple(per_turn),
    )


def attach_spans(
    events: Sequence[Mapping[str, Any]], *, require_spans: bool = True
) -> list[ScoredToken]:
    """The joined rows alone, for a caller that wants the facet and not the census.

    The assertions still run -- they are in ``join_spans``, which this calls, and there is no path
    to a row that skips them.
    """
    return list(join_spans(events, require_spans=require_spans).rows)


def join_record(path: Path, *, require_spans: bool = True) -> SpanJoin:
    """Read a capture record from disk, verifying its hash chain, and join it.

    ``read_record`` is imported inside the function on purpose. It lives in ``session.py``, which
    imports the architecture view and so maps Metal, and this module has to stay importable next to
    a run that holds the box lock. Callers that already hold the events should use ``join_spans``.
    """
    from local_llm_lab.pipeline.live_lens.session import read_record

    return join_spans(read_record(path), require_spans=require_spans)
