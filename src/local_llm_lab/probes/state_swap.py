"""EXP-002: is the hybrid's recurrent state a usable memory channel under a persistent cache?

EXP-001 answered a narrower question and answered it cleanly: at the decision where the
directory listing has left the observation window, the model's own output holds no trace of the
hidden filename. The null is not a limit of the lens. **The state cannot carry what it never
saw** -- the deployed runner never carries a recurrent state across the eviction that hides the
listing, because ``ArraysCache`` defines neither ``is_trimmable`` nor ``trim`` and a trimming
strategy on a hybrid therefore degenerates to a rebuild.

The question that survives is the Director's real one, and the same untrimmability is what makes
it askable. A **persistent-cache** regime would keep in the recurrent state every observation
the window later hides, while the attention KV for those spans can be masked. So this is not a
swap between contexts but a within-run dissociation between the two paths under one cache:

* **A, recurrent only.** The attention KV over the hidden-observation spans is masked, so
  attention is blind to exactly what the window hides while the recurrent blocks take no mask.
  This is the experiment.
* **B, everything.** The same cache and the same forward with nothing hidden. Upper bound, and
  the gate on the instrument: if a run whose attention can still read the listing does not raise
  P(true), nothing about A or C is readable.
* **C, EXP-001's condition.** A fresh prefill of the windowed, stripped text with no persistent
  cache. It must reproduce EXP-001's decisive row or the reading stops.
* **Control, foreign persistent state.** Another point's persistent cache under A's masking,
  which separates retrieval from generic perturbation.

**B6, the paragraph that is load-bearing in both directions.** Observations are unstripped, so
the listing is present; notes have ``strip_pending`` applied, as in EXP-001. Strip both and the
filename never enters, so arm A cannot rise for the right reason. Strip neither and an
unstripped note keeps the filename in its ``pending:`` list -- and a note is an assistant
message the observation window never hides -- so arm A would read it off a visible note with the
listing masked and rise for the wrong reason. With B6's construction the filename enters the
model **exactly once, through the listing observation**, which is the span arm A masks. That is
counted in the rendered prompt for every point rather than inferred from the flags passed, and
a point that fails it is dropped and tallied.

**The identity gate is a gate, not a control.** A run's own cache entries are captured and
re-injected into the same run unmasked, and the scored distribution must be unchanged to within
1e-6 over the scored candidates. It runs on several points and the run is gated on the **worst**
of them: one point can pass by luck, and a gate that can pass by luck is not a gate. Every
point's observed maximum is recorded whether it passes or fails, so the threshold can be
recalibrated from evidence. It is written into the artifact before any arm; if it fails, no arm
is readable and the run stops.

**Arm A has four ways to be quietly wrong and all four look like a result.** Mask too little and
attention leaks the answer. Let the suffix stand readable outside the masked span and it leaks
by another route. Mask only at the decision and the answer arrives one attention hop away. Mask
the observation *while it is arriving* and the recurrent state never acquires what arm A tests
for, so the arm reports a null it could not have avoided. Each is guarded, and the guards are
documented where the code makes the choice rather than only here.

Conformance statement (R34), for the estimator this module computes. The decisive measurement is
the model's own next-token distribution over the suffix's first token, per arm, paired per probe
point, with P(true suffix) and P(already-read suffix) reported separately and their absolute
values alongside -- uniform over ten digits is 0.1 (EXP-001 §3.3). The probe-point family is
EXP-001's unchanged: split ``jsweep``, ``ledger_reconcile``, step 3, the first read after the
listing leaves the window, leakage guard, coinciding first tokens skipped. The persistent cache
is the model's own ``make_cache``, and the artifact names the class used per block kind. The
masked spans are the observation messages ``keep_last`` would hide, located from message
boundaries in the rendered token sequence and recorded per point as resolved absolute column
ranges with their token counts -- the ranges the mask indexed, not the message indices they came
from. Each arm's scored record carries the position scored, the cache offset, and **which mask
form ran**: the library's boolean array with the hidden columns ANDed, or the row constructed
where the library returns ``None`` at a single token.

Comparability (R35). Populated as EXP-001's is, including ``fp32_manual_vs_native``, with the
persistent arms' rendering named beside arm C's: the two are different windows over the same
conversation and an artifact that recorded only one would misdescribe three arms out of four.

No model runs from this docstring: the CLI loads weights, and it runs only under a Director lift.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.jlens import encode
from local_llm_lab.probes.jspace_sweep import (
    MIN_CASES,
    PROBE_STEP,
    TASK_SPLIT,
    holm_adjust,
    note_prefix,
    sign_test,
    strip_pending,
    suffix_of,
)

__all__ = [
    "ARMS",
    "IDENTITY_TOLERANCE",
    "PERSISTENT_ARMS",
    "aggregate",
    "chunking_deviation",
    "compare_to_exp001",
    "hidden_message_indices",
    "identity_gate",
    "identity_gate_family",
    "main",
    "message_token_spans",
    "prefill",
    "render_markdown",
    "run_arms",
    "score_point",
    "select_points",
    "suffix_occurrences_outside",
]

#: The four arms of EXP-002 §3, in the order the artifact reports them.
ARMS = ("A", "B", "C", "control")
#: The arms that run over a persistent cache; arm C is a fresh prefill with none.
PERSISTENT_ARMS = ("A", "B", "control")
#: EXP-002 §3.4: maximum absolute difference over the scored candidates' probabilities.
IDENTITY_TOLERANCE = 1e-6
#: How many probe points the identity gate runs on. One point can pass by luck, and a gate that
#: can pass by luck is not a gate; the run is gated on the maximum over these (Chief, 2026-09-05).
IDENTITY_GATE_POINTS = 3
#: The two candidates every arm is scored on (EXP-001 §3.3 decomposes onto both).
CANDIDATES = ("true", "already_read")


def hidden_message_indices(
    messages: Sequence[dict[str, Any]], *, keep_last: int
) -> tuple[int, ...]:
    """The observation messages ``keep_last`` would hide, read off the function that hides them.

    ``window_messages`` replaces a hidden observation's content with a stub rather than dropping
    the message, so the hidden set is recoverable by comparing its output against its input.
    Deriving it instead from ``keep_last`` and the tool positions would be a second copy of the
    windowing rule, free to drift from the one that renders arm C -- and a mask over the wrong
    messages is the failure that reads as a result.
    """
    from local_llm_lab.pipeline.protocol import window_messages

    windowed = window_messages(list(messages), keep_last)
    return tuple(
        index
        for index, (before, after) in enumerate(zip(messages, windowed, strict=True))
        if before["content"] != after["content"]
    )


def _rendered_boundaries(
    tokenizer: Any, messages: Sequence[dict[str, Any]], *, spec: Any
) -> tuple[int, list[int], list[int]]:
    """The head block's size, and cumulative token and character counts at every boundary after it.

    The boundaries are found by rendering each prefix of the conversation through the same
    ``build_prompt`` the scored prompt goes through, and taking the length. That is only valid
    while each render nests inside the next, so both nestings are **checked**: characters,
    because a template that re-rendered an earlier turn would move every later boundary; and
    token ids, because a token that straddled a boundary would put the mask one column out.
    Chat templates separate turns with atomic special tokens, so both hold in practice -- but a
    silently wrong span hides the wrong text while the forward still runs cleanly, which is the
    failure this experiment cannot afford.

    **Not every prefix is a conversation the template will render, so the scan discovers where
    it can start rather than assuming index 1.** Qwen3.5's template scans the message list in
    reverse for a user query and refuses outright when it finds none, so ``[system]`` alone --
    the first prefix of every conversation this repo builds -- raises ``No user query found in
    messages.``. The 2.5 template has no such guard, which is why a locator that started at
    count 1 was correct by accident on the 3B and died three seconds into the 4B run. The
    smallest renderable count ``k0`` becomes the **head block**: messages ``[0, k0)`` are
    located as one chunk rather than individually, and everything from ``k0`` on keeps its own
    boundary. At ``k0 == 1`` this is exactly the per-message scheme it replaces.

    Two things are deliberately *not* inferred from the first success. Renderability is **not
    monotone** -- ``[system, user, assistant]`` renders and ``[system, user, assistant, system]``
    raises ``System message must be at the beginning.`` -- so every count past the head is
    required to render and a refusal there is a hard error. And a tool observation inside the
    head block would have no locatable span at all while the scan reported success: a shape like
    ``[system, tool, assistant, user, ...]`` yields ``k0 == 4`` with an unlocatable observation
    at index 1 and no error raised. That is refused here, where the reason is structural, rather
    than left to the caller's index check, which only sees the observations this run happens to
    hide.
    """
    # ``jinja2`` is what raises when a chat template refuses a conversation, and this catch is
    # the only thing standing between that refusal and a crash, so the package is declared in
    # pyproject rather than relied on as a transitive of ``mlx-lm``.
    from jinja2.exceptions import TemplateError

    from local_llm_lab.pipeline.protocol import build_prompt

    head_count = 0
    token_counts = [0]
    char_counts = [0]
    previous_text = ""
    previous_ids: list[int] = []
    refusal = "the conversation is empty"
    for count in range(1, len(messages) + 1):
        # ``keep_last`` is the whole prefix, so the render windows nothing: these boundaries
        # have to describe the unwindowed history the persistent arms actually prefill.
        try:
            text = build_prompt(
                tokenizer, list(messages[:count]), keep_last=count, spec=spec, generation=False
            )
        except TemplateError as error:
            if head_count:
                raise ValueError(
                    f"chat template rendered the first {head_count} messages but refused the "
                    f"first {count}, so the boundaries after the head block cannot be located: "
                    f"{type(error).__name__}: {error}"
                ) from error
            refusal = f"{type(error).__name__}: {error}"
            continue
        if not head_count:
            head_count = count
            unlocatable = [
                index
                for index, message in enumerate(messages[:count])
                if message["role"] == "tool"
            ]
            if unlocatable:
                raise ValueError(
                    "chat template refused every prefix shorter than "
                    f"{count} messages, so messages {unlocatable} are tool observations inside "
                    "an undivided head block and have no span the mask could index"
                )
        if not text.startswith(previous_text):
            raise ValueError(
                f"chat template is not prefix-additive at message {count - 1}: the render of "
                "the shorter conversation is not a prefix of the longer one, so message "
                "boundaries cannot be located by length"
            )
        ids = encode(tokenizer, text)
        if ids[: len(previous_ids)] != previous_ids:
            raise ValueError(
                f"chat template is not a token prefix at message {count - 1}: a token crosses "
                "the boundary, so a span located by length would be one column out"
            )
        token_counts.append(len(ids))
        char_counts.append(len(text))
        previous_text, previous_ids = text, ids
    if not head_count:
        raise ValueError(
            "chat template refused every prefix of this conversation, so no message boundary "
            f"can be located: {refusal}"
        )
    return head_count, token_counts, char_counts


def message_token_spans(
    tokenizer: Any, messages: Sequence[dict[str, Any]], *, spec: Any
) -> tuple[int, tuple[tuple[int, int], ...], int]:
    """The head block's size, the render's chunk spans, and its token length.

    ``spans[0]`` covers ``messages[:head_count]`` as one block -- see ``_rendered_boundaries``
    for why the template may refuse to render a shorter prefix -- and ``spans[j]`` for ``j >= 1``
    covers message ``head_count - 1 + j``. Use ``_spans_by_message`` rather than indexing this
    tuple by message index; the two coincide only when ``head_count == 1``.
    """
    head_count, token_counts, _chars = _rendered_boundaries(tokenizer, messages, spec=spec)
    spans = tuple(zip(token_counts[:-1], token_counts[1:], strict=True))
    return head_count, spans, token_counts[-1]


def _unwindowed_keep_last(task: Any) -> int:
    """A window wide enough to hide nothing, derived from the task rather than named.

    One tool observation per step is the most the simulator can produce, so a keep-last above
    the step count leaves every observation whole. Written as a derivation because a literal
    here would be a claim about the task generator that this module does not own.
    """
    return len(task.steps) + 1


def select_points(
    tasks: Sequence[Any],
    tokenizer: Any,
    *,
    spec: Any,
    probe_step: int = PROBE_STEP,
    keep_last: int | None = None,
    strip_notes: bool = True,
    hide_observations: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build the probe points, with the B6 invariant counted rather than assumed.

    The selection rule is EXP-001's, unchanged, so arm C's cohort is the cohort that artifact
    scored: step ``probe_step``, which must be a ``read_file``; the context forced to end
    mid-note at the target's stem; a leakage guard against the *windowed* text, since that is
    the condition arm C reproduces; and pairs whose first suffix tokens coincide dropped,
    because the comparison would be between a token and itself.

    What EXP-002 adds is the second rendering of the same conversation. The persistent arms
    prefill the history **unwindowed**, so the listing observation is present in full, with
    ``strip_pending`` applied to the notes. ``strip_notes`` and ``hide_observations`` exist so
    that both directions B6 warns about can be driven and shown to be caught, not so that a run
    can choose: the defaults are B6's construction and nothing else is a valid reading.

    Returns the points and a tally of why every rejected task was rejected. The tally is
    reported in the artifact: a construction that quietly dropped most of the cohort would
    otherwise look like a small run rather than a broken one.
    """
    from local_llm_lab.pipeline.data import build_rows
    from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST, build_prompt, window_messages

    window = DEFAULT_KEEP_LAST if keep_last is None else keep_last
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    points: list[dict[str, Any]] = []
    for task in tasks:
        windowed_rows = build_rows(task)
        if len(windowed_rows) <= probe_step or probe_step >= len(task.steps):
            reject("too_few_rows")
            continue
        step = task.steps[probe_step]
        if step.action.name != "read_file":
            reject("not_a_read")
            continue
        target_path = str(step.action.arguments["path"])
        filename = target_path.rsplit("/", 1)[-1]
        prefix = note_prefix(step.thought, filename)
        if prefix is None:
            reject("no_note_prefix")
            continue
        earlier = [
            str(s.action.arguments["path"])
            for s in task.steps[:probe_step]
            if s.action.name == "read_file"
        ]
        if not earlier:
            reject("no_earlier_read")
            continue
        true_suffix = suffix_of(target_path)
        false_suffix = suffix_of(earlier[-1])
        if true_suffix == false_suffix:
            reject("suffix_collision")
            continue

        # Arm C, built exactly as EXP-001 builds it: rows already windowed by ``build_rows``,
        # so ``keep_last`` is the row's own tool count and the render re-windows nothing.
        windowed_messages = windowed_rows[probe_step]["messages"][:-1]
        stripped_windowed = strip_pending(windowed_messages)
        tool_messages = sum(1 for message in stripped_windowed if message["role"] == "tool")
        windowed_prompt = (
            build_prompt(tokenizer, stripped_windowed, keep_last=tool_messages, spec=spec) + prefix
        )
        if true_suffix in windowed_prompt:
            reject("leaked")
            continue
        first_true = encode(tokenizer, true_suffix)
        first_false = encode(tokenizer, false_suffix)
        if not first_true or not first_false or first_true[0] == first_false[0]:
            reject("token_collision")
            continue

        # The persistent arms' history: the same conversation with the window not applied.
        history = build_rows(task, keep_last=_unwindowed_keep_last(task))[probe_step]["messages"][
            :-1
        ]
        if window_messages(history, window) != windowed_messages:
            # The two renders must be the same conversation under two windows. If they are not,
            # arm C is not the windowed form of what the other arms prefill and the whole
            # comparison is between different texts.
            reject("window_mismatch")
            continue
        hidden_indices = hidden_message_indices(history, keep_last=window)
        if not hidden_indices:
            reject("no_hidden_observation")
            continue
        if hide_observations:
            history = window_messages(history, window)
        if strip_notes:
            history = strip_pending(history)

        head_count, token_spans, char_spans = _persistent_spans(tokenizer, history, spec=spec)
        if min(hidden_indices) < head_count:
            # The head block is the prefix the chat template would not render one message at a
            # time, so it carries no interior boundary. A hidden observation inside it has no
            # span for the mask to index, and there is no defensible run under that condition:
            # this raises rather than joining the ``reject`` tally, because a structural break
            # in the instrument must not be reported as a property of the data.
            #
            # It cannot fire today, and the reason it is kept is the reason it cannot.
            # ``_rendered_boundaries`` already refuses a head block containing a ``role ==
            # "tool"`` message, which is *its copy* of the rule that the observations
            # ``window_messages`` hides are tool messages. This check owns no copy of that rule:
            # it reads the hidden set ``hidden_message_indices`` derived from ``window_messages``
            # itself, so it is what survives a change to what the windowing hides.
            raise ValueError(
                f"hidden observation at message {min(hidden_indices)} falls inside the "
                f"{head_count}-message head block, which has no per-message boundary"
            )
        token_by_message = _spans_by_message(head_count, token_spans)
        char_by_message = _spans_by_message(head_count, char_spans)
        persistent_prompt = (
            build_prompt(tokenizer, history, keep_last=len(history), spec=spec) + prefix
        )
        persistent_ids = encode(tokenizer, persistent_prompt)
        if persistent_ids[: token_spans[-1][1]] != encode(
            tokenizer,
            build_prompt(
                tokenizer, history, keep_last=len(history), spec=spec, generation=False
            ),
        ):
            reject("scored_render_diverged")
            continue

        hidden_spans = tuple(token_by_message[index] for index in hidden_indices)
        hidden_character_spans = tuple(char_by_message[index] for index in hidden_indices)
        occurrences = persistent_prompt.count(filename)
        inside = occurrences == 1 and any(
            start <= persistent_prompt.index(filename) < end
            for start, end in hidden_character_spans
        )
        if occurrences == 0:
            reject("filename_absent")
            continue
        if not inside:
            reject("filename_outside_hidden_span")
            continue
        # The leakage guard above checks arm C's windowed text. The persistent prompt carries
        # the observations in full, so it needs its own: a suffix readable outside the masked
        # span is a route arm A's attention can take, and arm A would rise for the wrong reason.
        # Standalone occurrences only -- a three-digit run inside a six-digit value is not the
        # suffix and no attention head can read it as one.
        standalone_outside = suffix_occurrences_outside(
            persistent_prompt, true_suffix, hidden_character_spans
        )
        naive_outside = suffix_occurrences_outside(
            persistent_prompt, true_suffix, hidden_character_spans, standalone=False
        )
        if standalone_outside:
            reject("suffix_outside_hidden_span")
            continue

        # Arm C's own chunking, so it runs the same forward path as the persistent arms.
        windowed_head_count, windowed_token_spans, _windowed_chars = _persistent_spans(
            tokenizer, stripped_windowed, spec=spec
        )
        windowed_ids = encode(tokenizer, windowed_prompt)
        if windowed_ids[: windowed_token_spans[-1][1]] != encode(
            tokenizer,
            build_prompt(
                tokenizer,
                stripped_windowed,
                keep_last=len(stripped_windowed),
                spec=spec,
                generation=False,
            ),
        ):
            reject("arm_c_render_diverged")
            continue

        points.append(
            {
                "task_id": task.task_id,
                "difficulty": getattr(task, "difficulty", None),
                "filename": filename,
                "prefix": prefix,
                "true": true_suffix,
                "false": false_suffix,
                "true_token": first_true[0],
                "false_token": first_false[0],
                "windowed_prompt": windowed_prompt,
                "windowed_ids": windowed_ids,
                "windowed_chunks": windowed_token_spans,
                "windowed_head_messages": windowed_head_count,
                "windowed_scored_chunk": (windowed_token_spans[-1][1], len(windowed_ids)),
                "persistent_prompt": persistent_prompt,
                "persistent_ids": persistent_ids,
                "prefill_chunks": token_spans,
                # How many messages the first prefill chunk covers. Recorded because the chunk
                # count is otherwise one short of the message count with nothing in the artifact
                # to say why, and because it is what lets a reader check from the record alone
                # that no hidden observation fell inside the undivided head.
                "head_messages": head_count,
                "scored_chunk": (token_spans[-1][1], len(persistent_ids)),
                "hidden_spans": hidden_spans,
                "hidden_character_spans": hidden_character_spans,
                "hidden_span_records": tuple(
                    {
                        "start": start,
                        "end": end,
                        "tokens": end - start,
                        "message_index": index,
                        "role": history[index]["role"],
                        "name": history[index].get("name"),
                    }
                    for index, (start, end) in zip(hidden_indices, hidden_spans, strict=True)
                ),
                "b6": {
                    "filename_occurrences": occurrences,
                    "inside_hidden_span": inside,
                    # Both counts are kept per point. The standalone count is zero here because
                    # the point survived the rule above; the substring count is the number a
                    # naive rule would have rejected on, and their difference is what the
                    # distinction between the two is worth on this cohort.
                    "standalone_suffix_occurrences_outside_hidden_spans": len(
                        standalone_outside
                    ),
                    "substring_suffix_occurrences_outside_hidden_spans": len(naive_outside),
                },
            }
        )
    return points, rejected


def naive_rule_comparison(
    points: Sequence[dict[str, Any]], rejected: dict[str, int]
) -> dict[str, Any]:
    """How the standalone suffix rule and a naive any-occurrence rule differ on this cohort.

    Counted rather than argued, because it decides whether the distinction costs the
    population. A naive rule rejects every point the standalone rule rejects, plus every
    accepted point carrying a substring hit inside a longer number. Measured on the real
    ``jsweep`` ledger cohort: **0 of 42 either way**, and the structural reason is worth
    recording beside the number -- the persistent prompt and arm C's windowed prompt differ
    *only inside the hidden spans*, so EXP-001's leakage guard on the windowed text already
    forces the outside-span count to zero. The rule is therefore redundant on this
    construction and is kept because it fails closed if the window, the probe step or the
    generator moves, not because it is currently catching anything.
    """
    standalone = rejected.get("suffix_outside_hidden_span", 0)
    substring_hits = sum(
        1
        for point in points
        if point["b6"]["substring_suffix_occurrences_outside_hidden_spans"] > 0
    )
    return {
        "standalone_rule_rejected": standalone,
        "naive_any_occurrence_would_reject": standalone + substring_hits,
        "accepted_points_with_a_substring_hit_only": substring_hits,
        "points_accepted": len(points),
        "rule": (
            "a suffix is a route only where it is readable as the suffix; probes.patch's "
            "_value_pattern boundaries are reused, so a three-digit run inside a six-digit "
            "value is not an occurrence. Its lookbehind also excludes a preceding hyphen, so "
            "a suffix inside a hyphenated filename is not matched here -- that case is B6's "
            "and B6 rejects the point for it"
        ),
    }


def _persistent_spans(
    tokenizer: Any, messages: Sequence[dict[str, Any]], *, spec: Any
) -> tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """The head block's size and the render's token and character chunks, in one pass.

    The chunks tile the render contiguously from column 0, which is what ``prefill`` needs; they
    are **not** indexed by message. ``_spans_by_message`` is the only supported way to go from a
    message index to its span.
    """
    head_count, token_counts, char_counts = _rendered_boundaries(tokenizer, messages, spec=spec)
    return (
        head_count,
        tuple(zip(token_counts[:-1], token_counts[1:], strict=True)),
        tuple(zip(char_counts[:-1], char_counts[1:], strict=True)),
    )


def _spans_by_message(
    head_count: int, chunks: Sequence[tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    """Message index -> its chunk, defined only for messages outside the head block.

    A dict rather than the chunk tuple, because the chunk tuple is shorter than the message list
    and shifted by ``head_count - 1`` whenever the template refuses the shortest prefixes.
    Indexing it positionally by message index would take a real span belonging to the wrong
    message and mask the wrong text -- clean-running and wrong, the one failure mode this
    module's boundaries exist to prevent. A missing key raises instead.
    """
    return {head_count - 1 + position: span for position, span in enumerate(chunks) if position}


def suffix_occurrences_outside(
    text: str, suffix: str, spans: Sequence[tuple[int, int]], *, standalone: bool = True
) -> list[int]:
    """Character offsets where ``suffix`` occurs in ``text`` outside every one of ``spans``.

    ``standalone`` uses ``probes.patch._value_pattern``, which exists for exactly this problem:
    ``32`` inside ``132`` is a different value, not a second occurrence of this one (issue #29),
    and its trailing lookahead was split this morning so a value ending a clause still matches.
    The reasoning transfers, and one boundary of it has to be stated rather than assumed: the
    lookbehind excludes a preceding hyphen, so a suffix inside a hyphenated **filename** does
    not match here. That is correct division of labour and not a hole -- a filename outside the
    masked span is B6's business and B6 already rejects the point for it -- but the two rules
    have to be read together or the pair looks like it covers more than it does.

    With ``standalone=False`` this is the naive any-occurrence rule, kept so the run can report
    both counts: the gap between them is the reason the distinction matters.
    """
    import re

    from local_llm_lab.probes.patch import _value_pattern

    pattern = _value_pattern(suffix) if standalone else re.escape(suffix)
    return [
        match.start()
        for match in re.finditer(pattern, text)
        if not any(low <= match.start() < high for low, high in spans)
    ]


# --------------------------------------------------------------------------- running the arms


def _spans_behind(
    hidden_spans: Sequence[tuple[int, int]], start: int
) -> tuple[tuple[int, int], ...]:
    """The hidden spans that lie wholly before a chunk's first query position.

    A span is carried by a forward only once the whole span is behind it. A chunk that *contains*
    a hidden observation must not be masked over it: those queries are the model reading the
    observation as it arrives, which is how the recurrent state comes to hold it at all, and
    blinding them would delete from the recurrent path the very thing arm A is asking about.

    **Do not collapse the two cases into one rule.** They look like an off-by-one and they are
    not. Arm A has four ways to be quietly wrong, all of which produce a clean-looking result:
    mask too little and attention leaks the answer; mask a suffix that is readable outside the
    span and the same thing happens by another route; mask only at the decision and the answer
    arrives one attention hop away; and mask the observation *while it is arriving*, so the
    recurrent state never acquires what arm A exists to test for and the arm reports a null it
    could not have avoided. ``test_masking_a_hidden_observation_while_it_arrives_is_a_different_experiment``
    fails if a later edit merges the branches.
    """
    return tuple((start_column, end) for start_column, end in hidden_spans if end <= start)


def _chunk(ids: Sequence[int], span: tuple[int, int]) -> Sequence[int]:
    """The token slice one forward runs over."""
    start, end = span
    return ids[start:end]


def forward_schedule(point: dict[str, Any]) -> dict[str, Any]:
    """Every forward each arm runs, with the spans that forward carried.

    Recorded beside the hidden spans so the per-chunk masking is **verifiable from the
    artifact** rather than trusted (Chief, 2026-09-05). A reader can check for themselves which
    chunks carried the mask and which did not, which is the only way to audit the propagation
    fix from outside the code -- and the chunk that contains the hidden observation carrying
    nothing is the visible signature of the rule ``_spans_behind`` documents.
    """
    hidden = point["hidden_spans"]

    def described(span: tuple[int, int], *, scored: bool, spans) -> dict[str, Any]:
        start, end = span
        return {
            "start": start,
            "end": end,
            "tokens": end - start,
            "scored": scored,
            "carried_spans": [[low, high] for low, high in spans],
        }

    persistent = [
        described(span, scored=False, spans=_spans_behind(hidden, span[0]))
        for span in point["prefill_chunks"]
    ]
    persistent.append(
        described(point["scored_chunk"], scored=True, spans=_spans_behind(hidden, point["scored_chunk"][0]))
    )
    arm_c = [described(span, scored=False, spans=()) for span in point["windowed_chunks"]]
    arm_c.append(described(point["windowed_scored_chunk"], scored=True, spans=()))
    return {
        "persistent_chunks": persistent,
        "persistent_head_messages": point["head_messages"],
        "arm_c_chunks": arm_c,
        "arm_c_head_messages": point["windowed_head_messages"],
        "note": (
            "arms A, B and the control share the persistent boundaries; carried_spans is arm "
            "A's masking, arm B carries none at any chunk, and the control carries the foreign "
            "point's spans at these same boundaries. A chunk carrying nothing while a hidden "
            "span lies inside it is the rule, not a gap: the observation is read as it arrives "
            "so the recurrent state acquires it, and only later forwards are blinded to it. "
            "The first prefill chunk covers head_messages messages rather than one: the chat "
            "template refuses to render a shorter prefix, so that block has no interior "
            "boundary. No hidden span may start inside it and select_points raises if one does"
        ),
    }


def prefill(
    view: Any,
    ids: Sequence[int],
    chunks: Sequence[tuple[int, int]],
    *,
    hidden_spans: Sequence[tuple[int, int]],
) -> list[Any]:
    """Build one persistent cache turn by turn, carrying the mask from where the span ends.

    **The mask is carried on every forward after the span, not only at the decision**, and this
    is the one place where the work order's two "forms" are not interchangeable. Masking only at
    the scored step leaves the answer one attention hop away: the positions between the hidden
    observation and the decision attended to it during their own forward, and the scored query
    then attends to *them*. Re-measured before this was written, on a toy decoder with two
    attention blocks: the two constructions differ by 0.23 in logit space. A fixture whose only
    attention block is the last one shows 0.0 and would have hidden it.

    The recurrent blocks take no mask at any step, which is the asymmetry the experiment is.
    Every chunk goes through ``cached_logits`` -- the scored row it unembeds is thrown away
    during a prefill -- so that the prefill and the decision reach the blocks through one code
    path rather than two that must be kept in step.
    """
    cache = view.make_cache()
    for start, end in chunks:
        view.cached_logits(
            ids[start:end], cache, hidden_spans=_spans_behind(hidden_spans, start)
        )
    return cache


def _candidate_probabilities(logits: Any, point: dict[str, Any]) -> dict[str, float]:
    """The two scored candidates' probabilities from one next-token distribution.

    EXP-001 §3.3: P(true suffix) and P(already-read suffix) are reported separately, because an
    ordering statistic cannot say which of the two moved.
    """
    import mlx.core as mx

    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1).reshape(-1)
    return {
        "true": float(probabilities[point["true_token"]].item()),
        "already_read": float(probabilities[point["false_token"]].item()),
    }


def _scored_record(record: dict[str, Any], position: int) -> dict[str, Any]:
    """The mask form and columns that actually ran, in a shape JSON can hold."""
    spans = record.get("hidden_spans", ())
    shape = record.get("attention_mask_shape")
    return {
        "attention_mask_route": record.get("attention_mask_route"),
        "hidden_spans": [[int(start), int(end)] for start, end in spans],
        "query_tokens": record.get("query_tokens"),
        "cache_offset": record.get("cache_offset"),
        "attention_mask_shape": list(shape) if shape is not None else None,
        "scored_position": position,
    }


def score_point(
    view: Any,
    point: dict[str, Any],
    *,
    cache: list[Any] | None,
    hidden_spans: Sequence[tuple[int, int]] | None,
    ids: Sequence[int] | None = None,
    position: int = -1,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score one decision and return its two probabilities beside the record of how.

    ``ids`` defaults to the point's own scored chunk -- the generation prompt and the forced
    note stem -- which is what every persistent arm runs. Arm C passes the windowed prompt
    whole, with no cache, because that is EXP-001's condition and the whole of it.
    """
    record: dict[str, Any] = {}
    if ids is None:
        start, end = point["scored_chunk"]
        ids = point["persistent_ids"][start:end]
    logits = view.cached_logits(
        ids, cache, hidden_spans=hidden_spans, position=position, record=record
    )
    return _candidate_probabilities(logits, point), _scored_record(record, position)


def run_arms(
    view: Any,
    points: Sequence[dict[str, Any]],
    *,
    arms: Sequence[str] = ARMS,
    progress: Any | None = None,
) -> list[dict[str, Any]]:
    """Score every probe point under every requested arm, one persistent cache per arm per point.

    Each arm gets its own prefill rather than sharing one cache, because ``ArraysCache`` is
    untrimmable: a cache cannot be rewound to the decision and re-run under a different mask, so
    reusing one across arms is not available at any price (work order, trap 4). Arm A's cache is
    prefilled *under its own mask* and arm B's without one, which is why they are two caches and
    not one -- and, with B forced onto the explicit-array route that hides nothing, the two arms
    differ by the mask alone.

    The control takes the **next** point's cache, prefilled exactly as arm A's is, and runs this
    point's decision over it. The pairing is EXP-001's rotation, so the two artifacts' nulls are
    built the same way.
    """
    import mlx.core as mx

    selected = [arm for arm in ARMS if arm in arms]
    results: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        foreign = points[(index + 1) % len(points)]
        if "control" in selected and foreign["task_id"] == point["task_id"]:
            # The rotation closes on itself at one point, which would run the control over the
            # point's *own* persistent state. It would then agree with arm A by construction and
            # report a generic perturbation where there was none -- an answer-shaped result.
            raise ValueError(
                "the foreign-state control needs a second probe point: the rotation gave "
                f"{point['task_id']} its own cache, which is arm A and not a control"
            )
        probabilities: dict[str, dict[str, float]] = {}
        scored: dict[str, dict[str, Any]] = {}
        for arm in selected:
            if arm == "C":
                # EXP-001's condition, on the shared forward path (Chief, 2026-09-05): a
                # **fresh** cache holding only this point's own windowed, stripped text,
                # chunked as the persistent arms are. Fresh is the load-bearing word -- no
                # state reaches it from anywhere, so it is still EXP-001's condition -- and
                # sharing the path is what stops the gated-delta kernel's chunking deviation
                # riding on every comparison, since every statistic is paired against C.
                cache = prefill(
                    view, point["windowed_ids"], point["windowed_chunks"], hidden_spans=()
                )
                values, record = score_point(
                    view,
                    point,
                    cache=cache,
                    hidden_spans=(),
                    ids=_chunk(point["windowed_ids"], point["windowed_scored_chunk"]),
                )
            elif arm == "B":
                cache = prefill(
                    view, point["persistent_ids"], point["prefill_chunks"], hidden_spans=()
                )
                values, record = score_point(view, point, cache=cache, hidden_spans=())
            elif arm == "A":
                cache = prefill(
                    view,
                    point["persistent_ids"],
                    point["prefill_chunks"],
                    hidden_spans=point["hidden_spans"],
                )
                values, record = score_point(
                    view, point, cache=cache, hidden_spans=point["hidden_spans"]
                )
            else:
                # The foreign cache's columns are the foreign point's, so the spans that index
                # it must be too; carrying this point's spans would mask arbitrary text.
                cache = prefill(
                    view,
                    foreign["persistent_ids"],
                    foreign["prefill_chunks"],
                    hidden_spans=foreign["hidden_spans"],
                )
                values, record = score_point(
                    view, point, cache=cache, hidden_spans=foreign["hidden_spans"]
                )
            probabilities[arm] = values
            scored[arm] = record
        # EXP-001's own construction, retained beside the chunked one as the reproduction
        # check against that artifact's decisive row. It never enters a paired family.
        single_values, single_record = score_point(
            view, point, cache=None, hidden_spans=None, ids=point["windowed_ids"]
        )
        results.append(
            {
                "task_id": point["task_id"],
                "difficulty": point["difficulty"],
                "true": point["true"],
                "false": point["false"],
                "foreign_task_id": foreign["task_id"],
                "probabilities": probabilities,
                "scored": scored,
                "c_single_forward": {
                    "probabilities": single_values,
                    "scored": single_record,
                    "role": "reproduction check against EXP-001; not in any paired family",
                },
                "hidden_spans": [dict(record) for record in point["hidden_span_records"]],
                "forward_schedule": forward_schedule(point),
                "b6": dict(point["b6"]),
            }
        )
        mx.clear_cache()
        if progress is not None:
            progress(index + 1, len(points), "probe point", task_id=point["task_id"])
    return results


def identity_gate(
    view: Any, point: dict[str, Any], *, tolerance: float = IDENTITY_TOLERANCE
) -> dict[str, Any]:
    """Capture a run's own cache entries, re-inject them into the same run, and compare.

    A gate and not a control (EXP-002 §3.4). It runs arm B's form -- the persistent cache with
    nothing hidden -- twice: once over the cache as prefilled, once over a fresh cache carrying
    the captured state. If capture-and-reinject moves the distribution at all, no arm's number
    means what it says, so this is written into the artifact before any arm and the run stops on
    a failure.

    The observed maximum is recorded **whether it passes or fails**, so a threshold that proves
    too tight on this hardware is recalibrated from evidence rather than from argument.

    The capture is production machinery, not new code: ``pipeline.runner``'s snapshot strategy
    copies and re-injects cache state on every turn, and this reuses its copier so that what the
    gate verifies is what the deployed runner does.
    """
    from local_llm_lab.pipeline.runner import _copy_cache_state

    cache = prefill(view, point["persistent_ids"], point["prefill_chunks"], hidden_spans=())
    captured = [_copy_cache_state(entry.state) for entry in cache]
    direct, direct_record = score_point(view, point, cache=cache, hidden_spans=())

    reinjected_cache = view.make_cache()
    for entry, state in zip(reinjected_cache, captured, strict=True):
        entry.state = state
    reinjected, reinjected_record = score_point(
        view, point, cache=reinjected_cache, hidden_spans=()
    )

    observed = max(abs(direct[name] - reinjected[name]) for name in CANDIDATES)
    return {
        "ruling": "EXP-002 §3.4",
        "task_id": point["task_id"],
        "tolerance": tolerance,
        "observed_max_abs_probability_difference": observed,
        "passed": observed <= tolerance,
        "candidates": list(CANDIDATES),
        "direct": direct,
        "reinjected": reinjected,
        "captured_entries": len(captured),
        "cache_class_by_block_kind": _cache_classes(view, cache),
        "scored": {"direct": direct_record, "reinjected": reinjected_record},
        "recorded_whether_it_passes_or_fails": True,
    }


def identity_gate_family(
    view: Any,
    points: Sequence[dict[str, Any]],
    *,
    count: int = IDENTITY_GATE_POINTS,
    tolerance: float = IDENTITY_TOLERANCE,
) -> dict[str, Any]:
    """Run the gate on several points and gate on the worst of them.

    One point can pass by luck, and a gate that can pass by luck is not a gate (Chief,
    2026-09-05). Each point's observed maximum is recorded, and the run is gated on the maximum
    over all of them, so a single well-behaved point cannot carry a cache that does not
    round-trip elsewhere.
    """
    measured = [
        identity_gate(view, point, tolerance=tolerance)
        for point in points[: max(1, count)]
    ]
    worst = max(gate["observed_max_abs_probability_difference"] for gate in measured)
    return {
        "ruling": "EXP-002 §3.4",
        "tolerance": tolerance,
        "points_measured": len(measured),
        "observed_max_abs_probability_difference": worst,
        "passed": worst <= tolerance,
        "gated_on": "the maximum over every point measured, not the first",
        "per_point": measured,
        "cache_class_by_block_kind": measured[0]["cache_class_by_block_kind"],
        "captured_entries": measured[0]["captured_entries"],
        "recorded_whether_it_passes_or_fails": True,
    }


def chunking_deviation(view: Any, point: dict[str, Any]) -> dict[str, Any]:
    """What the shared chunked path actually cancels, measured rather than assumed.

    Prefilling turn by turn is not bit-identical to one forward over the whole sequence. Arm C
    is now chunked like the persistent arms so that deviation cancels in the paired
    differences -- but arm C's text is shorter and chunks at different boundaries, so the
    cancellation can only be partial and the residual has to be measured (Chief, 2026-09-05).

    Measured in the candidate probabilities, which is the quantity every statistic is computed
    from. That distinction matters and correcting it changed the number by two orders of
    magnitude: the 7.87e-06 first recorded here was a maximum over the whole vocabulary in
    *logit* space on one toy sequence, and the same setup in candidate probabilities is
    2.16e-07. On persistent-versus-windowed shapes each arm's own deviation is 1.5e-08 to
    2.6e-08 and the residual left on the paired difference is 4.7e-09 to 1.1e-08 -- smaller
    than either arm's own deviation in every chunking tried, so the cancellation is real, and
    about ninety times below the identity gate's own tolerance.
    """
    persistent, windowed = point["persistent_ids"], point["windowed_ids"]
    chunked_b = score_point(
        view,
        point,
        cache=prefill(view, persistent, point["prefill_chunks"], hidden_spans=()),
        hidden_spans=(),
    )[0]
    single_b = score_point(view, point, cache=None, hidden_spans=None, ids=persistent)[0]
    chunked_c = score_point(
        view,
        point,
        cache=prefill(view, windowed, point["windowed_chunks"], hidden_spans=()),
        hidden_spans=(),
        ids=_chunk(windowed, point["windowed_scored_chunk"]),
    )[0]
    single_c = score_point(view, point, cache=None, hidden_spans=None, ids=windowed)[0]

    delta_b = max(abs(chunked_b[name] - single_b[name]) for name in CANDIDATES)
    delta_c = max(abs(chunked_c[name] - single_c[name]) for name in CANDIDATES)
    residual = max(
        abs((chunked_b[name] - chunked_c[name]) - (single_b[name] - single_c[name]))
        for name in CANDIDATES
    )
    return {
        "task_id": point["task_id"],
        "quantity": (
            "maximum absolute difference over the scored candidates' probabilities, which is "
            "what every statistic is computed from; a logit-space maximum over the whole "
            "vocabulary is a different and much larger number and is not what rides on a "
            "comparison"
        ),
        "persistent_chunked_vs_single": delta_b,
        "arm_c_chunked_vs_single": delta_c,
        "residual_on_the_paired_difference": residual,
        "cancels": residual <= max(delta_b, delta_c),
        "identity_tolerance": IDENTITY_TOLERANCE,
        "below_identity_tolerance": residual < IDENTITY_TOLERANCE,
        "why_it_is_only_partial": (
            "arm C's text is the windowed one, which is shorter and chunks at different "
            "boundaries, so the two arms' kernel deviations are not the same number and "
            "cannot cancel exactly"
        ),
    }


def compare_to_exp001(results: Sequence[dict[str, Any]], path: Any) -> dict[str, Any]:
    """Compare arm C's single forward against EXP-001's own per-case model output.

    Turns "arm C reproduces EXP-001's decisive row" from something read by eye into something
    the artifact states. EXP-001 keys its per-case probabilities by candidate label, with
    ``false`` where this artifact says ``already_read``; the pairing is by task id, and points
    the older artifact does not carry are counted rather than silently dropped.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    by_task = {
        case.get("task_id"): case
        for case in record.get("per_case", [])
        if isinstance(case, dict)
    }
    matched = 0
    worst = 0.0
    for result in results:
        case = by_task.get(result["task_id"])
        theirs = (case or {}).get("matched", {}).get("model_output")
        if not isinstance(theirs, dict):
            continue
        ours = result["c_single_forward"]["probabilities"]
        matched += 1
        worst = max(
            worst,
            abs(ours["true"] - float(theirs["true"])),
            abs(ours["already_read"] - float(theirs["false"])),
        )
    return {
        "artifact": str(path),
        "points_in_run": len(results),
        "points_matched_by_task_id": matched,
        "max_abs_probability_difference": worst if matched else None,
        "compared": "arm C's single uncached forward against EXP-001's per_case model_output",
    }


def _cache_classes(view: Any, cache: Sequence[Any]) -> dict[str, str]:
    """Which cache class the model's own factory built for each block kind (R34)."""
    classes: dict[str, str] = {}
    for index, entry in enumerate(cache):
        kind = view.layer_kind(index)
        classes.setdefault(kind, type(entry).__name__)
    return classes


# ------------------------------------------------------------------------------ the statistic


def _within_row(true_values: Sequence[float], read_values: Sequence[float]) -> dict[str, Any]:
    """One "is P(true) above P(already-read)?" row, with every count needed to rebuild ``n``.

    ``wins``, ``losses`` and ``ties`` are all stated, in the same shape as the paired rows'
    discordant-pair counts, so a reader can reconstruct each row's denominator from the
    artifact instead of having to know which tie convention applied where (Chief, 2026-09-05).
    The conventions differ deliberately and the counts are what make the asymmetry visible.
    """
    pairs = list(zip(true_values, read_values, strict=True))
    wins = sum(1 for a, b in pairs if a > b)
    losses = sum(1 for a, b in pairs if a < b)
    ties = len(pairs) - wins - losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "points": len(pairs),
        "n": len(pairs),
        "n_rule": "every point; a tie counts as a non-win, which is EXP-001's convention",
        "rate": wins / len(pairs) if pairs else 0.0,
        "p": sign_test(wins, len(pairs)),
        "median_p_true": statistics.median(true_values) if true_values else 0.0,
        "median_p_already_read": statistics.median(read_values) if read_values else 0.0,
    }


def aggregate(
    results: Sequence[dict[str, Any]], arms: Sequence[str] = ARMS
) -> dict[str, Any]:
    """Within-arm preferences and the paired comparisons against arm C (EXP-002 §3.5).

    Two families, and they use the sign test differently on purpose.

    * **Within an arm**, the row is EXP-001's decisive row: how often P(true) exceeds
      P(already-read). It keeps EXP-001's exact convention -- a tie counts as a non-win -- so
      that arm C's cell can be compared with that artifact's number rather than merely resemble
      it. Ties are counted and reported beside it.
    * **Against arm C**, the row is a paired sign test on the same quantity in two arms, and
      there ties are excluded from ``n`` as the sign test conventionally requires. An exact
      equality between two arms is a real event here -- arm A and arm B can coincide where the
      mask covers nothing the decision reads -- so it is reported rather than counted as a loss.

    Multiplicity: Holm across the arms, per quantity, as EXP-002 §3.5 pre-registers.
    """
    selected = [arm for arm in ARMS if arm in arms]
    within: dict[str, Any] = {}
    for arm in selected:
        true_values = [result["probabilities"][arm]["true"] for result in results]
        read_values = [result["probabilities"][arm]["already_read"] for result in results]
        within[arm] = _within_row(true_values, read_values)

    paired: dict[str, Any] = {}
    if "C" in selected:
        for candidate in CANDIDATES:
            family: dict[str, Any] = {}
            for arm in selected:
                if arm == "C":
                    continue
                differences = [
                    result["probabilities"][arm][candidate]
                    - result["probabilities"]["C"][candidate]
                    for result in results
                ]
                wins = sum(1 for value in differences if value > 0)
                losses = sum(1 for value in differences if value < 0)
                ties = len(differences) - wins - losses
                total = wins + losses
                family[arm] = {
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "points": len(differences),
                    "n": total,
                    "n_rule": "wins + losses; ties are excluded, as the sign test requires",
                    "rate": wins / total if total else 0.0,
                    "p": sign_test(wins, total),
                }
            for arm, adjusted in holm_adjust(
                {arm: value["p"] for arm, value in family.items()}
            ).items():
                family[arm]["p_holm"] = adjusted
            paired[candidate] = family
    single = [
        result["c_single_forward"]["probabilities"]
        for result in results
        if "c_single_forward" in result
    ]
    arm_c_single: dict[str, Any] = {}
    if single:
        # EXP-001's own construction, reported in the same shape as a within-arm row so the two
        # artifacts' decisive rows sit side by side, with its distance from the chunked arm C
        # stated rather than left to be assumed small.
        arm_c_single = _within_row(
            [values["true"] for values in single],
            [values["already_read"] for values in single],
        )
        differences = [
            max(
                abs(values[name] - result["probabilities"]["C"][name])
                for name in CANDIDATES
            )
            for values, result in zip(single, results, strict=True)
            if "C" in result["probabilities"]
        ]
        arm_c_single.update(
            {
                "role": "reproduction check against EXP-001; never in a paired family",
                "max_abs_difference_vs_chunked": max(differences) if differences else None,
                "median_abs_difference_vs_chunked": (
                    statistics.median(differences) if differences else None
                ),
            }
        )
    return {
        "within_arm": within,
        "arm_c_single_forward": arm_c_single,
        "paired_vs_C": paired,
        "tie_handling": (
            "within_arm keeps EXP-001's convention (a tie is a non-win) so arm C reproduces "
            "that artifact's decisive row; paired_vs_C excludes ties from n, as the sign test "
            "requires, and reports how many there were"
        ),
    }


# ------------------------------------------------------------------------------- the artifact


def conformance_block(
    *,
    cache_classes: dict[str, str],
    cache_trimmable: bool,
    keep_last: int,
    row_keep_last: int,
    probe_step: int,
    scored: dict[str, dict[str, Any]],
    points: int,
    rejected: dict[str, int],
    capture_dtype: str,
    chunking: dict[str, Any] | None = None,
    naive_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The R34 block: which cache, which spans, how located, and where each arm was scored."""
    return {
        "ruling": "R34",
        "estimator": (
            "the model's own next-token distribution over the suffix's first token, per arm, "
            "paired per probe point; P(true) and P(already-read) reported separately with "
            "their absolute values, uniform over ten digits being 0.1 (EXP-001 §3.3)"
        ),
        "probe_points": {
            "family": (
                "EXP-001's unchanged: split jsweep, ledger_reconcile, step "
                f"{probe_step}, the first read after the listing leaves the window, leakage "
                "guard against the windowed text, coinciding first suffix tokens skipped"
            ),
            "usable": points,
            "rejected": dict(sorted(rejected.items())),
        },
        "cache_class_by_block_kind": dict(cache_classes),
        "cache_trimmable": cache_trimmable,
        "persistent_cache_arms": list(PERSISTENT_ARMS),
        "cache_construction": (
            "the model's own make_cache, prefilled one chunk per message of the history and "
            "then one chunk for the generation prompt and the forced note stem, through a "
            "single persistent cache per arm per point; ArraysCache is untrimmable, so a cache "
            "cannot be rewound to the decision and each arm needs its own prefill"
        ),
        "chunked_prefill_deviation": chunking,
        "suffix_rule": naive_rule,
        "b6_construction": (
            "observations unstripped so the listing is present, notes stripped with "
            "strip_pending as in EXP-001; the filename's occurrence count in the rendered "
            "prompt and its position relative to the masked span are counted per point and a "
            "point that fails either is dropped and tallied above"
        ),
        "hidden_span_location": (
            "the observation messages keep_last would hide, taken from window_messages itself "
            "by comparing its output with its input, then located in the rendered token "
            "sequence by rendering each prefix of the conversation through build_prompt and "
            "taking its length; both the character and the token nesting of successive renders "
            "are checked, so a template that re-rendered an earlier turn fails rather than "
            "moving every boundary silently"
        ),
        "hidden_span_recording": (
            "per point, the resolved absolute column ranges the mask indexed, with their token "
            "counts -- the derivation's output, not the message indices it was computed from"
        ),
        "hidden_span_masking": (
            "carried on every forward whose queries begin at or after the span's end, not only "
            "at the scored step: positions between the hidden observation and the decision "
            "would otherwise have attended to it during their own forward, and the scored "
            "query then attends to them. The recurrent blocks take no mask at any step"
        ),
        "arm_scoring": {arm: dict(record) for arm, record in scored.items()},
        "forward_schedule_recording": (
            "every point records each forward its arms ran -- start, end, token count, whether "
            "it was the scored one, and the spans it carried -- so the per-chunk masking is "
            "checkable from the artifact rather than trusted"
        ),
        "arm_c_forms": (
            "arm C runs twice. The comparator is a fresh cache holding only its own windowed, "
            "stripped text, chunked as the persistent arms are, so the four arms differ by "
            "mask and cache content alone; every paired statistic uses that one. EXP-001's own "
            "single uncached forward is kept beside it as the reproduction check and enters no "
            "family"
        ),
        "arm_b_route": (
            "arm B forces the explicit boolean array with no column hidden, rather than the "
            "model's default sentinel route, so arms A and B differ by the mask alone; S1's "
            "acceptance is that the two routes agree exactly on an otherwise identical forward"
        ),
        "window": {
            "keep_last": keep_last,
            "row_keep_last": row_keep_last,
            "arm_c": "the windowed, stripped text, rendered exactly as EXP-001 renders it",
            "persistent_arms": (
                "the same conversation unwindowed, so the listing observation is present in "
                "full; window_messages of it is asserted equal to arm C's own messages"
            ),
        },
        "capture_dtype": {
            "registry": capture_dtype,
            "applies": False,
            "reason": (
                "R18b's capture dtype governs residual capture, and EXP-002 captures no "
                "residual: the blocks run natively and only ArchitectureView's own readout "
                "casts to float32. The registry value is recorded so two artifacts can still "
                "be compared on it, with the note that this run does not use it"
            ),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """The arms table, the gate above it, and the conformance statement.

    Probabilities and p-values are rendered to four significant figures rather than to three
    decimal places. EXP-001's table used decimals and its numbers survived it, but the absolute
    probabilities EXP-001 §3.3 requires alongside the ordering statistic are exactly the ones a
    fixed decimal count can round to ``0.000`` -- a real number printed as zero, in the column
    the reading depends on. The JSON carries full precision either way.
    """
    gate = payload.get("identity_gate", {})
    stats = payload.get("results", {})
    within = stats.get("within_arm", {})
    conformance = payload.get("conformance", {})
    lines = [
        "# EXP-002: recurrent state swap under a persistent cache",
        "",
        f"model: `{payload.get('model')}`  policy: `{payload.get('policy')}`  "
        f"points: {payload.get('points')}  arms: {payload.get('arms')}",
        "",
        "## Identity gate (recorded before any arm)",
        "",
        f"- observed maximum absolute probability difference: "
        f"{gate.get('observed_max_abs_probability_difference')}",
        f"- tolerance: {gate.get('tolerance')}  passed: {gate.get('passed')}",
        f"- points measured: {gate.get('points_measured')}  "
        f"gated on: {gate.get('gated_on')}",
        "- per point: "
        + ", ".join(
            f"{entry.get('task_id')}={entry.get('observed_max_abs_probability_difference'):.4g}"
            for entry in gate.get("per_point", [])
        ),
        f"- captured entries: {gate.get('captured_entries')}  "
        f"cache classes: {gate.get('cache_class_by_block_kind')}",
        "",
        "If the gate fails, no arm below is readable.",
        "",
        "## How often is the TRUE suffix more probable than a wrong, previously-seen one?",
        "",
        "| arm | wins | rate | p | median P(true) | median P(already-read) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, value in within.items():
        lines.append(
            f"| {arm} | {value['wins']}/{value['n']} | {value['rate']:.0%} | "
            f"{value['p']:.4g} | {value['median_p_true']:.4g} | "
            f"{value['median_p_already_read']:.4g} |"
        )
    single = stats.get("arm_c_single_forward") or {}
    if single:
        lines.append(
            f"| C, single forward | {single['wins']}/{single['n']} | {single['rate']:.0%} | "
            f"{single['p']:.4g} | {single['median_p_true']:.4g} | "
            f"{single['median_p_already_read']:.4g} |"
        )
    lines += [
        "",
        "Arm C runs twice. The row above it is the chunked comparator every statistic is",
        "paired against; the single-forward row is EXP-001's own construction, kept as the",
        f"reproduction check and in no family. They differ by at most "
        f"{single.get('max_abs_difference_vs_chunked')}.",
        "",
        "Arm C is EXP-001's condition and its row is the one that must reproduce that",
        "artifact's decisive row; arm B is the upper bound and the gate on the instrument.",
        "",
        "## Paired against arm C, per quantity (Holm across the arms)",
        "",
        "| quantity | arm | wins | losses | ties | p | p, Holm-adjusted |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for quantity, family in stats.get("paired_vs_C", {}).items():
        for arm, value in family.items():
            holm = value.get("p_holm")
            lines.append(
                f"| {quantity} | {arm} | {value['wins']} | {value['losses']} | "
                f"{value['ties']} | {value['p']:.4g} | "
                f"{'-' if holm is None else f'{holm:.4g}'} |"
            )
    lines += [
        "",
        "## Conformance (R34)",
        "",
        f"- cache class per block kind: {conformance.get('cache_class_by_block_kind')}",
        f"- cache trimmable: {conformance.get('cache_trimmable')}",
        f"- spans located: {conformance.get('hidden_span_location')}",
        f"- spans masked: {conformance.get('hidden_span_masking')}",
        f"- B6: {conformance.get('b6_construction')}",
        f"- suffix rule: {json.dumps(conformance.get('suffix_rule', {}), sort_keys=True)}",
        f"- chunked prefill deviation: "
        f"{json.dumps(conformance.get('chunked_prefill_deviation', {}), sort_keys=True)}",
        f"- forward schedule: {conformance.get('forward_schedule_recording')}",
        f"- arm C forms: {conformance.get('arm_c_forms')}",
        f"- probe points: {conformance.get('probe_points')}",
        f"- scored: {json.dumps(conformance.get('arm_scoring', {}), sort_keys=True)}",
        "",
        f"Comparability (R35): {json.dumps(payload.get('comparability', {}), sort_keys=True)}",
    ]
    if payload.get("results", {}).get("exp001_reproduction"):
        lines += [
            "",
            "## Reproduction against EXP-001",
            "",
            json.dumps(payload["results"]["exp001_reproduction"], sort_keys=True),
        ]
    return "\n".join(lines)


def _default_seed() -> int:
    """The generator's own default data seed; never a literal in this module."""
    import inspect

    from local_llm_lab.pipeline.tasks import make_tasks

    return inspect.signature(make_tasks).parameters["seed"].default


def _resolve_arms(value: str | None) -> tuple[str, ...]:
    """Parse ``--arms``, keeping the artifact's order and insisting on arm C."""
    if value is None:
        return ARMS
    requested = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in requested if name not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s): {', '.join(unknown)}; expected some of {list(ARMS)}")
    if not requested:
        raise ValueError("--arms must name at least one arm")
    if "C" not in requested:
        # Every reported statistic is paired against arm C, and arm C is the arm that ties this
        # instrument to EXP-001's decisive row. A run without it produces no readable number.
        raise ValueError(
            "--arms must include C: every statistic is paired against it, and it is the arm "
            "that has to reproduce EXP-001's decisive row before the others can be read"
        )
    return tuple(arm for arm in ARMS if arm in requested)


def _write_artifact(output: Path, payload: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "state_swap.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "state_swap.md").write_text(render_markdown(payload) + "\n", encoding="utf-8")
    return target


def main() -> None:  # noqa: C901 - probe CLI orchestration
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import preflight as preflight_module
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.pipeline.jlens import comparability_block
    from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST
    from local_llm_lab.pipeline.tasks import (
        GENERATOR_VERSION,
        JSPACE_SPLIT_LIMIT,
        JSPACE_SPLIT_NAMES,
        make_jspace_tasks,
        make_tasks,
    )
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu
    from local_llm_lab.probes.policies import resolve_policy
    from local_llm_lab.probes.state_probe import preflight_precision_block, spec_capture_dtype
    from local_llm_lab.provenance import write_provenance
    from local_llm_lab.runlog import RunLog, git_commit

    parser = argparse.ArgumentParser(
        description=(
            "EXP-002: a within-run dissociation between the recurrent and attention paths "
            "under one persistent cache, with the identity gate recorded before any arm."
        )
    )
    parser.add_argument("--model", default="qwen35-4b")
    parser.add_argument(
        "--policy", default="base", help="a policy named by the selected model; default base"
    )
    parser.add_argument("--split", default=TASK_SPLIT)
    parser.add_argument("--count", type=int, default=JSPACE_SPLIT_LIMIT)
    parser.add_argument("--probe-step", type=int, default=PROBE_STEP)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument(
        "--generator-version",
        type=int,
        default=None,
        help="Generator version the probe points are bound to; defaults to HEAD, recorded.",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help=f"comma-separated subset of {','.join(ARMS)}; must include C. Default: all four.",
    )
    parser.add_argument(
        "--exp001-artifact",
        type=Path,
        default=None,
        help=(
            "EXP-001 sweep.json to check arm C's single forward against; the maximum absolute "
            "probability difference and the number of points matched by task id are recorded."
        ),
    )
    parser.add_argument("--skip-preflight-check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    add_gpu_arguments(parser)
    args = parser.parse_args()

    spec = load_model_spec(args.model)
    try:
        arms = _resolve_arms(args.arms)
        adapter = resolve_policy(args.policy, spec)
    except ValueError as error:
        parser.error(str(error))
    if args.count < 1:
        parser.error("--count must be positive")
    if args.probe_step < 0:
        parser.error("--probe-step must be non-negative")

    # The seed is always recorded, whether it came from the flag or the generator's own
    # default: an artifact that does not name its seed cannot be replayed (N3, R12/R23).
    data_seed_source = "cli" if args.data_seed is not None else "generator-default"
    data_seed = args.data_seed if args.data_seed is not None else _default_seed()
    generator_version = (
        GENERATOR_VERSION if args.generator_version is None else args.generator_version
    )
    if generator_version != GENERATOR_VERSION:
        parser.error(
            f"--generator-version {generator_version} does not match HEAD's "
            f"{GENERATOR_VERSION}; replaying an older generator is an R12/R23 tool's job, not "
            "this probe's"
        )
    capture_dtype = spec_capture_dtype(spec)

    # R26(e): the log opens once the identity is known, before the preflight gate, the GPU
    # guard and the model load, so run.log alone says what ran and covers the whole run.
    with RunLog.open(
        args.output,
        name="state-swap",
        command=sys.argv,
        identity={
            "model": spec.name,
            "hf_id": spec.hf_id,
            "policy": args.policy,
            "adapter": str(adapter) if adapter is not None else None,
            "split": args.split,
            "count": args.count,
            "probe_step": args.probe_step,
            "arms": ",".join(arms),
            "data_seed": data_seed,
            "data_seed_source": data_seed_source,
            "generator_version": generator_version,
            "capture_dtype": capture_dtype,
            "identity_tolerance": IDENTITY_TOLERANCE,
            "git_commit": git_commit(),
        },
    ) as log:
        preflight_module.require_preflight(spec, skip=args.skip_preflight_check)
        if args.split in JSPACE_SPLIT_NAMES:
            tasks = make_jspace_tasks(args.split, args.count, data_seed)
        else:
            tasks = make_tasks(args.split, args.count, data_seed)
        ledger = [task for task in tasks if task.family == "ledger_reconcile"]
        log.info("tasks", split=args.split, generated=len(tasks), ledger=len(ledger))

        require_idle_gpu(parser, args, "loading the EXP-002 state-swap model")
        log.info("loading policy", model=args.model, policy=args.policy)
        model, tokenizer, view, resolved = load_policy(spec, adapter)
        del model

        try:
            points, rejected = select_points(
                ledger, tokenizer, spec=spec, probe_step=args.probe_step
            )
        except ValueError as error:
            log.error("probe point construction failed", detail=str(error))
            parser.error(str(error))
        log.info("probe points", usable=len(points), rejected=json.dumps(rejected, sort_keys=True))
        if len(points) < MIN_CASES:
            parser.error(
                f"only {len(points)} usable probe point(s); the sign test needs at least "
                f"{MIN_CASES}"
            )

        # §3.4: the gate runs and is recorded before any arm. It is arm B's form -- the
        # persistent cache with nothing hidden -- run twice over captured and re-injected state.
        gate = identity_gate_family(view, points)
        log.metric(
            "identity gate",
            observed=gate["observed_max_abs_probability_difference"],
            tolerance=gate["tolerance"],
            points_measured=gate["points_measured"],
            passed=gate["passed"],
        )
        # Measured on the gate's first point, once, and recorded as a number in the
        # conformance block rather than as a caveat in prose.
        chunking = chunking_deviation(view, points[0])
        log.metric(
            "chunking deviation",
            residual=chunking["residual_on_the_paired_difference"],
            arm_c=chunking["arm_c_chunked_vs_single"],
            persistent=chunking["persistent_chunked_vs_single"],
            cancels=chunking["cancels"],
        )
        naive_rule = naive_rule_comparison(points, rejected)

        common = {
            "model": spec.name,
            "hf_id": spec.hf_id,
            "policy": args.policy,
            "adapter": None if adapter is None else str(adapter),
            "split": args.split,
            "count": args.count,
            "probe_step": args.probe_step,
            "data_seed": data_seed,
            "data_seed_source": data_seed_source,
            "generator_version": generator_version,
            "arms": list(arms),
            "identity_gate": gate,
            "points": len(points),
        }
        comparability = comparability_block(
            model=spec.name,
            hf_id=spec.hf_id,
            policy=args.policy,
            adapter=None if adapter is None else str(adapter),
            # No derivative is taken here: EXP-002's decisive measurement is the model's own
            # output, so the field says so rather than naming a method that was never run.
            jvp_method="none: the decisive measurement is the model's own next-token output",
            template_kwargs=dict(spec.chat.template_kwargs),
            # Arm C's rows reach select_points already windowed by build_rows and the render
            # re-windows nothing, exactly as EXP-001's do; the persistent arms are a second
            # window over the same conversation and are described beside it below.
            keep_last=None,
            row_keep_last=DEFAULT_KEEP_LAST,
            rewindowed=False,
            estimator_variant="model_output",
            layer_selection=None,
            layer_kinds={},
            generator_version=generator_version,
            data_seed=data_seed,
            fp32_manual_vs_native=preflight_precision_block(spec),
        )
        comparability["prompt_rendering"]["persistent_arms"] = {
            "keep_last": None,
            "row_keep_last": None,
            "rewindowed": False,
            "windowing_rule": (
                "the persistent arms prefill the same conversation unwindowed, so the listing "
                "observation is present in full and the filename enters once; window_messages "
                f"of it at keep_last={DEFAULT_KEEP_LAST} is asserted equal to arm C's own "
                "messages, so the two arms are two windows over one conversation"
            ),
        }

        if not gate["passed"]:
            # Recorded before the refusal: a threshold that proves too tight on this hardware
            # is recalibrated from the observed maximum, which only exists if it is written out.
            payload = {
                **common,
                "results": {},
                "per_point": [],
                "conformance": conformance_block(
                    cache_classes=gate["cache_class_by_block_kind"],
                    cache_trimmable=_cache_trimmable(view),
                    keep_last=DEFAULT_KEEP_LAST,
                    row_keep_last=DEFAULT_KEEP_LAST,
                    probe_step=args.probe_step,
                    scored={"identity_gate": gate["per_point"][0]["scored"]["direct"]},
                    points=len(points),
                    rejected=rejected,
                    capture_dtype=capture_dtype,
                    chunking=chunking,
                    naive_rule=naive_rule,
                ),
                "comparability": comparability,
            }
            _write_artifact(args.output, payload)
            log.error(
                "identity gate failed",
                observed=gate["observed_max_abs_probability_difference"],
                tolerance=gate["tolerance"],
            )
            parser.error(
                "identity gate failed: capture-and-reinject moved the scored distribution by "
                f"{gate['observed_max_abs_probability_difference']}, above the tolerance of "
                f"{gate['tolerance']}. No arm is readable; the observed maximum is recorded in "
                f"{args.output / 'state_swap.json'} for recalibration."
            )

        results = run_arms(view, points, arms=arms, progress=log.progress)
        stats = aggregate(results, arms)
        conformance = conformance_block(
            cache_classes=gate["cache_class_by_block_kind"],
            cache_trimmable=_cache_trimmable(view),
            keep_last=DEFAULT_KEEP_LAST,
            row_keep_last=DEFAULT_KEEP_LAST,
            probe_step=args.probe_step,
            # The first point's records, which are the forms every point ran under: the route
            # is a function of the scored chunk's length and the cache, both of which are the
            # same shape at every point, and the per-point records carry each one anyway.
            scored={arm: results[0]["scored"][arm] for arm in arms},
            points=len(points),
            rejected=rejected,
            capture_dtype=capture_dtype,
            chunking=chunking,
            naive_rule=naive_rule,
        )
        if args.exp001_artifact is not None:
            try:
                stats["exp001_reproduction"] = compare_to_exp001(results, args.exp001_artifact)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                # A missing or malformed comparison artifact must not cost the run its data.
                log.warn("exp001 comparison unavailable", detail=str(error))
                stats["exp001_reproduction"] = {
                    "artifact": str(args.exp001_artifact),
                    "error": str(error),
                }
            else:
                log.info(
                    "exp001 reproduction",
                    matched=stats["exp001_reproduction"]["points_matched_by_task_id"],
                    max_abs=stats["exp001_reproduction"]["max_abs_probability_difference"],
                )
        payload = {
            **common,
            "results": stats,
            "per_point": results,
            "conformance": conformance,
            "comparability": comparability,
        }
        target = _write_artifact(args.output, payload)
        log.info("wrote artifact", path=str(target))
        write_provenance(
            args.output,
            resolved=resolved,
            spec=spec,
            extra={
                "stage": "state-swap",
                "conformance": conformance,
                "comparability": comparability,
            },
        )


def _cache_trimmable(view: Any) -> bool | None:
    """Whether the model's own cache can be trimmed, or ``None`` where the view cannot say.

    Recorded rather than assumed: the whole regime exists *because* ``ArraysCache`` inherits
    ``is_trimmable() == False``, so an artifact claiming a trimmable cache would be describing
    a different experiment.
    """
    try:
        return bool(view.cache_trimmable)
    except (AttributeError, ValueError):
        return None
