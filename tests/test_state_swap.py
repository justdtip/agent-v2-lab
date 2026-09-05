"""EXP-002 S3: the four arms, the identity gate and the artifact.

Fakes and tiny real models only (R10): no checkpoint, no tokenizer from the Hub, nothing on the
GPU. Two seams are driven for real because a stand-in cannot show what they do (R31):

* ``pipeline.data.build_rows``, ``pipeline.protocol.window_messages`` and
  ``pipeline.protocol.build_prompt`` over real generated tasks, because the B6 invariant is a
  claim about text those functions produce and a hand-written conversation could be made to
  satisfy it by construction;
* ``mlx_lm``'s own ``ArraysCache`` and ``KVCache`` through a toy Qwen3.5 for the identity gate,
  because whether captured cache state round-trips -- and whether re-injecting it restores the
  attention offset -- is a fact about those classes and not about a shape a fake could be given.

The decoder itself is faked wherever the arms are being compared, and deliberately so: the fake
has an attention route and a recurrent route that can be switched independently, so a test can
put the hidden filename on exactly one of them and check the pipeline reports it there. A real
model would put the answer wherever it happens to be, which is the run's question, not the
instrument's.
"""

from __future__ import annotations

import importlib
import inspect
import json
import tomllib
from pathlib import Path

import mlx.core as mx
import pytest
from test_arch import make_tiny_hybrid_model

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.protocol import DEFAULT_KEEP_LAST, window_messages
from local_llm_lab.pipeline.tasks import make_jspace_tasks, make_tasks
from local_llm_lab.probes import state_swap
from local_llm_lab.probes.jspace_sweep import PROBE_STEP

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="cpu_stream")
def _cpu_stream():
    """Keep every slice here off the GPU: toy weights, and nothing needs a device.

    The real toy Qwen3.5 comes from ``test_arch`` rather than being rebuilt -- the identity
    gate's claim is about ``ArraysCache`` and ``KVCache``, which is exactly what that fixture
    is -- but the stream fixture is declared here so no test argument shadows an import.
    """
    with mx.stream(mx.cpu):
        yield


def _default_seed() -> int:
    """The generator's own default; the tests name no seed literal either."""
    return inspect.signature(make_tasks).parameters["seed"].default


def _ledger_tasks(count: int = 60):
    return [
        task
        for task in make_jspace_tasks("jsweep", count, _default_seed())
        if task.family == "ledger_reconcile"
    ]


class _SwapTokenizer:
    """One id per character, and a chat template whose renders nest as prefixes.

    ``additive=False`` builds the one template shape the span locator cannot survive: a render
    of ``messages[:k]`` that is not a prefix of the render of ``messages[:k + 1]``. Real chat
    templates are prefix-additive over these conversations, but a template that merged or
    re-rendered earlier turns would silently move every boundary, so the locator has to check
    rather than assume.
    """

    bos_token = None

    def __init__(self, *, additive: bool = True) -> None:
        self.additive = additive
        self.template_calls: list[dict] = []

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % 64 for character in text]

    def decode(self, ids):
        return "".join(chr(65 + int(i) % 26) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        del tokenize
        self.template_calls.append(dict(kwargs))
        bodies = [str(message["content"]) for message in messages]
        if not self.additive and bodies:
            bodies[0] = f"{len(bodies)}{bodies[0]}"
        body = "\n".join(bodies)
        if not add_generation_prompt:
            return body
        from local_llm_lab.pipeline.protocol import generation_suffix

        return body + "\n" + generation_suffix(load_model_spec("qwen35-4b"))


# ------------------------------------------------------------------- the installed entry


def test_state_swap_is_an_installed_console_script() -> None:
    """EXP-002 §4 names the CLI, so it has to be in the wheel like every other probe."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["agent-v2-state-swap"] == (
        "local_llm_lab.probes.state_swap:main"
    )
    assert callable(importlib.import_module("local_llm_lab.probes.state_swap").main)


# --------------------------------------------------------- which observations are hidden


def test_hidden_message_indices_come_from_window_messages_itself() -> None:
    """The hidden set is read off the function that hides, not from a second copy of its rule.

    ``window_messages`` replaces a hidden observation's content rather than dropping it, so the
    set can be recovered by comparing its output with its input. Re-deriving it from
    ``keep_last`` and the tool positions would be a second implementation of the same rule,
    free to drift from the one that renders arm C.
    """
    task = _ledger_tasks()[0]
    from local_llm_lab.pipeline.data import build_rows

    messages = build_rows(task, keep_last=len(task.steps) + 1)[PROBE_STEP]["messages"][:-1]
    windowed = window_messages(messages, DEFAULT_KEEP_LAST)

    hidden = state_swap.hidden_message_indices(messages, keep_last=DEFAULT_KEEP_LAST)

    assert hidden == tuple(
        index
        for index, (before, after) in enumerate(zip(messages, windowed, strict=True))
        if before["content"] != after["content"]
    )
    assert all(messages[index]["role"] == "tool" for index in hidden)
    # The listing is the first observation and the window hides it at step 3; that is the whole
    # premise of the probe point, so it is asserted rather than assumed.
    assert hidden and "FILES:" in messages[hidden[0]]["content"]


def test_an_unwindowed_history_windows_back_to_arm_cs_own_text() -> None:
    """Arm C must be exactly ``window_messages`` of the history the other arms prefill."""
    task = _ledger_tasks()[0]
    from local_llm_lab.pipeline.data import build_rows

    unwindowed = build_rows(task, keep_last=len(task.steps) + 1)[PROBE_STEP]["messages"][:-1]
    windowed = build_rows(task)[PROBE_STEP]["messages"][:-1]

    assert window_messages(unwindowed, DEFAULT_KEEP_LAST) == windowed
    assert unwindowed != windowed


# ------------------------------------------------------------------ locating the spans


def test_message_token_spans_partition_the_rendered_prompt() -> None:
    """Spans are located in token space, and must tile the render with no gap or overlap."""
    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")
    messages = _ledger_tasks()[0]
    from local_llm_lab.pipeline.data import build_rows

    messages = build_rows(messages, keep_last=99)[PROBE_STEP]["messages"][:-1]

    spans, prompt_tokens = state_swap.message_token_spans(tokenizer, messages, spec=spec)

    assert len(spans) == len(messages)
    assert spans[0][0] == 0
    assert [span[0] for span in spans[1:]] == [span[1] for span in spans[:-1]]
    assert spans[-1][1] == prompt_tokens
    assert all(start < end for start, end in spans)


def test_message_token_spans_refuse_a_template_that_is_not_prefix_additive() -> None:
    """A template that rewrites earlier turns would move every boundary without a word."""
    tokenizer = _SwapTokenizer(additive=False)
    spec = load_model_spec("qwen35-4b")
    from local_llm_lab.pipeline.data import build_rows

    messages = build_rows(_ledger_tasks()[0], keep_last=99)[PROBE_STEP]["messages"][:-1]

    with pytest.raises(ValueError, match="prefix"):
        state_swap.message_token_spans(tokenizer, messages, spec=spec)


# ------------------------------------------------------------------------ B6, both ways


def test_b6_puts_the_filename_in_the_prompt_exactly_once_inside_the_masked_span() -> None:
    """The invariant is counted in the rendered text, not inferred from the flags passed.

    Spec B6: observations unstripped so the listing is present, notes stripped so the filename
    cannot be read off a visible assistant message. Counting the filename is worth more than
    checking that ``strip_pending`` was called, because it is the property the arms depend on
    and the only one a reader can re-check from the artifact.
    """
    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")

    points, rejected = state_swap.select_points(_ledger_tasks(), tokenizer, spec=spec)

    assert points, f"no usable probe point; rejections: {rejected}"
    for point in points:
        prompt = point["persistent_prompt"]
        assert prompt.count(point["filename"]) == 1
        where = prompt.index(point["filename"])
        assert any(
            start <= where < end for start, end in point["hidden_character_spans"]
        ), "the filename entered outside the span arm A masks"
        assert point["b6"]["filename_occurrences"] == 1
        assert point["b6"]["inside_hidden_span"] is True


def test_b6_fails_in_both_directions_and_the_point_is_dropped() -> None:
    """Strip both and the filename never enters; strip neither and a visible note carries it.

    Both directions produce an arm A that looks like an answer and is not, so both are checked
    against the same real tasks rather than argued about.
    """
    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")
    tasks = _ledger_tasks()

    hidden_both, both_rejected = state_swap.select_points(
        tasks, tokenizer, spec=spec, hide_observations=True
    )
    assert hidden_both == []
    assert both_rejected["filename_absent"] > 0

    kept_notes, neither_rejected = state_swap.select_points(
        tasks, tokenizer, spec=spec, strip_notes=False
    )
    assert kept_notes == []
    assert neither_rejected["filename_outside_hidden_span"] > 0


def test_arm_c_reproduces_exp_001s_own_cohort_and_prompt() -> None:
    """Arm C is EXP-001's condition, so its text must be what EXP-001 scored, point for point."""
    from local_llm_lab.probes.jspace_sweep import select_cases

    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")
    tasks = _ledger_tasks()

    points, _rejected = state_swap.select_points(tasks, tokenizer, spec=spec)
    cases = select_cases(tasks, tokenizer, spec=spec)

    assert [point["task_id"] for point in points] == [case["task_id"] for case in cases]
    for point, case in zip(points, cases, strict=True):
        assert point["windowed_prompt"] == case["prompt"]
        assert (point["true"], point["false"]) == (case["true"], case["false"])
        assert (point["true_token"], point["false_token"]) == (
            case["true_token"],
            case["false_token"],
        )


def test_the_scored_chunk_is_the_generation_prompt_and_the_forced_note_stem() -> None:
    """Every arm is scored at one decision, mid-note; the chunking has to end exactly there."""
    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")

    point = state_swap.select_points(_ledger_tasks(), tokenizer, spec=spec)[0][0]

    ids = point["persistent_ids"]
    start, end = point["scored_chunk"]
    assert end == len(ids)
    assert point["prefill_chunks"][-1][1] == start
    assert point["prefill_chunks"][0][0] == 0
    # No gap and no overlap: every token of the prompt is prefilled or scored exactly once.
    assert [chunk[0] for chunk in point["prefill_chunks"][1:]] == [
        chunk[1] for chunk in point["prefill_chunks"][:-1]
    ]
    assert point["prefix"].endswith("-")
    assert point["persistent_prompt"].endswith(point["prefix"])


def test_hidden_spans_are_recorded_as_column_ranges_with_their_token_counts() -> None:
    """The Head checks coverage against the columns the mask indexed, not against an index."""
    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")

    point = state_swap.select_points(_ledger_tasks(), tokenizer, spec=spec)[0][0]

    assert point["hidden_spans"]
    for span, described in zip(point["hidden_spans"], point["hidden_span_records"], strict=True):
        start, end = span
        assert described["start"] == start and described["end"] == end
        assert described["tokens"] == end - start
        assert described["role"] == "tool"
        assert 0 <= start < end <= point["scored_chunk"][0]


# ------------------------------------------------------------------ the arms, on a fake
#
# The decoder here is a stand-in with two independently switchable routes: an *attention* route
# that reads only the columns the mask leaves visible, and a *recurrent* route that reads every
# token the cache was ever prefilled with, mask or no mask. That is the dissociation EXP-002
# measures, reduced to something a test can plant an answer in. The token sequences are
# hand-built so that the true candidate occurs **only inside the span arm A masks**: with a real
# conversation the background frequency of a digit would swamp the effect and the test would
# assert nothing.


class _SwapEntry:
    """A cache entry that keeps every token prefilled through it, and its own offset."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.offset = 0

    @property
    def state(self):
        return [list(self.tokens), self.offset]

    @state.setter
    def state(self, value) -> None:
        self.tokens = list(value[0])
        self.offset = int(value[1])


class _ForgetfulEntry(_SwapEntry):
    """A cache whose captured state comes back empty: the identity gate's failure case.

    The coarsest way a snapshot can fail to round-trip, and deliberately coarse. A stand-in
    that counts tokens saturates its own softmax on a real-length prompt, so losing a single
    token is below its resolution -- the subtle direction is the one checked against the real
    ``ArraysCache`` and ``KVCache``, which round-trip exactly. What this fake is for is the
    other half of the gate's contract: that a failure stops the run and still leaves its number
    in the artifact.
    """

    @property
    def state(self):
        return [[], 0]

    @state.setter
    def state(self, value) -> None:
        self.tokens = list(value[0])
        self.offset = int(value[1])


class _SwapView:
    """Counts tokens on two routes, so an arm can be given the answer on exactly one of them.

    ``GAIN`` scales the counts before the softmax and exists for one reason: a real probe-point
    prompt is thousands of tokens, so raw counts put the most frequent character hundreds of
    logits above everything else and every candidate probability underflows to exactly 0.0.
    The statistics would then be computed over a column of zeros and assert nothing. It is a
    fixed constant, so it scales every arm's log-odds identically and no comparison below
    depends on its value.
    """

    num_layers = 4
    vocab_size = 64
    GAIN = 0.01

    def __init__(self, *, recurrent: float = 1.0, attention: float = 1.0, entry=_SwapEntry):
        self.recurrent = recurrent
        self.attention = attention
        self.entry = entry
        self.calls: list[dict] = []

    def layer_kind(self, index):
        return "linear_attention" if (index + 1) % self.num_layers else "attention"

    def make_cache(self):
        return [self.entry() for _ in range(self.num_layers)]

    def cached_logits(self, ids, cache, *, hidden_spans=None, position=-1, record=None):
        values = [int(value) for value in mx.array(ids).reshape(-1).tolist()]
        entry = cache[0] if cache is not None else self.entry()
        offset = entry.offset
        entry.tokens.extend(values)
        entry.offset = len(entry.tokens)
        for other in (cache or [])[1:]:
            other.tokens.extend(values)
            other.offset = len(other.tokens)
        spans = () if hidden_spans is None else tuple(tuple(span) for span in hidden_spans)
        hidden = {column for start, end in spans for column in range(start, end)}
        logits = [0.0] * self.vocab_size
        for index, token in enumerate(entry.tokens):
            logits[token] += self.recurrent * self.GAIN
            if index not in hidden:
                logits[token] += self.attention * self.GAIN
        self.calls.append(
            {"tokens": len(values), "offset": offset, "hidden_spans": spans, "position": position}
        )
        if record is not None:
            record["attention_mask_route"] = (
                "model helper, unmodified"
                if hidden_spans is None
                else (
                    "row constructed at a single step"
                    if len(values) == 1
                    else "library array, hidden columns ANDed"
                )
            )
            record["hidden_spans"] = spans
            record["query_tokens"] = len(values)
            record["cache_offset"] = offset
            record["attention_mask_shape"] = (len(values), offset + len(values))
        return mx.array(logits, dtype=mx.float32)[None, :]


def _synthetic_point(task_id="point-0", *, true_token=7, false_token=9, filler=1):
    """A probe point whose true candidate occurs only inside the span arm A masks."""
    hidden = [true_token, true_token, true_token]
    ids = [filler, filler] + hidden + [false_token, filler] + [filler, filler]
    hidden_span = (2, 2 + len(hidden))
    return {
        "task_id": task_id,
        "difficulty": None,
        "filename": f"invoice-{task_id}.txt",
        "prefix": "Reading invoice-",
        "true": "7",
        "false": "9",
        "true_token": true_token,
        "false_token": false_token,
        "persistent_ids": ids,
        "windowed_ids": [filler, filler, false_token, filler, filler, filler],
        "windowed_chunks": ((0, 2), (2, 4)),
        "windowed_scored_chunk": (4, 6),
        "windowed_prompt": "windowed",
        "persistent_prompt": "persistent",
        "prefill_chunks": ((0, 2), (2, 5), (5, 7)),
        "scored_chunk": (7, len(ids)),
        "hidden_spans": (hidden_span,),
        "hidden_character_spans": ((0, 1),),
        "hidden_span_records": (
            {
                "start": hidden_span[0],
                "end": hidden_span[1],
                "tokens": hidden_span[1] - hidden_span[0],
                "message_index": 1,
                "role": "tool",
                "name": "list_files",
            },
        ),
        "b6": {
            "filename_occurrences": 1,
            "inside_hidden_span": True,
            "suffix_occurrences_outside_hidden_spans": 0,
        },
    }


def test_the_mask_is_carried_on_every_forward_after_the_span_not_only_at_the_decision() -> None:
    """Masking only the scored step leaves the answer one attention hop away.

    Measured on a real two-attention-block toy decoder before this was written: masking only at
    the decision differs from masking from the span onward by 0.23 in logit space, because the
    positions between the span and the decision attended to it during their own forward and the
    scored query then attends to *them*. A fixture whose only attention block is the last one
    cannot show this, which is why the check is on the sequence of calls and not on a number.
    """
    view = _SwapView()
    point = _synthetic_point()

    state_swap.prefill(
        view,
        point["persistent_ids"],
        point["prefill_chunks"],
        hidden_spans=point["hidden_spans"],
    )

    spans = [call["hidden_spans"] for call in view.calls]
    # Chunk 0 and chunk 1 begin before the span ends, so they carry nothing; chunk 2 begins
    # after it and carries the span, and so must every later forward.
    assert spans == [(), (), point["hidden_spans"]]


def test_arm_b_forces_the_explicit_array_so_a_and_b_differ_by_the_mask_alone() -> None:
    """Arm B hides nothing but takes the array route, not the model's default sentinel route."""
    view = _SwapView()
    point = _synthetic_point()

    state_swap.prefill(view, point["persistent_ids"], point["prefill_chunks"], hidden_spans=())

    assert [call["hidden_spans"] for call in view.calls] == [(), (), ()]
    assert all(call["hidden_spans"] is not None for call in view.calls)


def test_arm_a_reads_the_hidden_candidate_only_through_the_recurrent_route() -> None:
    """The instrument must report a recurrent channel where there is one, and none where not.

    Compared as odds -- P(true) over P(already-read) -- and not as bare probabilities. The two
    worlds below are two different decoders, and a softmax renormalises over the whole
    vocabulary, so P(true) alone can rise between them while the preference the experiment
    measures falls. EXP-001 §3.3's decomposition is a within-context comparison for the same
    reason. That correction came out of the first draft of this test, which asserted on the
    probabilities and failed for exactly this.
    """

    def odds(values):
        return values["true"] / values["already_read"]

    point = _synthetic_point()

    # A, B and C only: the control needs a second point, and this test is about the two routes.
    three = ("A", "B", "C")
    carried = state_swap.run_arms(_SwapView(recurrent=1.0), [point], arms=three)[0][
        "probabilities"
    ]
    severed = state_swap.run_arms(_SwapView(recurrent=0.0), [point], arms=three)[0][
        "probabilities"
    ]

    # With the recurrent route carrying, arm A rises above EXP-001's condition and arm B, whose
    # attention can still read the listing, stays above arm A. That is §2's first outcome row.
    assert odds(carried["B"]) > odds(carried["A"]) > odds(carried["C"])
    # Sever it and arm A is *indistinguishable from* arm C while arm B still rises: §2's second
    # outcome row, the null the experiment has to be able to report.
    assert odds(severed["A"]) == pytest.approx(odds(severed["C"]))
    assert odds(severed["B"]) > odds(severed["A"])
    assert odds(severed["A"]) < odds(carried["A"])


def test_the_control_scores_the_decision_against_another_points_persistent_state() -> None:
    """Foreign state separates retrieval from generic perturbation, so it must be foreign."""
    points = [_synthetic_point("point-0"), _synthetic_point("point-1", true_token=11)]

    results = state_swap.run_arms(_SwapView(), points)

    assert [result["foreign_task_id"] for result in results] == ["point-1", "point-0"]
    # Point 0's true candidate is absent from point 1's history, so the foreign state cannot
    # deliver it: the control must sit below arm A, which is the whole point of running it.
    assert results[0]["probabilities"]["control"]["true"] < results[0]["probabilities"]["A"]["true"]


def test_a_single_point_run_refuses_the_control_rather_than_scoring_it_on_itself() -> None:
    """A rotation that closes on one point makes the control arm A, and it would look like one.

    The control exists to separate retrieval from generic perturbation. Run over the point's own
    cache it agrees with arm A by construction, so the fourth outcome row -- "arm A and the
    control rise together" -- would fire on an artefact of the pairing.
    """
    with pytest.raises(ValueError, match="second probe point"):
        state_swap.run_arms(_SwapView(), [_synthetic_point()])


def test_every_arm_records_the_mask_route_and_the_columns_it_used() -> None:
    """An arm-A number a reader cannot attribute to a code path is not a readable number."""
    points = [_synthetic_point("point-0"), _synthetic_point("point-1")]
    results = state_swap.run_arms(_SwapView(), points)

    scored = results[0]["scored"]
    assert set(scored) == set(state_swap.ARMS)
    assert scored["A"]["hidden_spans"] == [[2, 5]]
    assert scored["B"]["hidden_spans"] == []
    # Arm C now runs the same chunked, cached, explicit-array path as B, so the four arms
    # differ by mask and cache content alone. EXP-001's own route survives beside it.
    assert scored["C"]["attention_mask_route"] == "library array, hidden columns ANDed"
    assert results[0]["c_single_forward"]["scored"]["attention_mask_route"] == (
        "model helper, unmodified"
    )
    for arm in state_swap.ARMS:
        assert scored[arm]["attention_mask_route"]
        assert scored[arm]["scored_position"] == -1
        assert "cache_offset" in scored[arm]


def test_run_arms_reports_one_progress_line_per_probe_point() -> None:
    """R26 (g): the run is never silent for longer than one point."""
    seen: list[tuple] = []
    points = [_synthetic_point("point-0"), _synthetic_point("point-1")]

    state_swap.run_arms(
        _SwapView(), points, progress=lambda step, total, label, **fields: seen.append(
            (step, total, label, fields.get("task_id"))
        )
    )

    assert seen == [(1, 2, "probe point", "point-0"), (2, 2, "probe point", "point-1")]


# ----------------------------------------------------------------- the identity gate


def test_identity_gate_passes_on_the_real_cache_classes_and_records_the_observed_maximum(
    cpu_stream,
) -> None:
    """R31: whether captured state round-trips is a fact about ``ArraysCache`` and ``KVCache``.

    A stand-in cache would round-trip because it was written to; the real classes have to be
    asked. ``KVCache.state``'s setter restores ``offset`` from the keys it is given and
    ``ArraysCache`` carries no offset at all, and neither is visible from a fake.
    """
    from local_llm_lab.arch import ArchitectureView

    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    point = _synthetic_point(true_token=7, false_token=9, filler=1)

    gate = state_swap.identity_gate(view, point)

    assert gate["passed"] is True
    assert gate["observed_max_abs_probability_difference"] <= state_swap.IDENTITY_TOLERANCE
    assert gate["tolerance"] == state_swap.IDENTITY_TOLERANCE
    assert gate["cache_class_by_block_kind"] == {
        "linear_attention": "ArraysCache",
        "attention": "KVCache",
    }
    assert gate["captured_entries"] == view.num_layers
    assert gate["task_id"] == point["task_id"]


def test_identity_gate_records_the_observed_maximum_when_it_fails() -> None:
    """A failing threshold has to be recalibrated from evidence, so the number is kept."""
    view = _SwapView(entry=_ForgetfulEntry)

    gate = state_swap.identity_gate(view, _synthetic_point())

    assert gate["passed"] is False
    assert gate["observed_max_abs_probability_difference"] > state_swap.IDENTITY_TOLERANCE
    assert "observed_max_abs_probability_difference" in gate


# --------------------------------------------------------------- the statistic and the artifact


def test_arm_c_keeps_exp_001s_tie_convention_and_the_paired_rows_drop_ties() -> None:
    """Arm C's cell has to be comparable with EXP-001's row, and a tie is not a paired win."""
    results = [
        {
            "probabilities": {
                "A": {"true": 0.2, "already_read": 0.1},
                "B": {"true": 0.3, "already_read": 0.1},
                "C": {"true": 0.1, "already_read": 0.1},
                "control": {"true": 0.1, "already_read": 0.1},
            }
        },
        {
            "probabilities": {
                "A": {"true": 0.4, "already_read": 0.1},
                "B": {"true": 0.5, "already_read": 0.1},
                "C": {"true": 0.05, "already_read": 0.1},
                "control": {"true": 0.05, "already_read": 0.1},
            }
        },
    ]

    stats = state_swap.aggregate(results)

    # EXP-001's convention: point one's arm C is an exact tie and counts as a non-win.
    row = stats["within_arm"]["C"]
    assert (row["wins"], row["losses"], row["ties"], row["n"]) == (0, 1, 1, 2)
    assert row["median_p_true"] == pytest.approx(0.075)
    assert row["median_p_already_read"] == pytest.approx(0.1)
    # Every row's denominator is reconstructable from the counts the artifact states, so a
    # reader never has to know which convention applied where. The two conventions differ and
    # the counts are what make the asymmetry visible rather than something to be inferred.
    for family in (stats["within_arm"], *stats["paired_vs_C"].values()):
        for value in family.values():
            assert value["wins"] + value["losses"] + value["ties"] == value["points"]
            assert value["n_rule"]
    assert row["n"] == row["points"]
    assert stats["paired_vs_C"]["true"]["A"]["n"] == (
        stats["paired_vs_C"]["true"]["A"]["wins"] + stats["paired_vs_C"]["true"]["A"]["losses"]
    )
    # The control equals arm C at both points on P(true), so every pair is a tie and n is zero.
    control = stats["paired_vs_C"]["true"]["control"]
    assert (control["wins"], control["losses"], control["ties"], control["n"]) == (0, 0, 2, 0)
    assert stats["paired_vs_C"]["true"]["A"]["n"] == 2
    assert "p_holm" in stats["paired_vs_C"]["true"]["A"]


def _write_preflight(tmp_path, spec):
    """The preflight artifact in the shape ``run_preflight`` really writes.

    Nested under ``residual_equivalence``, which is the shape the audit found was not being
    read: the top level was the only thing consulted and ``run_preflight`` never wrote a key
    there, so every probe artifact recorded ``null`` for a number R18a requires in all of them.
    Writing only the nested shape here is what makes the assertion below mean something.
    """
    root = tmp_path / "preflight"
    root.mkdir(exist_ok=True)
    (root / f"{spec.name}.json").write_text(
        json.dumps(
            {
                "residual_equivalence": {
                    "fp32_manual_vs_native": {"frobenius_relative": 0.0285, "max_abs": 0.02}
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def _run_state_swap_cli(monkeypatch, tmp_path, extra=(), *, view=None, tokenizer=None):
    from local_llm_lab import models
    from local_llm_lab.pipeline import evaluate, preflight
    from local_llm_lab.pipeline.tasks import FAMILIES
    from local_llm_lab.probes import guard, policies
    from local_llm_lab.probes import state_swap as module

    spec = models.load_model_spec("qwen35-4b")
    monkeypatch.setattr(preflight, "_OUTPUT_DIRECTORY", _write_preflight(tmp_path, spec))
    monkeypatch.setattr(preflight, "require_preflight", lambda *_a, **_k: None)
    monkeypatch.setattr(guard, "require_idle_gpu", lambda *_a: None)
    monkeypatch.setattr(policies, "resolve_policy", lambda *_a: None)
    tokenizer = _SwapTokenizer() if tokenizer is None else tokenizer
    loaded = _SwapView() if view is None else view
    monkeypatch.setattr(
        evaluate, "load_policy", lambda *_a, **_k: (object(), tokenizer, loaded, None)
    )
    output = tmp_path / "swap"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-v2-state-swap",
            "--model",
            "qwen35-4b",
            "--count",
            str(len(FAMILIES) * 10),
            "--skip-preflight-check",
            "--output",
            str(output),
            *extra,
        ],
    )
    module.main()
    payload = json.loads((output / "state_swap.json").read_text(encoding="utf-8"))
    return payload, output, tokenizer, spec


def test_the_cli_writes_the_identity_gate_before_any_arm(monkeypatch, tmp_path) -> None:
    """§3.4: the gate is recorded first, and no arm is readable without it."""
    payload, output, _tokenizer, _spec = _run_state_swap_cli(monkeypatch, tmp_path)

    keys = list(payload)
    assert keys.index("identity_gate") < keys.index("results")
    assert payload["identity_gate"]["passed"] is True
    assert "observed_max_abs_probability_difference" in payload["identity_gate"]
    assert (output / "state_swap.md").exists()
    assert (output / "provenance.json").exists()


def test_the_run_stops_when_the_identity_gate_fails_but_keeps_its_number(
    monkeypatch, tmp_path
) -> None:
    """A failed gate has to leave evidence: the threshold is recalibrated from it, not argued."""
    with pytest.raises(SystemExit):
        _run_state_swap_cli(monkeypatch, tmp_path, view=_SwapView(entry=_ForgetfulEntry))

    payload = json.loads((tmp_path / "swap" / "state_swap.json").read_text(encoding="utf-8"))
    assert payload["identity_gate"]["passed"] is False
    assert payload["identity_gate"]["observed_max_abs_probability_difference"] > 0
    assert payload["results"] == {}
    assert payload["per_point"] == []


def test_the_artifact_records_r34_conformance_and_r35_comparability(monkeypatch, tmp_path) -> None:
    """R34/R35: which cache, which spans, how located, where scored -- and the precision block."""
    payload, _output, _tokenizer, spec = _run_state_swap_cli(monkeypatch, tmp_path)

    conformance = payload["conformance"]
    assert conformance["cache_class_by_block_kind"]
    assert conformance["hidden_span_location"]
    assert conformance["hidden_span_masking"]
    assert conformance["b6_construction"]
    assert set(conformance["arm_scoring"]) == set(state_swap.ARMS)
    for arm in state_swap.ARMS:
        assert conformance["arm_scoring"][arm]["scored_position"] == -1
        assert conformance["arm_scoring"][arm]["attention_mask_route"]
    assert conformance["window"]["keep_last"] == DEFAULT_KEEP_LAST

    comparability = payload["comparability"]
    assert comparability["ruling"] == "R35"
    # The reader the audit found defective now reads residual_equivalence.fp32_manual_vs_native,
    # and the fixture writes only that shape, so a null here would be the old defect returning.
    assert comparability["fp32_manual_vs_native"] == {
        "frobenius_relative": 0.0285,
        "max_abs": 0.02,
    }
    assert comparability["prompt_rendering"]["template_kwargs"] == dict(spec.chat.template_kwargs)
    assert comparability["prompt_rendering"]["persistent_arms"]["rewindowed"] is False


def test_the_artifact_records_the_resolved_columns_the_mask_used_per_point(
    monkeypatch, tmp_path
) -> None:
    """The Head checks coverage against what ran, so the spans in the artifact are the mask's."""
    payload, _output, _tokenizer, _spec = _run_state_swap_cli(monkeypatch, tmp_path)

    assert payload["per_point"]
    for record in payload["per_point"]:
        spans = record["hidden_spans"]
        assert spans and all(span["tokens"] == span["end"] - span["start"] for span in spans)
        assert record["scored"]["A"]["hidden_spans"] == [
            [span["start"], span["end"]] for span in spans
        ]
        assert record["scored"]["B"]["hidden_spans"] == []
        assert record["b6"]["filename_occurrences"] == 1
        assert record["b6"]["inside_hidden_span"] is True


def test_the_cli_logs_the_exp_001_identity_block_and_one_line_per_point(
    monkeypatch, tmp_path
) -> None:
    """R26: ``run.log`` alone must say what ran, and never go silent for longer than a point."""
    payload, output, _tokenizer, _spec = _run_state_swap_cli(monkeypatch, tmp_path)

    log = (output / "run.log").read_text(encoding="utf-8")
    for field in ("model=", "hf_id=", "split=", "count=", "data_seed=", "git_commit="):
        assert field in log
    assert log.count("[probe point ") == payload["points"]
    assert "identity gate" in log


def test_the_arms_flag_selects_arms_and_refuses_a_family_without_arm_c(
    monkeypatch, tmp_path
) -> None:
    """Every reported statistic is paired against arm C, so a run without it reports nothing."""
    payload, _output, _tokenizer, _spec = _run_state_swap_cli(
        monkeypatch, tmp_path, extra=("--arms", "A,C")
    )
    assert payload["arms"] == ["A", "C"]
    assert set(payload["results"]["within_arm"]) == {"A", "C"}

    with pytest.raises(SystemExit):
        _run_state_swap_cli(monkeypatch, tmp_path, extra=("--arms", "A,B"))


def test_masking_only_the_decision_leaves_the_answer_one_attention_hop_away(cpu_stream) -> None:
    """The measurement behind ``prefill``'s design, against real blocks rather than an argument.

    The work order offers two forms for the scored step and treats them as alternatives. They
    are alternatives about the mask's *shape*; they say nothing about which forwards carry it,
    and that is a separate choice with a separate failure. Masking only at the decision leaves
    every position between the hidden observation and the decision having attended to it during
    its own forward, and the scored query then attends to those positions. So arm A would rise
    through attention with the span nominally masked -- the same failure B6 guards against, by a
    different route.

    Eight blocks, so two are attention and the second reads what the first wrote. On the
    four-block fixture S1 was accepted against, this difference is exactly 0.0, because its one
    attention block is the last one and there is no earlier attention output to propagate. That
    is why this test does not reuse it.
    """
    from local_llm_lab.arch import ArchitectureView

    view = ArchitectureView.from_model(make_tiny_hybrid_model(8))
    assert [view.layer_kind(index) for index in range(view.num_layers)].count("attention") == 2
    point = _synthetic_point(true_token=7, false_token=9, filler=1)
    ids, chunks = point["persistent_ids"], point["prefill_chunks"]
    spans = point["hidden_spans"]
    start, end = point["scored_chunk"]

    carried = view.cached_logits(
        ids[start:end], state_swap.prefill(view, ids, chunks, hidden_spans=spans),
        hidden_spans=spans,
    )
    late = view.cached_logits(
        ids[start:end], state_swap.prefill(view, ids, chunks, hidden_spans=()),
        hidden_spans=spans,
    )
    open_arm = view.cached_logits(
        ids[start:end], state_swap.prefill(view, ids, chunks, hidden_spans=()), hidden_spans=()
    )

    assert float(mx.max(mx.abs(carried - late)).item()) > 0.0
    # Both really do mask something, so the difference above is between two masked forwards and
    # not between a masked one and an unmasked one.
    assert float(mx.max(mx.abs(open_arm - late)).item()) > 0.0
    assert float(mx.max(mx.abs(open_arm - carried)).item()) > 0.0


# ------------------------------------------- the Chief's requirement: arm C on the shared path


def test_arm_c_runs_the_same_chunked_path_over_a_cache_that_holds_only_its_own_text() -> None:
    """A fresh cache, chunked like the persistent arms, so only mask and content differ.

    "Fresh" is the load-bearing word: arm C is still EXP-001's condition because no state
    reaches it from anywhere else. What changes is the numerical path, which now matches the
    other three instead of being a single uncached forward.
    """
    view = _SwapView()
    points = [_synthetic_point("point-0"), _synthetic_point("point-1")]

    state_swap.run_arms(view, points, arms=("B", "C"))

    windowed = points[0]["windowed_ids"]
    chunk_calls = [call for call in view.calls if call["tokens"] == 2]
    # Arm C's three forwards -- two prefill chunks and the scored chunk -- all carry the
    # explicit array, and the first begins at offset zero because the cache is new.
    assert chunk_calls[0]["offset"] == 0
    assert all(call["hidden_spans"] == () for call in chunk_calls)
    assert len(windowed) == 6


def test_the_single_forward_arm_c_is_kept_as_the_tie_to_exp_001() -> None:
    """The comparisons use chunked C; EXP-001's own construction is retained beside it."""
    points = [_synthetic_point("point-0"), _synthetic_point("point-1")]

    results = state_swap.run_arms(_SwapView(), points)
    stats = state_swap.aggregate(results)

    for result in results:
        assert set(result["c_single_forward"]["probabilities"]) == {"true", "already_read"}
    row = stats["arm_c_single_forward"]
    assert row["n"] == len(results)
    assert "max_abs_difference_vs_chunked" in row
    assert "median_abs_difference_vs_chunked" in row
    # The paired families are computed against chunked C, never against this row.
    assert set(stats["paired_vs_C"]["true"]) == {"A", "B", "control"}


def test_chunking_deviation_is_measured_and_the_residual_is_reported(cpu_stream) -> None:
    """The Chief's measurement, made permanent: what the shared chunked path actually cancels.

    Arm C's text is shorter than the persistent arms' and chunks at different boundaries, so
    the kernel deviation need not cancel exactly. Measured on real blocks over toy weights,
    on an eight-block hybrid so two attention blocks are in play:

    * each arm's own chunked-versus-single deviation, in the candidate probabilities every
      statistic is computed from -- 1.5e-08 to 2.6e-08 across three chunkings;
    * the residual left on the paired difference, 4.7e-09 to 1.1e-08, smaller than either
      arm's own deviation in every case, so the cancellation is real and partial.

    The residual is what rides on a comparison, and it sits about ninety times below the
    identity gate's own 1e-6 tolerance.
    """
    from local_llm_lab.arch import ArchitectureView

    view = ArchitectureView.from_model(make_tiny_hybrid_model(8))
    point = _real_shaped_point()

    measured = state_swap.chunking_deviation(view, point)

    assert measured["quantity"]
    assert measured["residual_on_the_paired_difference"] < state_swap.IDENTITY_TOLERANCE
    assert measured["residual_on_the_paired_difference"] <= max(
        measured["persistent_chunked_vs_single"], measured["arm_c_chunked_vs_single"]
    )
    assert measured["cancels"] is True


def _real_shaped_point():
    """A point whose two texts differ in length and chunk at different boundaries.

    That difference is the whole of the Chief's concern, so the fixture has to have it: a
    windowed text the same length as the persistent one would make the cancellation look exact
    for a reason the real run does not have.
    """
    persistent = [3, 1, 4, 6, 5, 2, 7, 8, 1, 9, 2, 3, 5, 7, 4, 6, 2, 8, 1, 3]
    windowed = [3, 1, 4, 6, 5, 9, 2, 3, 5, 7, 4, 6, 2, 8, 1, 3]
    point = _synthetic_point()
    point.update(
        {
            "persistent_ids": persistent,
            "prefill_chunks": ((0, 4), (4, 11), (11, 15)),
            "scored_chunk": (15, 20),
            "hidden_spans": ((4, 11),),
            "windowed_ids": windowed,
            "windowed_chunks": ((0, 4), (4, 7), (7, 11)),
            "windowed_scored_chunk": (11, 16),
        }
    )
    return point


def test_masking_a_hidden_observation_while_it_arrives_is_a_different_experiment(
    cpu_stream,
) -> None:
    """The fourth way arm A could be quietly wrong, pinned so a later edit cannot merge it.

    ``_spans_behind`` masks a span only once it lies *wholly behind* the chunk. The chunk that
    contains the hidden observation therefore carries nothing, and that is not an off-by-one:
    those queries are the model reading the observation as it arrives, which is how the
    recurrent state comes to hold the filename at all. Mask them and the state never acquires
    what arm A exists to test for, and the arm reports a null it could not have avoided --
    indistinguishable, from the outside, from a real one.

    So the two rules are shown to give different numbers on real blocks. A future simplification
    that collapses them fails here.
    """
    from local_llm_lab.arch import ArchitectureView

    view = ArchitectureView.from_model(make_tiny_hybrid_model(8))
    point = _real_shaped_point()
    ids, chunks, spans = point["persistent_ids"], point["prefill_chunks"], point["hidden_spans"]
    start, end = point["scored_chunk"]

    # The rule: nothing is carried while the observation is still arriving.
    assert state_swap._spans_behind(spans, chunks[1][0]) == ()
    assert state_swap._spans_behind(spans, chunks[2][0]) == spans

    correct = view.cached_logits(
        ids[start:end], state_swap.prefill(view, ids, chunks, hidden_spans=spans),
        hidden_spans=spans,
    )
    # The collapsed rule: mask the span from its own chunk onward, blinding the observation to
    # itself. It runs cleanly and gives a different answer.
    blinded_cache = view.make_cache()
    for chunk_start, chunk_end in chunks:
        carried = tuple(span for span in spans if span[0] <= chunk_start)
        view.cached_logits(ids[chunk_start:chunk_end], blinded_cache, hidden_spans=carried)
    blinded = view.cached_logits(ids[start:end], blinded_cache, hidden_spans=spans)

    assert float(mx.max(mx.abs(correct - blinded)).item()) > 0.0


def test_the_identity_gate_runs_on_several_points_and_gates_on_the_worst(cpu_stream) -> None:
    """One point can pass by luck, and a gate that can pass by luck is not a gate."""
    from local_llm_lab.arch import ArchitectureView

    view = ArchitectureView.from_model(make_tiny_hybrid_model())
    points = [_synthetic_point("point-0"), _synthetic_point("point-1"), _synthetic_point("p-2")]

    family = state_swap.identity_gate_family(view, points)

    assert family["points_measured"] == state_swap.IDENTITY_GATE_POINTS
    assert len(family["per_point"]) == len(points)
    assert family["observed_max_abs_probability_difference"] == max(
        gate["observed_max_abs_probability_difference"] for gate in family["per_point"]
    )
    assert family["passed"] is True
    assert family["gated_on"]


def test_the_identity_gate_family_fails_when_any_measured_point_fails() -> None:
    """The maximum governs, so a good first point cannot carry a cache that does not round-trip."""

    class _OneGoodPoint(_SwapView):
        """Round-trips for the first point measured and forgets everything after it."""

        def __init__(self) -> None:
            super().__init__()
            self.seen = 0

        def make_cache(self):
            self.seen += 1
            entry = _SwapEntry if self.seen <= 2 else _ForgetfulEntry
            return [entry() for _ in range(self.num_layers)]

    points = [_synthetic_point("point-0"), _synthetic_point("point-1"), _synthetic_point("p-2")]

    family = state_swap.identity_gate_family(_OneGoodPoint(), points)

    assert family["per_point"][0]["passed"] is True
    assert family["passed"] is False


def test_suffix_occurrences_outside_matches_only_a_readable_suffix() -> None:
    """The standalone rule, on constructed text, at each boundary that decides a rejection."""
    spans = ((0, 10),)
    text = "0123456789 note: approved 751 and context 897751 plus invoice-2-751.txt end."

    standalone = state_swap.suffix_occurrences_outside(text, "751", spans)
    naive = state_swap.suffix_occurrences_outside(text, "751", spans, standalone=False)

    # One readable occurrence: the amount in the note. The run of digits inside 897751 is a
    # different value, and the hyphenated filename is B6's business, not this rule's.
    assert len(standalone) == 1
    assert text[standalone[0] : standalone[0] + 3] == "751"
    assert text[standalone[0] - 1] == " "
    assert len(naive) == 3
    # Anything inside a masked span is excluded from both, since that is the route arm A hides.
    assert state_swap.suffix_occurrences_outside("751xxxxxxx", "751", ((0, 3),)) == []


def test_the_standalone_suffix_rule_costs_this_cohort_nothing_and_says_why() -> None:
    """Measured on the real cohort: 0 of 42 either way, and the rule is redundant here.

    The persistent prompt and arm C's windowed prompt differ *only inside the hidden spans*, so
    EXP-001's leakage guard -- which runs first, on the windowed text -- already forces the
    outside-span count to zero. Any standalone occurrence would be caught as ``leaked`` before
    this rule is reached. It is kept because it fails closed if the window, the probe step or
    the generator moves, and the artifact says so rather than implying it is catching something.
    """
    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")

    points, rejected = state_swap.select_points(_ledger_tasks(720), tokenizer, spec=spec)
    comparison = state_swap.naive_rule_comparison(points, rejected)

    assert len(points) == 42  # EXP-001's own cohort size, and the spec's "expect ~42"
    assert rejected.get("suffix_outside_hidden_span", 0) == 0
    assert rejected["leaked"] > 0  # the earlier guard is what makes this rule redundant
    assert comparison["standalone_rule_rejected"] == 0
    assert comparison["naive_any_occurrence_would_reject"] == 0
    for point in points:
        assert point["b6"]["standalone_suffix_occurrences_outside_hidden_spans"] == 0
        assert point["true"] not in point["windowed_prompt"]


def test_the_artifact_records_the_forward_schedule_so_the_masking_is_auditable(
    monkeypatch, tmp_path
) -> None:
    """A reader must be able to check which chunks carried the mask without reading the code."""
    payload, _output, _tokenizer, _spec = _run_state_swap_cli(monkeypatch, tmp_path)

    for record in payload["per_point"]:
        schedule = record["forward_schedule"]
        chunks = schedule["persistent_chunks"]
        assert chunks[-1]["scored"] is True
        assert all(chunk["tokens"] == chunk["end"] - chunk["start"] for chunk in chunks)
        # No gap and no overlap across the whole forward sequence.
        assert [chunk["start"] for chunk in chunks[1:]] == [
            chunk["end"] for chunk in chunks[:-1]
        ]
        hidden = [[span["start"], span["end"]] for span in record["hidden_spans"]]
        carrying = [chunk for chunk in chunks if chunk["carried_spans"]]
        assert carrying and all(chunk["carried_spans"] == hidden for chunk in carrying)
        # The chunk that contains the hidden observation carries nothing: the observation is
        # read as it arrives, and only later forwards are blinded to it. This is the visible
        # signature of that rule, and it is what a reviewer checks from outside the code.
        containing = [
            chunk
            for chunk in chunks
            if any(chunk["start"] <= span[0] < chunk["end"] for span in hidden)
        ]
        assert containing and all(chunk["carried_spans"] == [] for chunk in containing)
        # Every forward at or after the span's end carries it; none before does.
        for chunk in chunks:
            expected = [span for span in hidden if span[1] <= chunk["start"]]
            assert chunk["carried_spans"] == expected
        assert schedule["arm_c_chunks"][-1]["scored"] is True
        assert all(chunk["carried_spans"] == [] for chunk in schedule["arm_c_chunks"])


def test_the_artifact_records_the_gate_family_and_the_chunking_measurement(
    monkeypatch, tmp_path
) -> None:
    """Both are numbers the reading depends on, so both are stated rather than described."""
    payload, _output, _tokenizer, _spec = _run_state_swap_cli(monkeypatch, tmp_path)

    gate = payload["identity_gate"]
    assert gate["points_measured"] == state_swap.IDENTITY_GATE_POINTS
    assert gate["observed_max_abs_probability_difference"] == max(
        entry["observed_max_abs_probability_difference"] for entry in gate["per_point"]
    )

    chunking = payload["conformance"]["chunked_prefill_deviation"]
    for key in (
        "persistent_chunked_vs_single",
        "arm_c_chunked_vs_single",
        "residual_on_the_paired_difference",
        "cancels",
        "quantity",
    ):
        assert key in chunking
    suffix_rule = payload["conformance"]["suffix_rule"]
    assert suffix_rule["standalone_rule_rejected"] == 0
    assert suffix_rule["naive_any_occurrence_would_reject"] == 0


def test_the_exp001_artifact_flag_states_the_reproduction_instead_of_leaving_it_to_the_eye(
    monkeypatch, tmp_path
) -> None:
    """Arm C's single forward is compared against EXP-001's per-case model output, by task id."""
    payload, _output, _tokenizer, _spec = _run_state_swap_cli(monkeypatch, tmp_path)

    # An EXP-001-shaped artifact carrying this run's own single-forward numbers, so a correct
    # comparison must report a maximum difference of zero over every point.
    earlier = tmp_path / "exp001.json"
    earlier.write_text(
        json.dumps(
            {
                "per_case": [
                    {
                        "task_id": record["task_id"],
                        "matched": {
                            "model_output": {
                                "true": record["c_single_forward"]["probabilities"]["true"],
                                "false": record["c_single_forward"]["probabilities"][
                                    "already_read"
                                ],
                            }
                        },
                    }
                    for record in payload["per_point"]
                ]
                # One task the older artifact never carried, so "matched" is a real count.
                + [{"task_id": "absent-from-this-run", "matched": {"model_output": {}}}]
            }
        ),
        encoding="utf-8",
    )

    compared, _out, _tok, _spec = _run_state_swap_cli(
        monkeypatch, tmp_path, extra=("--exp001-artifact", str(earlier))
    )
    reproduction = compared["results"]["exp001_reproduction"]

    assert reproduction["points_matched_by_task_id"] == compared["points"]
    assert reproduction["max_abs_probability_difference"] == 0.0


def test_a_missing_exp001_artifact_does_not_cost_the_run_its_data(monkeypatch, tmp_path) -> None:
    """The comparison is a convenience; losing it must not lose four arms of measurement."""
    payload, _output, _tokenizer, _spec = _run_state_swap_cli(
        monkeypatch, tmp_path, extra=("--exp001-artifact", str(tmp_path / "nope.json"))
    )

    assert "error" in payload["results"]["exp001_reproduction"]
    assert payload["results"]["within_arm"]
    assert payload["per_point"]
