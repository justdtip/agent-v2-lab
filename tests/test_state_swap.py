"""EXP-002 S3: the four arms, the identity gate and the artifact.

Fakes and tiny real models only (R10): no checkpoint, nothing on the GPU. Three seams are driven
for real because a stand-in cannot show what they do (R31):

* ``pipeline.data.build_rows``, ``pipeline.protocol.window_messages`` and
  ``pipeline.protocol.build_prompt`` over real generated tasks, because the B6 invariant is a
  claim about text those functions produce and a hand-written conversation could be made to
  satisfy it by construction;
* ``mlx_lm``'s own ``ArraysCache`` and ``KVCache`` through a toy Qwen3.5 for the identity gate,
  because whether captured cache state round-trips -- and whether re-injecting it restores the
  attention offset -- is a fact about those classes and not about a shape a fake could be given;
* **the real 4B chat template**, which the earlier revision of this file explicitly excluded and
  which that exclusion cost a run. Every test here rendered through a hand-written stand-in
  until EXP-002 died three seconds in on ``[system]`` alone -- a shape the stand-in accepted and
  the real Qwen3.5 template refuses -- with all 39 of them green. A reader tested against a
  stand-in passes exactly when the stand-in differs from the writer in the way its author
  assumed it would not (R38). A tokenizer is not a checkpoint: it loads from the project cache
  in seconds, with no weights and no device, so R10 does not reach it. The ``real_tokenizer``
  fixture loads it and skips **loudly** when it is not cached, and
  ``test_the_real_tokenizer_coverage_cannot_be_skipped_into_nothing`` never skips, so deleting
  or quietly disabling that coverage turns the suite red rather than green-with-an-``s``.

The decoder itself is faked wherever the arms are being compared, and deliberately so: the fake
has an attention route and a recurrent route that can be switched independently, so a test can
put the hidden filename on exactly one of them and check the pipeline reports it there. A real
model would put the answer wherever it happens to be, which is the run's question, not the
instrument's.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import tomllib
import warnings
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


#: The roles Qwen3.5's template knows. Anything else is ``Unexpected message role.``.
_TEMPLATE_ROLES = frozenset({"system", "user", "assistant", "tool"})


class _SwapTokenizer:
    """One id per character, a prefix-additive render, and the real template's refusals.

    ``additive=False`` builds the one template shape the span locator cannot survive: a render
    of ``messages[:k]`` that is not a prefix of the render of ``messages[:k + 1]``. Real chat
    templates are prefix-additive over these conversations, but a template that merged or
    re-rendered earlier turns would silently move every boundary, so the locator has to check
    rather than assume.

    **What this stand-in refuses, and why the list is here.** The earlier version rendered every
    conversation it was handed, including ``[system]`` alone. The real Qwen3.5 template refuses
    that -- it scans the message list in reverse for a user query and raises ``No user query
    found in messages.`` when there is none -- and the whole suite passed while the run died
    three seconds in on the first prefix ``_rendered_boundaries`` renders. A stand-in that
    accepts more than the writer does passes exactly when it differs from the writer in the way
    its author assumed it would not (R38), so every refusal the real template makes on a shape
    reachable from ``build_prompt`` is reproduced here, with the same ``TemplateError`` type:

    * an empty conversation (``transformers`` raises ``ValueError`` before Jinja is entered);
    * no user turn whose content, **once trimmed**, is not a ``<tool_response>`` wrapper -- the
      template trims first, so a padded wrapper is still a wrapper and still not a query;
    * a system message anywhere but index 0, or a second one;
    * a role the template does not know;
    * content that is not a string. Narrower in the writer, which also accepts ``None`` and a
      list of content items; this stand-in refuses both. That is an over-refusal, which can only
      fail a test that should pass, and closing it means reimplementing the multimodal
      ``render_content`` macro. Nothing in this repo puts anything but a string in ``content``.

    Divergences that are **not** refusals stay unmodelled and are recorded rather than fixed:
    the real template ``|trim``s each content, injects ``<think>\\n...\\n</think>\\n\\n`` into
    every assistant turn after the last user query, and frames tool messages as ``user`` turns
    wrapped in ``<tool_response>`` with consecutive ones merged. Modelling those would make this
    a second implementation of the template rather than a stand-in, and the two shapes where
    they break prefix-additivity -- adjacent tool messages, and a second user turn later -- are
    unreachable through ``_rendered_boundaries``: every producer appends assistant and tool in
    pairs, and ``window_messages``, ``strip_pending`` and ``_strip_messages`` all preserve count
    and role. The real-tokenizer tests below are what covers the rendered text itself.

    **The ``</think>`` handling, stated precisely, because a loose statement of it points a
    follow-up at the wrong file.** Template lines 94-96 split an assistant content on
    ``</think>``; lines 100-104 then choose what to do with the halves, and the choice is what
    matters. For an assistant *after* the last user query the pre-``</think>`` half is
    **relocated** into the emitted reasoning block -- moved, not lost. Only the ``else`` limb,
    taken for an assistant at or before the last user query, emits the remainder alone and
    **discards** everything before the tag. So the discard needs a conversation with an
    assistant turn followed by a later user turn, and the agent loop never builds one: the
    ``assistant_message`` appends in ``pipeline.runner.run_task`` and ``pipeline.branch`` sit in
    conversations with exactly one user turn, at index 1, so every assistant they append takes
    the relocating limb. The one producer of the discard-eligible shape in this repo is
    ``chat_replay.make_chat_prompts``, whose ``follow_up`` branch yields ``[user, assistant,
    user]`` -- and those are the ``pre-expansion-policy-replay`` rows ``render_rows``
    allow-lists. Nothing is lost there today, because that middle assistant is a fixed literal
    with no ``</think>`` in it; a follow-up that wants to test the mechanism has to go there,
    and there is no point looking at the agent loop.
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

    @staticmethod
    def _refuse(messages) -> None:
        """Raise exactly where the real Qwen3.5 template raises, on string content."""
        from jinja2.exceptions import TemplateError

        if not messages:
            raise ValueError("Cannot apply chat template to an empty conversation.")
        for index, message in enumerate(messages):
            role = message.get("role")
            if role not in _TEMPLATE_ROLES:
                raise TemplateError("Unexpected message role.")
            if role == "system" and index:
                raise TemplateError("System message must be at the beginning.")
            if not isinstance(message.get("content"), str):
                raise TemplateError("Unexpected content type.")
        if not any(
            # ``|trim`` before the wrapper test, as the template does -- its reverse scan reads
            # ``render_content(message.content, false)|trim`` and only then asks whether the
            # result opens and closes with the wrapper. Testing the untrimmed string made this
            # stand-in accept a padded ``<tool_response>`` the writer refuses: it read the
            # padding as ordinary user text and counted a query the template does not.
            message["role"] == "user"
            and not (
                message["content"].strip().startswith("<tool_response>")
                and message["content"].strip().endswith("</tool_response>")
            )
            for message in messages
        ):
            raise TemplateError("No user query found in messages.")

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        del tokenize
        self.template_calls.append(dict(kwargs))
        self._refuse(messages)
        bodies = [str(message["content"]) for message in messages]
        if not self.additive and bodies:
            bodies[0] = f"{len(bodies)}{bodies[0]}"
        body = "\n".join(bodies)
        if not add_generation_prompt:
            return body
        # Honour ``enable_thinking`` rather than hardcoding one spec's suffix. The real template
        # emits a closed ``<think>\n\n</think>\n\n`` only when it is False, and an *open*
        # ``<think>`` otherwise; a stand-in that ignored the kwarg could neither confirm nor
        # refute the model-agnostic rendering SPEC-001 is about.
        assistant = "<|im_start|>assistant\n"
        if kwargs.get("enable_thinking") is False:
            return body + "\n" + assistant + "<think>\n\n</think>\n\n"
        return body + "\n" + assistant


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
    """Spans are located in token space, and must tile the render with no gap or overlap.

    The head block is one chunk covering ``head_count`` messages, so the chunks are one shorter
    than the message list for each message the template refuses to render on its own. The
    partition property is unchanged: contiguous, starting at column 0, ending at the render's
    length.
    """
    tokenizer = _SwapTokenizer()
    spec = load_model_spec("qwen35-4b")
    messages = _ledger_tasks()[0]
    from local_llm_lab.pipeline.data import build_rows

    messages = build_rows(messages, keep_last=99)[PROBE_STEP]["messages"][:-1]

    head_count, spans, prompt_tokens = state_swap.message_token_spans(
        tokenizer, messages, spec=spec
    )

    assert head_count == 2, "the system prompt alone is not a conversation this template renders"
    assert len(spans) == len(messages) - head_count + 1
    assert spans[0][0] == 0
    assert [span[0] for span in spans[1:]] == [span[1] for span in spans[:-1]]
    assert spans[-1][1] == prompt_tokens
    assert all(start < end for start, end in spans)
    # Message index -> chunk, and only outside the head: the two lists are shifted, so indexing
    # the chunk tuple by message index would take a real span belonging to another message.
    by_message = state_swap._spans_by_message(head_count, spans)
    assert sorted(by_message) == list(range(head_count, len(messages)))
    assert by_message[len(messages) - 1] == spans[-1]
    with pytest.raises(KeyError):
        by_message[head_count - 1]


def test_message_token_spans_refuse_a_template_that_is_not_prefix_additive() -> None:
    """A template that rewrites earlier turns would move every boundary without a word."""
    tokenizer = _SwapTokenizer(additive=False)
    spec = load_model_spec("qwen35-4b")
    from local_llm_lab.pipeline.data import build_rows

    messages = build_rows(_ledger_tasks()[0], keep_last=99)[PROBE_STEP]["messages"][:-1]

    with pytest.raises(ValueError, match="prefix"):
        state_swap.message_token_spans(tokenizer, messages, spec=spec)


# ------------------------------------------- the real tokenizer, because a stand-in cannot


def _shaped(*roles: str) -> list[dict]:
    """A conversation of the given roles, with the fields ``build_prompt`` reads."""
    return [
        {
            "role": role,
            "content": f"{role} content",
            **({"name": "read_file"} if role == "tool" else {}),
        }
        for role in roles
    ]


#: ``(label, conversation, the writer renders it)``. Measured on the real 4B tokenizer; the
#: stand-in is required to agree row for row, which is the property whose absence let a
#: system-only prefix pass every test and kill the run.
_TEMPLATE_SHAPES = (
    ("system alone -- every conversation's first prefix", _shaped("system"), False),
    ("assistant alone", _shaped("assistant"), False),
    ("tool alone", _shaped("tool"), False),
    ("system then assistant", _shaped("system", "assistant"), False),
    ("system then tool", _shaped("system", "tool"), False),
    ("no user at any length", _shaped("system", "assistant", "tool"), False),
    ("user alone", _shaped("user"), True),
    ("system and user -- the shortest shape this repo renders", _shaped("system", "user"), True),
    ("assistant then user", _shaped("assistant", "user"), True),
    ("the probe's own opening", _shaped("system", "user", "assistant", "tool"), True),
    (
        "a system message that is not first",
        _shaped("system", "user", "assistant", "system"),
        False,
    ),
    ("a role the template does not know", _shaped("system", "user", "narrator"), False),
    (
        "a user turn that is only a tool_response wrapper",
        [
            {"role": "system", "content": "system content"},
            {"role": "user", "content": "<tool_response>listing</tool_response>"},
        ],
        False,
    ),
    (
        "content that is not a string",
        [{"role": "system", "content": "system content"}, {"role": "user", "content": 7}],
        False,
    ),
    # The template trims before it tests for the wrapper (``render_content(...)|trim`` at the
    # head of its reverse scan), so padding does not smuggle a tool response past the check.
    # These two rows are here because a stand-in that tested the untrimmed string accepted both
    # while the writer refused them -- the same over-acceptance as the system-only prefix, one
    # rule further in.
    (
        "a tool_response wrapper padded with spaces",
        [
            {"role": "system", "content": "system content"},
            {"role": "user", "content": "  <tool_response>listing</tool_response>  "},
        ],
        False,
    ),
    (
        "a tool_response wrapper padded with newlines",
        [
            {"role": "system", "content": "system content"},
            {"role": "user", "content": "\n<tool_response>listing</tool_response>\n"},
        ],
        False,
    ),
)


def _renders(tokenizer, messages, spec) -> bool:
    """Whether ``build_prompt`` gets a render out of this tokenizer for this conversation."""
    from jinja2.exceptions import TemplateError

    from local_llm_lab.pipeline.protocol import build_prompt

    try:
        build_prompt(
            tokenizer, list(messages), keep_last=len(messages), spec=spec, generation=False
        )
    except (TemplateError, ValueError):
        # ``generation=False`` skips ``build_prompt``'s own suffix assertion, so the only
        # ``ValueError`` reachable here is ``transformers`` refusing an empty conversation.
        return False
    return True


def _cached_tokenizer_directory(spec) -> Path:
    """Where ``configure_local_cache`` would have put this model's tokenizer."""
    from local_llm_lab.project import configure_local_cache

    return configure_local_cache() / "hub" / ("models--" + spec.hf_id.replace("/", "--"))


@pytest.fixture(name="real_tokenizer", scope="session")
def _real_tokenizer():
    """The tokenizer the run renders through, or a skip that says out loud what it cost.

    A tokenizer is not a model: it loads from the project cache in a couple of seconds with no
    weights and no GPU, so R10's ban on Hub checkpoints does not reach it. Every test in this
    file used a hand-written stand-in until EXP-002 died three seconds into a real run on the
    first prefix ``_rendered_boundaries`` renders, with all 39 of them green -- a reader tested
    against a stand-in passes precisely when the stand-in differs from the writer in the way its
    author assumed it would not (R38).

    The skip is deliberately noisy and deliberately narrow. Noisy, because a silent ``s`` in the
    pytest output is how this blind spot comes back. Narrow, because the *only* condition that
    skips is a snapshot that is demonstrably not on disk: the load itself is not wrapped, so a
    cached-but-broken tokenizer fails the suite instead of quietly disabling it.
    """
    spec = load_model_spec("qwen35-4b")
    directory = _cached_tokenizer_directory(spec)
    if not list(directory.glob("snapshots/*/tokenizer_config.json")):
        message = (
            f"the real {spec.hf_id} tokenizer is not cached under {directory}, so the only "
            "tests in this file that render through the writer rather than a stand-in did not "
            "run. That is the blind spot EXP-002 died in. Populate the cache with "
            "`uv run python -c \"from local_llm_lab.project import configure_local_cache; "
            "configure_local_cache(); from transformers import AutoTokenizer; "
            f"AutoTokenizer.from_pretrained('{spec.hf_id}')\"` and re-run."
        )
        warnings.warn(message, stacklevel=2)
        pytest.skip(message)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, local_files_only=True)


def test_the_real_template_refuses_the_system_only_prefix_and_the_locator_starts_past_it(
    real_tokenizer,
) -> None:
    """The failure itself, on the writer: prefix 1 is refused and prefix 2 is where it begins.

    Qwen2.5's template rendered ``[system]`` alone, so a locator that started at count 1 was
    correct by accident on the 3B. Qwen3.5's scans in reverse for a user query and refuses when
    it finds none. Both halves are asserted here -- the refusal, so the reason the head block
    exists is on the record, and the head block's size, so a template that later accepted the
    shorter prefix would show up as a change rather than pass unnoticed.
    """
    from jinja2.exceptions import TemplateError

    from local_llm_lab.pipeline.data import build_rows
    from local_llm_lab.pipeline.protocol import build_prompt

    spec = load_model_spec("qwen35-4b")
    task = _ledger_tasks()[0]
    history = build_rows(task, keep_last=len(task.steps) + 1)[PROBE_STEP]["messages"][:-1]

    assert [message["role"] for message in history[:2]] == ["system", "user"]
    with pytest.raises(TemplateError, match="No user query found in messages"):
        build_prompt(real_tokenizer, history[:1], keep_last=1, spec=spec, generation=False)

    head_count, chunks, total = state_swap.message_token_spans(
        real_tokenizer, history, spec=spec
    )

    assert head_count == 2
    assert len(chunks) == len(history) - 1
    assert chunks[0][0] == 0
    assert [chunk[0] for chunk in chunks[1:]] == [chunk[1] for chunk in chunks[:-1]]
    assert chunks[-1][1] == total
    assert all(start < end for start, end in chunks)


def test_the_real_locator_puts_every_hidden_observation_outside_the_head_block(
    real_tokenizer,
) -> None:
    """``select_points`` end to end on the writer: the spans it masks are real columns.

    The head block loses one boundary -- the split *inside* ``[system, user]`` -- and nothing
    indexes it, which is the claim this test makes rather than asserts in a comment: every
    hidden observation is a tool message at an index at or past the head, and the filename lands
    inside one of the character spans that came back.
    """
    spec = load_model_spec("qwen35-4b")

    points, rejected = state_swap.select_points(_ledger_tasks()[:6], real_tokenizer, spec=spec)

    assert points, f"no usable probe point on the real tokenizer; rejections: {rejected}"
    for point in points:
        assert point["head_messages"] == 2
        assert point["prefill_chunks"][0] == (0, point["prefill_chunks"][0][1])
        assert len(point["prefill_chunks"]) >= 2
        for start, end in point["hidden_spans"]:
            assert point["prefill_chunks"][0][1] <= start < end
        for record in point["hidden_span_records"]:
            assert record["role"] == "tool"
            assert record["message_index"] >= point["head_messages"]
        prompt = point["persistent_prompt"]
        assert prompt.count(point["filename"]) == 1
        where = prompt.index(point["filename"])
        assert any(start <= where < end for start, end in point["hidden_character_spans"])


def test_the_real_locator_refuses_the_two_shapes_the_head_block_cannot_carry(
    real_tokenizer,
) -> None:
    """Renderability is not monotone, and a silent head can swallow an observation.

    Both are measured facts about the 3.5 template rather than hypotheses. A late system message
    renders at three messages and raises at four, so "the first prefix that renders means the
    rest render" is wrong. And a tool observation before the first user query sits inside a
    four-message head with *no* error from the template -- the locator has to refuse it itself,
    because a span it never located is a mask that never happened.
    """
    spec = load_model_spec("qwen35-4b")

    with pytest.raises(ValueError, match="refused the first 4"):
        state_swap.message_token_spans(
            real_tokenizer, _shaped("system", "user", "assistant", "system"), spec=spec
        )

    with pytest.raises(ValueError, match="head block"):
        state_swap.message_token_spans(
            real_tokenizer,
            _shaped("system", "tool", "assistant", "user", "assistant", "tool"),
            spec=spec,
        )


def test_the_stand_in_and_the_writer_agree_row_for_row_on_the_shape_table(
    real_tokenizer,
) -> None:
    """The R38 gap itself, pinned against the writer over the table below.

    The stand-in's job is to be wrong in no way its user could not survive. Acceptance is the
    one property ``_rendered_boundaries`` depends on and the one the old stand-in got wrong, so
    it is compared row for row against the writer. When the writer is not cached this test does
    not run -- and ``test_the_stand_in_refuses_the_shapes_the_real_template_refuses`` keeps the
    table pinned against the stand-in alone, so a skipped session still fails on a regression.

    **The name says "the shape table" and not "exactly", because the equality is the table's,
    not the template's.** An earlier name claimed the stand-in refuses exactly what the writer
    refuses; a differential sweep over 195 conversations found that untrue in two directions.
    One was a real defect and is fixed: the template ``|trim``s before testing for the
    ``<tool_response>`` wrapper and this stand-in did not, so a padded wrapper was read as a
    user query here and as a tool response there -- the same over-acceptance that killed
    EXP-002, one rule further in. Those two shapes are rows in the table now.

    The other direction is recorded and left alone. On ``content`` that is ``None`` or a list of
    content items the writer renders (``''`` and the concatenated items) and this stand-in
    raises ``Unexpected content type.``. That is an over-*refusal*, the harmless direction: it
    can only fail a test that should pass, never pass a run that should fail. Closing it would
    mean reproducing the multimodal ``render_content`` macro here, which is the second
    implementation of the template this stand-in exists to avoid being -- and nothing in this
    repo puts anything but a string in ``content``. So the compared domain is the string-content
    conversations the locator can actually reach, plus the near-misses around them, and that
    domain is exactly what ``_TEMPLATE_SHAPES`` enumerates.
    """
    spec = load_model_spec("qwen35-4b")
    fake = _SwapTokenizer()

    real = {
        label: _renders(real_tokenizer, messages, spec)
        for label, messages, _ in _TEMPLATE_SHAPES
    }
    stand_in = {label: _renders(fake, messages, spec) for label, messages, _ in _TEMPLATE_SHAPES}
    expected = {label: accepted for label, _messages, accepted in _TEMPLATE_SHAPES}

    assert real == expected
    assert stand_in == real


def test_the_stand_in_refuses_the_shapes_the_real_template_refuses() -> None:
    """The same table without the writer, so the property survives an uncached machine."""
    spec = load_model_spec("qwen35-4b")
    fake = _SwapTokenizer()

    assert {label: _renders(fake, messages, spec) for label, messages, _ in _TEMPLATE_SHAPES} == {
        label: accepted for label, _messages, accepted in _TEMPLATE_SHAPES
    }
    # The stand-in refuses through the same exception type the writer uses, or the locator's
    # narrow catch would be exercised by nothing.
    from jinja2.exceptions import TemplateError

    from local_llm_lab.pipeline.protocol import build_prompt

    with pytest.raises(TemplateError, match="No user query found in messages"):
        build_prompt(fake, _shaped("system"), keep_last=1, spec=spec, generation=False)


def test_the_real_tokenizer_coverage_cannot_be_skipped_into_nothing() -> None:
    """Never skips, so deleting or disabling the writer-backed tests turns the suite red.

    A skip that is legitimate on a machine without the cache is indistinguishable, in pytest's
    output, from a skip caused by a typo, a renamed fixture or a swallowed import. This test is
    the difference: it reads this module's own source and asserts that the writer-backed
    coverage still exists, still loads through ``AutoTokenizer.from_pretrained``, and still
    skips on exactly one checked condition with nothing caught around the load.

    The first version of this guard asserted only that *some* writer-backed tests existed, and
    a verifier emptied the one test that compares the stand-in against the writer into a bare
    docstring with the suite still green -- the guard counted three surviving neighbours and was
    satisfied. Counting tests is not the property; the property is that the comparison itself is
    still being made, over a table that is still non-degenerate. Both are checked below, by
    what the test *does* rather than by its name, so a rename keeps it and a gutted body loses
    it.
    """
    module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }

    fixture = functions["_real_tokenizer"]
    fixture_source = ast.unparse(fixture)
    assert "AutoTokenizer.from_pretrained" in fixture_source
    assert "local_files_only=True" in fixture_source
    assert "warnings.warn" in fixture_source
    assert fixture_source.count("pytest.skip") == 1
    assert not [node for node in ast.walk(fixture) if isinstance(node, ast.Try)], (
        "a try/except around the load would turn a broken tokenizer into a silent skip"
    )

    users = {
        name: node
        for name, node in functions.items()
        if name.startswith("test_")
        and any(argument.arg == "real_tokenizer" for argument in node.args.args)
    }
    assert len(users) >= 3, f"the writer-backed tests have gone missing: {sorted(users)}"
    sources = {name: ast.unparse(node) for name, node in users.items()}
    assert any(
        "_rendered_boundaries" in source or "message_token_spans" in source
        for source in sources.values()
    )
    assert any("select_points" in source for source in sources.values())

    # The blind spot this guard exists to close is not "a writer-backed test disappeared" but
    # "the stand-in drifted from the writer and nothing noticed". Exactly one test in this
    # repository can notice that: the one rendering the same conversations through both. It is
    # identified by what it does -- builds a ``_SwapTokenizer`` and renders through the
    # ``real_tokenizer`` fixture -- because a name is the one part of a test a refactor is free
    # to change and a gutted body is the mutant that got past the first version of this guard.
    comparisons = [
        name
        for name in users
        if "_SwapTokenizer(" in sources[name] and "_renders(real_tokenizer" in sources[name]
    ]
    assert comparisons, (
        "no test renders the same conversations through both the stand-in and the writer, so "
        "nothing here would notice the stand-in accepting a shape the real template refuses -- "
        "the exact gap that killed EXP-002 with all 39 tests green"
    )
    # An equality over the shared table, not a spot check: the comparison has to fail when the
    # two disagree on any row, so ``==`` against something derived from ``_TEMPLATE_SHAPES`` is
    # required. Emptying the body drops both and this assertion is what turns red.
    assert any(
        "_TEMPLATE_SHAPES" in sources[name]
        and any(
            isinstance(statement, ast.Assert)
            and isinstance(statement.test, ast.Compare)
            and any(isinstance(operator, ast.Eq) for operator in statement.test.ops)
            for statement in ast.walk(users[name])
        )
        for name in comparisons
    ), (
        "the stand-in/writer comparison no longer asserts an equality over _TEMPLATE_SHAPES, so "
        "it can pass while the two disagree"
    )

    # And the table itself is non-degenerate, because two dicts built from an empty table are
    # equal for free -- deleting the rows would satisfy every assertion above. The row checked
    # by name is the shape the run actually died on; the rest are checked by kind.
    accepted = {label for label, _messages, renders in _TEMPLATE_SHAPES if renders}
    refused = {label for label, _messages, renders in _TEMPLATE_SHAPES if not renders}
    assert accepted, "a table with nothing the writer accepts cannot catch an over-refusal"
    assert refused, "a table with nothing the writer refuses cannot catch an over-acceptance"
    assert any("system alone" in label for label in refused), (
        "the system-only prefix is the shape EXP-002 died on; it must stay in the table"
    )


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
        "windowed_head_messages": 2,
        "windowed_scored_chunk": (4, 6),
        "windowed_prompt": "windowed",
        "persistent_prompt": "persistent",
        "prefill_chunks": ((0, 2), (2, 5), (5, 7)),
        # The first prefill chunk covers the two messages the chat template will not render
        # apart, as it does on the real 4B; see ``_rendered_boundaries``.
        "head_messages": 2,
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
        # The chunk count is one short of the message count for each message the template will
        # not render on its own, so the artifact says how many the first chunk covers -- without
        # it a reader cannot tell an undivided head from a lost boundary.
        assert schedule["persistent_head_messages"] >= 1
        assert schedule["arm_c_head_messages"] >= 1
        assert all(
            span["start"] >= chunks[0]["end"] for span in record["hidden_spans"]
        ), "a hidden span inside the undivided head block would be a mask over unlocated text"
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
