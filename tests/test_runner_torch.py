"""The torch generation path, proved against a stub view rather than a checkpoint.

Nothing here loads a model. The stub supplies exactly the three things the loop is allowed to
touch -- ``make_cache``, ``native_readout`` and the model itself -- and scripts what the model
returns, so the test can assert the tokens, the stop point, and that the cache was actually
used rather than the prompt re-fed every step.

The load-bearing test is the last one: the same piece stream through the MLX path and the
torch path must stop at the same index. Two backends that agree today because two copies of a
stop rule agree is the thing this file exists to prevent.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from local_llm_lab.agent_protocol import Action  # noqa: E402
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.protocol import parse_turn, render_turn  # noqa: E402
from local_llm_lab.pipeline.runner import (  # noqa: E402
    config_eos_ids,
    generate_turn_tokens,
    generate_turn_with_count,
    is_torch_model,
    make_turn_cache,
    torch_greedy_stream,
)

CALL_PIECES = [
    "n",
    "\n",
    "```",
    "json",
    "\n",
    '{"name": "finish", ',
    '"arguments": {"answer": "x"}}',
    "\n",
    "```",
    "\nJUNK",
]


#: Scripted token ids start here so they cannot collide with Gemma's real terminators, 1 and
#: 106. A stub whose ordinary tokens are also EOS ids stops generation for the wrong reason and
#: the test then measures the collision instead of the loop.
TOKEN_BASE = 1000


class _StubTokenizer:
    """Token ``TOKEN_BASE + i`` decodes to ``pieces[i]``, plus any extra ids given by id."""

    bos_token = None

    def __init__(
        self,
        pieces: list[str],
        prompt_ids: list[int] | None = None,
        extra: dict[int, str] | None = None,
    ) -> None:
        self.table = {TOKEN_BASE + index: piece for index, piece in enumerate(pieces)}
        self.table.update(extra or {})
        self.prompt_ids = prompt_ids if prompt_ids is not None else [7, 8, 9]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.table[i] for i in ids)

    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        return list(self.prompt_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        raise KeyError(token)


class _StubCache:
    """Records the width of every forward, which is how the test sees the cache working."""

    def __init__(self) -> None:
        self.widths: list[int] = []


class _StubView:
    """The whole surface the torch loop is permitted to use, and a log of how it used it.

    ``native_readout`` is present and must never be called: the model's own head is the
    producer and the readout is the thing compared against it, never the reverse.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.caches: list[_StubCache] = []
        self.readout_calls = 0

    def make_cache(self) -> _StubCache:
        cache = _StubCache()
        self.caches.append(cache)
        return cache

    def native_readout(self, hidden: torch.Tensor) -> torch.Tensor:
        self.readout_calls += 1
        raise AssertionError(
            "the generation loop must not read through native_readout; the readout is "
            "compared against the model's head, never substituted for it"
        )


class _StubConfig:
    def __init__(self, eos_token_id) -> None:
        self.eos_token_id = eos_token_id


class _StubModel(torch.nn.Module):
    """Returns logits whose last-row argmax is the next scripted token."""

    def __init__(self, script: list[int], vocab_size: int, eos_token_id=(1, 106)) -> None:
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.config = _StubConfig(
            eos_token_id if isinstance(eos_token_id, int) else list(eos_token_id)
        )
        self.step = 0

    def forward(self, tokens: torch.Tensor, *, cache: _StubCache | None = None) -> torch.Tensor:
        # Keyword-only on purpose. WS-A's wrapper is called as `wrapped(ids, cache=cache)`, and
        # that is the one part of the seam a stub cannot otherwise hold the loop to.
        if cache is None:
            raise AssertionError("the cache must reach the model, as a keyword, every forward")
        cache.widths.append(int(tokens.shape[1]))
        token = self.script[self.step] if self.step < len(self.script) else 0
        self.step += 1
        logits = torch.zeros(1, int(tokens.shape[1]), self.vocab_size)
        logits[0, -1, min(token, self.vocab_size - 1)] = 1.0
        return logits


#: `prefill_passes` splits any prompt under 2,049 tokens into one chunk plus the final token,
#: so two forwards precede the first decode and only the second one's logits are read. The
#: script therefore carries one leading entry that is never used.
_DISCARDED_PREFILL_LOGITS = [0]


def _stubs(pieces: list[str], eos_token_id=(1, 106)):
    tokenizer = _StubTokenizer(pieces)
    size = TOKEN_BASE + len(pieces) + 1
    view = _StubView(vocab_size=size)
    script = _DISCARDED_PREFILL_LOGITS + [TOKEN_BASE + index for index in range(len(pieces))]
    model = _StubModel(script, vocab_size=size, eos_token_id=eos_token_id)
    return model, view, tokenizer


def _generate(model, view, tokenizer, prompt_ids, max_tokens):
    from local_llm_lab.pipeline.runner import config_eos_ids

    return list(
        torch_greedy_stream(
            model, view, tokenizer, prompt_ids, max_tokens, eos_ids=config_eos_ids(model)
        )
    )


def test_is_torch_model_discovers_structurally() -> None:
    model, _, _ = _stubs(CALL_PIECES)
    assert is_torch_model(model)
    assert not is_torch_model(object())
    assert not is_torch_model(None)


def test_greedy_stream_yields_scripted_tokens_with_incremental_pieces() -> None:
    model, view, tokenizer = _stubs(CALL_PIECES)
    produced = _generate(model, view, tokenizer, [0, 1, 2], len(CALL_PIECES))
    assert [token for token, _ in produced] == [
        TOKEN_BASE + index for index in range(len(CALL_PIECES))
    ]
    assert [piece for _, piece in produced] == CALL_PIECES, (
        "the piece is the text the token added, which is what the stop gate inspects"
    )
    assert view.readout_calls == 0, "the model's head produces; the readout does not"


def test_greedy_stream_uses_the_native_prefill_partition() -> None:
    """Chunks of NATIVE_PREFILL_STEP_SIZE with the final prompt token always separate.

    A single-chunk prefill produces the same tokens and different forward rows, and
    ForwardLedger.validate asserts the partition on every forward, so the difference would
    surface as a failed assertion far from its cause.
    """
    model, view, tokenizer = _stubs(CALL_PIECES)
    prompt_ids = [0, 1, 2, 3, 4]
    _generate(model, view, tokenizer, prompt_ids, 3)
    assert view.caches[0].widths == [len(prompt_ids) - 1, 1, 1, 1, 1], (
        "four prompt tokens, then the final prompt token alone, then one lookahead forward "
        "per emitted token"
    )
    assert len(view.caches) == 1, "one within-turn cache per turn"


def test_the_lookahead_forward_matches_the_recorded_partition() -> None:
    """A turn of n emissions leaves n single-token forwards, the last one's argmax unused.

    That is what makes the recorded forward at offset p the prediction of position p+1, which
    the golden harness's whole join rests on.
    """
    model, view, tokenizer = _stubs(CALL_PIECES)
    produced = _generate(model, view, tokenizer, [0, 1, 2, 3, 4], 3)
    single_token_forwards = [width for width in view.caches[0].widths if width == 1]
    assert len(produced) == 3
    assert len(single_token_forwards) == 1 + 3, "the final prompt token, then one per emission"


def test_greedy_stream_honours_the_token_cap() -> None:
    model, view, tokenizer = _stubs(CALL_PIECES)
    produced = _generate(model, view, tokenizer, [0, 1], 3)
    assert len(produced) == 3


def test_generation_stops_on_a_config_eos_id_and_keeps_the_token() -> None:
    """Token 106 is emitted on six of the golden records' ninety-four turns, always last."""
    tokenizer = _StubTokenizer(["a", "b", "c", "d"], extra={106: "<end_of_turn>"})
    view = _StubView(vocab_size=TOKEN_BASE + 8)
    # Two ordinary tokens, then <end_of_turn>, then tokens that must never be reached.
    script = _DISCARDED_PREFILL_LOGITS + [
        TOKEN_BASE,
        TOKEN_BASE + 1,
        106,
        TOKEN_BASE + 2,
        TOKEN_BASE + 3,
    ]
    model = _StubModel(script, vocab_size=TOKEN_BASE + 8, eos_token_id=[1, 106])

    produced = list(
        torch_greedy_stream(model, view, tokenizer, [7, 8], 50, eos_ids=config_eos_ids(model))
    )
    assert [token for token, _ in produced] == [TOKEN_BASE, TOKEN_BASE + 1, 106], (
        "the terminator is yielded and then generation ends; the recorded emissions contain it"
    )


def test_eos_ids_come_from_the_config_and_never_from_the_tokenizer() -> None:
    model, _, _ = _stubs(CALL_PIECES, eos_token_id=[1, 106])
    assert config_eos_ids(model) == frozenset({1, 106})

    single = _StubModel([0], vocab_size=8, eos_token_id=7)
    assert config_eos_ids(single) == frozenset({7}), "a scalar declaration is still a set"

    naked = _StubModel([0], vocab_size=8)  # default eos, replaced below
    naked.config = _StubConfig(None)
    with pytest.raises(ValueError, match="token cap"):
        config_eos_ids(naked)


def test_torch_generation_stops_at_the_closing_fence() -> None:
    """The same script the MLX test uses, through the torch branch, stopping in the same place."""
    rendered = render_turn("n", Action("finish", {"answer": "x"}))
    assert "".join(CALL_PIECES) == rendered + "\nJUNK"
    model, view, tokenizer = _stubs(CALL_PIECES)
    text, count, think_tokens = generate_turn_with_count(
        model,
        tokenizer,
        "prompt",
        None,
        200,
        spec=load_model_spec("qwen25-coder-3b"),
        view=view,
    )
    assert text == rendered and "JUNK" not in text
    assert count == 9, "generation stops right after the closing fence"
    assert think_tokens == 0
    assert parse_turn(text).action == Action("finish", {"answer": "x"})


def test_generate_turn_tokens_reports_why_the_turn_stopped() -> None:
    model, view, tokenizer = _stubs(CALL_PIECES)
    ids, reason, _ = generate_turn_tokens(
        model,
        tokenizer,
        [0, 1, 2],
        200,
        view=view,
        spec=load_model_spec("qwen25-coder-3b"),
    )
    assert len(ids) == 9 and reason == "turn_complete"

    model, view, tokenizer = _stubs(CALL_PIECES)
    ids, reason, _ = generate_turn_tokens(
        model,
        tokenizer,
        [0, 1, 2],
        4,
        view=view,
        spec=load_model_spec("qwen25-coder-3b"),
    )
    assert len(ids) == 4 and reason == "token_cap", (
        "a turn cut off by the cap is a truncation, not a wrong answer, and the caller "
        "cannot tell the two apart without this"
    )


def test_torch_generation_refuses_what_it_has_not_ported() -> None:
    model, view, tokenizer = _stubs(CALL_PIECES)
    spec = load_model_spec("qwen25-coder-3b")

    with pytest.raises(ValueError, match="architecture view"):
        generate_turn_with_count(model, tokenizer, "prompt", None, 20, spec=spec)

    sampler = types.SimpleNamespace(sampling_temperature=0.7)
    with pytest.raises(NotImplementedError, match="greedy"):
        generate_turn_with_count(model, tokenizer, "prompt", sampler, 20, spec=spec, view=view)

    greedy = types.SimpleNamespace(sampling_temperature=0.0)
    text, _, _ = generate_turn_with_count(
        model, tokenizer, "prompt", greedy, 20, spec=spec, view=view
    )
    assert text, "a greedy sampler is accepted"


@pytest.mark.parametrize("strategy", ["trim", "snapshot", "history"])
def test_make_turn_cache_refuses_unported_reuse_strategies_on_torch(strategy: str) -> None:
    model, view, _ = _stubs(CALL_PIECES)
    resolved = types.SimpleNamespace(
        cache_strategy=strategy,
        cache_strategy_reason="config",
        layer_types=("full_attention",),
        spec=types.SimpleNamespace(name="stub"),
    )
    with pytest.raises(NotImplementedError, match=strategy):
        make_turn_cache(model, view, resolved, prefix_tokens=1)


def test_make_turn_cache_allows_none_on_torch() -> None:
    model, view, _ = _stubs(CALL_PIECES)
    resolved = types.SimpleNamespace(
        cache_strategy="none",
        cache_strategy_reason="config",
        layer_types=("full_attention",),
        spec=types.SimpleNamespace(name="stub"),
    )
    assert make_turn_cache(model, view, resolved, prefix_tokens=1) is None


@pytest.mark.parametrize(
    "pieces",
    [
        CALL_PIECES,
        ["a", "b", "c", "d"],
        ["note ", "```json\n", '{"name": "finish", "arguments": {}}', "\n```", " tail"],
        ["x", "`", "``", "json", '{"a": 1}', "```", "```", "after"],
    ],
)
def test_both_backends_stop_at_the_same_index(monkeypatch, pieces: list[str]) -> None:
    """One stop rule, two backends. This is the property the golden records depend on."""
    spec = load_model_spec("qwen25-coder-3b")

    class _Response:
        def __init__(self, token: int, text: str) -> None:
            self.token = token
            self.text = text

    def fake_stream_generate(model, tokenizer, *, prompt, max_tokens, sampler):
        for index, piece in enumerate(pieces):
            yield _Response(TOKEN_BASE + index, piece)

    fake_module = types.ModuleType("mlx_lm")
    fake_module.stream_generate = fake_stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    # Both sides get the same budget. The MLX fake stream is finite by construction and the
    # torch stub is not, so an unequal budget would compare the stubs and not the stop rule.
    budget = len(pieces)
    mlx_text, mlx_count, _ = generate_turn_with_count(
        None, _StubTokenizer(pieces), "prompt", None, budget, spec=spec
    )

    model, view, tokenizer = _stubs(pieces)
    torch_text, torch_count, _ = generate_turn_with_count(
        model, tokenizer, "prompt", None, budget, spec=spec, view=view
    )

    assert torch_count == mlx_count, "the two backends stopped at different tokens"
    assert torch_text == mlx_text


def test_both_backends_say_when_the_model_ended_the_turn(monkeypatch) -> None:
    """The label stays ``token_cap``; the fact beside it says the model stopped.

    Only the tool-call id is in the stop set, so a stream that ends itself on a terminator
    falls through to the cap label on both backends. The records are compared on the label,
    so it is not moved; ``ended_on_eos`` is added beside it so a record never says a turn was
    capped when the model ended it, and a turn that really hit the cap says False.
    """
    spec = load_model_spec("qwen25-coder-3b")

    class _Response:
        def __init__(self, token: int, text: str) -> None:
            self.token = token
            self.text = text

    def fake_stream(tokens):
        def fake_stream_generate(model, tokenizer, *, prompt, max_tokens, sampler):
            for token in tokens[:max_tokens]:
                yield _Response(token, tokenizer.decode([token]))

        return fake_stream_generate

    def mlx_turn(tokens, budget):
        fake_module = types.ModuleType("mlx_lm")
        fake_module.stream_generate = fake_stream(tokens)
        monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
        tokenizer = _StubTokenizer(["a", "b", "c"], extra={106: "<end_of_turn>"})
        tokenizer.eos_token_ids = {1, 106}
        return generate_turn_with_count(None, tokenizer, "prompt", None, budget, spec=spec)

    def torch_turn(script, budget):
        tokenizer = _StubTokenizer(["a", "b", "c"], extra={106: "<end_of_turn>"})
        view = _StubView(vocab_size=TOKEN_BASE + 8)
        model = _StubModel(
            _DISCARDED_PREFILL_LOGITS + script, vocab_size=TOKEN_BASE + 8, eos_token_id=[1, 106]
        )
        return generate_turn_with_count(
            model, tokenizer, "prompt", None, budget, spec=spec, view=view
        )

    ended = [TOKEN_BASE, TOKEN_BASE + 1, 106]
    for output in (mlx_turn(ended, 10), torch_turn(ended, 10)):
        text, count, think = output
        assert count == 3 and text.endswith("<end_of_turn>")
        assert output.reason == "token_cap", "the label the records are compared on is unmoved"
        assert output.ended_on_eos is True

    capped = [TOKEN_BASE, TOKEN_BASE + 1, TOKEN_BASE + 2]
    for output in (mlx_turn(capped, 2), torch_turn(capped, 2)):
        assert output.reason == "token_cap" and output.ended_on_eos is False
        assert len(output) == 3, "still unpacks as three for every existing caller"


class _HFStyleOutput:
    """What an HF causal-LM forward actually returns: an object carrying `.logits`."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.past_key_values = None


class _WrappedModel(_StubModel):
    """A model behind the capture wrapper: same logits, delivered in an HF output object."""

    def forward(self, tokens: torch.Tensor, *, cache: _StubCache | None = None):
        return _HFStyleOutput(super().forward(tokens, cache=cache))


class _ViewWithIds(_StubView):
    """A view that owns token-to-tensor conversion, as WS-A's real one does."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.id_calls: list[list[int]] = []

    def _ids(self, token_ids) -> torch.Tensor:
        self.id_calls.append(list(token_ids))
        return torch.tensor([list(token_ids)], dtype=torch.long)


def test_the_loop_unwraps_an_hf_output_and_a_bare_tensor_alike() -> None:
    tokenizer = _StubTokenizer(CALL_PIECES)
    size = TOKEN_BASE + len(CALL_PIECES) + 1
    script = _DISCARDED_PREFILL_LOGITS + [TOKEN_BASE + index for index in range(len(CALL_PIECES))]

    wrapped = _WrappedModel(script, vocab_size=size)
    view = _StubView(vocab_size=size)
    from_output = _generate(wrapped, view, tokenizer, [7, 8, 9], 4)

    bare = _StubModel(script, vocab_size=size)
    plain_view = _StubView(vocab_size=size)
    from_tensor = _generate(bare, plain_view, tokenizer, [7, 8, 9], 4)

    assert from_output == from_tensor, (
        "the wrapper returns the HF output object and a stub returns the tensor; the loop must "
        "read the same native logits out of both"
    )


def test_the_view_converts_the_tokens_when_it_can() -> None:
    """The view carries device and dtype; building the tensor here would strand it on the CPU."""
    tokenizer = _StubTokenizer(CALL_PIECES)
    size = TOKEN_BASE + len(CALL_PIECES) + 1
    script = _DISCARDED_PREFILL_LOGITS + [TOKEN_BASE + index for index in range(len(CALL_PIECES))]
    view = _ViewWithIds(vocab_size=size)
    model = _WrappedModel(script, vocab_size=size)

    prompt_ids = [7, 8, 9, 10]
    _generate(model, view, tokenizer, prompt_ids, 3)

    assert view.id_calls[0] == prompt_ids[:-1], "the prefill chunk goes through the view"
    assert view.id_calls[1] == prompt_ids[-1:], "so does the final prompt token, on its own"
    assert all(len(call) == 1 for call in view.id_calls[2:]), "and every decode step after it"
