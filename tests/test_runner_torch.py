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


class _StubTokenizer:
    """Token id ``i`` decodes to ``pieces[i]``, so a scripted id list is a scripted string."""

    bos_token = None

    def __init__(self, pieces: list[str], prompt_ids: list[int] | None = None) -> None:
        self.pieces = pieces
        self.prompt_ids = prompt_ids if prompt_ids is not None else [0, 1, 2]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] for i in ids)

    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        return list(self.prompt_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        raise KeyError(token)


class _StubCache:
    """Records the width of every forward, which is how the test sees the cache working."""

    def __init__(self) -> None:
        self.widths: list[int] = []


class _StubView:
    """The whole surface the torch loop is permitted to use, and a log of how it used it."""

    def __init__(self, vocab_size: int, hidden_size: int = 4) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.caches: list[_StubCache] = []

    def make_cache(self) -> _StubCache:
        cache = _StubCache()
        self.caches.append(cache)
        return cache

    def native_readout(self, hidden: torch.Tensor) -> torch.Tensor:
        """Channel zero of each row names that row's argmax token."""
        index = hidden[..., 0].long().clamp(0, self.vocab_size - 1)
        logits = torch.zeros(hidden.shape[0], hidden.shape[1], self.vocab_size)
        return logits.scatter(2, index.unsqueeze(-1), 1.0)


class _StubModel(torch.nn.Module):
    """Emits a scripted token per call, and records how wide each forward was."""

    def __init__(self, script: list[int], hidden_size: int = 4) -> None:
        super().__init__()
        self.script = list(script)
        self.hidden_size = hidden_size
        self.step = 0

    def forward(self, tokens: torch.Tensor, cache: _StubCache | None = None) -> torch.Tensor:
        if cache is not None:
            cache.widths.append(int(tokens.shape[1]))
        token = self.script[self.step] if self.step < len(self.script) else 0
        self.step += 1
        hidden = torch.zeros(1, int(tokens.shape[1]), self.hidden_size)
        hidden[0, -1, 0] = float(token)
        return hidden


def _stubs(pieces: list[str], script: list[int] | None = None):
    tokenizer = _StubTokenizer(pieces)
    view = _StubView(vocab_size=len(pieces))
    model = _StubModel(script if script is not None else list(range(len(pieces))))
    return model, view, tokenizer


def test_is_torch_model_discovers_structurally() -> None:
    model, _, _ = _stubs(CALL_PIECES)
    assert is_torch_model(model)
    assert not is_torch_model(object())
    assert not is_torch_model(None)


def test_greedy_stream_yields_scripted_tokens_with_incremental_pieces() -> None:
    model, view, tokenizer = _stubs(CALL_PIECES)
    produced = list(
        torch_greedy_stream(model, view, tokenizer, [0, 1, 2], max_tokens=len(CALL_PIECES))
    )
    assert [token for token, _ in produced] == list(range(len(CALL_PIECES)))
    assert [piece for _, piece in produced] == CALL_PIECES, (
        "the piece is the text the token added, which is what the stop gate inspects"
    )


def test_greedy_stream_prefills_once_then_feeds_single_tokens() -> None:
    model, view, tokenizer = _stubs(CALL_PIECES)
    prompt_ids = [0, 1, 2, 3, 4]
    list(torch_greedy_stream(model, view, tokenizer, prompt_ids, max_tokens=4))
    assert view.caches[0].widths == [len(prompt_ids), 1, 1, 1], (
        "the prompt is forwarded once and then one token at a time through the cache; "
        "re-feeding the prompt would be quadratic and would still produce the right tokens"
    )
    assert len(view.caches) == 1, "one within-turn cache per turn"


def test_greedy_stream_honours_the_token_cap() -> None:
    model, view, tokenizer = _stubs(CALL_PIECES)
    produced = list(torch_greedy_stream(model, view, tokenizer, [0], max_tokens=3))
    assert len(produced) == 3


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
    ids, reason = generate_turn_tokens(
        model,
        tokenizer,
        [0, 1, 2],
        200,
        view=view,
        spec=load_model_spec("qwen25-coder-3b"),
    )
    assert len(ids) == 9 and reason == "turn_complete"

    model, view, tokenizer = _stubs(CALL_PIECES)
    ids, reason = generate_turn_tokens(
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
            yield _Response(index, piece)

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
