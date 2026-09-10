"""The sampled decoding path, held to the ruling that created it, on stubs and no checkpoint.

Five things the ruling asked a test to say (plan §16.16; the WS-B order's amendment of
2026-09-10): a fixed seed reproduces the draw on CPU, two seeds differ, the greedy path is
unchanged, a gate handed the sampled mode refuses, and the manifest fields are present and
typed. Beside them, the refusals the ruling scoped -- another temperature, any truncation --
and the property that makes the sampled loop the greedy loop's sibling rather than a fork:
the same prefill partition, the same lookahead, and the same stop rule downstream.

The stubs draw from two shapes of logits. A *coin* puts equal mass on two tokens so that the
draw carries information, which is what reproducibility and difference are claims about. A
*peak* puts all but a vanishing share on the scripted token, so the sampled loop follows a
script and can be compared with the greedy loop on the very same model.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from local_llm_lab import device  # noqa: E402
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline import runner  # noqa: E402
from local_llm_lab.pipeline.runner import (  # noqa: E402
    STOP_TURN_COMPLETE,
    config_eos_ids,
    generate_turn_tokens,
    generate_turn_with_count,
    torch_greedy_stream,
    torch_sampled_stream,
)
from local_llm_lab.pipeline.sampled_decode import (  # noqa: E402
    REPRODUCIBILITY_NOTE,
    RULED_TEMPERATURE,
    SampledDecoding,
    decoding_manifest,
    decoding_mode,
    require_greedy,
)

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "scripts", _ROOT / "research" / "acceptance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import acceptance_gates as gates  # noqa: E402
import golden_trajectories as golden  # noqa: E402
import tolerance  # noqa: E402
import tolerance_baseline as baseline  # noqa: E402

#: Scripted ids start here so they cannot collide with Gemma's real terminators, 1 and 106.
TOKEN_BASE = 1000
HEADS, TAILS = TOKEN_BASE, TOKEN_BASE + 1

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


class _Tokenizer:
    bos_token = None

    def __init__(self, pieces: dict[int, str]) -> None:
        self.table = dict(pieces)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.table.get(i, "?") for i in ids)

    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        return [7, 8, 9]

    def convert_tokens_to_ids(self, token: str) -> int:
        raise KeyError(token)


class _Cache:
    def __init__(self) -> None:
        self.widths: list[int] = []


class _View:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.caches: list[_Cache] = []

    def make_cache(self) -> _Cache:
        cache = _Cache()
        self.caches.append(cache)
        return cache

    def native_readout(self, hidden):
        raise AssertionError("neither loop may read through native_readout")


class _Config:
    def __init__(self, eos_token_id) -> None:
        self.eos_token_id = list(eos_token_id)


class _CoinModel(torch.nn.Module):
    """Equal mass on HEADS and TAILS every step, and a vanishing share on everything else."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = _Config((1, 106))

    def forward(self, tokens, *, cache=None):
        if cache is None:
            raise AssertionError("the cache must reach the model, as a keyword, every forward")
        cache.widths.append(int(tokens.shape[1]))
        logits = torch.zeros(1, int(tokens.shape[1]), self.vocab_size)
        logits[0, -1, HEADS] = 30.0
        logits[0, -1, TAILS] = 30.0
        return logits


class _PeakModel(torch.nn.Module):
    """The next scripted token at a logit that leaves the rest a 1e-15 share between them."""

    def __init__(self, script: list[int], vocab_size: int) -> None:
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.config = _Config((1, 106))
        self.step = 0

    def forward(self, tokens, *, cache=None):
        if cache is None:
            raise AssertionError("the cache must reach the model, as a keyword, every forward")
        cache.widths.append(int(tokens.shape[1]))
        token = self.script[self.step] if self.step < len(self.script) else 0
        self.step += 1
        logits = torch.zeros(1, int(tokens.shape[1]), self.vocab_size)
        logits[0, -1, min(token, self.vocab_size - 1)] = 40.0
        return logits


#: One prefill chunk plus the final token precede the first decode; the first logits go unread.
_DISCARDED_PREFILL_LOGITS = [0]


def _coin(vocab_size: int = TOKEN_BASE + 4):
    tokenizer = _Tokenizer({HEADS: "H", TAILS: "T", 106: "<eot>"})
    return _CoinModel(vocab_size), _View(vocab_size), tokenizer


def _peak(pieces: list[str], *, tail: list[int] = ()):
    table = {TOKEN_BASE + i: piece for i, piece in enumerate(pieces)}
    table[106] = "<end_of_turn>"
    size = TOKEN_BASE + len(pieces) + 1
    script = _DISCARDED_PREFILL_LOGITS + [TOKEN_BASE + i for i in range(len(pieces))] + list(tail)
    return _PeakModel(script, size), _View(size), _Tokenizer(table)


def _sampled(model, view, tokenizer, decoding, max_tokens=32, prompt=(7, 8, 9)):
    return [
        token
        for token, _ in torch_sampled_stream(
            model,
            view,
            tokenizer,
            list(prompt),
            max_tokens,
            eos_ids=config_eos_ids(model),
            decoding=decoding,
        )
    ]


def _greedy(model, view, tokenizer, max_tokens=32, prompt=(7, 8, 9)):
    return [
        token
        for token, _ in torch_greedy_stream(
            model, view, tokenizer, list(prompt), max_tokens, eos_ids=config_eos_ids(model)
        )
    ]


def _body_without_docstring(function) -> str:
    """The code of a function, with its prose removed, so a word in a docstring is not a branch."""
    module = ast.parse(inspect.getsource(function))
    definition = module.body[0]
    assert isinstance(definition, ast.FunctionDef)
    statements = definition.body
    if isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        statements = statements[1:]
    return "\n".join(ast.unparse(statement) for statement in statements)


@pytest.fixture
def pinned(monkeypatch):
    """A process that has pinned, with a known seed, and will not re-pin under the test.

    The loops call ``pin_torch_determinism`` once per process, and that call would overwrite
    the seed this fixture sets; marking it done keeps the pin the test's own.
    """
    monkeypatch.setattr(runner, "_DETERMINISM_PINNED", True)
    monkeypatch.setattr(
        device, "_pinned", {"seed": 11, "deterministic": True, "attn_implementation": "eager"}
    )
    return 11


# --- the ruling's five tests -----------------------------------------------------------------


def test_a_fixed_seed_reproduces_the_draw_on_cpu() -> None:
    first = _sampled(*_coin(), SampledDecoding(seed=7))
    second = _sampled(*_coin(), SampledDecoding(seed=7))
    assert first == second
    assert len(first) == 32
    assert {HEADS, TAILS} <= set(first), (
        "a draw that never varies says nothing about reproducibility; the coin must have "
        "landed both ways for equality to be a claim"
    )


def test_two_seeds_differ() -> None:
    assert _sampled(*_coin(), SampledDecoding(seed=7)) != _sampled(
        *_coin(), SampledDecoding(seed=8)
    )


def test_the_greedy_path_is_unchanged_and_never_draws(monkeypatch) -> None:
    """The greedy loop has no sampled branch inside it, and its output is what it was."""

    def refuse(*args, **kwargs):
        raise AssertionError("the greedy path drew a sample")

    monkeypatch.setattr(torch, "multinomial", refuse)

    model, view, tokenizer = _peak(CALL_PIECES)
    expected = [TOKEN_BASE + i for i in range(len(CALL_PIECES))]
    assert _greedy(model, view, tokenizer, max_tokens=len(CALL_PIECES)) == expected

    spec = load_model_spec("qwen25-coder-3b")
    model, view, tokenizer = _peak(CALL_PIECES)
    ids, reason, _ = generate_turn_tokens(model, tokenizer, [7, 8, 9], 20, view=view, spec=spec)
    assert reason == STOP_TURN_COMPLETE and ids == expected[:9]

    model, view, tokenizer = _peak(CALL_PIECES)
    greedy = types.SimpleNamespace(sampling_temperature=0.0)
    text, count, _ = generate_turn_with_count(
        model, tokenizer, "prompt", greedy, 20, spec=spec, view=view
    )
    assert count == 9 and text.endswith("```")

    body = _body_without_docstring(torch_greedy_stream)
    assert "argmax" in body
    for word in ("multinomial", "decoding", "draw(", "SampledDecoding"):
        assert word not in body, f"the greedy loop carries a sampled branch inside it: {word}"


@pytest.mark.parametrize(
    "handed",
    ["sampled", SampledDecoding(seed=1)],
    ids=["the word", "a declaration"],
)
def test_a_gate_handed_the_sampled_mode_refuses_by_name(handed) -> None:
    with pytest.raises(ValueError, match="gate 5.*stays greedy.*refuses the sampled mode"):
        golden.torch_generator(None, None, None, None, decoding=handed)
    with pytest.raises(ValueError, match="tolerance runner.*stays greedy"):
        tolerance.run_tolerance(None, None, decoding=handed)


def test_both_device_scripts_refuse_the_flag_before_touching_anything(tmp_path, capsys) -> None:
    """Refused at the parser, before records are read, a window is checked or weights load."""
    with pytest.raises(SystemExit) as gates_exit:
        gates.main(["--records", str(tmp_path), "--decoding", "sampled"])
    assert gates_exit.value.code == 2
    assert "acceptance kit stays greedy" in capsys.readouterr().err

    with pytest.raises(SystemExit) as baseline_exit:
        baseline.main(["--records", str(tmp_path), "--decoding", "sampled"])
    assert baseline_exit.value.code == 2
    assert "tolerance runner stays greedy" in capsys.readouterr().err


def test_the_manifest_fields_are_present_and_typed(pinned) -> None:
    decoding = SampledDecoding()
    _sampled(*_coin(), decoding, max_tokens=5)
    manifest = decoding_manifest(decoding)

    assert manifest["mode"] == "sampled"
    assert isinstance(manifest["temperature"], float) and manifest["temperature"] == 1.0
    assert manifest["truncation"] == "none"
    assert manifest["sampler"]["name"] == torch_sampled_stream.__name__
    assert manifest["sampler"]["backend"] == "torch"
    assert isinstance(manifest["sampler"]["function"], str)
    assert isinstance(manifest["seed"], int) and manifest["seed"] == pinned
    assert manifest["seed_source"] == "device.pin"
    assert isinstance(manifest["draws"], int) and manifest["draws"] == 5
    assert isinstance(manifest["kernel_set"]["torch"], str)
    assert manifest["note"] == REPRODUCIBILITY_NOTE
    assert "one backend and kernel set" in manifest["note"]
    json.dumps(manifest)

    greedy = decoding_manifest(None)
    assert greedy["mode"] == "greedy"
    assert greedy["sampler"]["name"] == torch_greedy_stream.__name__
    assert "seed" not in greedy and "temperature" not in greedy, (
        "greedy has neither, and a manifest that carried them would let a greedy record be "
        "read as a sampled one"
    )
    assert decoding_manifest("greedy") == greedy


# --- what the ruling scoped ------------------------------------------------------------------


def test_another_temperature_or_any_truncation_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="refuses temperature 0.7"):
        SampledDecoding(temperature=0.7)
    with pytest.raises(ValueError, match="refuses top-p truncation"):
        SampledDecoding(top_p=0.9)
    with pytest.raises(ValueError, match="refuses top-k truncation"):
        SampledDecoding(top_k=40)
    assert SampledDecoding().temperature == RULED_TEMPERATURE == 1.0
    with pytest.raises(ValueError, match="unknown decoding"):
        decoding_mode("warm")
    assert require_greedy(None, where="here") == "greedy"


def test_the_seed_is_read_from_the_device_pin_when_none_is_stated(pinned) -> None:
    from_pin = SampledDecoding()
    assert _sampled(*_coin(), from_pin) == _sampled(*_coin(), SampledDecoding(seed=pinned))
    assert from_pin.seed == pinned and from_pin.seed_source == "device.pin"
    assert SampledDecoding(seed=3).seed_source == "explicit"


def test_an_unpinned_process_has_no_seed_and_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_DETERMINISM_PINNED", True)
    monkeypatch.setattr(device, "_pinned", None)
    with pytest.raises(RuntimeError, match="nothing is pinned"):
        SampledDecoding().resolved_seed()
    with pytest.raises(RuntimeError, match="nothing is pinned"):
        _sampled(*_coin(), SampledDecoding(), max_tokens=1)


def test_one_instance_is_one_stream_and_a_fresh_one_restarts_it() -> None:
    """Across turns the same declaration keeps drawing; a new one with the seed starts over."""
    shared = SampledDecoding(seed=5)
    first_turn = _sampled(*_coin(), shared, max_tokens=8)
    second_turn = _sampled(*_coin(), shared, max_tokens=8)
    assert shared.draws == 16
    restarted = _sampled(*_coin(), SampledDecoding(seed=5), max_tokens=8)
    assert restarted == first_turn
    assert first_turn + second_turn == _sampled(*_coin(), SampledDecoding(seed=5), max_tokens=16)


# --- the sibling property --------------------------------------------------------------------


def test_the_sampled_loop_shares_the_partition_and_the_stop_rule_with_the_greedy_one() -> None:
    """Same forward widths, same tokens on a peaked model, same reason for stopping."""
    model, view, tokenizer = _peak(CALL_PIECES)
    greedy_tokens = _greedy(model, view, tokenizer, max_tokens=len(CALL_PIECES))
    greedy_widths = view.caches[0].widths

    model, view, tokenizer = _peak(CALL_PIECES)
    sampled_tokens = _sampled(
        model, view, tokenizer, SampledDecoding(seed=1), max_tokens=len(CALL_PIECES)
    )
    assert sampled_tokens == greedy_tokens
    assert view.caches[0].widths == greedy_widths, "the forward partition must not differ"
    assert greedy_widths[:2] == [2, 1], "one chunk plus the final prompt token, then singles"

    spec = load_model_spec("qwen25-coder-3b")
    model, view, tokenizer = _peak(CALL_PIECES)
    ids, reason, ended_on_eos = generate_turn_tokens(
        model, tokenizer, [7, 8, 9], 20, view=view, spec=spec, decoding=SampledDecoding(seed=1)
    )
    assert reason == STOP_TURN_COMPLETE and len(ids) == 9, "the shared stop rule closed the fence"
    assert ended_on_eos is False


def test_the_sampled_loop_keeps_a_terminator_and_stops_on_it() -> None:
    """EOS ends the stream and is kept, and the turn reports it exactly as the greedy turn does.

    The reason is compared with the greedy path's rather than pinned to a value, because the
    stop rule is shared and this test is about the sibling property, not about the label.
    """
    model, view, tokenizer = _peak(["a", "b"], tail=[106, TOKEN_BASE])
    tokens = _sampled(model, view, tokenizer, SampledDecoding(seed=1), max_tokens=10)
    assert tokens == [TOKEN_BASE, TOKEN_BASE + 1, 106]

    spec = load_model_spec("qwen25-coder-3b")
    model, view, tokenizer = _peak(["a", "b"], tail=[106, TOKEN_BASE])
    greedy_ids, greedy_reason, greedy_eos = generate_turn_tokens(
        model, tokenizer, [7, 8, 9], 10, view=view, spec=spec
    )
    model, view, tokenizer = _peak(["a", "b"], tail=[106, TOKEN_BASE])
    ids, reason, ended_on_eos = generate_turn_tokens(
        model, tokenizer, [7, 8, 9], 10, view=view, spec=spec, decoding=SampledDecoding(seed=1)
    )
    assert ids == greedy_ids == [TOKEN_BASE, TOKEN_BASE + 1, 106]
    assert reason == greedy_reason
    assert ended_on_eos is greedy_eos is True, "the fact beside the label, on the sampled path too"


def test_the_sampled_loop_honours_the_token_cap() -> None:
    assert len(_sampled(*_coin(), SampledDecoding(seed=2), max_tokens=3)) == 3


def test_the_torch_turn_takes_a_declaration_as_its_sampler_and_nothing_else_sampled() -> None:
    spec = load_model_spec("qwen25-coder-3b")
    model, view, tokenizer = _peak(CALL_PIECES)
    text, count, _ = generate_turn_with_count(
        model, tokenizer, "prompt", SampledDecoding(seed=3), 20, spec=spec, view=view
    )
    assert count == 9 and text.endswith("```")

    warm = types.SimpleNamespace(sampling_temperature=1.0)
    model, view, tokenizer = _peak(CALL_PIECES)
    with pytest.raises(NotImplementedError, match="greedy.*SampledDecoding"):
        generate_turn_with_count(model, tokenizer, "prompt", warm, 20, spec=spec, view=view)

    model, view, tokenizer = _peak(CALL_PIECES)
    with pytest.raises(ValueError, match="bare word 'sampled' declares nothing"):
        generate_turn_tokens(
            model, tokenizer, [7, 8, 9], 20, view=view, spec=spec, decoding="sampled"
        )
