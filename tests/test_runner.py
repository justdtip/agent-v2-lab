from __future__ import annotations

import json
import sys
import types

from local_llm_lab.agent_protocol import Action
from local_llm_lab.pipeline.protocol import parse_turn, render_turn
from local_llm_lab.pipeline.runner import Trajectory, generate_turn, generate_turn_with_count


def test_trajectory_as_dict_json_round_trips_through_constructor() -> None:
    trajectory = Trajectory(
        task_id="test-ledger-0001-clean",
        family="ledger",
        variant="clean",
        label="unit",
        prompt="test prompt",
        steps=[{"thinking": None, "action": "read_file"}],
    )

    record = json.loads(json.dumps(trajectory.as_dict()))
    assert Trajectory(**record) == trajectory


def test_trajectory_accepts_write_report_fields_with_backward_compatible_defaults() -> None:
    legacy = Trajectory(
        task_id="test-ledger-0001-clean",
        family="ledger",
        variant="clean",
        label="unit",
        prompt="test prompt",
    )
    record = legacy.as_dict() | {"difficulty": 2, "integrity": {"clean": True}}

    restored = Trajectory(**json.loads(json.dumps(record)))

    assert restored.difficulty == 2
    assert restored.integrity == {"clean": True}
    assert Trajectory(**legacy.as_dict()).difficulty == -1
    assert Trajectory(**legacy.as_dict()).integrity == {}


class _FakeResponse:
    def __init__(self, token: int, text: str) -> None:
        self.token = token
        self.text = text


class _FakeTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        raise KeyError(token)


def test_generate_turn_stops_at_closing_fence(monkeypatch) -> None:
    rendered = render_turn("n", Action("finish", {"answer": "x"}))
    pieces = [
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
    assert "".join(pieces) == rendered + "\nJUNK"
    consumed: list[int] = []

    def fake_stream_generate(model, tokenizer, *, prompt, max_tokens, sampler):
        for index, piece in enumerate(pieces):
            consumed.append(index)
            yield _FakeResponse(index, piece)

    fake_module = types.ModuleType("mlx_lm")
    fake_module.stream_generate = fake_stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    text = generate_turn(None, _FakeTokenizer(pieces), "prompt", None, 200)
    assert "JUNK" not in text
    assert text == rendered
    assert consumed == list(range(9)), "generation stops right after the closing fence"
    assert parse_turn(text).action == Action("finish", {"answer": "x"})
    consumed.clear()
    counted, n_tokens = generate_turn_with_count(None, _FakeTokenizer(pieces), "prompt", None, 200)
    assert counted == rendered and n_tokens == 9 == len(consumed)


class _FakeKV:
    """Minimal stand-in for an mlx-lm KV cache entry: tracks an offset and trims from the end."""

    def __init__(self) -> None:
        self.offset = 0
        self.trimmable = True

    def is_trimmable(self) -> bool:
        return self.trimmable

    def trim(self, n: int) -> int:
        n = min(n, self.offset)
        self.offset -= n
        return n


def _fake_cache_module(monkeypatch, entries):
    """Patch just the three cache helpers on the real module, leaving its other exports intact."""
    from mlx_lm.models import cache as kv

    monkeypatch.setattr(kv, "make_prompt_cache", lambda model: list(entries))
    monkeypatch.setattr(
        kv, "can_trim_prompt_cache", lambda cache: all(c.is_trimmable() for c in cache)
    )
    monkeypatch.setattr(kv, "trim_prompt_cache", lambda cache, n: [c.trim(n) for c in cache][0])
    return kv


def test_common_prefix_length_counts_shared_leading_tokens() -> None:
    from local_llm_lab.pipeline.runner import common_prefix_length

    assert common_prefix_length([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert common_prefix_length([], [1]) == 0
    assert common_prefix_length([5], [6]) == 0


def test_turn_cache_reuses_shared_prefix_and_encodes_only_the_rest(monkeypatch) -> None:
    from local_llm_lab.pipeline.runner import TurnCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TurnCache(model=object())

    first = [1, 2, 3, 4, 5]
    assert cache.prepare(first) == first, "an empty cache must encode the whole prompt"
    entry.offset = len(first)
    cache.commit(first, [90, 91])
    entry.offset = len(first) + 2

    # Next prompt shares 1,2,3 then diverges; the generated tail must be trimmed away too.
    second = [1, 2, 3, 7, 8]
    assert cache.prepare(second) == [7, 8]
    assert entry.offset == 3, "cache must be trimmed back to exactly the shared prefix"
    assert cache.reused_tokens == 3


def test_turn_cache_never_consumes_the_entire_prompt(monkeypatch) -> None:
    """The model needs at least one token to run, so an identical prompt still encodes one."""
    from local_llm_lab.pipeline.runner import TurnCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TurnCache(model=object())
    prompt = [1, 2, 3]
    cache.prepare(prompt)
    entry.offset = 3
    cache.commit(prompt, [])
    assert cache.prepare(prompt) == [3]
    assert entry.offset == 2


def test_turn_cache_rebuilds_when_it_cannot_be_trimmed(monkeypatch) -> None:
    from local_llm_lab.pipeline.runner import TurnCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TurnCache(model=object())
    cache.prepare([1, 2, 3, 4])
    entry.offset = 4
    cache.commit([1, 2, 3, 4], [])
    entry.trimmable = False
    # Divergence would need a trim, which is impossible: fall back to encoding everything.
    assert cache.prepare([1, 2, 9]) == [1, 2, 9]


def test_turn_cache_rebuilds_on_offset_mismatch(monkeypatch) -> None:
    """A cache whose offset disagrees with the shared prefix would attend to stale keys."""
    from local_llm_lab.pipeline.runner import TurnCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TurnCache(model=object())
    cache.prepare([1, 2, 3, 4])
    cache.commit([1, 2, 3, 4], [])
    entry.offset = 99  # corrupted state
    assert cache.prepare([1, 2, 3, 9]) == [1, 2, 3, 9]
