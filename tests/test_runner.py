from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from local_llm_lab.agent_protocol import Action
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.protocol import parse_turn, render_turn
from local_llm_lab.pipeline.runner import Trajectory, generate_turn, generate_turn_with_count
from local_llm_lab.pipeline.tasks import Task


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
    old_record = legacy.as_dict()
    old_record.pop("think_tokens", None)
    old_record.pop("model", None)
    restored_old = Trajectory(**old_record)
    assert restored_old.think_tokens == 0
    assert restored_old.model == {}


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
    counted, n_tokens, think_tokens = generate_turn_with_count(
        None,
        _FakeTokenizer(pieces),
        "prompt",
        None,
        200,
        spec=load_model_spec("qwen25-coder-3b"),
    )
    assert counted == rendered and n_tokens == 9 == len(consumed)
    assert think_tokens == 0


@pytest.mark.parametrize("thinking", ["inference", "trained"])
def test_generate_turn_force_closes_thinking_at_budget_and_continues(
    monkeypatch, thinking: str
) -> None:
    rendered = render_turn("Visible note", Action("finish", {"answer": "x"}))
    pieces = [
        "<",
        "think>",
        "reason",
        "Visible note\n",
        "```",
        "json\n",
        '{"name": "finish", "arguments": {"answer": "x"}}',
        "\n```",
        "\nJUNK",
    ]
    consumed: list[int] = []

    def fake_stream_generate(model, tokenizer, *, prompt, max_tokens, sampler):
        for index, piece in enumerate(pieces):
            consumed.append(index)
            yield _FakeResponse(index, piece)

    fake_module = types.ModuleType("mlx_lm")
    fake_module.stream_generate = fake_stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    base = load_model_spec("qwen35-4b")
    spec = replace(base, chat=replace(base.chat, thinking=thinking, max_think_tokens=3))

    raw, total_tokens, think_tokens = generate_turn_with_count(
        None, _FakeTokenizer(pieces), "prompt", None, 200, spec=spec
    )

    assert raw == "<think>reason\n</think>\n\n" + rendered
    assert (total_tokens, think_tokens) == (8, 3)
    assert consumed == list(range(8)), "forced close continues only until the note and call close"


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


def test_trim_cache_reuses_shared_prefix_and_encodes_only_the_rest(monkeypatch) -> None:
    from local_llm_lab.pipeline.runner import TrimCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TrimCache(model=object())

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


def test_trim_cache_never_consumes_the_entire_prompt(monkeypatch) -> None:
    """The model needs at least one token to run, so an identical prompt still encodes one."""
    from local_llm_lab.pipeline.runner import TrimCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TrimCache(model=object())
    prompt = [1, 2, 3]
    cache.prepare(prompt)
    entry.offset = 3
    cache.commit(prompt, [])
    assert cache.prepare(prompt) == [3]
    assert entry.offset == 2


def test_trim_cache_rebuilds_when_it_cannot_be_trimmed(monkeypatch) -> None:
    from local_llm_lab.pipeline.runner import TrimCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TrimCache(model=object())
    cache.prepare([1, 2, 3, 4])
    entry.offset = 4
    cache.commit([1, 2, 3, 4], [])
    entry.trimmable = False
    # Divergence would need a trim, which is impossible: fall back to encoding everything.
    assert cache.prepare([1, 2, 9]) == [1, 2, 9]


def test_trim_cache_rebuilds_on_offset_mismatch(monkeypatch) -> None:
    """A cache whose offset disagrees with the shared prefix would attend to stale keys."""
    from local_llm_lab.pipeline.runner import TrimCache

    entry = _FakeKV()
    _fake_cache_module(monkeypatch, [entry])
    cache = TrimCache(model=object())
    cache.prepare([1, 2, 3, 4])
    cache.commit([1, 2, 3, 4], [])
    entry.offset = 99  # corrupted state
    assert cache.prepare([1, 2, 3, 9]) == [1, 2, 3, 9]


class _FakeArray:
    def __init__(self, values) -> None:
        self.values = list(values)


class _FakeArraysCache:
    def __init__(self) -> None:
        self.state = [[_FakeArray([1, 2])]]


class _FakeTupleCache:
    def __init__(self) -> None:
        self.state = (_FakeArray([3]), _FakeArray([4]))


class _FakeInput:
    def __init__(self, values) -> None:
        self.values = list(values)

    def __getitem__(self, key):
        assert key == (None, slice(None, None, None))
        return self


class _SnapshotModel:
    def __init__(self) -> None:
        self.prefixes: list[list[int]] = []

    def __call__(self, token_ids, *, cache):
        self.prefixes.append(list(token_ids.values))
        cache[0].state[0][0].values.append(10)
        cache[1].state[0].values.append(11)


class _SnapshotView:
    def __init__(self, entries) -> None:
        self.entries = entries
        self.make_cache_calls = 0

    def make_cache(self):
        self.make_cache_calls += 1
        return self.entries


def _fake_mx(monkeypatch):
    import mlx

    fake = types.ModuleType("mlx.core")
    fake.array = _FakeInput
    fake.eval = lambda *values: None
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    monkeypatch.setattr(mlx, "core", fake)


def test_snapshot_cache_primes_once_and_restores_isolated_prefix_state(monkeypatch) -> None:
    from local_llm_lab.pipeline.runner import SnapshotCache

    _fake_mx(monkeypatch)
    entries = [_FakeArraysCache(), _FakeTupleCache()]
    view = _SnapshotView(entries)
    model = _SnapshotModel()
    cache = SnapshotCache(model, view, prefix_tokens=2)

    assert cache.prepare([5, 6, 7, 8]) == [7, 8]
    assert model.prefixes == [[5, 6]]
    assert view.make_cache_calls == 1
    assert (cache.reused_tokens, cache.encoded_tokens) == (0, 4)

    first_live_array = cache.cache[0].state[0][0]
    first_live_array.values.append(99)
    cache.cache[1].state[0].values.append(99)
    cache.commit([5, 6, 7, 8], [90])

    assert cache.prepare([5, 6, 9]) == [9]
    assert cache.cache[0].state[0][0].values == [1, 2, 10]
    assert cache.cache[1].state[0].values == [3, 11]
    assert cache.cache[0].state[0][0] is not first_live_array
    assert model.prefixes == [[5, 6]]
    assert (cache.reused_tokens, cache.encoded_tokens) == (2, 5)

    cache.cache[0].state[0][0].values.append(88)
    assert cache.prepare([5, 6, 10, 11]) == [10, 11]
    assert cache.cache[0].state[0][0].values == [1, 2, 10]
    assert (cache.reused_tokens, cache.encoded_tokens) == (4, 7)


@pytest.mark.parametrize(("prefix_tokens", "tokens"), [(0, [1, 2]), (2, [1, 2])])
def test_snapshot_cache_rejects_invalid_prefix_boundaries(
    monkeypatch, prefix_tokens: int, tokens: list[int]
) -> None:
    from local_llm_lab.pipeline.runner import SnapshotCache

    _fake_mx(monkeypatch)
    cache = SnapshotCache(_SnapshotModel(), _SnapshotView([_FakeArraysCache()]), prefix_tokens)

    with pytest.raises(ValueError, match="prefix_tokens"):
        cache.prepare(tokens)


def test_snapshot_cache_rejects_changed_immutable_prefix(monkeypatch) -> None:
    from local_llm_lab.pipeline.runner import SnapshotCache

    _fake_mx(monkeypatch)
    cache = SnapshotCache(
        _SnapshotModel(), _SnapshotView([_FakeArraysCache(), _FakeTupleCache()]), 2
    )
    cache.prepare([1, 2, 3])

    with pytest.raises(ValueError, match="immutable prefix"):
        cache.prepare([1, 9, 3])


class _FactoryView:
    def __init__(self, layer_types: tuple[str, ...]) -> None:
        self.layer_types = layer_types

    def layer_kind(self, index: int) -> str:
        return self.layer_types[index]


def _resolved(strategy: str, reason: str, layer_types: tuple[str, ...]):
    return SimpleNamespace(
        spec=SimpleNamespace(name="qwen35-4b"),
        cache_strategy=strategy,
        cache_strategy_reason=reason,
        layer_types=layer_types,
    )


def test_make_turn_cache_selects_strategy_and_warns_only_for_unverified_hybrid(capsys) -> None:
    from local_llm_lab.pipeline.runner import SnapshotCache, TrimCache, make_turn_cache

    dense_types = ("attention",) * 4
    hybrid_types = ("linear_attention", "linear_attention", "linear_attention", "attention")
    dense_view = _FactoryView(dense_types)
    hybrid_view = _FactoryView(hybrid_types)
    model = object()

    trim = make_turn_cache(
        model,
        dense_view,
        _resolved("trim", "auto:trimmable", dense_types),
        prefix_tokens=4,
    )
    snapshot = make_turn_cache(
        model,
        hybrid_view,
        _resolved("snapshot", "auto:equivalence_verified", hybrid_types),
        prefix_tokens=4,
    )
    disabled = make_turn_cache(
        model,
        hybrid_view,
        _resolved("none", "auto:equivalence_unverified", hybrid_types),
        prefix_tokens=4,
    )

    assert isinstance(trim, TrimCache)
    assert isinstance(snapshot, SnapshotCache)
    assert disabled is None
    assert capsys.readouterr().err.splitlines() == [
        "WARNING: qwen35-4b cache auto-resolution disabled reuse: equivalence is unverified"
    ]

    assert (
        make_turn_cache(
            model,
            hybrid_view,
            _resolved("none", "explicit:none", hybrid_types),
            prefix_tokens=4,
        )
        is None
    )
    assert (
        make_turn_cache(
            model,
            dense_view,
            _resolved("none", "auto:equivalence_unverified", dense_types),
            prefix_tokens=4,
        )
        is None
    )
    assert capsys.readouterr().err == ""


class _IntegrationCache:
    def __init__(self) -> None:
        self.state: list[int] = []
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        removed = min(count, self.offset)
        self.state = self.state[: self.offset - removed]
        self.offset -= removed
        return removed


class _IntegrationView:
    def __init__(self, entry: _IntegrationCache) -> None:
        self.entry = entry
        self.make_cache_calls = 0

    def make_cache(self):
        self.make_cache_calls += 1
        return [self.entry]


class _IntegrationModel:
    def __init__(self) -> None:
        self.primed: list[list[int]] = []

    def __call__(self, token_ids, *, cache):
        prefix = list(token_ids.values)
        self.primed.append(prefix)
        cache[0].state = prefix
        cache[0].offset = len(prefix)


class _IntegrationTokenizer:
    prompt_tokens = {
        "turn-one": [10, 11, 12, 13],
        "turn-two": [10, 11, 14, 15],
    }

    def __init__(self, responses: dict[int, str]) -> None:
        self.responses = responses

    def encode(self, prompt: str) -> list[int]:
        return list(self.prompt_tokens[prompt])

    def decode(self, ids: list[int]) -> str:
        return "".join(self.responses[token] for token in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        raise KeyError(token)


@pytest.mark.parametrize("strategy", ["trim", "snapshot", "none"])
def test_fake_greedy_generation_is_equivalent_for_cache_strategies(
    monkeypatch, strategy: str
) -> None:
    """A disconnected or stale prompt cache must change this context-sensitive fake output."""
    import mlx_lm

    from local_llm_lab.pipeline.runner import make_turn_cache

    _fake_mx(monkeypatch)
    entry = _IntegrationCache()
    _fake_cache_module(monkeypatch, [entry])
    view = _IntegrationView(entry)
    model = _IntegrationModel()
    responses = {
        1: render_turn("first", Action("read_file", {"path": "a"})),
        2: render_turn("second", Action("finish", {"answer": "done"})),
    }
    tokenizer = _IntegrationTokenizer(responses)
    response_for_prompt = {
        (10, 11, 12, 13): 1,
        (10, 11, 14, 15): 2,
    }
    prompt_cache_flags = []

    def fake_stream_generate(
        model, tokenizer, *, prompt, max_tokens, sampler, prompt_cache=None
    ):
        suffix = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt.values)
        prefix = [] if prompt_cache is None else list(prompt_cache[0].state)
        full_prompt = prefix + suffix
        response_token = response_for_prompt[tuple(full_prompt)]
        prompt_cache_flags.append(prompt_cache is not None)
        if prompt_cache is not None:
            prompt_cache[0].state = full_prompt + [response_token]
            prompt_cache[0].offset = len(prompt_cache[0].state)
        yield _FakeResponse(response_token, responses[response_token])

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    layer_types = ("linear_attention", "attention") if strategy == "snapshot" else ("attention",)
    turn_cache = make_turn_cache(
        model,
        view,
        _resolved(strategy, f"explicit:{strategy}", layer_types),
        prefix_tokens=2,
    )
    spec = load_model_spec("qwen35-4b")

    def generate(cache):
        raw_turns = []
        actions = []
        for prompt in ("turn-one", "turn-two"):
            raw, total_tokens, think_tokens = generate_turn_with_count(
                model, tokenizer, prompt, "greedy", 20, cache, spec=spec
            )
            raw_turns.append(raw)
            actions.append(parse_turn(raw).action)
            assert (total_tokens, think_tokens) == (1, 0)
        return raw_turns, actions

    cached_raw, cached_actions = generate(turn_cache)
    plain_raw, plain_actions = generate(None)

    assert cached_raw == plain_raw
    assert cached_actions == plain_actions
    if strategy == "none":
        assert prompt_cache_flags == [False, False, False, False]
    else:
        assert prompt_cache_flags == [True, True, False, False]
    if strategy == "snapshot":
        assert model.primed == [[10, 11]]
        assert view.make_cache_calls == 1


def _finish_task() -> Task:
    return Task(
        task_id="test-runner-0001-clean",
        family="runner",
        variant="clean",
        prompt="finish with x",
        files={},
        steps=(),
        expected_answer="x",
        required_tools=frozenset(),
    )


def test_run_task_passes_active_spec_without_forwarding_tools_and_records_thinking(
    monkeypatch,
) -> None:
    from local_llm_lab.pipeline import runner

    spec = load_model_spec("qwen35-4b")
    prompt_calls = []

    def fake_build_prompt(tokenizer, messages, *, spec, keep_last, generation=True):
        prompt_calls.append((spec, keep_last, generation, tuple(m["role"] for m in messages)))
        return "turn" if generation else "prefix"

    raw = "<think>first\nlast</think>\n\n" + render_turn(
        "done", Action("finish", {"answer": "x"})
    )

    def fake_generate(model, tokenizer, prompt, sampler, max_tokens, turn_cache, *, spec):
        assert prompt == "turn"
        assert turn_cache is None
        return raw, 8, 3

    monkeypatch.setattr(runner, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(runner, "generate_turn_with_count", fake_generate)
    tokenizer = SimpleNamespace(encode=lambda text: list(range(len(text))))

    trajectory = runner.run_task(
        object(),
        tokenizer,
        _finish_task(),
        sampler=None,
        spec=spec,
        view=None,
        resolved=None,
        max_steps=1,
        use_cache=False,
    )

    assert [call[:3] for call in prompt_calls] == [(spec, 2, False), (spec, 2, True)]
    assert trajectory.steps[0]["thinking"] == "first\nlast"
    assert trajectory.steps[0]["think_tokens"] == 3
    assert trajectory.steps[0]["thought"] == "done"
    assert trajectory.think_tokens == 3
    assert trajectory.generated_tokens == 8


def test_run_task_uses_resolved_cache_factory_and_records_model(monkeypatch) -> None:
    from local_llm_lab.pipeline import runner

    spec = load_model_spec("qwen35-4b")
    view = object()
    model_payload = {"cache_strategy": "none", "cache_strategy_reason": "explicit:none"}
    resolved = SimpleNamespace(as_dict=lambda: model_payload)
    cache = SimpleNamespace(cache=None, reused_tokens=0, encoded_tokens=0)
    factory_calls = []

    def fake_build_prompt(tokenizer, messages, *, spec, keep_last, generation=True):
        return "turn" if generation else "fixed"

    def fake_factory(model, actual_view, actual_resolved, *, prefix_tokens):
        factory_calls.append((model, actual_view, actual_resolved, prefix_tokens))
        return cache

    def fake_generate(model, tokenizer, prompt, sampler, max_tokens, turn_cache, *, spec):
        assert turn_cache is cache
        return render_turn("done", Action("finish", {"answer": "x"})), 4, 0

    monkeypatch.setattr(runner, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(runner, "make_turn_cache", fake_factory)
    monkeypatch.setattr(runner, "generate_turn_with_count", fake_generate)
    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3, 4] if text == "fixed" else [9])
    model = object()

    trajectory = runner.run_task(
        model,
        tokenizer,
        _finish_task(),
        sampler=None,
        spec=spec,
        view=view,
        resolved=resolved,
        max_steps=1,
    )

    assert factory_calls == [(model, view, resolved, 4)]
    assert trajectory.model == model_payload
