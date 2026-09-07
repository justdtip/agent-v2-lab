"""Cache contracts on fixed tokens and a tiny installed model; no checkpoint loads."""

import copy

import mlx.core as mx
import numpy as np
import pytest

from local_llm_lab.arch import ArchitectureView


def test_forward_ledger_counts_lookahead_not_emissions():
    from local_llm_lab.forward import ForwardLedger

    ledger = ForwardLedger([1, 2, 3])
    ledger.record(0, [1, 2])
    ledger.record(2, [3])
    ledger.record(3, [4])  # generator lookahead happens before yield
    ledger.emitted(4)
    ledger.emitted(5)  # emission itself never advances the model cursor
    assert ledger.offset == 4
    assert ledger.tokens == [1, 2, 3, 4]
    assert ledger.generated == [4, 5]
    ledger.record(4, [5])
    assert [(p.offset, p.input_ids) for p in ledger.passes] == [
        (0, (1, 2)),
        (2, (3,)),
        (3, (4,)),
        (4, (5,)),
    ]


@pytest.mark.parametrize("offset, ids", [(1, [1]), (0, [9]), (0, []), (0, [True])])
def test_forward_ledger_rejects_invalid_forward_without_advancing(offset, ids):
    from local_llm_lab.forward import ForwardLedger

    ledger = ForwardLedger([1, 2])
    with pytest.raises(ValueError):
        ledger.record(offset, ids)
    assert ledger.offset == 0


def test_forward_ledger_detects_a_different_consumed_or_emitted_continuation():
    from local_llm_lab.forward import ForwardLedger

    ledger = ForwardLedger([1])
    ledger.record(0, [1, 2])
    with pytest.raises(ValueError, match="emitted"):
        ledger.emitted(3)
    assert ledger.generated == []
    ledger.emitted(2)
    ledger.emitted(4)
    with pytest.raises(ValueError, match="emitted"):
        ledger.record(2, [5])
    assert ledger.tokens == [1, 2]


def test_forward_ledger_validates_restored_prefix_and_bounds():
    from local_llm_lab.forward import ForwardLedger, ForwardPass

    saved = (ForwardPass(0, (1, 2)),)
    ledger = ForwardLedger([1, 2, 3], restored_passes=saved)
    ledger.record(2, [3])
    assert ledger.offset == 3
    with pytest.raises(ValueError, match="prompt"):
        ForwardLedger([1, 9, 3], restored_passes=saved)
    with pytest.raises(ValueError, match="context"):
        ForwardLedger([1, 2, 3], max_tokens=2)


def _history(**kwargs):
    from test_live_lens_native import tiny_model

    from local_llm_lab.pipeline.runner import HistoryCache

    model = tiny_model()
    view = ArchitectureView.from_model(model)
    return model, HistoryCache(model, view, prefill_step_size=4, **kwargs)


def _forward(cache, ids):
    result = cache.generation_model(mx.array([ids]), cache=cache.cache)
    mx.eval(result)
    return np.array(result)


def test_history_reuses_intermediate_state_and_replays_a_rewritten_observation_exactly():
    model, cache = _history()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert cache.prepare(prompt) == prompt
    _forward(cache, prompt[:4])
    _forward(cache, prompt[4:8])
    _forward(cache, prompt[8:])
    _forward(cache, [10])
    cache.emitted(10)
    cache.commit(prompt, [10])
    assert cache.ledger.offset == 10

    # The first eight tokens include intermediate history; the opening was only four.
    extended = prompt + [10, 11, 12, 13]
    assert cache.prepare(extended) == [9, 10, 11, 12, 13]
    assert cache.ledger.offset == 8
    assert cache.reused_tokens == 8
    _forward(cache, [9, 10, 11, 12])
    _forward(cache, [13])
    cache.commit(extended, [])

    # An expired tool result changes token 4. All later recurrence must be rebuilt.
    changed = [1, 2, 3, 4, 20, 6, 7, 8, 9]
    assert cache.prepare(changed) == [20, 6, 7, 8, 9]
    assert cache.ledger.offset == 4
    actual = [_forward(cache, changed[4:8]), _forward(cache, changed[8:])]
    fresh = model.make_cache()
    model(mx.array([changed[:4]]), cache=fresh)
    expected = [
        np.array(model(mx.array([chunk]), cache=fresh)) for chunk in (changed[4:8], changed[8:])
    ]
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert all(entry.offset == 9 for entry in cache.cache if hasattr(entry, "offset"))


def test_history_does_not_reuse_a_token_match_from_a_different_forward_partition():
    _, cache = _history()
    # A three-token partial prefill cannot substitute for the next four-token prefill.
    cache.prepare([1, 2, 3, 4])
    _forward(cache, [1, 2, 3])
    _forward(cache, [4])
    _forward(cache, [5])
    cache.commit([1, 2, 3, 4], [5])
    assert cache.prepare([1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]
    assert cache.reused_tokens == 0


def test_history_restores_isolated_full_native_cache_after_repeated_advancement():
    _, cache = _history()
    prompt = [1, 2, 3, 4, 5, 6]
    cache.prepare(prompt)
    _forward(cache, prompt[:4])
    _forward(cache, [5])
    first = _forward(cache, [6])
    cache.commit(prompt, [])
    for _ in range(2):
        assert cache.prepare(prompt) == [6]
        np.testing.assert_array_equal(_forward(cache, [6]), first)
        cache.commit(prompt, [])


def test_history_checkpoint_budget_falls_back_to_recomputation():
    _, cache = _history(max_bytes=1)
    prompt = [1, 2, 3, 4, 5]
    cache.prepare(prompt)
    _forward(cache, prompt[:4])
    _forward(cache, [5])
    cache.commit(prompt, [])
    assert cache.snapshot_bytes <= 1
    assert cache.prepare(prompt + [6]) == prompt + [6]
    assert cache.reused_tokens == 0


def test_history_rejects_foreign_or_misaligned_cache_before_forward():
    model, cache = _history()
    cache.prepare([1, 2, 3])
    with pytest.raises(ValueError, match="cache"):
        cache.generation_model(mx.array([[1]]), cache=model.make_cache())
    _forward(cache, [1, 2])
    attention = next(entry for entry in cache.cache if hasattr(entry, "offset"))
    attention.offset = 1
    with pytest.raises(ValueError, match="offset"):
        _forward(cache, [3])


def test_snapshot_copy_preserves_offsetless_cache_metadata():
    from mlx_lm.models.cache import ArraysCache

    from local_llm_lab.pipeline.runner import clone_prompt_cache

    entry = ArraysCache(2)
    entry.state = [mx.array([1.0]), mx.array([2.0])]
    entry.lengths = mx.array([7])
    entry.left_padding = mx.array([2])
    saved = clone_prompt_cache([entry])
    entry.lengths = mx.array([99])
    entry.state[0] = mx.array([99.0])
    assert saved[0].lengths.tolist() == [7]
    assert saved[0].left_padding.tolist() == [2]
    restored = copy.deepcopy(saved)
    restored[0].state[0] = mx.array([42.0])
    assert saved[0].state[0].tolist() == [1.0]


def test_native_runner_uses_history_ledger_through_early_tool_stop_and_next_turn():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.runner import generate_turn_with_count

    model, cache = _history()
    raw = '```json\n{"name": "finish", "arguments": {"answer": "ok"}}\n```'
    backend = Tokenizer(WordLevel({"[UNK]": 0, "prompt": 1, raw: 2, "[EOS]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        clean_up_tokenization_spaces=False,
    )
    prompt = " ".join(["prompt"] * 9)
    spec = load_model_spec("qwen25-coder-3b")

    def sampler(logprobs):
        return mx.array([2])

    expected = generate_turn_with_count(model, tokenizer, prompt, sampler, 5, spec=spec)
    for _ in range(2):
        actual = generate_turn_with_count(model, tokenizer, prompt, sampler, 5, cache, spec=spec)
        assert actual == expected == (raw, 1, 0)
        assert cache.ledger.tokens == [1] * 9 + [2]
        assert cache.ledger.generated == [2]
        assert cache.ledger.offset == 10
    assert cache.reused_tokens == 8


def test_partial_forward_failure_cannot_leave_live_state_reusable():
    model, cache = _history()
    cache.prepare([1, 2, 3, 4, 5])
    original = cache.model

    class Broken:
        def __call__(self, ids, **kwargs):
            original(ids, **kwargs)
            raise RuntimeError("failed after advancing cache")

    cache.model = Broken()
    with pytest.raises(RuntimeError, match="after advancing"):
        _forward(cache, [1, 2, 3, 4])
    assert cache.ledger is None
    assert cache.cache is None
    cache.model = model
    assert cache.prepare([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_native_prefill_cadence_is_the_installed_generator_default():
    """The registry constant plans the history cache's partition; it must be the library's own
    default (``generate_step``'s ``prefill_step_size``), or reuse would match a schedule the
    ordinary runner never takes. Reviewer's amendment while landing the history-cache patch."""
    import inspect

    from mlx_lm.generate import generate_step

    from local_llm_lab.models import NATIVE_PREFILL_STEP_SIZE

    assert (
        inspect.signature(generate_step).parameters["prefill_step_size"].default
        == NATIVE_PREFILL_STEP_SIZE
    )
