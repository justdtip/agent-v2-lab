"""Saving a cache entry's contents is not saving its position.

``mlx_lm`` splits the job between two properties and its two cache kinds split it differently.
``KVCache.state``'s setter recovers the offset from the restored array's own length.
``RotatingKVCache.state``'s setter assigns keys and values and nothing else, keeping ``offset``
and ``_idx`` in ``meta_state``.

So a snapshot that saves ``state`` alone round-trips correctly on a full-attention model and
restores a rotating layer to the right contents at the wrong position. Gemma 3 4B runs a
rotating cache on 29 of its 34 blocks. The failure mode is wrong attention, not an exception,
which is why it survived until someone looked at the two setters side by side.

The fakes below reproduce those two contracts exactly rather than importing ``mlx_lm``, which
keeps this file off the model library and runnable beside a held box window.
"""

from __future__ import annotations

import pytest

from local_llm_lab.pipeline.runner import restore_cache, snapshot_cache


class _FakeKVCache:
    """``mlx_lm.models.cache.KVCache``: the state setter recovers the offset from the length."""

    def __init__(self, length: int = 0) -> None:
        self.keys = list(range(length))
        self.values = list(range(length))
        self.offset = length

    @property
    def state(self):
        return (list(self.keys), list(self.values))

    @state.setter
    def state(self, value):
        self.keys, self.values = value
        self.offset = len(self.keys)

    def advance(self, count: int) -> None:
        self.keys.extend(range(count))
        self.values.extend(range(count))
        self.offset += count


class _FakeRotatingCache:
    """``mlx_lm.models.cache.RotatingKVCache``: position lives only in ``meta_state``."""

    def __init__(self, length: int = 0, max_size: int = 1024) -> None:
        self.keep = 0
        self.max_size = max_size
        self.keys = list(range(length))
        self.values = list(range(length))
        self.offset = length
        self._idx = length

    @property
    def state(self):
        return (list(self.keys), list(self.values))

    @state.setter
    def state(self, value):
        self.keys, self.values = value

    @property
    def meta_state(self):
        return tuple(map(str, (self.keep, self.max_size, self.offset, self._idx)))

    @meta_state.setter
    def meta_state(self, value):
        self.keep, self.max_size, self.offset, self._idx = map(int, value)

    def advance(self, count: int) -> None:
        self.keys.extend(range(count))
        self.values.extend(range(count))
        self.offset += count
        self._idx += count


class _StatelessCache:
    """The base class's contract: no state, no metadata, and assigning either raises."""

    @property
    def state(self):
        return []

    @state.setter
    def state(self, value):
        if value:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, value):
        if value:
            raise ValueError("This cache has no meta_state but a meta_state was set.")


def test_a_rotating_entry_comes_back_to_the_position_it_was_saved_at() -> None:
    entry = _FakeRotatingCache(length=100)
    saved = snapshot_cache([entry])

    entry.advance(45)
    assert entry.offset == 145 and entry._idx == 145

    restore_cache([entry], saved)
    assert entry.offset == 100, (
        "restoring contents without position leaves the layer attending at the wrong offset, "
        "which produces plausible wrong logits rather than an error"
    )
    assert entry._idx == 100, "the write index rotates with the offset and must come back too"
    assert entry.keys == list(range(100))


def test_saving_state_alone_is_the_defect_this_replaces() -> None:
    """The old behaviour, written out, so the next reader meets it already refuted."""
    entry = _FakeRotatingCache(length=100)
    contents_only = entry.state

    entry.advance(45)
    entry.state = contents_only

    assert entry.keys == list(range(100)), "contents come back"
    assert entry.offset == 145, "and the position does not; that is the whole bug"


def test_a_full_attention_entry_was_never_broken_and_still_is_not() -> None:
    """The asymmetry is why this was invisible on a model without rotating layers."""
    entry = _FakeKVCache(length=100)
    saved = snapshot_cache([entry])
    entry.advance(45)
    restore_cache([entry], saved)
    assert entry.offset == 100 and entry.keys == list(range(100))


def test_a_mixed_stack_restores_every_kind_together() -> None:
    """Gemma's shape: a few full-attention blocks among many rotating ones."""
    # The real layout, read from the stage-two manifest: global attention at 1-based layers
    # 6, 12, 18, 24 and 30, so a full-attention cache exactly where (index + 1) divides by six.
    entries = [
        _FakeKVCache(length=100) if (index + 1) % 6 == 0 else _FakeRotatingCache(length=100)
        for index in range(34)
    ]
    saved = snapshot_cache(entries)
    for entry in entries:
        entry.advance(45)

    restore_cache(entries, saved)
    assert all(entry.offset == 100 for entry in entries)
    assert sum(isinstance(entry, _FakeRotatingCache) for entry in entries) == 29, (
        "29 of 34 blocks rotate on this model, which is what makes the defect load-bearing"
    )


def test_a_stateless_entry_is_left_alone_rather_than_assigned_an_empty_metadata() -> None:
    entry = _StatelessCache()
    saved = snapshot_cache([entry])
    restore_cache([entry], saved)


def test_a_snapshot_is_a_copy_and_not_a_view() -> None:
    entry = _FakeKVCache(length=10)
    saved = snapshot_cache([entry])
    entry.keys.append(999)
    restore_cache([entry], saved)
    assert entry.keys == list(range(10)), "a snapshot that aliased the live cache saves nothing"


def test_restoring_the_wrong_number_of_entries_raises() -> None:
    entries = [_FakeKVCache(length=4), _FakeKVCache(length=4)]
    saved = snapshot_cache(entries)
    with pytest.raises(ValueError):
        restore_cache(entries[:1], saved)
