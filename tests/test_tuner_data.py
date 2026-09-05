from __future__ import annotations

import json

import pytest
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import iterate_batches

from local_llm_lab.pipeline.data import DatasetManifestMissingError
from local_llm_lab.tuner_data import RenderedRowsDataset, load_rendered_splits


class _MergingTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls.append((text, add_special_tokens))
        assert not add_special_tokens
        return {
            "prompt": [11],
            "completion": [12],
            "long_completion": [12, 13],
            "promptcompletion": [99],
        }[text]


def test_rendered_rows_dataset_tokenizes_each_side_before_the_joint_boundary_can_merge() -> None:
    tokenizer = _MergingTokenizer()
    dataset = RenderedRowsDataset(
        [{"prompt": "prompt", "completion": "completion", "metadata": {}}],
        tokenizer,
        max_seq_length=8,
    )

    assert dataset.process(dataset[0]) == ([11, 12], 1)
    assert tokenizer.calls == [("prompt", False), ("completion", False)]


def test_rendered_rows_dataset_rejects_rows_that_would_mask_every_completion_token() -> None:
    with pytest.raises(ValueError, match="leaves no room"):
        RenderedRowsDataset(
            [{"prompt": "prompt", "completion": "completion"}],
            _MergingTokenizer(),
            max_seq_length=1,
        )


def test_rendered_rows_dataset_truncates_only_completion_tokens_at_the_sequence_limit() -> None:
    dataset = RenderedRowsDataset(
        [{"prompt": "prompt", "completion": "long_completion"}],
        _MergingTokenizer(),
        max_seq_length=2,
    )

    assert dataset.process(dataset[0]) == ([11, 12], 1)


def test_rendered_rows_dataset_sorts_unsorted_rows_by_sequence_length_stably() -> None:
    class _LengthTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return {
                "p1": [1],
                "p2": [2],
                "p3": [3],
                "p4": [4],
                "c1": [11],
                "c2": [12, 13],
                "c3": [14, 15, 16],
                "c4": [17, 18],
            }[text]

    dataset = RenderedRowsDataset(
        [
            {"prompt": "p1", "completion": "c3"},
            {"prompt": "p2", "completion": "c1"},
            {"prompt": "p3", "completion": "c2"},
            {"prompt": "p4", "completion": "c4"},
        ],
        _LengthTokenizer(),
        max_seq_length=8,
    )

    assert [dataset[index].tokens for index in range(len(dataset))] == [
        [2, 11],
        [3, 12, 13],
        [4, 17, 18],
        [1, 14, 15, 16],
    ]


def _write_splits(directory, splits: tuple[str, ...], *, stamped: bool = True) -> None:
    """Rendered rows this tokenizer can read, plus the manifest the writer stamps last.

    ``write_dataset`` renders rows through a real chat template, which ``_MergingTokenizer``
    -- deliberately a four-word vocabulary, so a merge across the prompt/completion boundary
    is visible in the source -- could not encode. Only the manifest's presence is read, so it
    is stamped here and the rows stay the ones these tests are about.
    """
    for split in splits:
        (directory / f"{split}.jsonl").write_text(
            json.dumps({"prompt": "prompt", "completion": "completion"}) + "\n", encoding="utf-8"
        )
    if stamped:
        (directory / "manifest.json").write_text("{}\n", encoding="utf-8")


def test_load_rendered_splits_validates_each_named_file(tmp_path) -> None:
    _write_splits(tmp_path, ("train", "valid", "test"))

    train, valid, test = load_rendered_splits(tmp_path, _MergingTokenizer(), max_seq_length=8)

    assert [len(split) for split in (train, valid, test)] == [1, 1, 1]


def test_load_rendered_splits_reads_only_the_named_splits(tmp_path) -> None:
    """The train stage never trains on test, so an unreadable test file must not stop it."""
    _write_splits(tmp_path, ("train", "valid"))

    loaded = load_rendered_splits(
        tmp_path, _MergingTokenizer(), max_seq_length=8, splits=("train", "valid")
    )

    assert [len(split) for split in loaded] == [1, 1]
    with pytest.raises(FileNotFoundError):
        load_rendered_splits(tmp_path, _MergingTokenizer(), max_seq_length=8)


def test_load_rendered_splits_refuses_a_dataset_that_was_never_stamped(tmp_path) -> None:
    """Ruled on #73: this is the seam where a dataset becomes weights, so it checks first.

    Every row can be present and readable while the write that produced them died before the
    manifest, and there is nothing about the rows themselves that says which. Refusing here
    is what keeps an unrepeatable training run from starting; the refusal precedes the
    tokenization pass, so it costs nothing and reports the directory rather than a row.
    """
    _write_splits(tmp_path, ("train", "valid", "test"), stamped=False)

    with pytest.raises(DatasetManifestMissingError, match="manifest.json"):
        load_rendered_splits(tmp_path, _MergingTokenizer(), max_seq_length=8)


# ------------------------------------------------------- the pinned mlx-lm dataset protocol


class _PairTokenizer:
    """One token per whitespace-separated word, so a row's length is readable in the source."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [len(word) for word in text.split()]


def test_rendered_rows_dataset_satisfies_the_pinned_cache_dataset_protocol() -> None:
    """mlx-lm 0.31.3 wraps every split in ``CacheDataset``, which calls ``process`` itself.

    ``mlx_lm/lora.py:301-302`` passes ``CacheDataset(train_set)`` to ``train``, and
    ``mlx_lm/tuner/datasets.py:158-160`` reads ``self._data.process(self._data[idx])``. A
    dataset that returns pre-tokenized pairs from ``__getitem__`` and has no ``process``
    raises ``AttributeError`` at the first training step.
    """
    dataset = RenderedRowsDataset(
        [{"prompt": "aa b", "completion": "ccc"}], _PairTokenizer(), max_seq_length=8
    )
    cached = CacheDataset(dataset)

    assert len(cached) == 1
    assert cached[0] == ([2, 1, 3], 2)
    # datasets.py:155-156 sorts batches by ``len(dataset[idx])``, so the raw item's length
    # must be its token count and not, say, a dict's key count.
    assert cached.itemlen(0) == 3


def test_rendered_rows_dataset_feeds_the_pinned_iterate_batches(tmp_path) -> None:
    """The trainer's own batching must produce padded rows and our prompt-mask offsets."""
    dataset = RenderedRowsDataset(
        [
            {"prompt": "aa b", "completion": "ccc dddd"},
            {"prompt": "aa", "completion": "ccc"},
        ],
        _PairTokenizer(),
        max_seq_length=8,
    )

    batch, offsets_lengths = next(
        iterate_batches(CacheDataset(dataset), batch_size=2, max_seq_length=8)
    )

    # Rows are sorted by token count at construction: the two-token row batches first.
    assert batch.shape == (2, 8)
    assert batch.tolist() == [[2, 3, 0, 0, 0, 0, 0, 0], [2, 1, 3, 4, 0, 0, 0, 0]]
    assert offsets_lengths.tolist() == [[1, 2], [2, 4]]
