from __future__ import annotations

import json

import pytest

from local_llm_lab.tuner_data import RenderedRowsDataset, load_rendered_splits


class _MergingTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {
            "prompt": [11],
            "completion": [12],
            "long_completion": [12, 13],
            "promptcompletion": [99],
        }[text]


def test_rendered_rows_dataset_tokenizes_each_side_before_the_joint_boundary_can_merge() -> None:
    dataset = RenderedRowsDataset(
        [{"prompt": "prompt", "completion": "completion", "metadata": {}}],
        _MergingTokenizer(),
        max_seq_length=8,
    )

    tokens, offset = dataset[0]

    assert tokens == [11, 12]
    assert offset == 1


def test_rendered_rows_dataset_rejects_rows_that_would_mask_every_completion_token() -> None:
    with pytest.raises(ValueError, match="leaves no room"):
        RenderedRowsDataset(
            [{"prompt": "prompt", "completion": "completion"}], _MergingTokenizer(), max_seq_length=1
        )


def test_rendered_rows_dataset_truncates_only_completion_tokens_at_the_sequence_limit() -> None:
    dataset = RenderedRowsDataset(
        [{"prompt": "prompt", "completion": "long_completion"}],
        _MergingTokenizer(),
        max_seq_length=2,
    )

    assert dataset[0] == ([11, 12], 1)


def test_load_rendered_splits_validates_each_named_file(tmp_path) -> None:
    for split in ("train", "valid", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"prompt": "prompt", "completion": "completion"}) + "\n", encoding="utf-8"
        )

    train, valid, test = load_rendered_splits(tmp_path, _MergingTokenizer(), max_seq_length=8)

    assert [len(split) for split in (train, valid, test)] == [1, 1, 1]
