"""Tokenized rendered-row datasets for the in-process MLX-LM training seam."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.data import require_dataset_manifest

__all__ = ["RenderedRow", "RenderedRowsDataset", "load_rendered_splits"]

CANONICAL_SPLITS = ("train", "valid", "test")


class RenderedRow:
    """One tokenized rendered row: the joined tokens and the prompt-mask offset.

    ``len`` is the token count because mlx-lm 0.31.3 sorts a run's batches by the length of
    the *unprocessed* item (``CacheDataset.itemlen`` in ``mlx_lm/tuner/datasets.py:153-155``
    returns ``len(self._data[idx])``). A raw dict there would sort every row by its key count
    and silently destroy the length bucketing training depends on for its padding budget.
    """

    __slots__ = ("tokens", "offset")

    def __init__(self, tokens: list[int], offset: int) -> None:
        self.tokens = tokens
        self.offset = offset

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"RenderedRow(tokens={self.tokens!r}, offset={self.offset!r})"


class RenderedRowsDataset:
    """MLX-LM dataset protocol using separately tokenized prompt and completion fields.

    mlx-lm 0.31.3 wraps every split it trains on in ``CacheDataset``
    (``mlx_lm/lora.py:301-302``), which reads ``self._data.process(self._data[idx])``
    (``datasets.py:156-160``). So ``__getitem__`` yields the unprocessed row and ``process``
    returns the ``(tokens, offset)`` pair ``CompletionsDataset.process`` returns
    (``datasets.py:203-221``); ``iterate_batches`` unpacks that pair into the batch and the
    prompt mask (``trainer.py:143-146``).

    Tokenization stays eager, at construction: it is what refuses a row the trainer could not
    learn from before any weights are touched, and what orders the rows by length. ``process``
    is therefore a pure lookup, which is trivially the deterministic function the caching
    wrapper assumes.
    """

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, *, max_seq_length: int) -> None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        tokenized = [
            _tokenize_row(row, tokenizer, max_seq_length=max_seq_length, index=index)
            for index, row in enumerate(rows)
        ]
        self._items = [
            item
            for _, item in sorted(enumerate(tokenized), key=lambda pair: (len(pair[1]), pair[0]))
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> RenderedRow:
        return self._items[index]

    def process(self, row: RenderedRow) -> tuple[list[int], int]:
        """Return the trainer's ``(tokens, prompt offset)`` pair for one row it asked for."""
        return (row.tokens, row.offset)


def load_rendered_splits(
    data_dir: Path,
    tokenizer: Any,
    *,
    max_seq_length: int,
    splits: Sequence[str] = CANONICAL_SPLITS,
) -> tuple[RenderedRowsDataset, ...]:
    """Load the named rendered-row files, in the order named, one dataset each.

    The default is the canonical train, validation and test triple. A caller that will not
    train on a split must not ask for it: reading a split costs a full tokenization pass and
    fails closed on any row that cannot be tokenized, and a row nothing reads is no reason to
    refuse a run. Every split that *is* named still fails closed, unchanged.

    The manifest is required first (ruling on #73). This is the seam where a dataset becomes
    weights, so it is the one place where "the rows are all here" must not be allowed to stand
    in for "the write that produced them finished": a dataset stamped by nothing is a run that
    died between its last role file and its commit point, and training on it is unrepeatable.
    """
    require_dataset_manifest(Path(data_dir))
    return tuple(
        RenderedRowsDataset(
            _read_rows(data_dir / f"{split}.jsonl"), tokenizer, max_seq_length=max_seq_length
        )
        for split in splits
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _tokenize_row(
    row: dict[str, Any], tokenizer: Any, *, max_seq_length: int, index: int
) -> RenderedRow:
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError(f"row {index}: prompt and completion must be strings")
    prompt_tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    completion_tokens = list(tokenizer.encode(completion, add_special_tokens=False))
    if not completion_tokens:
        raise ValueError(f"row {index}: completion must tokenize to at least one token")
    if len(prompt_tokens) >= max_seq_length:
        raise ValueError(f"row {index}: prompt leaves no room for a completion token")
    completion_tokens = completion_tokens[: max_seq_length - len(prompt_tokens)]
    return RenderedRow([*prompt_tokens, *completion_tokens], len(prompt_tokens))
