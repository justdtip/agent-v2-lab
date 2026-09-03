"""Tokenized rendered-row datasets for the in-process MLX-LM training seam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["RenderedRowsDataset", "load_rendered_splits"]


class RenderedRowsDataset:
    """MLX-LM dataset protocol using separately tokenized prompt and completion fields."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, *, max_seq_length: int) -> None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        tokenized = [
            _tokenize_row(row, tokenizer, max_seq_length=max_seq_length, index=index)
            for index, row in enumerate(rows)
        ]
        self._items = [
            item
            for _, item in sorted(
                enumerate(tokenized), key=lambda pair: (len(pair[1][0]), pair[0])
            )
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self._items[index]


def load_rendered_splits(
    data_dir: Path, tokenizer: Any, *, max_seq_length: int
) -> tuple[RenderedRowsDataset, RenderedRowsDataset, RenderedRowsDataset]:
    """Load the canonical train, validation, and test rendered-row files."""
    splits = tuple(
        RenderedRowsDataset(
            _read_rows(data_dir / f"{split}.jsonl"), tokenizer, max_seq_length=max_seq_length
        )
        for split in ("train", "valid", "test")
    )
    return splits  # type: ignore[return-value]


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
) -> tuple[list[int], int]:
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
    return [*prompt_tokens, *completion_tokens], len(prompt_tokens)
