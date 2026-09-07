"""Exact forward inputs, independent of text rendering and generation yield timing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from local_llm_lab.models import NATIVE_PREFILL_STEP_SIZE


@dataclass(frozen=True)
class ForwardPass:
    offset: int
    input_ids: tuple[int, ...]


class ForwardLedger:
    """One logical sequence, including restored prefixes and un-emitted lookahead.

    Record only successful native forward calls. Emissions may precede or follow their
    consuming forward, but they never advance its cursor. Restored passes are validated
    against the new prompt before they can establish a starting position.
    """

    def __init__(
        self,
        prompt_ids: Sequence[int],
        *,
        restored_passes: Sequence[ForwardPass] = (),
        max_tokens: int = 65536,
    ):
        self.prompt_ids = list(prompt_ids)
        self.max_tokens = max_tokens
        if not self.prompt_ids or not 0 < len(self.prompt_ids) <= max_tokens:
            raise ValueError("prompt is empty or exceeds context limit")
        self._validate_ids(self.prompt_ids)
        self.tokens: list[int] = []
        self.generated: list[int] = []
        self.passes: list[ForwardPass] = []
        for row in restored_passes:
            if row.offset + len(row.input_ids) > len(self.prompt_ids):
                raise ValueError("restored prefix exceeds prompt")
            self.record(row.offset, row.input_ids)

    @property
    def offset(self) -> int:
        return len(self.tokens)

    @staticmethod
    def _validate_ids(ids: Sequence[int]) -> None:
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("forward input must contain nonnegative integer token IDs")

    def validate(self, offset: int, ids: Sequence[int]) -> None:
        self._validate_ids(ids)
        if offset != self.offset:
            raise ValueError("non-contiguous forward offset")
        if offset + len(ids) > self.max_tokens:
            raise ValueError("forward exceeds context limit")
        expected = self.prompt_ids + self.generated
        for local, token in enumerate(ids):
            position = offset + local
            if position < len(expected) and token != expected[position]:
                kind = "prompt" if position < len(self.prompt_ids) else "emitted continuation"
                raise ValueError(f"forward tokens disagree with {kind} at position {position}")

    def record(self, offset: int, ids: Sequence[int]) -> None:
        self.validate(offset, ids)
        self.passes.append(ForwardPass(offset, tuple(ids)))
        self.tokens.extend(ids)

    def emitted(self, token_id: int) -> None:
        self._validate_ids([token_id])
        position = len(self.prompt_ids) + len(self.generated)
        if position >= self.max_tokens:
            raise ValueError("emission exceeds context limit")
        if position < self.offset and self.tokens[position] != token_id:
            raise ValueError(f"emitted token disagrees with forward at position {position}")
        self.generated.append(token_id)


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Match the installed generator's string encoding in every consumer."""
    bos = tokenizer.bos_token
    return list(
        tokenizer.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos))
    )


def prefill_passes(
    prompt_ids: Sequence[int], step_size: int = NATIVE_PREFILL_STEP_SIZE
) -> list[ForwardPass]:
    """Native prefill partitions, with the final prompt token always separate."""
    if type(step_size) is not int or step_size < 1 or not prompt_ids:
        raise ValueError("prefill requires tokens and a positive integer step size")
    result = []
    offset = 0
    while offset < len(prompt_ids) - 1:
        end = min(offset + step_size, len(prompt_ids) - 1)
        result.append(ForwardPass(offset, tuple(prompt_ids[offset:end])))
        offset = end
    result.append(ForwardPass(offset, (prompt_ids[-1],)))
    return result
