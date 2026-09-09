"""Batch rendered rows for a torch causal LM, with the two things MLX did not need.

mlx-lm pads with literal token id 0 and passes **no attention mask**: padded positions are attended
over and their logits are thrown away by the loss mask. That is harmless there and is not harmless
here. On `transformers`, and especially on Gemma 3 whose sliding-window masks are built from the
attention mask, unmasked pad positions are attended to by the real tokens and change their
representations. So this collator builds an attention mask, and marks pads `-100` in the labels.

It also reproduces, behind a flag that is off by default, an mlx-lm artefact worth naming rather
than inheriting. mlx-lm's loss mask is `(steps >= offset) & (steps <= true_length)` over
`steps = arange(1, T)`, and target position `j` reads `batch[j]`; real tokens occupy `0..length-1`,
so the target at `steps == length` is `batch[length]`, which is padding -- token id 0. Padding to
`1 + 32*ceil(L/32)` guarantees at least one pad column, so **every untruncated row supervises
exactly one pad token**, training the model to emit token id 0 after its final end-of-turn. It is
an artefact of an off-by-one meeting a padding rule, not a design choice, and reproducing it is a
decision for whoever is comparing against those records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import torch

#: What `torch.nn.functional.cross_entropy` ignores, and what `transformers` writes for a pad.
IGNORE_INDEX = -100

#: mlx-lm's padded width for a batch whose longest row is `L`: `1 + 32*ceil(L/32)`
#: (`mlx_lm/tuner/trainer.py`). The `1 +` is what guarantees the supervised pad column above.
MLX_PAD_MULTIPLE = 32


def mlx_padded_width(longest: int, multiple: int = MLX_PAD_MULTIPLE) -> int:
    """The width mlx-lm would pad a batch to, so a memory profile can be compared like for like."""
    return 1 + multiple * -(-longest // multiple)


@dataclass(frozen=True)
class CausalCollator:
    """Turn ``{"input_ids", "prompt_length"}`` rows into a padded, masked batch.

    ``prompt_length`` is the number of leading positions that are context rather than target; it is
    mlx-lm's ``offset``. Labels are the input ids with everything before the offset, and every pad,
    set to :data:`IGNORE_INDEX`; the loss does the causal shift itself.
    """

    pad_token_id: int = 0
    #: Reproduce mlx-lm's supervised pad column. Off by default: the departure is recorded, not
    #: inherited. Turn it on only to reproduce an MLX training record deliberately.
    supervise_one_pad: bool = False
    #: `mlx` reproduces `1 + 32*ceil(L/32)`; `None` pads to the batch's longest row.
    pad_strategy: str = "longest"

    def __post_init__(self) -> None:
        if self.pad_strategy not in ("longest", "mlx"):
            raise ValueError(f"pad_strategy must be 'longest' or 'mlx'; got {self.pad_strategy!r}")

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty batch")
        rows = [list(feature["input_ids"]) for feature in features]
        offsets = [int(feature.get("prompt_length", 0)) for feature in features]
        longest = max(len(row) for row in rows)
        width = mlx_padded_width(longest) if self.pad_strategy == "mlx" else longest

        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)

        for index, (row, offset) in enumerate(zip(rows, offsets)):
            length = len(row)
            if offset > length:
                raise ValueError(
                    f"row {index}: prompt_length {offset} exceeds its {length} tokens, so the row "
                    "supervises nothing and the batch would silently carry a dead example"
                )
            input_ids[index, :length] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, :length] = 1
            # Supervised targets are the real tokens from the offset onward. The pad column is
            # never attended to; whether it is *supervised* is the flag above.
            supervised_end = length + 1 if self.supervise_one_pad and length < width else length
            labels[index, offset:supervised_end] = input_ids[index, offset:supervised_end]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
