"""Does mlx-lm supervise a padding token at the end of every row? Transcribed from its own source.

`mlx_lm/tuner/trainer.py` builds a batch (lines 156-167) and masks the loss (lines 86-99). Neither
step is long, and neither mentions the other, which is how the interaction below survived: the
padding rule guarantees a column that the mask rule then supervises.

    batch:  pad_to = 32
            width = min(1 + 32 * ceil(max(lengths) / 32), max_seq_length)
            batch_arr = np.zeros((rows, width), np.int32)      # the pad value is literally 0
    loss:   targets = batch[:, 1:]
            steps   = arange(1, targets.shape[1] + 1)
            mask    = (steps >= offset) & (steps <= true_length)

Target index `k` carries `steps[k] = k + 1` and reads `batch[k + 1]`, so the last supervised target
is `batch[true_length]`. Real tokens occupy `0 .. true_length - 1`. The `1 +` in the width therefore
guarantees that `batch[true_length]` exists and is padding, for every row that was not truncated at
the cap.

This script reconstructs both rules in numpy and reports, per row, how many supervised targets are
padding. It loads no model and imports no mlx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAD_TO = 32


def batch_and_mask(lengths: list[int], offsets: list[int], max_seq_length: int):
    """mlx-lm's construction, transcribed."""
    width = 1 + PAD_TO * ((max(lengths) + PAD_TO - 1) // PAD_TO)
    width = min(width, max_seq_length)
    batch = np.zeros((len(lengths), width), np.int32)
    truncated = []
    for row, length in enumerate(lengths):
        kept = min(length, max_seq_length)
        # Real tokens are 1..kept so that a zero in a supervised slot can only be padding.
        batch[row, :kept] = np.arange(1, kept + 1)
        truncated.append(kept)
    targets = batch[:, 1:]
    steps = np.arange(1, targets.shape[1] + 1)
    mask = (steps >= np.array(offsets)[:, None]) & (steps <= np.array(truncated)[:, None])
    return batch, targets, mask, width, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    cases = [
        ("typical row", [100], 10),
        ("length an exact multiple of 32", [32], 10),
        ("length one over a multiple", [33], 10),
        ("row truncated at the cap", [args.max_seq_length + 904], 10),
    ]
    rows = []
    for label, lengths, offset in cases:
        _, targets, mask, width, truncated = batch_and_mask(lengths, [offset], args.max_seq_length)
        supervised = targets[0][mask[0]]
        rows.append({
            "case": label,
            "true_length": lengths[0],
            "kept_after_cap": truncated[0],
            "padded_width": width,
            "supervised_tokens": int(mask[0].sum()),
            "supervised_pad_tokens": int((supervised == 0).sum()),
            "truncated": truncated[0] != lengths[0],
        })

    payload = {
        "max_seq_length": args.max_seq_length,
        "rule": "last supervised target is batch[true_length]; padding to 1 + 32*ceil(L/32) "
                "guarantees that column exists for an untruncated row",
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.strict:
        for row in rows:
            expected = 0 if row["truncated"] else 1
            if row["supervised_pad_tokens"] != expected:
                raise SystemExit(f"{row['case']}: expected {expected} pad tokens, got {row}")


if __name__ == "__main__":
    main()
