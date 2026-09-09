"""Residual captures for the state programme, to the contract the pre-registration fixes.

One decision position per distinct decision, every layer, native precision, batch width one. The
contract is not advice here: every field below is required, and a capture that cannot state one is
refused rather than written, because a capture whose width or position is unrecorded cannot be read
against a lens afterwards and nobody will discover that until the reading disagrees.

**Width one, and why it is a refusal rather than a default.** The bf16 forward is not
batch-invariant — measured on the card, no block bitwise identical between two widths
(`research/records/WSD-FD-CALIBRATION-2026-09-10`) — so a residual is a residual *of a width*. A
batched capture would carry a width-dependent arithmetic term into every probe fitted on it, and the
probe would have no way to know. `forward_batch` is declared per cell and only 1 is accepted.

**Native precision, and no promotion anywhere.** The capture records the deployed computation. A
promoted capture is a different measurement and belongs to `residual_precision_probe`, not here.

**The prompt digest is checked, not trusted.** Each decision names the sha256 of the rendered prompt
it was enumerated under. The capture recomputes it from the corpus row it is about to forward and
refuses on a mismatch, so a capture set and a corpus that have drifted apart cannot silently produce
captures attributed to rows that no longer exist.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Every field a captured cell must carry. Checked by `assert_contract`, which is called on the
#: first cell of every run rather than trusted: a contract that is only documented is a contract
#: nothing enforces.
REQUIRED_CELL_FIELDS = (
    "task_id", "step", "split", "family", "variant", "difficulty", "recovery",
    "rendered_rows", "prompt_sha256", "token_index", "seq_len",
    "forward_batch", "anchor_batch", "capture_dtype", "checkpoint_sha256",
    "layers", "d_model", "device", "decoding", "shard", "index_in_shard", "basis",
)

#: The only accepted width, and the only accepted arithmetic path. Both are refusals.
FORWARD_BATCH = 1
CAPTURE_DTYPE = "native"

DEFAULT_SHARD_SIZE = 256

#: What a corpus row must carry for this module to be able to capture from it. Declared, and checked
#: against a real row by `tests/test_state_capture.py`, because the first version of this module
#: read `row["ids"]` — the *lens* corpus's shape — and the agent corpus has no such key. Every test
#: passed, because every test built its own row and supplied one. A fixture that constructs its own
#: input can validate code against a corpus that does not exist.
REQUIRED_ROW_FIELDS = ("messages", "prompt", "metadata")


class ContractViolation(ValueError):
    """A cell the capture contract does not describe, refused before anything is written."""


@dataclass(frozen=True, kw_only=True)
class CaptureTarget:
    """Where a capture goes and what it is a capture of."""

    directory: Path
    entry: str
    checkpoint_sha256: str
    decoding: str
    shard_size: int = DEFAULT_SHARD_SIZE


def prompt_digest(messages: Sequence[dict]) -> str:
    """The digest the capture set was enumerated under: the prompt, excluding the model's turn.

    Identical to `prereg_inputs.sha` over `messages[:-1]`, and the two must stay identical — a
    capture whose digest is computed differently from the enumeration's would refuse every row for a
    reason that has nothing to do with the rows.
    """
    payload = json.dumps(list(messages)[:-1], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def assert_contract(cell: dict) -> None:
    """Refuse a cell the contract does not fully describe, naming every missing field at once."""
    missing = [field for field in REQUIRED_CELL_FIELDS if field not in cell]
    if missing:
        raise ContractViolation(
            f"the capture cell declares no {missing}. Every field in REQUIRED_CELL_FIELDS is "
            "required: a capture that cannot say its width, its position or its arithmetic path "
            "cannot be read against a lens later, and the disagreement then looks like a result. "
            "See research/records/STATE-PLAN-PROGRESS-2026-09-09/PREREGISTRATION.md §2."
        )
    if cell["forward_batch"] != FORWARD_BATCH or cell["anchor_batch"] != FORWARD_BATCH:
        raise ContractViolation(
            f"forward_batch={cell['forward_batch']!r} anchor_batch={cell['anchor_batch']!r}: the "
            f"capture contract accepts {FORWARD_BATCH} only. The bf16 forward is not "
            "batch-invariant, so a residual captured at another width is a residual of another "
            "function (WSD-FD-CALIBRATION-2026-09-10)."
        )
    if cell["capture_dtype"] != CAPTURE_DTYPE:
        raise ContractViolation(
            f"capture_dtype={cell['capture_dtype']!r}: the capture records the deployed "
            f"computation and accepts {CAPTURE_DTYPE!r} only. A promoted capture is a different "
            "measurement and belongs to residual_precision_probe."
        )
    if not isinstance(cell["token_index"], int) or cell["token_index"] < 0:
        raise ContractViolation(
            f"token_index={cell['token_index']!r}: the position must be recorded as the integer it "
            "resolved to. 'the last prompt token' is a rule for finding it, not a record of which "
            "one was found."
        )
    if cell["token_index"] >= cell["seq_len"]:
        raise ContractViolation(
            f"token_index={cell['token_index']} is outside seq_len={cell['seq_len']}"
        )


def rows_by_decision(corpus_rows: Iterable[dict]) -> dict[tuple[str, int], dict]:
    """Index the corpus by `(task_id, step)`, keeping the first row of each repeated key.

    The repeats are byte-identical copies made by `recovery_repeats` in the training split, so the
    first is the decision and the rest are the training mix's weight on it. Taking the first is
    correct **because** they are identical, and `prereg_inputs.py` refuses if they ever are not.
    """
    index: dict[tuple[str, int], dict] = {}
    for row in corpus_rows:
        meta = row.get("metadata", {})
        if meta.get("family") is None:
            continue
        index.setdefault((str(meta["task_id"]), int(meta["step"])), row)
    return index


def capture_decisions(
    *,
    decisions: Sequence[dict],
    corpus: dict[tuple[str, int], dict],
    forward: Callable[[Sequence[int]], Any],
    target: CaptureTarget,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Capture one decision position per decision, writing each shard as it completes.

    `forward` is the seam: it takes the **corpus row** and returns a mapping with `residuals`,
    `seq_len`, `token_index`, `layers`, `d_model`, `device` and `dtype`, where `residuals` is the
    stack of every layer's residual at the decision position. It takes the row rather than a list of
    ids because this corpus carries the rendered `prompt` as a string and no ids, so tokenization
    belongs to the caller that owns the tokenizer. Passing it in keeps this function testable
    without a model, and keeps the contract checks in one place rather than in the caller.

    Shards are written as they fill, and the manifest line for a cell is appended only after its
    shard is on disk, so an interrupted run leaves a manifest that describes exactly what exists.
    """
    target.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = target.directory / "manifest.jsonl"
    done, highest_shard = _already_captured(manifest_path)
    pending = [d for d in decisions if (d["task_id"], d["step"]) not in done]

    # The next shard is one past the highest the manifest names, never `len(done) // shard_size`.
    # A run that ended on a partial shard leaves a count that is not a multiple of the shard size,
    # and the division would then hand the next run the partial shard's own number: it would
    # overwrite a shard whose manifest lines already point at it, and the `shard_sha256` beside
    # those lines would stop matching. That is detectable afterwards and it should not be possible.
    written, shard, buffer, cells = 0, highest_shard + 1, [], []
    for decision in pending:
        key = (decision["task_id"], decision["step"])
        row = corpus.get(key)
        if row is None:
            raise ContractViolation(
                f"{key} is in the capture set and not in the corpus: the two have drifted apart "
                "and a capture attributed to a row that does not exist is worse than no capture."
            )
        observed = prompt_digest(row["messages"])
        if observed != decision["prompt_sha256"]:
            raise ContractViolation(
                f"{key}: the corpus row's prompt digest {observed[:12]} is not the "
                f"{decision['prompt_sha256'][:12]} the capture set was enumerated under. The "
                "corpus has changed since the set was fixed; re-enumerate rather than capture."
            )
        absent = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if absent:
            raise ContractViolation(
                f"{key}: the corpus row declares no {absent}. This corpus carries the rendered "
                "`prompt` as a string and no token ids; the caller's `forward` owns tokenization "
                "and is handed the row, not an id list."
            )
        result = forward(row)
        cell = {
            **{k: decision[k] for k in
               ("task_id", "step", "split", "family", "variant", "difficulty", "recovery",
                "rendered_rows", "prompt_sha256")},
            "token_index": int(result["token_index"]),
            "seq_len": int(result["seq_len"]),
            "forward_batch": FORWARD_BATCH,
            "anchor_batch": FORWARD_BATCH,
            "capture_dtype": CAPTURE_DTYPE,
            "checkpoint_sha256": target.checkpoint_sha256,
            "layers": int(result["layers"]),
            "d_model": int(result["d_model"]),
            "device": str(result["device"]),
            "dtype": str(result["dtype"]),
            "decoding": target.decoding,
            "shard": shard,
            "index_in_shard": len(buffer),
            "basis": "measured-here",
        }
        assert_contract(cell)
        buffer.append(result["residuals"])
        cells.append(cell)
        if len(buffer) >= target.shard_size:
            written += _flush(target, shard, buffer, cells, manifest_path)
            shard, buffer, cells = shard + 1, [], []
            if progress is not None:
                progress({"event": "shard", "shard": shard - 1, "captured": written + len(done)})
    if buffer:
        written += _flush(target, shard, buffer, cells, manifest_path)
    return {"captured": written, "already_present": len(done), "shards": shard + (1 if cells else 0)}


def _already_captured(manifest_path: Path) -> tuple[set[tuple[str, int]], int]:
    """What the manifest says exists, and the highest shard number it names (-1 if none)."""
    if not manifest_path.exists():
        return set(), -1
    done: set[tuple[str, int]] = set()
    highest = -1
    for line in manifest_path.read_text().splitlines():
        if line.strip():
            cell = json.loads(line)
            done.add((cell["task_id"], int(cell["step"])))
            highest = max(highest, int(cell["shard"]))
    return done, highest


def _flush(target: CaptureTarget, shard: int, buffer: list, cells: list[dict],
           manifest_path: Path) -> int:
    """Write the shard, then its manifest lines. Never the other way round.

    A manifest line that names a shard which is not on disk describes a capture nobody has. The
    reverse — a shard with no manifest lines — is recoverable, because the run resumes from the
    manifest and simply rewrites it.
    """
    import torch

    path = target.directory / f"residuals-{shard:05d}.pt"
    torch.save(torch.stack([torch.as_tensor(item) for item in buffer]), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with manifest_path.open("a", encoding="utf-8") as stream:
        for cell in cells:
            stream.write(json.dumps({**cell, "shard_sha256": digest}, sort_keys=True) + "\n")
        stream.flush()
    return len(cells)


__all__ = [
    "CAPTURE_DTYPE",
    "FORWARD_BATCH",
    "REQUIRED_CELL_FIELDS",
    "REQUIRED_ROW_FIELDS",
    "CaptureTarget",
    "ContractViolation",
    "assert_contract",
    "capture_decisions",
    "prompt_digest",
    "rows_by_decision",
]
