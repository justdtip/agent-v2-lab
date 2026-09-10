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
    "rendered_rows", "row_ordinals",
    # Three digests, and they are three different claims. `messages_sha256` is the semantic record
    # the capture set was enumerated under — canonical JSON over `messages[:-1]`, which is what
    # `prereg_inputs.py` computed and is **not** the model's input. `rendered_prompt_sha256` is the
    # byte digest of the string that was tokenized. `token_ids_sha256` is the ids actually consumed.
    # The first was called `prompt_sha256`, which read as the last two and is neither.
    "messages_sha256", "rendered_prompt_sha256", "token_ids_sha256", "token_ids_length",
    "token_index", "seq_len",
    "forward_batch", "anchor_batch", "capture_dtype",
    "checkpoint_sha256", "config_sha256", "weight_files",
    "layers", "d_model", "device", "dtype", "decoding", "shard", "index_in_shard", "basis",
)

#: What the `forward` seam must report about the pass it actually ran, as opposed to what the
#: contract wishes were true. The writer compares these to the contract and refuses on a mismatch;
#: it does **not** write the contract's constants into the cell. The first version did exactly
#: that — stamped `forward_batch = FORWARD_BATCH` and then asserted it equalled `FORWARD_BATCH` —
#: so a seam that expanded a batch or promoted its arithmetic was recorded as width 1, native, and
#: the guard erased the evidence it existed to check (Codex C1).
REQUIRED_SEAM_FIELDS = (
    "residuals", "seq_len", "token_index", "layers", "d_model", "device", "dtype",
    "forward_batch", "anchor_batch", "capture_dtype",
    "rendered_prompt_sha256", "token_ids_sha256", "token_ids_length",
)

#: The only accepted width, and the only accepted arithmetic path. Both are refusals.
FORWARD_BATCH = 1
CAPTURE_DTYPE = "native"

DEFAULT_SHARD_SIZE = 256

#: The enumeration fields a reused cell must still agree with the request about. A capture set
#: re-enumerated after a pass began can change any of them, and a cell that disagrees describes a
#: decision the request no longer makes.
ENUMERATION_FIELDS = (
    "split", "family", "variant", "difficulty", "recovery", "rendered_rows", "row_ordinals",
)

#: What a corpus row must carry for this module to be able to capture from it. Declared, and checked
#: against a real row by `tests/test_state_capture.py`, because the first version of this module
#: read `row["ids"]` — the *lens* corpus's shape — and the agent corpus has no such key. Every test
#: passed, because every test built its own row and supplied one. A fixture that constructs its own
#: input can validate code against a corpus that does not exist.
REQUIRED_ROW_FIELDS = ("messages", "prompt", "metadata")


class ContractViolation(ValueError):
    """A cell the capture contract does not describe, refused before anything is written."""


class ResumeUnverified(ContractViolation):
    """An existing capture that cannot be shown to be the capture this run would have made."""


def checkpoint_identity(sha256_map: object) -> dict:
    """The checkpoint's identity: every file's digest, and one digest over all of them.

    `report["sha256"]` from the loader is a mapping of file name to digest covering the config
    **and every weight shard**. The first version of this module took `sha256["config.json"]` alone
    and called it the checkpoint's identity, which two checkpoints sharing a config and differing in
    every weight would satisfy identically — an identity that cannot tell apart the thing it
    identifies. The config digest is still recorded, named as the config's and not as the
    checkpoint's.
    """
    if not isinstance(sha256_map, dict) or not sha256_map:
        raise ContractViolation(
            "the loader reported no file digests, so this capture cannot say which checkpoint it "
            "is of. A capture without a checkpoint identity is unattributable and is refused "
            "rather than written."
        )
    if "config.json" not in sha256_map:
        raise ContractViolation(
            f"the digest manifest names {sorted(sha256_map)[:4]}… and no config.json; the loader's "
            "complete manifest is required, not a subset of it."
        )
    weights = sorted(k for k in sha256_map if k != "config.json")
    if not weights:
        raise ContractViolation(
            "the digest manifest names config.json and no weight files. Two checkpoints with one "
            "config and different weights would then share an identity."
        )
    canonical = json.dumps(dict(sorted(sha256_map.items())), sort_keys=True)
    return {
        "checkpoint_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "config_sha256": sha256_map["config.json"],
        "checkpoint_files_sha256": dict(sorted(sha256_map.items())),
        "weight_files": len(weights),
    }


@dataclass(frozen=True, kw_only=True)
class CaptureTarget:
    """Where a capture goes and what it is a capture of."""

    directory: Path
    entry: str
    identity: dict          # `checkpoint_identity(report["sha256"])`, never one file's digest
    decoding: str
    shard_size: int = DEFAULT_SHARD_SIZE


def prompt_digest(messages: Sequence[dict]) -> str:
    """The digest the capture set was enumerated under: the **messages**, excluding the model's turn.

    Recorded as `messages_sha256`, because that is what it is: canonical JSON over the semantic
    record and not the bytes the model reads. Two rows with one message record and different
    rendered prompts share it, which is why the cell also carries the rendered prompt's byte digest
    and the ids consumed, and why the capture boundary checks those separately.

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
    # `isinstance(True, int)` is True, and a float coerced by the writer before this check would
    # arrive already conforming — 1.9 becomes 1 and passes (Codex C1). The type is checked on the
    # value the seam reported, before any coercion, and bools are excluded explicitly.
    index = cell["token_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ContractViolation(
            f"token_index={index!r} of type {type(index).__name__}: the position must be recorded "
            "as the integer the seam resolved to, checked before any coercion. 'the last prompt "
            "token' is a rule for finding it, not a record of which one was found."
        )
    if cell["token_index"] >= cell["seq_len"]:
        raise ContractViolation(
            f"token_index={cell['token_index']} is outside seq_len={cell['seq_len']}"
        )


def assert_seam_ran_the_declared_pass(observed: dict, row: dict, key: tuple[str, int]) -> None:
    """Compare what the seam says it did against what the contract requires, and refuse on a gap.

    This is the check the first version could not perform, because the writer supplied the answers.
    Every value here is the seam's own report of the pass it ran; nothing in this function comes
    from a constant in this module except the constants being compared *against*.
    """
    missing = [field for field in REQUIRED_SEAM_FIELDS if field not in observed]
    if missing:
        raise ContractViolation(
            f"{key}: the forward seam reports no {missing}. It must attest to the pass it actually "
            "ran — its widths, its arithmetic path, the bytes it tokenized and the ids it "
            "consumed — because a writer that supplies those values checks only itself."
        )
    for field in ("forward_batch", "anchor_batch"):
        if observed[field] != FORWARD_BATCH:
            raise ContractViolation(
                f"{key}: the seam ran at {field}={observed[field]!r} and the capture contract "
                f"accepts {FORWARD_BATCH} only. The bf16 forward is not batch-invariant, so this "
                "is a residual of another function and is refused rather than relabelled."
            )
    if observed["capture_dtype"] != CAPTURE_DTYPE:
        raise ContractViolation(
            f"{key}: the seam ran the {observed['capture_dtype']!r} arithmetic path and the "
            f"contract accepts {CAPTURE_DTYPE!r} only."
        )
    expected = hashlib.sha256(row["prompt"].encode()).hexdigest()
    if observed["rendered_prompt_sha256"] != expected:
        raise ContractViolation(
            f"{key}: the seam tokenized bytes digesting to "
            f"{observed['rendered_prompt_sha256'][:12]} and the corpus row's rendered prompt "
            f"digests to {expected[:12]}. The capture would be of an input this row does not "
            "carry. `messages_sha256` cannot catch this: it is a digest of the semantic record "
            "and not of the model's input."
        )
    if observed["token_ids_length"] != observed["seq_len"]:
        raise ContractViolation(
            f"{key}: the seam consumed {observed['token_ids_length']} ids and reports a sequence "
            f"length of {observed['seq_len']}."
        )
    shape = tuple(getattr(observed["residuals"], "shape", ()))
    if shape != (observed["layers"], observed["d_model"]):
        raise ContractViolation(
            f"{key}: the residuals have shape {shape} and the seam declares "
            f"{(observed['layers'], observed['d_model'])}. A capture whose shape is not the shape "
            "it claims cannot be indexed by layer afterwards."
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
    forward: Callable[..., Any],
    target: CaptureTarget,
    prepare: Callable[[dict], dict] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Capture one decision position per decision, writing each shard as it completes.

    `forward` is the seam: it takes the **corpus row** and returns a mapping with `residuals`,
    `seq_len`, `token_index`, `layers`, `d_model`, `device` and `dtype`, where `residuals` is the
    stack of every layer's residual at the decision position. It takes the row rather than a list of
    ids because this corpus carries the rendered `prompt` as a string and no ids, so tokenization
    belongs to the caller that owns the tokenizer. Passing it in keeps this function testable
    without a model, and keeps the contract checks in one place rather than in the caller.

    `prepare` is the second seam and it exists for resume. It turns a row into the **exact model
    input** — the ids, their digest and length, and the rendered bytes' digest — *before* this
    function decides whether an existing capture may be reused. Without it a resumed pass cannot
    check that this run's tokenizer produces the ids the existing cells record, and two tokenizers
    give different ids for the same bytes while the checkpoint identity covers no tokenizer asset at
    all (Codex R2). When it is absent, resume verifies everything except the consumed ids and the
    summary says exactly that, rather than reporting a verification it did not perform.

    Shards are written as they fill, and the manifest line for a cell is appended only after its
    shard is on disk, so an interrupted run leaves a manifest that describes exactly what exists.
    """
    target.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = target.directory / "manifest.jsonl"
    existing, highest_shard = _existing_cells(manifest_path)

    # Every **requested** decision is validated, whether it is to be captured or reused. The first
    # version validated a decision only on the path that captured it, so a resumed pass reported
    # complete and verified without ever consulting the request it was given (Codex R1).
    pending, reused = [], 0
    for decision in decisions:
        key = (decision["task_id"], int(decision["step"]))
        row = _row_for(corpus, key)
        observed_messages = prompt_digest(row["messages"])
        if observed_messages != decision["messages_sha256"]:
            raise ContractViolation(
                f"{key}: the corpus row's message digest {observed_messages[:12]} is not the "
                f"{decision['messages_sha256'][:12]} the capture set was enumerated under. The "
                "corpus has changed since the set was fixed; re-enumerate rather than capture."
            )
        cell = existing.get(key)
        if cell is None:
            pending.append((decision, row))
            continue
        _assert_reusable(cell, decision, row, target,
                         prepare(row) if prepare is not None else None)
        reused += 1

    # The next shard is one past the highest the manifest names, never `len(done) // shard_size`.
    # A run that ended on a partial shard leaves a count that is not a multiple of the shard size,
    # and the division would then hand the next run the partial shard's own number: it would
    # overwrite a shard whose manifest lines already point at it, and the `shard_sha256` beside
    # those lines would stop matching. That is detectable afterwards and it should not be possible.
    written, shard, buffer, cells = 0, highest_shard + 1, [], []
    for decision, row in pending:
        key = (decision["task_id"], int(decision["step"]))
        observed_messages = prompt_digest(row["messages"])
        prepared = prepare(row) if prepare is not None else None
        result = forward(row, prepared) if prepare is not None else forward(row)
        assert_seam_ran_the_declared_pass(result, row, key)
        cell = {
            **{k: decision[k] for k in
               ("task_id", "step", "split", "family", "variant", "difficulty", "recovery",
                "rendered_rows", "row_ordinals")},
            "messages_sha256": observed_messages,
            # Every field below that describes the pass comes from the seam's own report, not from
            # this module's constants. `assert_seam_ran_the_declared_pass` has already refused any
            # that disagree with the contract, so recording them is a record and not a relabelling.
            "rendered_prompt_sha256": result["rendered_prompt_sha256"],
            "token_ids_sha256": result["token_ids_sha256"],
            "token_ids_length": result["token_ids_length"],
            "token_index": result["token_index"],
            "seq_len": result["seq_len"],
            "forward_batch": result["forward_batch"],
            "anchor_batch": result["anchor_batch"],
            "capture_dtype": result["capture_dtype"],
            "layers": result["layers"],
            "d_model": result["d_model"],
            "device": str(result["device"]),
            "dtype": str(result["dtype"]),
            "checkpoint_sha256": target.identity["checkpoint_sha256"],
            "config_sha256": target.identity["config_sha256"],
            "weight_files": target.identity["weight_files"],
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

    # The coverage beside the verdict, per method entry thirty-four: a summary that says how many
    # cells were written and not how many were asked for cannot distinguish a complete pass from a
    # truncated one, and both look like success. `complete` is computed against the decisions this
    # call was given — a caller that passed a subset gets `requested` equal to that subset, which is
    # why `requested` is reported beside it rather than assumed to be the whole set.
    final, _ = _existing_cells(manifest_path)
    outstanding = [d for d in decisions if (d["task_id"], int(d["step"])) not in final]
    return {
        "captured": written,
        "already_present": reused,
        "requested": len(decisions),
        "outstanding": len(outstanding),
        "complete": not outstanding,
        "reused": reused,
        "verified_scope": (
            "checkpoint identity, semantic record, rendered input, requested enumeration, file "
            "integrity, and the consumed ids"
            if prepare is not None else
            "checkpoint identity, semantic record, rendered input, requested enumeration and file "
            "integrity — **not** the consumed ids, because no `prepare` seam was supplied and this "
            "run cannot know what ids its tokenizer would produce"
        ),
        "consumed_ids_verified_on_reuse": prepare is not None,
        "shards": shard + (1 if cells else 0),
        "coverage_note": "requested is what this call was given, not necessarily the whole capture "
                         "set; complete says every requested decision now has a manifest line; "
                         "reused is how many were accepted from a previous pass, each checked "
                         "against the scope named in verified_scope",
    }


def _row_for(corpus: dict[tuple[str, int], dict], key: tuple[str, int]) -> dict:
    row = corpus.get(key)
    if row is None:
        raise ContractViolation(
            f"{key} is in the capture set and not in the corpus: the two have drifted apart and a "
            "capture attributed to a row that does not exist is worse than no capture."
        )
    absent = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if absent:
        raise ContractViolation(
            f"{key}: the corpus row declares no {absent}. This corpus carries the rendered "
            "`prompt` as a string and no token ids; tokenization belongs to the caller's seams."
        )
    return row


def _existing_cells(manifest_path: Path) -> tuple[dict[tuple[str, int], dict], int]:
    """The manifest as a map, and the highest shard it names. No judgement, only reading."""
    if not manifest_path.exists():
        return {}, -1
    cells: dict[tuple[str, int], dict] = {}
    highest = -1
    for line in manifest_path.read_text().splitlines():
        if line.strip():
            cell = json.loads(line)
            cells[(cell["task_id"], int(cell["step"]))] = cell
            highest = max(highest, int(cell["shard"]))
    return cells, highest


def _assert_reusable(
    cell: dict, decision: dict, row: dict, target: CaptureTarget, prepared: dict | None
) -> None:
    """Refuse an existing capture that is not the capture this run would have made.

    Membership in a manifest is a claim. Every condition below was, at some point, something this
    module took on trust: the checkpoint, the semantic record, the rendered bytes, the shard's
    existence, the shard's contents, the request's own enumeration, and the ids actually consumed.
    Each is refused by name, because a manifest that disagrees with its own files or with the
    request is a fact the operator needs, not one for this function to paper over.
    """
    key = (decision["task_id"], int(decision["step"]))
    if cell.get("checkpoint_sha256") != target.identity["checkpoint_sha256"]:
        raise ResumeUnverified(
            f"{key} was captured under checkpoint {str(cell.get('checkpoint_sha256'))[:12]} and "
            f"this run is {target.identity['checkpoint_sha256'][:12]}. Resuming would mix two "
            "checkpoints' residuals under one manifest. Capture into a new directory."
        )
    if prompt_digest(row["messages"]) != cell.get("messages_sha256"):
        raise ResumeUnverified(
            f"{key}: the corpus row's message digest has moved since it was captured."
        )
    if hashlib.sha256(row["prompt"].encode()).hexdigest() != cell.get("rendered_prompt_sha256"):
        raise ResumeUnverified(
            f"{key}: the corpus row's rendered prompt has moved since it was captured, even though "
            "its message record has not. The existing capture is of different bytes."
        )
    differing = [f for f in ENUMERATION_FIELDS if cell.get(f) != decision.get(f)]
    if differing:
        raise ResumeUnverified(
            f"{key}: the existing capture and the requested decision disagree about {differing}. "
            "The capture set has been re-enumerated since the pass began, and reusing the cell "
            "would attribute it to a decision this request does not make."
        )
    if prepared is not None:
        if prepared["token_ids_sha256"] != cell.get("token_ids_sha256"):
            raise ResumeUnverified(
                f"{key}: this run's tokenizer produces ids digesting to "
                f"{prepared['token_ids_sha256'][:12]} and the existing capture records "
                f"{str(cell.get('token_ids_sha256'))[:12]}. Two tokenizers give different ids for "
                "the same bytes, and the checkpoint identity covers no tokenizer asset."
            )
        if prepared["token_ids_length"] != cell.get("token_ids_length"):
            raise ResumeUnverified(
                f"{key}: this run tokenizes to {prepared['token_ids_length']} ids and the existing "
                f"capture records {cell.get('token_ids_length')}."
            )
    shard_path = target.directory / f"residuals-{int(cell['shard']):05d}.pt"
    if not shard_path.exists():
        raise ResumeUnverified(
            f"{key} names {shard_path.name}, which is not on disk. The manifest describes a "
            "capture nobody has."
        )
    if hashlib.sha256(shard_path.read_bytes()).hexdigest() != cell.get("shard_sha256"):
        raise ResumeUnverified(
            f"{shard_path.name} does not hash to the {str(cell.get('shard_sha256'))[:12]} the "
            "manifest records. Its bytes have changed since it was written."
        )


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
    "ENUMERATION_FIELDS",
    "ResumeUnverified",
    "checkpoint_identity",
    "FORWARD_BATCH",
    "REQUIRED_CELL_FIELDS",
    "REQUIRED_ROW_FIELDS",
    "REQUIRED_SEAM_FIELDS",
    "CaptureTarget",
    "ContractViolation",
    "assert_contract",
    "assert_seam_ran_the_declared_pass",
    "capture_decisions",
    "prompt_digest",
    "rows_by_decision",
]
