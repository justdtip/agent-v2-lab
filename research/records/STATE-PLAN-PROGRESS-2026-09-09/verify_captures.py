"""The first permitted reading: the captures against the sealed capture set, and nothing else.

This reads the manifests. It does not read a residual, so no estimand is touched and no result can
be shaped by it. It runs through `read_gate.require_seal`, which is the refusal §11's precondition 5
asks for: without the seal there is no reading, including this one.

What it checks, each separately because each can fail alone:

* the seal is present and still true of the record (the gate does this and names what it verified);
* every sealed decision was captured, and nothing was captured that the seal did not enumerate;
* each cell's semantic record matches the sealed row's `messages_sha256` — the capture is of the
  decision the pre-registration enumerated, not merely of a decision;
* the contract fields §2 requires are present and carry the declared values, read from the cell
  rather than written into it;
* the identity fields are constant within a model, so no cell came from another checkpoint;
* the two models were given the same input at every decision, re-derived here from the consumed
  ids rather than cited from the record that first measured it;
* the shards on disk still digest to what the cells say they do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from local_llm_lab.pipeline.state_programme.read_gate import require_seal  # noqa: E402

#: Fields the contract fixes and their required values (§2). Read from the cell, never supplied.
DECLARED = {"forward_batch": 1, "anchor_batch": 1, "basis": "measured-here",
            "decoding": "teacher-forced", "capture_dtype": "native"}
#: Fields that identify the checkpoint and the geometry: constant across every cell of one model.
CONSTANT = ("checkpoint_sha256", "config_sha256", "layers", "d_model", "dtype", "device", "weight_files")
#: Fields without which a cell cannot be checked at all.
REQUIRED = ("task_id", "step", "messages_sha256", "rendered_prompt_sha256", "token_ids_sha256",
            "token_ids_length", "token_index", "seq_len", "shard", "shard_sha256")


def read_manifest(path: Path) -> tuple[dict, list[str]]:
    cells, problems, seen = {}, [], Counter()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        cell = json.loads(line)
        missing = [f for f in REQUIRED if f not in cell]
        if missing:
            problems.append(f"{path.name}:{number} is missing {missing}")
            continue
        key = (cell["task_id"], cell["step"])
        seen[key] += 1
        if seen[key] > 1:
            problems.append(f"{path.name}:{number} repeats the decision {key}")
            continue
        cells[key] = cell
    return cells, problems


def check_model(name: str, directory: Path, sealed: dict) -> tuple[dict, list[str]]:
    cells, problems = read_manifest(directory / "manifest.jsonl")

    missing = sorted(set(sealed) - set(cells))
    extra = sorted(set(cells) - set(sealed))
    if missing:
        problems.append(f"{name}: {len(missing)} sealed decision(s) never captured, first {missing[0]}")
    if extra:
        problems.append(f"{name}: {len(extra)} captured decision(s) the seal does not enumerate, first {extra[0]}")

    for key in sorted(set(sealed) & set(cells)):
        cell, row = cells[key], sealed[key]
        if cell["messages_sha256"] != row["messages_sha256"]:
            problems.append(f"{name} {key}: the captured decision is not the enumerated one")
        for field, value in DECLARED.items():
            if cell.get(field) != value:
                problems.append(f"{name} {key}: {field} is {cell.get(field)!r}, the contract declares {value!r}")
        if cell["token_index"] != cell["seq_len"] - 1:
            problems.append(f"{name} {key}: token_index {cell['token_index']} is not the last of {cell['seq_len']}")
        if cell["token_ids_length"] != cell["seq_len"]:
            problems.append(f"{name} {key}: token_ids_length disagrees with seq_len")

    for field in CONSTANT:
        values = {cell.get(field) for cell in cells.values()}
        if len(values) > 1:
            problems.append(f"{name}: {field} is not constant across cells: {sorted(map(str, values))[:4]}")

    return cells, problems


def check_shards(name: str, directory: Path, cells: dict) -> list[str]:
    expected: dict[int, str] = {}
    for cell in cells.values():
        expected.setdefault(cell["shard"], cell["shard_sha256"])
    problems = []
    for shard, want in sorted(expected.items()):
        path = directory / f"residuals-{shard:05d}.pt"
        if not path.exists():
            problems.append(f"{name}: shard {shard} is missing at {path}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            problems.append(f"{name}: shard {shard} digests to {got[:12]}…, the cells say {want[:12]}…")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=HERE)
    parser.add_argument("--captures", type=Path, required=True, help="directory holding the per-model capture dirs")
    parser.add_argument("--expect-seal", help="refuse unless the seal digests to this")
    parser.add_argument("--skip-shards", action="store_true", help="report the shard bytes as unchecked, not as passing")
    args = parser.parse_args(argv)

    seal = require_seal(args.record, expected_digest=args.expect_seal)
    print("seal verified:")
    for key, value in seal["verified"].items():
        print(f"  {key:<14} {value}")

    sealed = {}
    for line in (args.record / "capture-set.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            sealed[(row["task_id"], row["step"])] = row
    print(f"\nsealed capture set: {len(sealed)} decisions")

    problems, per_model = [], {}
    for name in sorted(p.name for p in args.captures.iterdir() if p.is_dir()):
        directory = args.captures / name
        if not (directory / "manifest.jsonl").exists():
            continue
        cells, found = check_model(name, directory, sealed)
        per_model[name] = cells
        problems += found
        print(f"  {name:<26} {len(cells)} cells")
        if args.skip_shards:
            print(f"  {'':<26} shard bytes NOT checked (--skip-shards)")
        else:
            problems += check_shards(name, directory, cells)

    names = sorted(per_model)
    if len(names) == 2:
        a, b = (per_model[n] for n in names)
        shared = sorted(set(a) & set(b))
        same_ids = sum(1 for k in shared if a[k]["token_ids_sha256"] == b[k]["token_ids_sha256"])
        same_pos = sum(1 for k in shared if a[k]["token_index"] == b[k]["token_index"])
        print(f"\nprecondition 2, re-derived here from the consumed ids:")
        print(f"  identical ids       {same_ids} of {len(shared)}")
        print(f"  identical position  {same_pos} of {len(shared)}")
        if same_ids != len(shared) or same_pos != len(shared):
            problems.append(f"the two models were not given the same input at every decision")

    if problems:
        print(f"\nREFUSED, {len(problems)} problem(s):")
        for problem in problems[:40]:
            print(f"  - {problem}")
        if len(problems) > 40:
            print(f"  … and {len(problems) - 40} more")
        return 1
    print("\nthe captures are the sealed capture set, and nothing was read but the manifests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
