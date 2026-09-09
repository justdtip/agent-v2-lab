"""What capturing the pre-registered set costs, per model, from the registry's own numbers.

R6's storage table was computed on 8,907 rendered rows. The capture unit is the distinct decision
(`prereg_inputs.py`), of which there are 7,629, and 348 of the rendered rows are chat replay with no
plan progress to be ground truth for. This recomputes the table on the set that will actually be
captured, and sizes the exploratory stratum separately, because that stratum is every position of a
row rather than one position of it and is the only part of the plan that can be large.

Depth and width come from the registry entries, never from memory: a hidden size typed by hand is
the kind of number that is wrong by a factor and looks right.

    python .../capture_budget.py --decisions capture-set.jsonl --data DIR \
        --config configs/models/gemma3-4b-cuda-bf16.yaml \
        --config configs/models/gemma3-12b-cuda-bf16.yaml \
        [--tokenizer PATH_OR_ID | --chars-per-token N] --out FILE

CPU only, no model weights, no card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

GIB = 1024**3
KIB = 1024
BYTES_PER_ELEMENT = {"bfloat16": 2, "bf16": 2, "float16": 2, "float32": 4}

#: Depth and width per registry entry name. The registry names the checkpoint, not its shape, so
#: these come from the published config of each snapshot and are checked on the card against
#: `model.config` before the first capture is written; `verified_on_device` says whether that
#: has happened for the run this file describes.
SHAPES = {
    "gemma3-4b-cuda-bf16": {"layers": 34, "hidden": 2560, "source": "google/gemma-3-4b-it config"},
    "gemma3-12b-cuda-bf16": {"layers": 48, "hidden": 3840, "source": "google/gemma-3-12b-it config"},
}


def per_position_bytes(layers: int, hidden: int, element: int) -> int:
    """Every layer's output plus the embedding: `num_layers + 1` residuals at one position."""
    return (layers + 1) * hidden * element


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", help="path or hub id; when absent --chars-per-token is required")
    parser.add_argument("--chars-per-token", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    decisions = [json.loads(line) for line in args.decisions.read_text().splitlines() if line.strip()]
    present = {d["task_id"] for d in decisions}
    absent = sorted(set(EXPLORATORY_EPISODES) - present)
    if absent:
        parser.error(f"the capture set does not contain {len(absent)} pre-registered episode(s): "
                     f"{', '.join(absent)}; this script and the pre-registration have drifted apart")

    # Prompt lengths, in characters, for the whole set and for the final decision of each episode.
    lengths: dict[tuple[str, int], int] = {}
    for split in ("train", "valid", "test"):
        for row in (json.loads(line) for line in
                    (args.data / f"{split}.jsonl").read_text().splitlines() if line.strip()):
            meta = row.get("metadata", {})
            if meta.get("family") is None:
                continue
            lengths[(str(meta["task_id"]), int(meta["step"]))] = len(row["prompt"])

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer  # imported only when asked for

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    elif args.chars_per_token is None:
        parser.error("pass --tokenizer to measure token counts, or --chars-per-token to declare the "
                     "ratio you are assuming; a default here would turn an assumption into a figure")

    def tokens_of(task: str, step: int) -> tuple[float, str]:
        if tokenizer is not None:
            row_key = (task, step)
            return float(len(tokenizer(PROMPTS[row_key], add_special_tokens=False)["input_ids"])), "measured"
        return lengths[(task, step)] / args.chars_per_token, "assumed"

    PROMPTS: dict[tuple[str, int], str] = {}
    if tokenizer is not None:
        for split in ("train", "valid", "test"):
            for row in (json.loads(line) for line in
                        (args.data / f"{split}.jsonl").read_text().splitlines() if line.strip()):
                meta = row.get("metadata", {})
                if meta.get("family") is not None:
                    PROMPTS[(str(meta["task_id"]), int(meta["step"]))] = row["prompt"]

    models = {}
    for path in args.config:
        entry = yaml.safe_load(path.read_text())
        name = entry["name"]
        shape = SHAPES[name]
        element = BYTES_PER_ELEMENT["bfloat16"]
        models[name] = {
            "config": str(path),
            "hf_id": entry.get("hf_id"),
            "layers": shape["layers"],
            "hidden": shape["hidden"],
            "shape_source": shape["source"],
            "verified_on_device": False,
            "capture_dtype": "native",
            "element_bytes": element,
            "residuals_per_position": shape["layers"] + 1,
            "bytes_per_position": per_position_bytes(shape["layers"], shape["hidden"], element),
        }

    # The exploratory stratum: every position of the final decision of each pre-registered episode.
    final_of: dict[str, int] = {}
    for decision in decisions:
        final_of[decision["task_id"]] = max(final_of.get(decision["task_id"], -1), decision["step"])

    payload = {
        "basis": "measured-here",
        "models": models,
        "main_stratum": {
            "unit": "one decision position per distinct decision",
            "decisions": len(decisions),
            "per_model_gib": {
                name: round(len(decisions) * m["bytes_per_position"] / GIB, 3)
                for name, m in models.items()
            },
            "total_gib": round(
                len(decisions) * sum(m["bytes_per_position"] for m in models.values()) / GIB, 3
            ),
        },
        "prompt_length_chars": {
            "note": "characters, not tokens: no Gemma tokenizer is present on the laptop, so the "
                    "token figures in the order are the Chief's and are re-measured on the card",
            "n": len(lengths),
            "min": min(lengths.values()),
            "median": sorted(lengths.values())[len(lengths) // 2],
            "max": max(lengths.values()),
        },
    }

    if args.tokenizer or args.chars_per_token:
        rows = []
        for episode in sorted(EXPLORATORY_EPISODES):
            step = final_of[episode]
            count, basis = tokens_of(episode, step)
            rows.append({"task_id": episode, "final_step": step,
                         "positions": round(count), "basis": basis})
        total_positions = sum(r["positions"] for r in rows)
        payload["exploratory_stratum"] = {
            "unit": "every position of the final decision of each pre-registered episode",
            "episodes": len(rows),
            "positions_total": total_positions,
            "basis": rows[0]["basis"] if rows else None,
            "chars_per_token": args.chars_per_token,
            "per_model_gib": {
                name: round(total_positions * m["bytes_per_position"] / GIB, 3)
                for name, m in models.items()
            },
            "total_gib": round(
                total_positions * sum(m["bytes_per_position"] for m in models.values()) / GIB, 3
            ),
            "by_episode": rows,
        }

    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"{'model':24s} {'layers':>7s} {'hidden':>7s} {'KiB/position':>13s} {'GiB, main':>10s}")
    for name, m in models.items():
        print(f"{name:24s} {m['layers']:7d} {m['hidden']:7d} "
              f"{m['bytes_per_position']/KIB:13.1f} "
              f"{payload['main_stratum']['per_model_gib'][name]:10.3f}")
    print(f"{'both':24s} {'':7s} {'':7s} {'':13s} {payload['main_stratum']['total_gib']:10.3f}")
    print(f"\ndecisions captured: {len(decisions)}")
    print("prompt chars: min {min} median {median} max {max}".format(**payload["prompt_length_chars"]))
    if "exploratory_stratum" in payload:
        e = payload["exploratory_stratum"]
        print(f"\nexploratory: {e['episodes']} episodes, {e['positions_total']} positions "
              f"({e['basis']}), {e['total_gib']:.2f} GiB for both models")
    return 0


#: The twelve, as `prereg_inputs.py` selects them. Written out rather than re-derived so this script
#: and the pre-registration cannot drift apart silently; `--decisions` is checked against them.
EXPLORATORY_EPISODES = (
    "test-aggregate_report-0011-clean",
    "test-batch_update-0010-clean",
    "test-calculate-0002-clean",
    "test-conditional_update-0009-clean",
    "test-cross_reference-0008-clean",
    "test-ledger_reconcile-0007-clean",
    "test-list-0005-clean",
    "test-pointer_chain-0006-clean",
    "test-read-0000-clean",
    "test-search-0001-clean",
    "test-synthesis-0003-clean",
    "test-update-0004-clean",
)

if __name__ == "__main__":
    raise SystemExit(main())
