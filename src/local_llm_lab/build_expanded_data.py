from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from local_llm_lab.project import PROJECT_ROOT

DECISION_REPEATS = {
    "set_plan": 8,
    "update_plan": 5,
    "finish": 3,
    "calculate": 4,
    "search_files": 3,
    "replace_text": 2,
    "list_files": 2,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_action(row: dict[str, Any]) -> str | None:
    """Return the supervised tool name, or None for ordinary chat targets."""
    messages = row.get("messages", [])
    if not messages or messages[-1].get("role") != "assistant":
        return None
    calls = messages[-1].get("tool_calls", [])
    if len(calls) != 1:
        return None
    return calls[0].get("function", {}).get("name")


def decision_balanced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Oversample scarce high-level decisions without altering held-out distributions."""
    return [row for row in rows for _ in range(DECISION_REPEATS.get(target_action(row), 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix complex, replay, and chat SFT data.")
    parser.add_argument("--complex", type=Path, default=PROJECT_ROOT / "data" / "complex_agent_sft")
    parser.add_argument("--original", type=Path, default=PROJECT_ROOT / "data" / "agent_sft")
    parser.add_argument("--chat", type=Path, default=PROJECT_ROOT / "data" / "chat_replay")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "expanded_agent_sft")
    parser.add_argument("--chat-repeats", type=int, default=2)
    parser.add_argument(
        "--decision-balanced",
        action="store_true",
        help="Oversample scarce complex decisions in the training split only.",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.chat_repeats < 1:
        parser.error("chat-repeats must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "chat_repeats": args.chat_repeats,
        "decision_balanced": args.decision_balanced,
        "decision_repeats": DECISION_REPEATS if args.decision_balanced else {},
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        complex_rows = _read_jsonl(args.complex.resolve() / f"{split}.jsonl")
        original_rows = _read_jsonl(args.original.resolve() / f"{split}.jsonl")
        chat_rows = _read_jsonl(args.chat.resolve() / f"{split}.jsonl")
        weighted_complex = (
            decision_balanced(complex_rows)
            if args.decision_balanced and split == "train"
            else complex_rows
        )
        rows = [*weighted_complex, *original_rows, *(chat_rows * args.chat_repeats)]
        random.Random(f"{args.seed}:{split}").shuffle(rows)
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["splits"][split] = {
            "rows": len(rows),
            "complex_rows": len(complex_rows),
            "complex_rows_after_balancing": len(weighted_complex),
            "original_agent_rows": len(original_rows),
            "chat_rows_after_replay": len(chat_rows) * args.chat_repeats,
            "sha256": _sha256(path),
        }
        print(
            f"{split}: {len(rows)} rows = {len(weighted_complex)} weighted complex + "
            f"{len(original_rows)} original + {len(chat_rows) * args.chat_repeats} chat",
            flush=True,
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote mixed expansion data to {output}", flush=True)
