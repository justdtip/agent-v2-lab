from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_llm_lab.compare_chat import CHAT_SYSTEM_PROMPT
from local_llm_lab.evaluate_agent import DEFAULT_MODEL
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache


@dataclass(frozen=True)
class ChatPrompt:
    prompt_id: str
    category: str
    messages: list[dict[str, str]]


def make_chat_prompts(split: str, count: int, seed: int = 20260902) -> list[ChatPrompt]:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if count < 1:
        raise ValueError("count must be positive")
    concepts = (
        "caching",
        "unit testing",
        "compound interest",
        "photosynthesis",
        "database indexes",
        "probability",
        "encryption",
        "feedback loops",
    )
    projects = (
        "organizing a community workshop",
        "learning basic Python",
        "migrating a small website",
        "preparing a research interview",
        "debugging an intermittent service",
        "writing a technical tutorial",
    )
    comparisons = (
        ("lists", "sets"),
        ("unit tests", "integration tests"),
        ("SQL", "NoSQL"),
        ("breadth-first search", "depth-first search"),
        ("precision", "recall"),
        ("threads", "processes"),
    )
    raw_sentences = (
        "we tested the change and it worked but the notes are scattered",
        "the meeting covered risks milestones owners and next steps",
        "this function is slow because it repeats the same expensive lookup",
        "the draft is accurate although it is longer than it needs to be",
    )
    categories = ("explain", "plan", "compare", "rewrite", "code", "follow_up")
    prompts = []
    for index in range(count):
        rng = random.Random(f"chat:{seed}:{split}:{index}")
        category = categories[index % len(categories)]
        if category == "explain":
            concept = rng.choice(concepts)
            user = (
                f"Explain {concept} to a curious beginner in two short paragraphs, then give one "
                "concrete example."
            )
            messages = [{"role": "user", "content": user}]
        elif category == "plan":
            project = rng.choice(projects)
            user = (
                f"Create a practical numbered plan for {project}. Include dependencies, one risk, "
                "and a final verification checkpoint."
            )
            messages = [{"role": "user", "content": user}]
        elif category == "compare":
            left, right = rng.choice(comparisons)
            user = (
                f"Compare {left} and {right}. Give two differences, one similarity, and a concise "
                "rule of thumb for choosing between them."
            )
            messages = [{"role": "user", "content": user}]
        elif category == "rewrite":
            sentence = rng.choice(raw_sentences)
            user = f"Rewrite this as a clear professional update without adding facts: {sentence!r}"
            messages = [{"role": "user", "content": user}]
        elif category == "code":
            limit = rng.randrange(5, 20)
            user = (
                "Write a small typed Python function that returns the even integers from zero "
                f"through {limit}. Include a one-sentence explanation and one example call."
            )
            messages = [{"role": "user", "content": user}]
        else:
            topic = rng.choice(projects)
            messages = [
                {"role": "user", "content": f"Help me think through {topic}."},
                {
                    "role": "assistant",
                    "content": (
                        "Start by clarifying the desired outcome, constraints, and deadline."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Assume the deadline is two weeks and the team has three people. Turn that "
                        "into a concise checklist with owners represented as A, B, and C."
                    ),
                },
            ]
        messages[-1]["content"] += f"\nScenario identifier: {split}-{index:04d}."
        prompts.append(
            ChatPrompt(
                prompt_id=f"chat-{split}-{category}-{index:04d}",
                category=category,
                messages=messages,
            )
        )
    return prompts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chat-retention targets from the pre-expansion policy."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agent-3b" / "best-adapter",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "chat_replay")
    parser.add_argument("--train-prompts", type=int, default=240)
    parser.add_argument("--valid-prompts", type=int, default=48)
    parser.add_argument("--test-prompts", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    counts = {
        "train": args.train_prompts,
        "valid": args.valid_prompts,
        "test": args.test_prompts,
    }
    if any(value < 1 for value in (*counts.values(), args.max_tokens)):
        parser.error("split sizes and max-tokens must be positive")

    configure_local_cache()
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model, adapter_path=str(args.adapter.resolve()))
    sampler = make_sampler(temp=0.0)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "teacher_model": args.model,
        "teacher_adapter": str(args.adapter.resolve()),
        "seed": 20260902,
        "splits": {},
    }
    for split, count in counts.items():
        rows = []
        for index, item in enumerate(make_chat_prompts(split, count), 1):
            context = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *item.messages]
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False
            )
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                sampler=sampler,
                max_tokens=args.max_tokens,
                verbose=False,
            ).strip()
            rows.append(
                {
                    "messages": [*context, {"role": "assistant", "content": response}],
                    "metadata": {
                        "prompt_id": item.prompt_id,
                        "category": item.category,
                        "source": "pre-expansion-policy-replay",
                    },
                }
            )
            if index % 20 == 0 or index == count:
                print(f"{split}: generated {index}/{count}", flush=True)
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["splits"][split] = {
            "rows": len(rows),
            "sha256": _sha256(path),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote chat replay data to {output}", flush=True)
