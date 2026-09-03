from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from local_llm_lab.project import PROJECT_ROOT, configure_local_cache
from local_llm_lab.rewards import combined_reward

DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
SYSTEM_PROMPT = (
    "Solve the problem carefully. Finish with exactly one line in the form FINAL: <answer>."
)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row: Any = json.loads(line)
            valid = (
                isinstance(row, dict)
                and isinstance(row.get("prompt"), str)
                and isinstance(row.get("answer"), str)
            )
            if not valid:
                raise ValueError(
                    f"{path}:{line_number} must contain string prompt and answer fields"
                )
            rows.append({"prompt": row["prompt"], "answer": row["answer"]})
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def _preview(rows: list[dict[str, str]], model: str, steps: int) -> None:
    print(f"Model:       {model}")
    print(f"Rows:        {len(rows)}")
    print(f"Steps:       {steps}")
    print("Generations: 4 per prompt")
    print("Output:      outputs/grpo")
    print("Reward checks:")
    answer = rows[0]["answer"]
    for response in (f"FINAL: {answer}", answer, "FINAL: definitely-wrong"):
        print(f"  {combined_reward(response, answer):.2f}  {response!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview or run a small native-MLX GRPO experiment."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "rewards" / "train.jsonl",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--run", action="store_true", help="Start training; otherwise only preview."
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset does not exist: {data_path}")
    rows = load_jsonl(data_path)
    _preview(rows, args.model, args.steps)
    if not args.run:
        print("Preview only. Add --run to download/load the model and train.")
        return

    cache = configure_local_cache()
    print(f"Model cache: {cache}")

    from mlx_tune import FastLanguageModel, GRPOConfig, GRPOTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=512,
        load_in_4bit=True,
    )
    # QLoRA: keep the quantized base frozen and train only these low-rank adapters.
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
    )

    training_rows = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        training_rows.append({"prompt": prompt, "answer": row["answer"]})

    trainer = GRPOTrainer(
        model=model,
        train_dataset=training_rows,
        tokenizer=tokenizer,
        reward_fn=combined_reward,
        args=GRPOConfig(
            loss_type="grpo",
            beta=0.04,
            num_generations=4,
            temperature=0.9,
            max_completion_length=64,
            learning_rate=5e-6,
            max_steps=args.steps,
            logging_steps=1,
            output_dir=str(PROJECT_ROOT / "outputs" / "grpo"),
        ),
    )
    trainer.train()
    model.save_pretrained(str(PROJECT_ROOT / "outputs" / "grpo" / "adapters"))
