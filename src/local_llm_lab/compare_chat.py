from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from local_llm_lab.agent_protocol import SYSTEM_PROMPT, TOOL_SPECS
from local_llm_lab.evaluate_agent import DEFAULT_MODEL
from local_llm_lab.project import PROJECT_ROOT, configure_local_cache

CHAT_SYSTEM_PROMPT = (
    "You are a concise, capable assistant. Be honest about uncertainty and follow "
    "the user's request."
)


def _reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    sampler: Any,
    max_tokens: int,
    with_tools: bool,
) -> str:
    from mlx_lm import generate

    template_args: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if with_tools:
        template_args["tools"] = TOOL_SPECS
    prompt = tokenizer.apply_chat_template(messages, **template_args)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=sampler,
        max_tokens=max_tokens,
        verbose=False,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with base and adapted 3B models side by side."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agent-3b" / "best-adapter",
    )
    parser.add_argument("--prompt", help="Run one comparison instead of an interactive session.")
    parser.add_argument("--tools", action="store_true", help="Expose the training tool schemas.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    adapter = args.adapter.resolve()
    if not (adapter / "adapters.safetensors").is_file():
        raise SystemExit(f"Trained adapter not found: {adapter}")
    configure_local_cache()

    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    print("Loading untouched base model...")
    base_model, tokenizer = load(args.model)
    print("Loading the same base with the trained adapter...")
    tuned_model, _ = load(args.model, adapter_path=str(adapter))
    sampler = make_sampler(temp=args.temperature)
    system = SYSTEM_PROMPT if args.tools else CHAT_SYSTEM_PROMPT
    base_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    tuned_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    prompt = args.prompt
    while True:
        if prompt is None:
            try:
                prompt = input("\nYou (or 'exit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not prompt or prompt.casefold() in {"exit", "quit"}:
            break

        base_messages.append({"role": "user", "content": prompt})
        tuned_messages.append({"role": "user", "content": prompt})
        base_reply = _reply(
            base_model, tokenizer, base_messages, sampler, args.max_tokens, args.tools
        )
        tuned_reply = _reply(
            tuned_model, tokenizer, tuned_messages, sampler, args.max_tokens, args.tools
        )
        base_messages.append({"role": "assistant", "content": base_reply})
        tuned_messages.append({"role": "assistant", "content": tuned_reply})
        print(f"\nBASE:\n{base_reply}")
        print(f"\nAGENT ADAPTER:\n{tuned_reply}")

        if args.prompt is not None:
            break
        prompt = None
