"""One renderer, used by extraction, data generation, training and evaluation.

Three separate renderers is how the injection site drifts. The site is an absolute token index
recorded at generation time and applied months later inside a training forward; if the string that
produced it was built by a different code path than the one the trainer uses, the index points
somewhere else and nothing says so. So: one function, and everything calls it.
"""

from __future__ import annotations

from typing import Any


def apply_template(tokenizer: Any, messages: list[dict], *, generation_prompt: bool,
                   thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=generation_prompt, tokenize=False,
            enable_thinking=thinking)
    except TypeError:                      # a template that takes no such flag
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=generation_prompt, tokenize=False)


def render_prompt(tokenizer: Any, text: str, *, thinking: bool = False,
                  system: str | None = None) -> list[int]:
    """The ids a user turn becomes, up to and including the generation prompt."""
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": text}]
    rendered = apply_template(tokenizer, messages, generation_prompt=True, thinking=thinking)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if bos is not None and ids.count(bos) > 1:
        raise ValueError(f"{ids.count(bos)} BOS tokens in one render")
    return ids


def render_supervised(tokenizer: Any, text: str, target: str, *, thinking: bool = False,
                      system: str | None = None) -> tuple[list[int], int]:
    """(input_ids, prompt_length). Everything from prompt_length on is supervised.

    The target is tokenised separately and concatenated rather than rendered as an assistant turn,
    so `prompt_length` is exact by construction. Rendering the pair and then searching for the
    boundary is where an off-by-one becomes a systematic over-supervision.
    """
    prompt = render_prompt(tokenizer, text, thinking=thinking, system=system)
    body = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not body:
        raise ValueError("the target tokenised to nothing")
    return prompt + body, len(prompt)


def final_prompt_position(prompt_ids: list[int]) -> int:
    """The site for `scope="final_token"`: the last prompt token, never inside the answer."""
    return len(prompt_ids) - 1
