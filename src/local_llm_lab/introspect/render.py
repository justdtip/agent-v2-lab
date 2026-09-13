"""One renderer, used by extraction, data generation, training and evaluation.

Three separate renderers is how the injection site drifts. The site is an absolute token index
recorded at generation time and applied months later inside a training forward; if the string that
produced it was built by a different code path than the one the trainer uses, the index points
somewhere else and nothing says so. So: one function, and everything calls it.

Two things about Gemma 4's template that are not what the argument names suggest, both measured on
the real tokenizer (2026-09-13):

`enable_thinking=False` does not remove a thinking channel, it OPENS one. The generation prompt
ends `<|turn>model\n<|channel>thought\n<channel|>`, so a supervised target concatenated onto it is
trained as thought-channel content, while the canonical rendering of a finished assistant turn
(`<|turn>model\nYES. ok.<turn|>`) carries no channel header at all. With `enable_thinking=True` the
generation prompt instead ends at `<|turn>model\n` and a `<|think|>` marker is added to the system
turn. We use False everywhere, so training and evaluation agree with each other; they do not agree
with the canonical assistant turn, and that is a deliberate, recorded choice rather than an
oversight.

A system turn shifts the injection site. `EXPERIMENT_SYSTEM` is 114 tokens on this tokenizer, so
the site moves from 13 to 132 on a one-word user turn and from 95 to 214 on the real detect prompt.
Every scale and every recorded site is therefore specific to whether a system prompt was present,
and a bank built without one cannot be reused with one without re-measuring.
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


def turn_end_id(tokenizer: Any) -> int:
    """The id that ends an assistant turn, verified by round-trip rather than by name.

    On Gemma 4 this is `<turn|>` (106). `<end_of_turn>` is NOT in this vocabulary and resolves to
    `<unk>` (3), which is why the name is checked back rather than trusted.
    """
    for name in ("<turn|>", "<end_of_turn>", "<|im_end|>"):
        got = tokenizer.convert_tokens_to_ids(name)
        if isinstance(got, int) and got >= 0 and got != tokenizer.unk_token_id \
                and tokenizer.convert_ids_to_tokens(got) == name:
            return got
    if tokenizer.eos_token_id is None:
        raise ValueError("no end-of-turn token and no eos: the model cannot be taught to stop")
    return int(tokenizer.eos_token_id)


def render_supervised(tokenizer: Any, text: str, target: str, *, thinking: bool = False,
                      system: str | None = None,
                      stop: bool = True) -> tuple[list[int], int]:
    """(input_ids, prompt_length). Everything from prompt_length on is supervised.

    The target is tokenised separately and concatenated rather than rendered as an assistant turn,
    so `prompt_length` is exact by construction. Rendering the pair and then searching for the
    boundary is where an off-by-one becomes a systematic over-supervision.

    `stop` appends the end-of-turn token to the supervised span. Without it nothing in 12,000 rows
    teaches the model to finish: the 2026-09-13 adapter answered and then repeated its own answer
    until the decoder's token cap, and the tail of every reply was noise the scorer had to ignore.
    """
    prompt = render_prompt(tokenizer, text, thinking=thinking, system=system)
    body = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not body:
        raise ValueError("the target tokenised to nothing")
    if stop:
        body = body + [turn_end_id(tokenizer)]
    return prompt + body, len(prompt)


def final_prompt_position(prompt_ids: list[int]) -> int:
    """The site for `scope="final_token"`: the last prompt token, never inside the answer."""
    return len(prompt_ids) - 1
