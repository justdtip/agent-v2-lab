"""A chat with a steering desk attached, and an instrument for watching the steering work.

Plain text is a message to the model. Anything starting with `/` is a command. The steering state
persists between turns and is printed on the prompt line, so what is currently being done to the
model is in front of you on every line you type.

WATCHING THE STEERING, WHICH IS THE POINT. Seeing the reply change shows you the *consequence*. To
see the *intervention*, `/ab` runs the same message twice on the same history — once with the desk
muted and once as set — and `/why` replays the steered reply through the unsteered model, forcing the
same tokens, and reports for each one what the clean model would have said instead. Because the
tokens are forced rather than resampled, the two runs visit an identical context at every position,
so the difference between them is the intervention and nothing else. A pair of free-running loops
would diverge after the first differing token and compare two different conversations.

THE FADER IS IN PER CENT OF THE RESIDUAL NORM, not in alpha. That is the number that predicts
behaviour across models, layers and prompts: on Gemma 3 12B at layer 32, a concept at 121% of the
residual flips the reply into Bengali while the same direction at 61% does nothing, and a *sustained*
injection collapses the model at 27%. Alpha means nothing without the vector's norm beside it, and
the norm is re-measured at the injection site on every turn rather than assumed.

SCOPE REPLACES `sustain`. A conversation has four cases and they behave differently:

    turn    inject over your message, release before the first generated token   (the default)
    reply   inject only while the model is generating
    both    inject over both, which is the paper's protocol and the destructive one
    all     inject over the whole context including earlier turns

HISTORY IS RE-PREFILLED EVERY TURN rather than carried in a cache. The render is prefix-monotone so a
cache *could* be carried, but a cache that was live during an earlier steered turn holds steered keys
and values forever, which makes later "clean" turns quietly not clean. Re-prefilling costs a fraction
of a second on a conversation this size and buys `/again`, `/undo` and an honest `/ab`.

    python scripts/steer_chat.py --model /path/to/snapshot --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

from inject_repl import BASELINE_WORDS, Console, Injection  # noqa: E402


class Desk:
    """One steering channel: a slot, a layer, a fader in per cent of residual norm, and a scope."""

    def __init__(self):
        self.slot = None
        self.layer = None
        self.percent = 40.0
        self.scope = "turn"

    @property
    def live(self) -> bool:
        return self.slot is not None

    def label(self) -> str:
        if not self.live:
            return "clean"
        return f"{self.slot} {self.percent:g}% L{self.layer} {self.scope}"


class Chat(Console):
    """The console's machinery, with a conversation on top of it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: list[dict] = []
        self.desk = Desk()
        self.log_path = None

    # -- rendering -----------------------------------------------------------------------------
    def render_chat(self, messages: list[dict]) -> list[int]:
        """The whole conversation with a generation prompt, one BOS, tokenized as the model expects."""
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and ids.count(bos) > 1:
            raise ValueError(f"{ids.count(bos)} BOS tokens in one render; the template and the "
                             "tokenizer are both adding one")
        return ids

    def turn_span(self, messages: list[dict]) -> tuple[int, int]:
        """Where the newest user turn starts, so `scope=turn` can inject over it and nothing else."""
        if len(messages) == 1:
            before = 0
        else:
            prior = self.tokenizer.apply_chat_template(
                messages[:-1], add_generation_prompt=False, tokenize=False)
            before = len(self.tokenizer(prior, add_special_tokens=False)["input_ids"])
        return before, len(self.render_chat(messages))

    # -- the intervention ----------------------------------------------------------------------
    def build_injection(self, ids: list[int], span: tuple[int, int]):
        """Translate the desk into a hook, with the fader read as per cent of the residual norm."""
        if not self.desk.live:
            return None, "clean"
        entry = self.slots[self.desk.slot]
        layer = self.desk.layer
        start, end = span
        site = end - 1 if self.desk.scope in {"turn", "both"} else end - 1
        if self.desk.scope == "all":
            start = 0
        here = float(self.residual_at(ids, layer, site).norm())
        scale = (self.desk.percent / 100.0) * here / float(entry["vector"].norm())
        sustain = self.desk.scope in {"reply", "both"}
        from_position = end - 1 if self.desk.scope == "reply" else start
        hook = Injection(self.view.blocks[layer - 1], entry["vector"], scale=scale,
                         from_position=from_position, sustain=sustain)
        note = (f"{self.desk.slot} {self.desk.percent:g}% of |h|={here:,.0f} at L{layer}, "
                f"scope {self.desk.scope}, from {from_position}")
        return hook, note

    def speak(self, messages: list[dict], max_tokens: int, *, steered: bool = True):
        """One reply. Returns (text, token ids, prompt ids, note)."""
        ids = self.render_chat(messages)
        span = self.turn_span(messages)
        hook, note = self.build_injection(ids, span) if steered else (None, "clean")
        started = time.time()
        if hook is None:
            emitted = self.generate_tokens(ids, max_tokens)
        else:
            with hook:
                emitted = self.generate_tokens(ids, max_tokens)
        return (self.tokenizer.decode(emitted), emitted, ids, note, time.time() - started)

    def generate_tokens(self, ids: list[int], max_tokens: int) -> list[int]:
        from transformers import DynamicCache

        cache = DynamicCache()
        emitted = []
        with torch.no_grad():
            logits = self.model(input_ids=self.view._ids(list(ids)),
                                past_key_values=cache, use_cache=True).logits
            for _ in range(max_tokens):
                token = int(logits[0, -1].float().argmax())
                if token in self.stop_ids:
                    break
                emitted.append(token)
                logits = self.model(input_ids=self.view._ids([token]),
                                    past_key_values=cache, use_cache=True).logits
        return emitted

    # -- the instrument ------------------------------------------------------------------------
    def clean_replay(self, prompt_ids: list[int], emitted: list[int]) -> list[dict]:
        """What the unsteered model would have said at each context the steered run visited.

        The steered tokens are forced rather than resampled, so both runs see an identical context
        at every position. A free-running clean loop would diverge after the first differing token
        and would be answering a different question.
        """
        from transformers import DynamicCache

        rows, cache = [], DynamicCache()
        with torch.no_grad():
            logits = self.model(input_ids=self.view._ids(list(prompt_ids)),
                                past_key_values=cache, use_cache=True).logits
            for index, token in enumerate(emitted):
                step = logits[0, -1].float().log_softmax(-1)
                order = step.argsort(descending=True)
                rank = int((order == token).nonzero()[0])
                rows.append({
                    "position": index,
                    "steered_token": self.tokenizer.decode([token]),
                    "clean_top": self.tokenizer.decode([int(order[0])]),
                    "clean_rank_of_steered": rank,
                    "logprob_steered_token": float(step[token]),
                    "logprob_clean_top": float(step[int(order[0])]),
                })
                logits = self.model(input_ids=self.view._ids([token]),
                                    past_key_values=cache, use_cache=True).logits
        return rows

    # -- commands ------------------------------------------------------------------------------
    def do_message(self, text: str, max_tokens: int, *, commit: bool = True) -> None:
        messages = self.history + [{"role": "user", "content": text}]
        reply, emitted, _ids, note, seconds = self.speak(messages, max_tokens)
        print(f"\n{reply.strip() or '(nothing)'}")
        mark = "·" if not self.desk.live else "↯"
        print(f"    {mark} {note} · {len(emitted)} tok · {seconds:.1f}s")
        if commit:
            self.history = messages + [{"role": "assistant", "content": reply}]
            self._log({"kind": "turn", "user": text, "reply": reply, "desk": self.desk.label(),
                       "note": note, "tokens": len(emitted)})

    def do_ab(self, text: str | None, max_tokens: int) -> None:
        """The same message, clean and steered, on the SAME history.

        With no argument this re-runs the last user turn — and it must roll the history back past
        that exchange first. Running the clean arm on a history that already contains the steered
        reply compares a clean continuation of a steered conversation against a steered one, which
        is not a control: the model is reacting to its own steered output in both arms.
        """
        base = list(self.history)
        if text is None:
            for index in range(len(base) - 1, -1, -1):
                if base[index]["role"] == "user":
                    text, base = base[index]["content"], base[:index]
                    break
        if text is None:
            print("nothing to re-run"); return
        messages = base + [{"role": "user", "content": text}]
        clean, _ce, _ci, _cn, _cs = self.speak(messages, max_tokens, steered=False)
        steered, _se, _si, note, _ss = self.speak(messages, max_tokens, steered=True)
        print(f"\n  on {len(base)} message(s) of history, both arms identical up to the desk")
        print(f"\n  A  clean\n     {clean.strip() or '(nothing)'}")
        print(f"\n  B  {note}\n     {steered.strip() or '(nothing)'}")
        same = clean.strip() == steered.strip()
        print(f"\n    {'identical — the desk changed nothing' if same else 'they differ'}")

    def do_why(self, max_tokens: int, show: int) -> None:
        """Replay the last steered reply through the clean model, token by token."""
        if not self.history or self.history[-1]["role"] != "assistant":
            print("no reply to explain"); return
        messages = self.history[:-1]
        ids = self.render_chat(messages)
        emitted = self.tokenizer(self.history[-1]["content"], add_special_tokens=False)["input_ids"]
        if not emitted:
            print("the last reply has no tokens"); return
        rows = self.clean_replay(ids, emitted)
        changed = [r for r in rows if r["clean_rank_of_steered"] != 0]
        print(f"\n  {len(changed)} of {len(rows)} tokens were not the clean model's first choice")
        print(f"  {'pos':>4} {'steered':<14} {'clean top':<14} {'rank':>5}  {'Δ logprob':>9}")
        for row in changed[:show]:
            delta = row["logprob_steered_token"] - row["logprob_clean_top"]
            print(f"  {row['position']:>4} {row['steered_token']!r:<14} {row['clean_top']!r:<14} "
                  f"{row['clean_rank_of_steered']:>5}  {delta:>9.2f}")
        if len(changed) > show:
            print(f"  ... and {len(changed) - show} more")

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        with open(self.log_path, "a") as handle:
            handle.write(json.dumps({"t": time.time(), **record}) + "\n")

    def do_desk(self) -> None:
        print(f"  channel   {self.desk.label()}")
        if self.desk.live:
            entry = self.slots[self.desk.slot]
            print(f"  vector    {self.desk.slot}: {entry['kind']}, norm {entry['norm']:,.0f}")
        print(f"  history   {len(self.history)} message(s)")
        print(f"  log       {self.log_path or '(none)'}")


HELP = """
  <text>                 say it to the model
  /on <slot> [n] [@L]    light a channel: fader n% of residual norm, layer L
  /off                   mute
  /+ [n]  /- [n]         nudge the fader (default 10 points)
  /scope turn|reply|both|all
  /again                 regenerate the last reply under the current desk
  /ab [text]             the same message clean and steered, side by side
  /why [n]               replay the last reply through the clean model, token by token
  /concept <slot> @<L> <word>      build Lindsey's concept vector
  /extract <slot> @<L> <text>      a residual row into a slot
  /slots  /desk  /undo  /clear  /save <f>  /load <f>  /help  /quit
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--concept-baseline", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    chat = Chat(args.model, device=args.device, dtype=args.dtype,
                baseline_words=args.concept_baseline, seed=args.seed)
    chat.log_path = args.log
    print(HELP)

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        try:
            if not line.startswith("/"):
                print(f"\n[{chat.desk.label()}] > {line}")
                chat.do_message(line, args.max_tokens)
            else:
                head, _, body = line[1:].partition(" ")
                body = body.strip()
                if head in {"quit", "exit"}:
                    return 0
                elif head == "help":
                    print(HELP)
                elif head == "desk":
                    chat.do_desk()
                elif head == "slots":
                    chat.cmd_slots()
                elif head == "on":
                    parts = body.split()
                    chat.desk.slot = parts[0]
                    for part in parts[1:]:
                        if part.startswith("@"):
                            chat.desk.layer = int(part[1:])
                        else:
                            chat.desk.percent = float(part.rstrip("%"))
                    if chat.desk.layer is None:
                        chat.desk.layer = chat.slots[chat.desk.slot]["layer"]
                    print(f"  {chat.desk.label()}")
                elif head == "off":
                    chat.desk.slot = None; print("  clean")
                elif head in {"+", "-"}:
                    step = float(body) if body else 10.0
                    chat.desk.percent += step if head == "+" else -step
                    print(f"  {chat.desk.label()}")
                elif head == "scope":
                    chat.desk.scope = body or "turn"; print(f"  {chat.desk.label()}")
                elif head == "again":
                    if chat.history and chat.history[-1]["role"] == "assistant":
                        last_user = chat.history[-2]["content"]
                        chat.history = chat.history[:-2]
                        print(f"\n[{chat.desk.label()}] > {last_user}")
                        chat.do_message(last_user, args.max_tokens)
                    else:
                        print("nothing to regenerate")
                elif head == "ab":
                    chat.do_ab(body or None, args.max_tokens)
                elif head == "why":
                    chat.do_why(args.max_tokens, int(body) if body else 20)
                elif head == "concept":
                    parts = body.split()
                    slot = parts.pop(0)
                    layer = int(parts.pop(0)[1:]) if parts and parts[0].startswith("@") else None
                    chat.cmd_concept(slot, layer, " ".join(parts))
                elif head == "extract":
                    parts = body.split()
                    slot = parts.pop(0)
                    layer = int(parts.pop(0)[1:]) if parts and parts[0].startswith("@") else None
                    chat.cmd_extract(slot, layer, None, " ".join(parts))
                elif head == "undo":
                    chat.history = chat.history[:-2]; print(f"  {len(chat.history)} message(s)")
                elif head == "clear":
                    chat.history = []; print("  cleared")
                elif head == "save":
                    chat.cmd_save(body)
                elif head == "load":
                    chat.cmd_load(body)
                else:
                    print(f"  unknown command /{head} — /help")
        except Exception as exc:
            print(f"  !! {type(exc).__name__}: {exc}")
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
