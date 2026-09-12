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

#: Gemma 4 puts its reasoning in a channel and its answer after it. The markers are literals in the
#: chat template rather than a documented API, so they are named here and detected at runtime: a
#: model without them simply has no thought to split off.
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

#: What a carried summary is labelled as inside the system turn.
SUMMARY_HEADING = "Notes carried over from the earlier part of this conversation:"

#: So `render_messages(head=None)` can mean "no system turn" and be told apart from "not given".
_UNSET = object()


def opens_thought(text: str) -> bool:
    """Whether this reply begins inside the reasoning channel.

    Either marker is enough. The template usually opens the channel in the prompt, so only the
    close comes back in the reply; but the model sometimes writes the open itself, and a reply cut
    off by the token cap carries the open and no close at all.
    """
    return THOUGHT_OPEN in text or THOUGHT_CLOSE in text


def split_thought(text: str) -> tuple[str, str]:
    """Separate reasoning from answer. Returns (thought, answer); thought is '' when there is none.

    A reply that opens the channel and never closes it is a thought the token cap cut off, not an
    answer: returning it as the answer would print the raw channel marker as if the model had said
    it, and would count every reasoning token as an answer token.
    """
    if THOUGHT_CLOSE not in text:
        if THOUGHT_OPEN in text:
            return text.partition(THOUGHT_OPEN)[2].strip("\n"), ""
        return "", text
    head, _, tail = text.partition(THOUGHT_CLOSE)
    if THOUGHT_OPEN in head:
        head = head.partition(THOUGHT_OPEN)[2]
    return head.strip("\n"), tail.lstrip("\n")


class Desk:
    """One steering channel: a slot, a layer, a fader in per cent of residual norm, and a scope."""

    def __init__(self):
        self.slot = None
        self.layer = None
        self.percent = 40.0
        self.scope = "turn"

    @property
    def live(self) -> bool:
        """A fader at zero is clean whatever is patched into it -- but the patch stays visible.

        Zeroing the slot instead would leave the rack and the fader disagreeing with the run: the
        desk would read `paris 120%` while every reply came out clean.
        """
        return self.slot is not None and self.percent > 0

    def label(self) -> str:
        if not self.live:
            return "clean"
        return f"{self.slot} {self.percent:g}% L{self.layer} {self.scope}"


class Chat(Console):
    """The console's machinery, with a conversation on top of it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: list[dict] = []
        #: The last reply's prompt ids and the ids the model actually emitted. Every replay reads
        #: these rather than re-tokenising the decoded text: `decode` then `encode` is not the
        #: identity, so re-tokenising forces the clean model through a token sequence the steered
        #: run never visited -- which is the one thing the replay exists to rule out.
        self.last_turn: dict | None = None
        self._keep_kw: dict | None = None
        #: Server state, never `history[0]`. The system prompt and any carried summary are
        #: re-rendered every turn; putting either in the history would make `undo`, `clear`, `ab`
        #: and the page's message count all disagree about what a conversation contains.
        self.system: str | None = None
        self.summary: str | None = None
        self.window: int = 0
        self.desk = Desk()
        self.log_path = None
        self._thinking = False
        self.supports_thinking = THOUGHT_OPEN in (self.tokenizer.chat_template or "")
        # Stop ids come from the tokenizer's own specials, never from another model's names: Gemma 4
        # closes a turn with <turn|> where Gemma 3 used <end_of_turn>.
        # `convert_tokens_to_ids` answers with the UNK id for a token the vocabulary does not have,
        # and UNK is an ordinary non-negative id: taken at face value on Gemma 3, whose vocabulary
        # has neither of these names, that puts UNK in the stop set and ends any reply the moment
        # the model emits it. The id has to name the token back.
        unk = self.tokenizer.unk_token_id
        for name in ("<turn|>", "<|turn|>"):
            got = self.tokenizer.convert_tokens_to_ids(name)
            if not isinstance(got, int) or got < 0 or got == unk:
                continue
            if self.tokenizer.convert_ids_to_tokens(got) != name:
                continue
            self.stop_ids.add(got)

    # -- rendering -----------------------------------------------------------------------------
    @staticmethod
    def compose_head(system: str | None, summary: str | None) -> str | None:
        """The system turn's text for a given prompt and summary, without touching instance state.

        Separate from `system_head` so a compaction can price the window it is about to install
        before installing it.
        """
        if not summary:
            return system
        carried = f"{SUMMARY_HEADING}\n{summary}"
        return carried if not system else f"{system}\n\n{carried}"

    def system_head(self) -> str | None:
        """The one system turn: the standing prompt, then any summary carried from an earlier window.

        Composed at render time rather than stored, so the system prompt stays independently
        editable and the summary is never a message. A summary in the history would be found by
        `ab`'s backward scan for the last user turn, eaten by `undo`'s `history[:-2]`, and counted
        by `state()["history"]`, which drives two buttons on the page.
        """
        return self.compose_head(self.system, self.summary)

    def _apply(self, messages: list[dict], *, generation_prompt: bool, thinking: bool) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=generation_prompt, tokenize=False,
                enable_thinking=thinking)
        except TypeError:
            # a template that takes no such flag — Gemma 3 and most others
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=generation_prompt, tokenize=False)

    def render_messages(self, messages: list[dict], *, generation_prompt: bool = True,
                        thinking: bool | None = None, head: str | None = _UNSET) -> list[int]:
        """The ids the model actually sees: one system turn, then the conversation.

        Every render in this class goes through here. `turn_span` used to call
        `apply_chat_template` itself and omit `enable_thinking`, so with reasoning ON the prefix it
        measured was seven tokens shorter than the real prompt AND was not a prefix of it: the
        template puts a `<|think|>` marker in the system turn that the omitted flag never produced.
        The injection span for scope `turn` and `both` was computed from that disagreement.
        """
        think = self.thinking if thinking is None else thinking
        head = self.system_head() if head is _UNSET else head
        full = ([{"role": "system", "content": head}] if head else []) + list(messages)
        ids = self.tokenizer(self._apply(full, generation_prompt=generation_prompt,
                                         thinking=think), add_special_tokens=False)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and ids.count(bos) > 1:
            raise ValueError(f"{ids.count(bos)} BOS tokens in one render; the template and the "
                             "tokenizer are both adding one")
        return ids

    def render_chat(self, messages: list[dict]) -> list[int]:
        """The whole conversation with a generation prompt, one BOS, tokenized as the model expects."""
        return self.render_messages(messages, generation_prompt=True)

    @property
    def thinking(self) -> bool:
        return self._thinking

    @thinking.setter
    def thinking(self, value: bool) -> None:
        self._thinking = bool(value)

    def turn_span(self, messages: list[dict]) -> tuple[int, int]:
        """Where the newest user turn starts, so `scope=turn` can inject over it and nothing else.

        Both renders now use the same reasoning flag, the same system prompt and the same carried
        summary, and the result is asserted to be a real token prefix. A common-prefix scan would
        not do: if a marker appears in one render and not the other the agreed prefix collapses
        towards nothing and the hook silently covers almost the whole context instead of one turn.
        """
        full = self.render_messages(messages, generation_prompt=True)
        prior_msgs = messages[:-1]
        prior = ([] if not prior_msgs and self.system_head() is None
                 else self.render_messages(prior_msgs, generation_prompt=False))
        if full[:len(prior)] != prior:
            raise ValueError(
                f"the prior render ({len(prior)} tokens) is not a prefix of the full render "
                f"({len(full)} tokens); the injection span cannot be computed from it")
        return len(prior), len(full)

    # -- the intervention ----------------------------------------------------------------------
    def build_injection(self, ids: list[int], span: tuple[int, int]):
        """Translate the desk into a hook, with the fader read as per cent of the residual norm."""
        if not self.desk.live:
            return None, "clean"
        entry = self.slots[self.desk.slot]
        layer = self.desk.layer
        start, end = span
        # The fader is calibrated at the last prompt token whatever the scope covers. For scope
        # "all" that means one position's norm sets the size of a perturbation applied over the
        # whole context, which the note says out loud rather than leaving implicit.
        site = end - 1
        if self.desk.scope == "all":
            start = 0
        here = float(self.residual_at(ids, layer, site).norm())
        scale = (self.desk.percent / 100.0) * here / float(entry["vector"].norm())
        sustain = self.desk.scope in {"reply", "both"}
        # `end` is the length of the rendered prompt, so `end - 1` is its LAST token and `end` is
        # the first token the model generates. `reply` scope means the reply, so it starts at `end`.
        from_position = end if self.desk.scope == "reply" else start
        if layer != entry["layer"]:
            note = (f"{self.desk.slot} was read at L{entry['layer']} and is being injected at "
                    f"L{layer} — a residual from one layer is not a direction at another")
            print(f"    ! {note}")
        hook = Injection(self.view.blocks[layer - 1], entry["vector"], scale=scale,
                         from_position=from_position, sustain=sustain)
        note = (f"{self.desk.slot} {self.desk.percent:g}% of |h|={here:,.0f} (read at the last "
                f"prompt token, position {site}) at L{layer}, scope {self.desk.scope}, "
                f"from {from_position}")
        return hook, note

    def speak(self, messages: list[dict], max_tokens: int, *, steered: bool = True,
              on_token=None):
        """One reply. Returns (text, token ids, prompt ids, note, seconds, hook).

        `on_token(index, id, text)` is called as each token is produced, for a caller that wants to
        show the reply arriving rather than after. It runs inside the generation loop and inside the
        service's lock, so it must not block on anything slow.
        """
        ids = self.render_chat(messages)
        span = self.turn_span(messages)
        hook, note = self.build_injection(ids, span) if steered else (None, "clean")
        started = time.time()
        if hook is None:
            emitted = self.generate_tokens(ids, max_tokens, on_token=on_token)
        else:
            with hook:
                emitted = self.generate_tokens(ids, max_tokens, on_token=on_token)
        # The hook comes back so a caller can say whether it actually fired. It counts its own
        # applications, and a hook that never fired is a clean reply that the desk still describes
        # as steered -- which reads as "this vector has no effect", a research conclusion rather
        # than a UI state.
        return (self.tokenizer.decode(emitted), emitted, ids, note, time.time() - started, hook)

    def generate_tokens(self, ids: list[int], max_tokens: int, *, on_token=None) -> list[int]:
        from transformers import DynamicCache

        cache = DynamicCache()
        emitted = []
        with torch.no_grad():
            # Only the last row is ever read, but the default keeps one 262,144-wide row per
            # position and this model then makes three more full-size copies for its logit
            # softcapping. At eight thousand tokens that is gigabytes allocated to read one row.
            logits = self.model(input_ids=self.view._ids(list(ids)), past_key_values=cache,
                                use_cache=True, **self._keep_last()).logits
            for _ in range(max_tokens):
                token = int(logits[0, -1].float().argmax())
                if token in self.stop_ids:
                    break
                emitted.append(token)
                if on_token is not None:
                    # Decoded one at a time. A tokenizer can split a character across two tokens, so
                    # the pieces are what the model emitted, not necessarily printable on their own;
                    # the caller re-decodes the whole reply at the end and replaces what it showed.
                    on_token(len(emitted) - 1, token, self.tokenizer.decode([token]))
                logits = self.model(input_ids=self.view._ids([token]),
                                    past_key_values=cache, use_cache=True).logits
        return emitted

    # -- the instrument ------------------------------------------------------------------------
    def clean_replay(self, prompt_ids: list[int], emitted: list[int], chunk: int = 0) -> list[dict]:
        """What the unsteered model would have said at each context the steered run visited.

        The steered tokens are forced rather than resampled, so both runs see an identical context
        at every position. A free-running clean loop would diverge after the first differing token
        and would be answering a different question.
        """
        from transformers import DynamicCache

        rows, cache = [], DynamicCache()
        with torch.no_grad():
            logits = self.model(input_ids=self.view._ids(list(prompt_ids)),
                                past_key_values=cache, use_cache=True, **self._keep_last()).logits
            if chunk and len(emitted) > 1:
                # Teacher forcing: every token is already fixed, so the whole reply goes through in
                # a few wide passes instead of one pass per token. Causal masking means position k
                # sees exactly the prefix it saw before.
                #
                # It is not BITWISE the same. A bf16 forward is not batch-invariant on this stack
                # and the steered run decoded one token at a time, so a wide replay does not share
                # its arithmetic schedule. For the interactive bench that is the right trade: the
                # replay is dead time after the stream has stopped, and a two-thousand-token reply
                # was paying two thousand extra forward passes for it. Anything written down as a
                # result passes chunk=0 and takes the slow path.
                #
                # `preds[k]` is the row that predicts `emitted[k]`: the prompt's last row predicts
                # the first token, and the row produced by consuming emitted[i] predicts
                # emitted[i+1], so feeding emitted[:-1] supplies exactly the rest.
                parts, done = [], 0
                # The prompt's last row predicts emitted[0]; the row produced by consuming
                # emitted[i] predicts emitted[i+1], so feeding emitted[:-1] supplies the rest.
                parts.append(self._score(logits[:, -1:, :], emitted[:1]))
                done += 1
                fed = list(emitted[:-1])
                for begin in range(0, len(fed), chunk):
                    out = self.model(input_ids=self.view._ids(fed[begin:begin + chunk]),
                                     past_key_values=cache, use_cache=True).logits
                    width = out.shape[1]
                    parts.append(self._score(out, emitted[done:done + width]))
                    done += width
                    del out  # a view into this would pin the whole 262k-wide tensor for the run
                if done != len(emitted):
                    raise ValueError(f"replay scored {done} of {len(emitted)} tokens")
                return self._rows_from(parts, emitted)
            for index, token in enumerate(emitted):
                rows.extend(self._rows_from([self._score(logits[:, -1:, :], [token])], [token],
                                            start=index))
                logits = self.model(input_ids=self.view._ids([token]),
                                    past_key_values=cache, use_cache=True).logits
        return rows

    def _keep_last(self) -> dict:
        """Ask for one row of logits where only one row is used, if this transformers knows how.

        Probed once and cached rather than assumed: the keyword has been spelled differently across
        versions, and passing an unknown one raises inside the model rather than being ignored.
        """
        if getattr(self, "_keep_kw", None) is None:
            import inspect
            names = set(inspect.signature(type(self.model).forward).parameters)
            self._keep_kw = ({"logits_to_keep": 1} if "logits_to_keep" in names
                             else {"num_logits_to_keep": 1} if "num_logits_to_keep" in names
                             else {})
        return self._keep_kw

    def _score(self, logits, tokens: list[int]):
        """Four small tensors per position, computed on device and brought back once.

        RANK IS THE NUMBER OF TOKENS STRICTLY ABOVE, changed 2026-09-12 (method entry 36). It was
        the steered token's index in a full argsort, which for a tie broke in unspecified order --
        so one of two tokens the clean model valued identically was reported as displaced, and
        `N of M changed` counted it. Measured over 16,886 saved replay rows, 45 of 1,847 changed
        rows (2.4%) were exact ties of this kind. This checkpoint makes them: final_logit_softcapping
        30.0 saturates the top of the distribution and bf16 rounding then collapses neighbours onto
        one value. Counts from before that date read about 2.4% high.
        """
        rows = logits[0].float().log_softmax(-1)
        want = torch.tensor(list(tokens), device=rows.device)
        tok_lp = rows.gather(1, want[:, None])[:, 0]
        top_lp, top_id = rows.max(-1)
        rank = (rows > tok_lp[:, None]).sum(-1)
        return (want.cpu(), tok_lp.cpu(), top_lp.cpu(), top_id.cpu(), rank.cpu())

    def _rows_from(self, parts, emitted: list[int], start: int = 0) -> list[dict]:
        want = torch.cat([p[0] for p in parts])
        tok_lp = torch.cat([p[1] for p in parts])
        top_lp = torch.cat([p[2] for p in parts])
        top_id = torch.cat([p[3] for p in parts])
        rank = torch.cat([p[4] for p in parts])
        steered = self.tokenizer.batch_decode(want[:, None].tolist())
        tops = self.tokenizer.batch_decode(top_id[:, None].tolist())
        return [{
            "position": start + i,
            "steered_token": steered[i],
            "clean_top": tops[i],
            "clean_rank_of_steered": int(rank[i]),
            "logprob_steered_token": float(tok_lp[i]),
            "logprob_clean_top": float(top_lp[i]),
        } for i in range(len(emitted))]

    # -- commands ------------------------------------------------------------------------------
    def do_message(self, text: str, max_tokens: int, *, commit: bool = True) -> None:
        messages = self.history + [{"role": "user", "content": text}]
        reply, emitted, _ids, note, seconds, _hook = self.speak(messages, max_tokens)
        print(f"\n{reply.strip() or '(nothing)'}")
        mark = "·" if not self.desk.live else "↯"
        print(f"    {mark} {note} · {len(emitted)} tok · {seconds:.1f}s")
        self.last_turn = {"prompt_ids": _ids, "emitted": emitted}
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
        clean, *_c = self.speak(messages, max_tokens, steered=False)
        steered, *_s, note, _ss, _sh = self.speak(messages, max_tokens, steered=True)
        print(f"\n  on {len(base)} message(s) of history, both arms identical up to the desk")
        print(f"\n  A  clean\n     {clean.strip() or '(nothing)'}")
        print(f"\n  B  {note}\n     {steered.strip() or '(nothing)'}")
        same = clean.strip() == steered.strip()
        print(f"\n    {'identical — the desk changed nothing' if same else 'they differ'}")

    def do_why(self, max_tokens: int, show: int) -> None:
        """Replay the last steered reply through the clean model, token by token."""
        if not self.history or self.history[-1]["role"] != "assistant":
            print("no reply to explain"); return
        if not self.last_turn:
            print("no recorded tokens for the last reply"); return
        ids, emitted = self.last_turn["prompt_ids"], self.last_turn["emitted"]
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
