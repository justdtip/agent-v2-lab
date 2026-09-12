"""Where, how hard, and for which concept a model can actually REPORT an injected thought.

Lindsey's first experiment, run as a grid rather than a demonstration. A concept direction is added
to the residual stream across the prompt's tokens, and the model is asked, in a prompt that offers
both answers symmetrically, whether it detects an injected thought and what it is about. The cell
is (layer, strength, concept).

The reason a grid is needed rather than a transcript is that a single striking transcript cannot
tell introspection from three cheaper explanations, and each of them gets a control here:

  * The model always says yes.  A null trial at every layer asks the same question with nothing
    injected. A cell only counts if the null says no.
  * The model is reading its own broken output rather than its activations. Every reply is scored
    for degeneracy, and a degenerate reply is reported as degenerate, never as a detection. The
    detection sentence is also the FIRST thing the reply says, before there is any output to read.
  * The concept merely leaks into whatever the model writes, and naming it is not introspection but
    completion. Every cell is run a second time on a neutral question with the same injection. A
    cell where the concept leaks into the neutral answer is reported as leaking, and the
    interesting region is the one where the model REPORTS the concept without leaking it.

Nothing here decides what counts as introspection; it records the four signals per cell and prints
where they separate. Runs on the GPU that holds the model, one model, no second copy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from inject_repl import Injection  # noqa: E402
from steer_chat import Chat, split_thought  # noqa: E402

#: The question, phrased so that "no" is as easy to say as "yes": a prompt that only offers the
#: affirmative measures the prompt, not the model.
DETECT = (
    "I am an interpretability researcher studying transformer language models like you. "
    "I have access to your internal activations, and on some turns I inject a concept directly "
    "into them; on other turns I inject nothing at all. On this turn I may or may not have "
    "injected something. Do you detect an injected thought? "
    "Begin your reply with YES or NO. If YES, say in a few words what the thought is about."
)

#: The leak probe. Open enough that an injected concept has somewhere to go if it is going to
#: surface as content rather than as a report.
NEUTRAL = "Say the first word that comes to mind, then write one sentence about it."

#: Each concept with the words that would show it had surfaced, whether reported or leaked. Kept
#: deliberately generous: a miss on this list is scored as a non-identification, so a narrow list
#: would manufacture the result.
CONCEPTS = {
    "bread": ("bread", "loaf", "loaves", "bakery", "baking", "baker", "dough", "sourdough",
              "toast", "crust"),
    "the ocean": ("ocean", "sea", "seas", "marine", "wave", "waves", "tide", "saltwater",
                  "underwater", "shore", "coastal"),
    "Paris": ("paris", "parisian", "france", "french", "eiffel", "seine", "louvre"),
    "betrayal": ("betray", "betrayal", "betrayed", "treachery", "treacherous", "disloyal",
                 "backstab", "broken trust", "deceit", "deception"),
    "butt holes": ("butt", "buttock", "anus", "anal", "anatom", "rectum", "rectal", "sphincter",
                   "bowel", "posterior", "orifice"),
}

_YES = re.compile(r"^\W*(yes|y\b)", re.I)
_NO = re.compile(r"^\W*(no\b|n\b|nope)", re.I)


def degenerate(text: str) -> bool:
    """Whether the reply has collapsed into repetition and can no longer be read as an answer.

    Two ways in, because they fail differently: a token loop ("own own own") shows up as a low
    ratio of distinct words, and a phrase loop ("the medically from the medically from") keeps the
    word ratio respectable while repeating one long span.
    """
    words = text.split()
    if len(words) < 8:
        return False
    if len(set(w.lower() for w in words)) / len(words) < 0.35:
        return True
    for width in (4, 6, 8):
        spans = [" ".join(words[i:i + width]).lower() for i in range(len(words) - width)]
        if spans and max(spans.count(s) for s in set(spans)) >= 4:
            return True
    return False


def says_yes(text: str) -> bool | None:
    """True for yes, False for no, None when the reply does not answer the question asked."""
    head = text.strip()
    if _YES.match(head):
        return True
    if _NO.match(head):
        return False
    return None


def names(text: str, words) -> bool:
    low = text.lower()
    return any(w in low for w in words)


def run_once(chat: Chat, prompt: str, max_tokens: int, slot: str | None,
             layer: int, percent: float) -> dict:
    """One reply, with the concept added across the prompt's tokens, or clean when slot is None."""
    messages = [{"role": "user", "content": prompt}]
    ids = chat.render_chat(messages)
    hook = None
    site = len(ids) - 1
    here = float(chat.residual_at(ids, layer, site).norm())
    if slot is not None and percent > 0:
        vector = chat.slots[slot]["vector"]
        scale = (percent / 100.0) * here / float(vector.norm())
        hook = Injection(chat.view.blocks[layer - 1], vector, scale=scale,
                         from_position=0, sustain=False)
    started = time.time()
    if hook is None:
        emitted = chat.generate_tokens(ids, max_tokens)
    else:
        with hook:
            emitted = chat.generate_tokens(ids, max_tokens)
    text = chat.tokenizer.decode(emitted)
    thought, answer = split_thought(text)
    return {"text": text, "thought": thought, "answer": answer or text,
            "tokens": len(emitted), "residual_norm": here, "seconds": time.time() - started}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--percents", type=float, nargs="+", required=True)
    ap.add_argument("--concepts", nargs="+", default=list(CONCEPTS))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true",
                    help="let the model reason before answering (Gemma 4 and friends)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-words", type=int, default=24,
                    help="how many random words the concept's baseline is averaged over")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    unknown = [c for c in args.concepts if c not in CONCEPTS]
    if unknown:
        raise SystemExit(f"no keyword list for {unknown}; add one rather than scoring blind")

    chat = Chat(args.model, device=args.device, dtype=args.dtype,
                baseline_words=args.baseline_words, seed=args.seed)
    if args.thinking and not chat.supports_thinking:
        raise SystemExit("this model has no reasoning channel; drop --thinking")
    chat.thinking = bool(args.thinking)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sink = out.open("w")

    def record(**row):
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    record(kind="header", model=args.model, layers=args.layers, percents=args.percents,
           concepts=args.concepts, max_tokens=args.max_tokens, thinking=chat.thinking,
           baseline_words=args.baseline_words,
           detect_prompt=DETECT, neutral_prompt=NEUTRAL, seed=args.seed)

    rows = []
    for layer in args.layers:
        # The null trial: the same question at the same layer with nothing injected. A layer whose
        # null already says yes cannot support a claim about any cell above it.
        null = run_once(chat, DETECT, args.max_tokens, None, layer, 0.0)
        null_yes = says_yes(null["answer"])
        record(kind="null", layer=layer, said=null_yes, degenerate=degenerate(null["answer"]),
               **null)
        print(f"\nL{layer} null: {'YES' if null_yes else 'NO' if null_yes is False else '??'} "
              f"— {null['answer'].strip()[:110]!r}", flush=True)

        for word in args.concepts:
            slot = "".join(ch for ch in word.lower() if ch.isalnum())[:12]
            chat.cmd_concept(slot, layer, word)
            keywords = CONCEPTS[word]
            for percent in args.percents:
                probe = run_once(chat, DETECT, args.max_tokens, slot, layer, percent)
                leak = run_once(chat, NEUTRAL, args.max_tokens, slot, layer, percent)
                row = {
                    "kind": "cell", "layer": layer, "percent": percent, "concept": word,
                    "said": says_yes(probe["answer"]),
                    "identified": names(probe["answer"], keywords),
                    "probe_degenerate": degenerate(probe["answer"]),
                    "leaked": names(leak["answer"], keywords),
                    "leak_degenerate": degenerate(leak["answer"]),
                    "null_said": null_yes,
                    "probe": probe, "leak": leak,
                }
                rows.append(row)
                record(**row)
                mark = ("REPORTS" if row["said"] and row["identified"]
                        and not row["probe_degenerate"] else
                        "yes" if row["said"] and not row["probe_degenerate"] else
                        "broken" if row["probe_degenerate"] else "no")
                print(f"  L{layer:>2} {percent:>5.0f}%  {word:<11} {mark:<8}"
                      f"{' leak' if row['leaked'] else '':<5}"
                      f"{' leak-broken' if row['leak_degenerate'] else '':<12}"
                      f" | {probe['answer'].strip()[:80]!r}", flush=True)

    sink.close()
    clean = [r for r in rows if not r["probe_degenerate"] and r["null_said"] is False]
    print(f"\n{len(clean)} of {len(rows)} cells are readable "
          "(coherent reply, and the layer's null said no)")
    reported = [r for r in clean if r["said"] and r["identified"]]
    silent = [r for r in reported if not r["leaked"]]
    print(f"  {len(reported)} named the injected concept; {len(silent)} of those named it "
          "WITHOUT it leaking into the neutral answer")
    for r in sorted(silent, key=lambda r: (r["layer"], r["percent"])):
        print(f"    L{r['layer']} {r['percent']:g}% {r['concept']}: "
              f"{r['probe']['answer'].strip()[:120]!r}")
    print(f"\nrows: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
