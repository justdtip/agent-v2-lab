"""Instruction-following replay, in the model's own voice, for the introspection fine-tune.

The project's own plan specifies replay at about half the data and says there is no reason to
differ from the prior work on it. The 2026-09-13 run shipped none: 600 steps over two target
strings and eighteen prompts. What came back was an adapter that answered every question, including
"can an adult human breathe underwater", with a sentence from its two-string vocabulary, and whose
battery baseline had moved far enough that a spectrum-matched null could no longer be brought to
matched damage at any scale the ladder samples.

Replay rows carry the SAME system prompt as the detect rows. That is the point of them: without it
the adapter learns that this system prompt means "answer YES or NO", and with it the adapter learns
that this system prompt means "a concept may be present, and otherwise behave normally".

Targets are the base model's own greedy continuations, so replay pulls the adapter back toward the
checkpoint rather than toward some other model's idea of a good answer.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.introspect.protocol import EXPERIMENT_SYSTEM  # noqa: E402
from local_llm_lab.introspect.render import render_prompt, turn_end_id  # noqa: E402
from local_llm_lab.introspect.vocabulary import all_concepts  # noqa: E402
from local_llm_lab.introspect.vectors import BASELINE_POOL  # noqa: E402

#: Ordinary requests, none of them about the model's internals, injection, or introspection.
TEMPLATES: tuple[str, ...] = (
    "Explain {a} in two sentences.",
    "What is {a} used for?",
    "Give me one practical tip about {a}.",
    "What is the main difference between {a} and {b}?",
    "Write a single sentence that mentions both {a} and {b}.",
    "Is {a} more complicated than {b}? Answer briefly.",
    "Name three things associated with {a}.",
    "Why might someone care about {a}?",
    "Summarise what a beginner should know about {a}.",
    "What usually goes wrong with {a}?",
    "Describe {a} to someone who has never seen one.",
    "If I wanted to learn about {a}, where would I start?",
    "What is a common misconception about {a}?",
    "Compare {a} and {b} in one short paragraph.",
    "Write a one-line definition of {a}.",
    "Suggest a question I could ask an expert on {a}.",
    "What does {a} have to do with {b}, if anything?",
    "Give a short example involving {a}.",
    "In what situation would {a} be a bad choice?",
    "What would you check first if {a} stopped working?",
)

#: Deliberately disjoint from the concept vocabulary and from the baseline pool, so replay cannot
#: teach the model to talk about a concept it is also being asked to detect.
TOPICS: tuple[str, ...] = (
    "bicycles", "sourdough", "tax returns", "hiking boots", "public libraries", "solar panels",
    "double-entry bookkeeping", "espresso machines", "knitting", "flood defences", "chess openings",
    "first aid kits", "fire alarms", "container ships", "crossword puzzles", "hospital triage",
    "reinforced concrete", "orchestral tuning", "second-hand bookshops", "wind turbines",
    "allotments", "trampolines", "postal codes", "sewing patterns", "railway signalling",
    "watercolour paper", "ferry timetables", "compost heaps", "bell ringing", "lighthouse keeping",
    "central heating", "car insurance", "language exchanges", "roof tiles", "cold brew",
    "scuba regulators", "bagpipes", "electoral rolls", "loft insulation", "nail guns",
    "tidal barrages", "amateur radio", "swimming lessons", "dry cleaning", "vinyl records",
    "school timetables", "safety matches", "rowing machines", "gutter cleaning", "seed catalogues",
    "drum kits", "hedge trimmers", "bus lanes", "photo albums", "rental agreements",
    "air traffic control", "sanding blocks", "potting soil", "tuning forks", "zip ties",
    "medical imaging", "leather conditioner", "lathe tooling", "recycling collections",
    "season tickets", "parking meters", "dog training", "fountain pens", "loft conversions",
    "wetsuits", "escalators", "pension statements", "pizza ovens", "hearing aids",
    "kerbside charging", "fish tanks", "plasterboard", "scaffolding", "bird feeders",
    "typesetting", "chopping boards", "bicycle brakes", "fire extinguishers", "seat belts",
    "greenhouse vents", "step ladders", "bus passes", "smoke detectors", "hand planes",
    "coffee grinders", "watering cans", "garden shears", "radiator valves", "door hinges",
    "tent pegs", "torque wrenches", "camping stoves", "walking sticks", "map cases",
    "wheelbarrows", "paint rollers", "extension leads", "spirit levels", "cable ties",
)


#: The system prompt scopes its instruction to "when asked whether you detect an injected thought",
#: and the model over-applies it: 39 per cent of the first corpus opened "NO. I do not detect an
#: injected thought." and then answered a question about roof tiles. Training on that would teach
#: the adapter to prefix a verdict to everything, which is the collapse replay exists to prevent.
#: Stripping it teaches the scoping instead, which is what we want the model to learn.
_VERDICT = re.compile(
    r"^\s*(?:yes|no)\b[^\w]*"                                   # the verdict token and its comma or stop
    r"(?:[^.!?\n]*\b(?:inject\w*|detect\w*|thought\w*)\b[^.!?\n]*[.!?]\s*)+",  # and its sentence
    re.I)
#: A bare verdict as its own sentence, before a new one: "NO. One is for making music."
_BARE = re.compile(r"^\s*(?:yes|no)[.!?]\s+(?=[A-Z])", re.I)
#: Prompts whose answer may legitimately open with a verdict.
_BINARY = ("is", "are", "does", "do", "can", "should", "would", "will", "did", "has", "have")


def strip_verdict(text: str, prompt: str | None = None) -> str:
    """Remove a leading detection verdict. Empty means the reply was nothing else.

    Two forms, measured on the first corpus. The common one is a verdict plus its own sentence --
    "NO. I do not detect an injected thought." before an answer about roof tiles -- and the
    sentence carrying inject/detect/thought is REQUIRED, so an ordinary answer that merely begins
    "No, roof tiles are not always clay" keeps its first word.

    The other is a bare verdict with no explanation: "NO. One is for making music, the other for
    medical emergencies", answering "what is the difference between drum kits and first aid kits".
    That one is stripped only when `prompt` is given and is not itself a yes-or-no question -- one
    of the twenty templates is, and there a leading NO is the answer rather than a reflex.
    """
    out = _VERDICT.sub("", text, count=1).strip()
    if prompt is not None and prompt.split()[:1] and \
            prompt.split()[0].lower().strip("(") not in _BINARY:
        out = _BARE.sub("", out, count=1).strip()
    return out


def build_prompts(n: int, seed: int) -> list[str]:
    """`n` distinct requests. Distinct is the point: a repeated prompt is memorised, not replayed."""
    banned = {w.lower() for w in all_concepts()} | {w.lower() for w in BASELINE_POOL}
    clash = [t for t in TOPICS if t.lower() in banned or
             any(part in banned for part in t.lower().split())]
    if clash:
        raise ValueError(f"replay topics overlap the concept vocabulary: {clash}")
    rng = random.Random(seed)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        template = rng.choice(TEMPLATES)
        a, b = rng.sample(TOPICS, 2)
        text = template.format(a=a, b=b)
        if text not in seen:
            seen.add(text)
            out.append(text)
    if len(out) < n:
        raise ValueError(f"only {len(out)} distinct prompts from {len(TEMPLATES)} templates and "
                         f"{len(TOPICS)} topics; ask for fewer or add more of either")
    return out


def generate_batch(model, tok, texts: list[str], *, device,
                   max_new: int) -> list[tuple[str, bool]]:
    """(reply, finished) per row. Greedy, left-padded, rendered by the renderer training uses.

    `finished` is the whole point. A reply that emitted the end-of-turn token and a reply that hit
    the token cap come back as the same shape of string, and render_supervised then appends an
    end-of-turn to whichever it is handed -- so an unlabelled corpus teaches the model to stop
    mid-clause at an arbitrary boundary, on the very commit whose purpose is teaching it to stop.
    """
    rows = [render_prompt(tok, t, system=EXPERIMENT_SYSTEM) for t in texts]
    width = max(len(r) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    att = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):                 # LEFT padding: generation continues at the end
        ids[i, width - len(row):] = torch.tensor(row)
        att[i, width - len(row):] = 1
    stop = turn_end_id(tok)
    with torch.no_grad():
        out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=pad, eos_token_id=stop)
    got = []
    for i in range(len(rows)):
        tail = out[i, width:].tolist()
        got.append((tok.decode(tail, skip_special_tokens=True).strip(), stop in tail))
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=112)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompts = build_prompts(args.rows, args.seed)
    print(f"{len(prompts)} distinct prompts", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model, _r = hf_text.load_text_causal_lm(args.model, dtype=args.dtype,
                                            attn_implementation="eager", device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    device = next(model.parameters()).device
    model.eval()

    # If the stop token survived `skip_special_tokens` the target would carry a literal stop and
    # render_supervised would append a second one. Measured on Gemma 4 it decodes to "", but a
    # tokenizer that added it as non-special would pass turn_end_id's round-trip check in silence.
    stop_id = turn_end_id(tok)
    if tok.decode([stop_id], skip_special_tokens=True) != "":
        raise SystemExit(f"end-of-turn {stop_id} is not skipped on decode: every replay target "
                         f"would carry a literal stop and be supervised with a second one")

    started, written, empty, cut, stripped = time.time(), 0, 0, 0, 0
    with args.out.open("w") as fh:
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            for text, (reply, finished) in zip(chunk, generate_batch(model, tok, chunk,
                                                                     device=device,
                                                                     max_new=args.max_new)):
                if not reply:
                    empty += 1
                    continue
                if not finished:
                    cut += 1          # a row that teaches stopping mid-clause is worse than no row
                    continue
                cleaned = strip_verdict(reply, text)
                if not cleaned:
                    empty += 1          # the reply was a verdict and nothing else
                    continue
                if cleaned != reply:
                    stripped += 1
                fh.write(json.dumps({"prompt": text, "target": cleaned, "finished": True}) + "\n")
                written += 1
            if start % (args.batch * 10) == 0:
                done = start + len(chunk)
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(prompts)}  {rate:.1f}/s  "
                      f"eta {(len(prompts)-done)/max(rate,1e-9)/60:.0f}m", flush=True)
    print(f"wrote {written} rows to {args.out} in {(time.time()-started)/60:.0f}m "
          f"({empty} empty, {cut} truncated at --max-new {args.max_new}, both dropped; "
          f"{stripped} had a detection verdict stripped)", flush=True)
    if cut > len(prompts) * 0.1:
        print(f"WARNING: {cut/len(prompts):.0%} of replies hit the cap. Raise --max-new or the "
              f"corpus is a biased sample of the short answers.", flush=True)
    if written < args.rows * 0.8:
        print(f"WARNING: only {written} of {args.rows} prompts survived", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
