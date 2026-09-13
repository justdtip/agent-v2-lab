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


def generate_batch(model, tok, texts: list[str], *, device, max_new: int) -> list[str]:
    """Greedy, left-padded, one batch. The prompts are rendered by the SAME renderer training uses."""
    rows = [render_prompt(tok, t, system=EXPERIMENT_SYSTEM) for t in texts]
    width = max(len(r) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    att = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):                 # LEFT padding: generation continues at the end
        ids[i, width - len(row):] = torch.tensor(row)
        att[i, width - len(row):] = 1
    with torch.no_grad():
        out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=pad, eos_token_id=turn_end_id(tok))
    return [tok.decode(out[i, width:], skip_special_tokens=True).strip() for i in range(len(rows))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
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

    started, written, empty = time.time(), 0, 0
    with args.out.open("w") as fh:
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            for text, reply in zip(chunk, generate_batch(model, tok, chunk, device=device,
                                                         max_new=args.max_new)):
                if not reply:
                    empty += 1
                    continue
                fh.write(json.dumps({"prompt": text, "target": reply}) + "\n")
                written += 1
            if start % (args.batch * 10) == 0:
                done = start + len(chunk)
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(prompts)}  {rate:.1f}/s  "
                      f"eta {(len(prompts)-done)/max(rate,1e-9)/60:.0f}m", flush=True)
    print(f"wrote {written} rows to {args.out} in {(time.time()-started)/60:.0f}m "
          f"({empty} empty replies dropped)", flush=True)
    if empty > len(prompts) * 0.02:
        print(f"WARNING: {empty/len(prompts):.0%} of replies were empty", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
