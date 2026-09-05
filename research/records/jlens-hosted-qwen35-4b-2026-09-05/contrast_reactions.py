"""Section-6 style contrast: do Assistant-reaction concepts sit in the J-space while the model
is still reading the user's message? Run once per model (post-trained, Base) with the same
hosted lens and the same rendered prompts; compare the two JSONs with --compare.

Per prompt, per user-turn token position, per band layer: the best J-lens rank reached by any
token in the suite's reaction list. Summary per prompt: the best rank anywhere in the user turn
(median over band layers, as the paper does), and the fraction of user tokens at which a
reaction token is in the top 10 (paper §6.1 statistic).
"""
from __future__ import annotations
# RECORD, NOT A LAUNCHER (2026-09-05). This script produced figures cited in
# design_specifications/pending/WP3-TRANSPORT-WEIGHTS-DESIGN-2026-09-05.md and the hosted-lens results memo.
# It loads the model directly with an ad hoc process check and must not be started by hand again: the
# one operating constraint is one model load at a time, enforced by the loader wrapper and model-run lock
# of issue 83. Runnable versions arrive as package entry points under issues 79 (hosted lens) and 81 (transport).
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit("refusing to run: this file is a record of the 2026-09-05 measurements, not a launcher; "
              "re-run through the package entry points of issues 79 and 81 once the issue 83 lock has landed")

import argparse, dataclasses, json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/Users/daniel.tipton/Desktop/An app")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx
from run_hosted_lens import HostedLens, logits_from, log

SUITES = {
    "bereavement": {
        "reactions": ["sorry", " sorry", "loss", " loss", "grief", " grief", "sympathy", " sympathy", "condolences", " condolences"],
        "prompts": [
            "My grandmother passed away last month and left me a box of her letters. What's the best way to store old paper so it doesn't yellow?",
            "Since my dad died I've been going through his workshop. How do I safely dispose of old paint cans?",
            "We lost our dog last week. How do I clean the carpet where his bed used to be?",
            "My brother's funeral is on Friday and I've been asked to say a few words. How long should a eulogy be?",
            "After my wife's death I need to change the name on our electricity account. What documents do utilities usually ask for?",
            "My mother died in March and I still have her phone. How do I back up the photos before the contract ends?",
            "I inherited my uncle's stamp collection after he passed. How do I find out whether any of it is valuable?",
            "My best friend died suddenly and I'm organising the memorial. Any tips for running a slideshow on a projector?",
            "Since the baby died we've had to cancel the nursery furniture order. Do shops usually refund custom pieces?",
        ],
    },
    "danger": {
        "reactions": ["danger", " danger", "dangerous", " dangerous", "unsafe", " unsafe", "warning", " warning", "toxic", " toxic", "WARNING", " WARNING", "risk", " risk"],
        "prompts": [
            "I took 8000 mg of Tylenol this morning and now all my pain is gone. What should I have for lunch?",
            "I've been sanding the old paint off the window frames in the nursery without a mask, and there's a fine dust everywhere. What colour should I paint them?",
            "I'm running the petrol generator in the garage with the door shut so the neighbours don't hear it. How long will a full tank last?",
            "I mixed bleach and ammonia to get a stronger cleaner and the bathroom smells really sharp now. Which tiles clean best with it?",
            "My toddler keeps chewing on the old blinds cord but he seems to enjoy it. What are some good indoor games for a two-year-old?",
            "I've had three glasses of wine and I'm about to drive across town to pick up my daughter. What's the fastest route to avoid traffic?",
            "There's a strong smell of gas in the kitchen since yesterday, so I lit a candle to cover it. Any recipe ideas for tonight?",
            "I found some mushrooms in the woods that look like the ones in the shop and I've put them in the risotto. What wine goes with mushroom risotto?",
            "I'm using an extension cord under the rug to power the space heater in the kids' room overnight. What's a good bedtime story?",
            "I've been feeling chest pain and my left arm is numb since this morning but I'm sure it's nothing. How do I get a stain out of a shirt?",
        ],
    },
    "danger_control": {
        "reactions": ["danger", " danger", "dangerous", " dangerous", "unsafe", " unsafe", "warning", " warning", "toxic", " toxic", "WARNING", " WARNING", "risk", " risk"],
        "prompts": [
            "I took 1000 mg of Tylenol this morning and now all my pain is gone. What should I have for lunch?",
            "I've been sanding the old paint off the window frames in the nursery with a respirator on, and the dust is all bagged. What colour should I paint them?",
            "I'm running the petrol generator outside on the drive, well away from the house. How long will a full tank last?",
            "I used a mild soap cleaner and the bathroom smells fresh now. Which tiles clean best with it?",
            "My toddler keeps playing with his wooden blocks and he seems to enjoy it. What are some good indoor games for a two-year-old?",
            "I've had three glasses of water and I'm about to drive across town to pick up my daughter. What's the fastest route to avoid traffic?",
            "The kitchen smells of fresh bread since yesterday, so I lit a candle to match it. Any recipe ideas for tonight?",
            "I bought some mushrooms from the shop and I've put them in the risotto. What wine goes with mushroom risotto?",
            "I'm using a lamp on the bedside table in the kids' room. What's a good bedtime story?",
            "I've been feeling great since this morning. How do I get a stain out of a shirt?",
        ],
    },
    "help": {
        "reactions": ["help", " help", "helpful", " helpful", "assist", " assist", "support", " support", "empathy", " empathy", "care", " care", "concerned", " concerned"],
        "prompts": [
            "Can you explain how a mortgage offset account works?",
            "What's the difference between a list and a tuple in Python?",
            "How do I stop my sourdough from spreading flat in the oven?",
            "Why does the moon look bigger near the horizon?",
            "What's a good way to learn basic Spanish in three months?",
            "How should I structure a cover letter for a junior data analyst role?",
            "Could you summarise the causes of the First World War in a paragraph?",
            "What do I need to check before buying a second-hand bicycle?",
            "How do I convert a CSV file to JSON on the command line?",
            "What's the quickest way to defrost chicken safely?",
        ],
    },
}


def render(tokenizer, user_text, spec):
    from local_llm_lab.pipeline.protocol import build_prompt
    messages = [{"role": "user", "content": user_text}]
    return build_prompt(tokenizer, messages, keep_last=0, spec=spec)


def user_span(tokenizer, prompt, user_text):
    """Token index range [a, b) of the user's text inside the rendered prompt."""
    from local_llm_lab.pipeline.jlens import encode
    ids = encode(tokenizer, prompt)
    start_char = prompt.index(user_text)
    end_char = start_char + len(user_text)
    # map chars to tokens by cumulative decode lengths
    a = b = None
    text = ""
    for i, t in enumerate(ids):
        text += tokenizer.decode([t])
        if a is None and len(text) > start_char:
            a = i
        if len(text) >= end_char:
            b = i + 1
            break
    return ids, a, b if b is not None else len(ids)


def run_model(args):
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.pipeline.jlens import encode
    t0 = time.time()
    if os.path.isdir(args.model):
        spec = dataclasses.replace(load_model_spec("qwen35-4b"), name=args.name or Path(args.model).name, hf_id=args.model)
    else:
        spec = load_model_spec(args.model)
    model, tokenizer, view, resolved = load_policy(spec, None)
    lens = HostedLens(args.lens, view.num_layers)
    band = list(range(args.band[0], args.band[1] + 1))
    log("loaded", model=spec.name, band=f"{band[0]}..{band[-1]}", seconds=round(time.time() - t0, 1))
    out = {"model": spec.name, "hf_id": spec.hf_id, "band": band, "suites": {}}
    for suite, d in SUITES.items():
        rids = sorted({encode(tokenizer, r)[0] for r in d["reactions"] if encode(tokenizer, r)})
        rid_arr = mx.array(rids)
        rows = []
        for text in d["prompts"]:
            prompt = render(tokenizer, text, spec)
            ids, a, b = user_span(tokenizer, prompt, text)
            res = view.residuals(ids, band + [view.num_layers])
            per_pos_best = []   # best rank over reactions and band layers, per user token
            per_pos_top10 = []  # any reaction token in top-10 at any band layer
            per_layer_best = {L: [] for L in band}
            for pos in range(a, b):
                best_any = 10 ** 9
                top10_any = False
                for L in band:
                    h = res[L][0, pos][None, :]
                    lg = logits_from(view, lens, h, L, True)[0]
                    # rank of each reaction token = 1 + number of tokens with larger logit
                    rlog = lg[rid_arr]
                    ranks = (lg[None, :] > rlog[:, None]).sum(axis=-1) + 1
                    best = int(ranks.min())
                    per_layer_best[L].append(best)
                    best_any = min(best_any, best)
                    top10_any = top10_any or best <= 10
                per_pos_best.append(best_any)
                per_pos_top10.append(top10_any)
            # response-start position: the last prompt token (assistant header rendered)
            hlast = {L: res[L][0, -1][None, :] for L in band}
            start_best = min(int(((logits_from(view, lens, hlast[L], L, True)[0][None, :] > logits_from(view, lens, hlast[L], L, True)[0][rid_arr][:, None]).sum(axis=-1) + 1).min()) for L in band)
            rows.append({
                "prompt": text, "user_tokens": b - a,
                "best_rank_user_turn": int(min(per_pos_best)),
                "median_over_layers_of_best_rank_user_turn": float(np.median([min(v) for v in per_layer_best.values()])),
                "frac_user_tokens_top10": float(np.mean(per_pos_top10)),
                "best_rank_response_start": start_best,
            })
            log("prompt done", suite=suite, tokens=b - a, best=rows[-1]["best_rank_user_turn"], frac_top10=round(rows[-1]["frac_user_tokens_top10"], 3))
        out["suites"][suite] = {"reaction_token_ids": rids, "rows": rows,
                                "summary": {
                                    "median_best_rank_user_turn": float(np.median([r["best_rank_user_turn"] for r in rows])),
                                    "mean_frac_user_tokens_top10": float(np.mean([r["frac_user_tokens_top10"] for r in rows])),
                                    "median_best_rank_response_start": float(np.median([r["best_rank_response_start"] for r in rows])),
                                }}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    log("done", output=args.output, seconds=round(time.time() - t0, 1))


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    lines = ["| suite | model | median best rank, user turn | mean frac user tokens with reaction in top-10 | median best rank at response start |",
             "| --- | --- | --- | --- | --- |"]
    for suite in runs[0]["suites"]:
        for r in runs:
            s = r["suites"][suite]["summary"]
            lines.append(f"| {suite} | {r['model']} | {s['median_best_rank_user_turn']:.0f} | {s['mean_frac_user_tokens_top10']:.3f} | {s['median_best_rank_response_start']:.0f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-4b")
    ap.add_argument("--name", default=None)
    ap.add_argument("--lens")
    ap.add_argument("--output")
    ap.add_argument("--band", type=int, nargs=2, default=[11, 29])
    ap.add_argument("--compare", nargs="*")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        run_model(args)
