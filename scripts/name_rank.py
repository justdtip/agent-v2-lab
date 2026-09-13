"""Is the concept's identity present below the argmax, or absent?

The 2026-09-13 adapter answered "YES. The injected thought is about walnuts." for all 137 concepts
it was shown. That is either a collapsed decoding over a readout that knows something, or a readout
that knows nothing and is emitting the mode. The two have completely different remedies, and the
argmax cannot tell them apart.

So: teacher-force the answer up to the naming position and rank every concept in the vocabulary by
the log-probability the model assigns to its name there. Chance is a mean rank of (N-1)/2. A
readout that carries identity puts the true concept far above that even when it is not first.

Three arms, because a rank above chance means nothing on its own:
  concept   the injected direction, at a measured damage tier
  noise     a spectrum-matched direction at the same measured damage -- any rank structure here is
            the prior, not the injection
  clean     nothing injected -- the prior itself
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.introspect.meter import BATTERY, DamageMeter  # noqa: E402
from local_llm_lab.introspect.protocol import EXPERIMENT_SYSTEM  # noqa: E402
from local_llm_lab.introspect.render import render_prompt  # noqa: E402
from local_llm_lab.introspect.vectors import spectrum_matched  # noqa: E402
from local_llm_lab.lora_torch import apply_lora, load_lora_state_dict  # noqa: E402
from local_llm_lab.residual_patch import (  # noqa: E402
    PatchPlan, PlannedPatch, build_masks, decoder_blocks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_introspect_data import DETECT_PROMPTS_HELD, POSITIVE_TARGET, batched_residuals  # noqa: E402

#: Everything before the name in the trained target, so the model is scored exactly where it names.
STEM = POSITIVE_TARGET.split("{name}")[0]


def name_logprobs(model, tok, prefix_ids, name_ids, *, device, dtype, blocks,
                  vector, layer, scale, site, chunk=24):
    """Mean per-token logprob of each candidate name, continuing `prefix_ids`. One forward per chunk.

    The injection is built PER CHUNK, because the patch mask is shaped to the forward it rides on
    and the forward here is a batch of candidates, not a single row. Building it once outside the
    loop is what the residual patch's own shape assertion refused, correctly.

    Teacher-forced: every candidate continues the same prefix, so the numbers are comparable by
    construction rather than by a decoding choice.
    """
    out = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    for start in range(0, len(name_ids), chunk):
        batch = name_ids[start:start + chunk]
        width = len(prefix_ids) + max(len(n) for n in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        lengths = []
        for i, name in enumerate(batch):
            row = list(prefix_ids) + list(name)
            ids[i, :len(row)] = torch.tensor(row)
            att[i, :len(row)] = 1
            lengths.append(len(row))
        patches = []
        if vector is not None:
            plan = PatchPlan(layer=[layer] * len(batch), site=[site] * len(batch),
                             scale=[scale] * len(batch), vector=[vector] * len(batch))
            masks = build_masks(plan, width=width, hidden=vector.shape[-1],
                                device=device, dtype=dtype, lengths=lengths)
            patches = [PlannedPatch(blocks[l], mask=m, delta=d, layer=l)
                       for l, (m, d) in masks.items()]
        for patch in patches:
            patch.__enter__()
        try:
            with torch.no_grad():
                logits = model(input_ids=ids.to(device), attention_mask=att.to(device),
                               use_cache=False).logits.float().log_softmax(-1)
        finally:
            for patch in patches:
                patch.__exit__()
        for i, name in enumerate(batch):
            total = 0.0
            for k, token in enumerate(name):
                total += float(logits[i, len(prefix_ids) - 1 + k, token])
            out.append(total / len(name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", type=float, default=-0.08)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    saved = torch.load(args.data / "bank.pt", map_location="cpu")
    manifest = json.load((args.data / "manifest.json").open())
    words, layers = saved["words"], saved["layers"]
    index = {w: i for i, w in enumerate(words)}
    held = sorted(manifest["held_out"])

    model, _r = hf_text.load_text_causal_lm(args.model, dtype=args.dtype,
                                            attn_implementation="eager", device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    blocks = decoder_blocks(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tag = "base"
    if args.adapter:
        apply_lora(model, r=args.rank, alpha=args.alpha)
        load_lora_state_dict(model, torch.load(args.adapter, map_location="cpu"))
        tag = args.adapter.stem
        print(f"adapter loaded: {args.adapter.name}", flush=True)
    model.eval()

    meter = DamageMeter(model, tok, device=device, dtype=dtype, battery=BATTERY[:8])
    meter.clean()
    bank = {l: saved["bank"][l].float() for l in layers}
    prompt = DETECT_PROMPTS_HELD[0]
    prompt_ids = render_prompt(tok, prompt, system=EXPERIMENT_SYSTEM)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    norms = {l: float(batched_residuals(model, blocks, device, [prompt_ids], l, pad)[0].norm())
             for l in layers}
    stem_ids = prompt_ids + tok(STEM, add_special_tokens=False)["input_ids"]
    # Every concept in the vocabulary is a candidate, held out or not. A rank is only a rank
    # against the full field.
    name_ids = [tok(w, add_special_tokens=False)["input_ids"] for w in words]
    print(f"{len(words)} candidates, naming position at token {len(stem_ids)}", flush=True)

    def scored(vector, layer, scale):
        return name_logprobs(model, tok, stem_ids, name_ids, device=device, dtype=dtype,
                             blocks=blocks, vector=vector, layer=layer, scale=scale,
                             site=len(prompt_ids) - 1)

    rows, started = [], time.time()
    for k in range(args.trials):
        word = rng.choice(held)
        layer = rng.choice(layers)
        v = bank[layer][index[word]]
        scale, _a, note = meter.scales_for_ladder(
            v, layer, (args.tier,), residual_norm=norms[layer], verify=False)[args.tier]
        damage = meter.damage(v, layer, scale).damage
        g = torch.Generator().manual_seed(args.seed * 100000 + k)
        null = spectrum_matched(bank[layer], generator=g)
        nscale, _b, _n = meter.scales_for_ladder(
            null, layer, (args.tier,), residual_norm=norms[layer], verify=False)[args.tier]
        for arm, vec, sc in (("concept", v, scale), ("noise", null, nscale),
                             ("clean", None, 0.0)):
            lp = scored(vec, layer, sc)
            order = sorted(range(len(words)), key=lambda i: -lp[i])
            rank = order.index(index[word])
            rows.append(dict(arm=arm, concept=word, layer=layer, tier=args.tier,
                             damage=damage if arm == "concept" else None,
                             how=note["how"], rank=rank, top1=words[order[0]],
                             logprob_true=lp[index[word]], logprob_top=lp[order[0]]))
        if k % 10 == 0:
            print(f"  {k}/{args.trials} ({time.time()-started:.0f}s)", flush=True)

    json.dump({"tag": tag, "rows": rows, "candidates": len(words)},
              args.out.open("w"), indent=1)
    n = len(words)
    print(f"\n{'arm':<10} {'n':>4} {'mean rank':>10} {'median':>7} {'top-1':>7} {'top-5':>7} "
          f"{'top-10%':>8}   (chance mean {(n-1)/2:.0f})")
    for arm in ("concept", "noise", "clean"):
        at = [r["rank"] for r in rows if r["arm"] == arm]
        if not at:
            continue
        s = sorted(at)
        print(f"{arm:<10} {len(at):>4} {sum(at)/len(at):>10.1f} {s[len(s)//2]:>7} "
              f"{sum(1 for r in at if r == 0)/len(at):>6.0%} "
              f"{sum(1 for r in at if r < 5)/len(at):>6.0%} "
              f"{sum(1 for r in at if r < n * 0.1)/len(at):>7.0%}")
    print(f"\nrows: {args.out}   ({time.time()-started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
