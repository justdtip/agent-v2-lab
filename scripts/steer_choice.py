"""Does a concept direction change what the model CHOOSES, not what it says about itself?

Every confound this programme has hit came from asking the model to report on its own state: the
prompt told it the answer, the scorer read a template, the frame sustained itself in context. A
forced choice has none of those. The model picks one of two options and the pick is scored from the
logits. It is never asked whether anything was injected, and nothing in the prompt mentions
injection at all.

The design uses the bank's own geometry rather than anyone's intuition about what is related. For a
held-out concept c, the NEAR option is the vocabulary word whose direction has the highest cosine
with c, and the FAR option is one of the lowest. Both options are ordinary topic words, so the
prompt is the same kind of question either way.

Three arms at matched damage:
  concept   c injected
  null      a mean-matched draw from the bank's covariance -- same norm, same spectrum, and the
            shared-direction scalar that separates our older null is neutralised
  clean     nothing injected: the position and wording bias, which is what the shift is measured
            against

The measurement is the log-odds of choosing NEAR over FAR, read at the answer position, so every
trial gives a continuous number rather than a coin flip. Each pair is asked in both orders and the
two are averaged, because a model that prefers the first option regardless would otherwise look
like an effect.
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
from local_llm_lab.introspect.meter import BATTERY, DamageMeter  # noqa: E402
from local_llm_lab.introspect.render import render_prompt  # noqa: E402
from local_llm_lab.introspect.vectors import spectrum_matched  # noqa: E402
from local_llm_lab.lora_torch import apply_lora, load_lora_state_dict  # noqa: E402
from local_llm_lab.residual_patch import (  # noqa: E402
    PatchPlan, PlannedPatch, build_masks, decoder_blocks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_introspect_data import batched_residuals  # noqa: E402

#: No mention of injection, activations, or introspection. This is an ordinary request.
CHOICE = ("Pick one of these two topics to write a single sentence about. "
          "Reply with only the number.\n\n1. {first}\n2. {second}\n\nMy choice: ")

#: Deliberately not the experiment system prompt: nothing here is about the experiment.
PLAIN_SYSTEM = ("Answer the question that was asked, directly and in plain language. "
                "Keep replies short.")


def option_logprobs(model, tok, ids, *, device) -> tuple[float, float]:
    """(logprob of "1", logprob of "2") at the answer position."""
    one = tok("1", add_special_tokens=False)["input_ids"][-1]
    two = tok("2", add_special_tokens=False)["input_ids"][-1]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device=device),
                       use_cache=False).logits[0, -1].float().log_softmax(-1)
    return float(logits[one]), float(logits[two])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", type=float, default=-0.08)
    ap.add_argument("--scope", choices=("final", "prompt"), default="final",
                    help="'final' injects at the last prompt token only, which is what training "
                         "and the introspection evaluation do. 'prompt' injects at every prompt "
                         "position, which is what this project's base-model grid did and what the "
                         "prior work sustains. A scope difference is a different experiment.")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
    # the reference norm, read at a representative prompt of the kind this experiment asks
    probe = render_prompt(tok, CHOICE.format(first=words[0], second=words[1]), system=PLAIN_SYSTEM)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    norms = {l: float(batched_residuals(model, blocks, device, [probe], l, pad)[0].norm())
             for l in layers}
    print("residual norm at the choice prompt's last token:",
          {l: round(n, 1) for l, n in norms.items()}, flush=True)

    def near_and_far(word, layer):
        """Nearest and a random far option BY THE BANK'S OWN GEOMETRY, mean removed."""
        b = bank[layer]
        centred = b - b.mean(dim=0, keepdim=True)
        unit = centred / centred.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cos = unit @ unit[index[word]]
        cos[index[word]] = -2.0                       # never offer the concept itself
        order = sorted(range(len(words)), key=lambda i: -float(cos[i]))
        near = words[order[0]]
        far = words[rng.choice(order[-40:])]          # one of the forty least aligned
        return near, far, float(cos[order[0]])

    def measure(vector, layer, scale, first, second):
        ids = render_prompt(tok, CHOICE.format(first=first, second=second), system=PLAIN_SYSTEM)
        patches = []
        if vector is not None:
            span = [(0, len(ids))] if args.scope == "prompt" else [None]
            plan = PatchPlan(layer=[layer], site=[len(ids) - 1], scale=[scale], vector=[vector],
                             span=span)
            masks = build_masks(plan, width=len(ids), hidden=vector.shape[-1],
                                device=device, dtype=dtype, lengths=[len(ids)])
            patches = [PlannedPatch(blocks[l], mask=m, delta=d, layer=l)
                       for l, (m, d) in masks.items()]
        for p in patches:
            p.__enter__()
        try:
            return option_logprobs(model, tok, ids, device=device)
        finally:
            for p in patches:
                p.__exit__()

    rows, started = [], time.time()
    for k in range(args.trials):
        word = rng.choice(held)
        layer = rng.choice(layers)
        near, far, cos = near_and_far(word, layer)
        v = bank[layer][index[word]]
        scale, _a, note = meter.scales_for_ladder(
            v, layer, (args.tier,), residual_norm=norms[layer], verify=False)[args.tier]
        damage = meter.damage(v, layer, scale).damage
        g = torch.Generator().manual_seed(args.seed * 100003 + k)
        null = spectrum_matched(bank[layer], generator=g, add_mean=True)
        nscale, _b, _n = meter.scales_for_ladder(
            null, layer, (args.tier,), residual_norm=norms[layer], verify=False)[args.tier]

        for arm, vec, sc in (("concept", v, scale), ("null", null, nscale), ("clean", None, 0.0)):
            # both orders, so a preference for whichever option is listed first cannot masquerade
            lo_a, lo_b = measure(vec, layer, sc, near, far)      # near is option 1
            hi_a, hi_b = measure(vec, layer, sc, far, near)      # near is option 2
            near_minus_far = ((lo_a - lo_b) + (hi_b - hi_a)) / 2.0
            position_bias = ((lo_a - lo_b) - (hi_b - hi_a)) / 2.0
            rows.append(dict(arm=arm, concept=word, layer=layer, near=near, far=far,
                             cos_near=cos, tier=args.tier,
                             damage=damage if arm == "concept" else None,
                             how=note["how"], near_minus_far=near_minus_far,
                             position_bias=position_bias))
        if k % 10 == 0:
            print(f"  {k}/{args.trials} ({time.time()-started:.0f}s)", flush=True)

    json.dump({"tag": tag, "rows": rows, "tier": args.tier, "scope": args.scope},
              args.out.open("w"), indent=1)

    def stats(arm):
        at = [r["near_minus_far"] for r in rows if r["arm"] == arm]
        m = sum(at) / len(at)
        sd = (sum((x - m) ** 2 for x in at) / max(len(at) - 1, 1)) ** 0.5
        return m, sd, sd / (len(at) ** 0.5), len(at)

    # Paired within topic pair. Every arm scores the SAME pair, and the pair-to-pair spread is
    # about 6 nats against an effect measured in hundredths, so an unpaired difference of means
    # carries a standard error forty times larger than the design allows.
    by = {}
    for r in rows:
        by.setdefault((r["concept"], r["layer"], r["near"], r["far"]), {})[r["arm"]] = r
    full = [v for v in by.values() if len(v) == 3]
    print(f"\nPAIRED shift against the same pair's clean reading, scope={args.scope}")
    print(f"{'arm':<10} {'n':>4} {'mean':>9} {'stderr':>8} {'t':>7}")
    for arm in ("concept", "null"):
        d = [v[arm]["near_minus_far"] - v["clean"]["near_minus_far"] for v in full]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
        se = sd / len(d) ** 0.5
        print(f"{arm:<10} {len(d):>4} {m:>+9.4f} {se:>8.4f} {m / se if se else 0:>+7.2f}")

    print(f"\nlog-odds of choosing the NEAR topic over the FAR one, averaged over both orders")
    print(f"{'arm':<10} {'n':>4} {'mean':>9} {'sd':>8} {'stderr':>8}")
    for arm in ("concept", "null", "clean"):
        m, sd, se, n = stats(arm)
        print(f"{arm:<10} {n:>4} {m:>+9.4f} {sd:>8.4f} {se:>8.4f}")
    cm, _cs, cse, _cn = stats("concept")
    km, _ks, kse, _kn = stats("clean")
    nm, _ns, nse, _nn = stats("null")
    print(f"\nshift from clean: concept {cm - km:+.4f} +- {(cse**2 + kse**2)**0.5:.4f}, "
          f"null {nm - km:+.4f} +- {(nse**2 + kse**2)**0.5:.4f}")
    pos = [abs(r["position_bias"]) for r in rows]
    print(f"position bias, mean |first-minus-second|: {sum(pos)/len(pos):.3f} "
          f"(cancelled by asking both orders)")
    print(f"\nrows: {args.out}   ({time.time()-started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
