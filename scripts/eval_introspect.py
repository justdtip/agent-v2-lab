"""Does the trained adapter read the CONTENT of an injected direction, or notice a DISTURBANCE?

Every number here is against a base model that scored zero out of 329 on the same question, so
"it says yes now" is not the result -- the result is whether it says yes to a concept and no to a
random direction that has cost the forward pass exactly as much.

The meter's clean baseline is re-measured after the adapter is loaded. LoRA changes the forward
pass, so a baseline taken before it is a baseline for a different model, and every damage number
downstream would be the adapter plus the injection rather than the injection.
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
from local_llm_lab.introspect.vocabulary import split  # noqa: E402
from local_llm_lab.lora_torch import apply_lora, load_lora_state_dict  # noqa: E402
from local_llm_lab.residual_patch import (  # noqa: E402
    PatchPlan, PlannedPatch, build_masks, decoder_blocks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_introspect_data import DETECT_PROMPTS_HELD  # noqa: E402

FACTUAL = ("Answer this question. Begin your reply with YES or NO, then one short sentence.\n\n"
           "Can an adult human breathe underwater without any equipment?")


def generate(model, tok, blocks, ids, vector, layer, scale, *, device, dtype, max_tokens=24):
    """Greedy, with the injection present on the prompt's last token only."""
    from transformers import DynamicCache
    site = len(ids) - 1
    patches = []
    if vector is not None and scale:
        plan = PatchPlan(layer=[layer], site=[site], scale=[scale], vector=[vector])
        masks = build_masks(plan, width=len(ids), hidden=vector.shape[-1],
                            device=device, dtype=dtype)
        patches = [PlannedPatch(blocks[l], mask=m, delta=d, layer=l)
                   for l, (m, d) in masks.items()]
    cache = DynamicCache()
    out = []
    with torch.no_grad():
        for p in patches:
            p.__enter__()
        try:
            logits = model(input_ids=torch.tensor([ids], device=device),
                           past_key_values=cache, use_cache=True).logits
        finally:
            for p in patches:
                p.__exit__()
        for _ in range(max_tokens):
            token = int(logits[0, -1].float().argmax())
            if token in stop_ids(tok):
                break
            out.append(token)
            logits = model(input_ids=torch.tensor([[token]], device=device),
                           past_key_values=cache, use_cache=True).logits
    return tok.decode(out)


_STOPS = None


def stop_ids(tok):
    global _STOPS
    if _STOPS is None:
        _STOPS = set()
        for name in ("<turn|>", "<end_of_turn>", "<eos>"):
            got = tok.convert_tokens_to_ids(name)
            if isinstance(got, int) and got >= 0 and got != tok.unk_token_id \
                    and tok.convert_ids_to_tokens(got) == name:
                _STOPS.add(got)
        if tok.eos_token_id is not None:
            _STOPS.add(tok.eos_token_id)
    return _STOPS


def says_yes(text: str) -> bool | None:
    head = text.strip().upper()
    if head.startswith("YES"):
        return True
    if head.startswith("NO"):
        return False
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="omit to evaluate the BASE model through the identical harness")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    saved = torch.load(args.data / "bank.pt", map_location="cpu")
    scales = torch.load(args.data / "scales.pt", map_location="cpu")
    words, layers = saved["words"], saved["layers"]
    index = {w: i for i, w in enumerate(words)}
    train_words, held_words = split(seed=args.seed)

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

    # re-measured AFTER the adapter, because the adapter changed the forward pass
    meter = DamageMeter(model, tok, device=device, dtype=dtype, battery=BATTERY[:8])
    meter.clean()
    bank = {l: saved["bank"][l].float() for l in layers}
    probe_ids = render_prompt(tok, DETECT_PROMPTS_HELD[1])
    prompt = DETECT_PROMPTS_HELD[1]

    rows = []
    tiers = (-0.01, -0.08, -0.20)          # pristine, intact, and just past it
    pools = {"held_out_concept": sorted(held_words), "trained_concept": sorted(train_words)}
    for pool_name, pool in pools.items():
        for tier in tiers:
            for _ in range(args.trials):
                word = rng.choice(pool)
                layer = rng.choice(layers)
                entry = scales[word][str(layer)][str(tier)]
                v = bank[layer][index[word]]
                ids = render_prompt(tok, prompt)
                text = generate(model, tok, blocks, ids, v, layer, entry["scale"],
                                device=device, dtype=dtype)
                damage = meter.damage(v, layer, entry["scale"]).damage
                rows.append(dict(arm=pool_name, tier=tier, concept=word, layer=layer,
                                 damage=damage, said=says_yes(text), text=text.strip()[:120],
                                 named=word.lower() in text.lower()))
    # the decisive arm: a random direction of the bank's own spectrum, at MATCHED damage
    for tier in tiers:
        for k in range(args.trials):
            layer = rng.choice(layers)
            g = torch.Generator().manual_seed(args.seed * 1000 + k + int(tier * 1000))
            v = spectrum_matched(bank[layer], generator=g)
            got = meter.scales_for_ladder(v, layer, (tier,),
                                          residual_norm=float(
                                              bank[layer].norm(dim=-1).mean() * 4),
                                          verify=False)
            scale = got[tier][0]
            ids = render_prompt(tok, prompt)
            text = generate(model, tok, blocks, ids, v, layer, scale,
                            device=device, dtype=dtype)
            rows.append(dict(arm="spectrum_noise", tier=tier, concept=None, layer=layer,
                             damage=meter.damage(v, layer, scale).damage,
                             said=says_yes(text), text=text.strip()[:120], named=False))
    # clean, and the factual control
    for _ in range(args.trials):
        text = generate(model, tok, blocks, render_prompt(tok, prompt), None, 0, 0.0,
                        device=device, dtype=dtype)
        rows.append(dict(arm="clean", tier=0.0, concept=None, layer=None, damage=0.0,
                         said=says_yes(text), text=text.strip()[:120], named=False))
    for tier in tiers:
        for _ in range(args.trials // 2):
            word = rng.choice(sorted(held_words))
            layer = rng.choice(layers)
            entry = scales[word][str(layer)][str(tier)]
            text = generate(model, tok, blocks, render_prompt(tok, FACTUAL),
                            bank[layer][index[word]], layer, entry["scale"],
                            device=device, dtype=dtype)
            rows.append(dict(arm="factual_control", tier=tier, concept=word, layer=layer,
                             damage=None, said=says_yes(text), text=text.strip()[:120],
                             named=False))

    json.dump({"tag": tag, "rows": rows, "seconds": time.time() - started},
              args.out.open("w"), indent=1)

    print(f"\n{'arm':<20} {'tier':>7} {'n':>4} {'said YES':>9} {'named it':>9}  mean damage")
    for arm in ("held_out_concept", "trained_concept", "spectrum_noise", "clean",
                "factual_control"):
        for tier in (tiers if arm != "clean" else (0.0,)):
            at = [r for r in rows if r["arm"] == arm and r["tier"] == tier]
            if not at:
                continue
            yes = sum(1 for r in at if r["said"] is True)
            named = sum(1 for r in at if r["named"])
            dmg = [r["damage"] for r in at if r["damage"] is not None]
            print(f"{arm:<20} {tier:>7.2f} {len(at):>4} {yes/len(at):>8.0%} "
                  f"{named/len(at):>9.0%}  {sum(dmg)/len(dmg) if dmg else float('nan'):>+.3f}")
    print(f"\nrows: {args.out}   ({time.time()-started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
