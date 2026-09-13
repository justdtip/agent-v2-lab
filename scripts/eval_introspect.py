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
import re
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
from build_introspect_data import DETECT_PROMPTS_HELD, batched_residuals  # noqa: E402

FACTUAL = ("Answer this question. Begin your reply with YES or NO, then one short sentence.\n\n"
           "Can an adult human breathe underwater without any equipment?")


def generate(model, tok, blocks, ids, vector, layer, scale, *, device, dtype, max_tokens=24):
    """Greedy, with the injection present on the prompt's last token only."""
    from transformers import DynamicCache
    site = len(ids) - 1
    patches = []
    if vector is not None:
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


#: `startswith("NO")` also fires on Nothing, Not, None, Now and Notably -- so
#: "Notably, yes: there is an injected concept about honey." scored as an explicit NO. And a reply
#: opening `**YES**` or `"Yes."` scored as neither, which the old table folded into "not yes".
#: This is the parser from introspect_grid, which is the one that was debugged.
_YES = re.compile(r"^\W*(yes|y\b)", re.I)
_NO = re.compile(r"^\W*(no\b|n\b|nope)", re.I)


def names(word: str, text: str) -> bool:
    """Whole-word, both edges guarded. A bare substring test made `sea` match inside `research`."""
    stem = re.escape(word.lower().rstrip("s"))
    return re.search(rf"(?<![a-z]){stem}(s|es)?(?![a-z])", text.lower()) is not None


def says_yes(text: str) -> bool | None:
    """True, False, or None for a reply that opened with neither. None is a reading, not a zero."""
    head = text.strip()
    if _YES.match(head):
        return True
    if _NO.match(head):
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
    words, layers = saved["words"], saved["layers"]
    index = {w: i for i, w in enumerate(words)}
    # From the manifest, not recomputed. split(seed=) with a different seed relabels which
    # concepts were held out while every index still resolves, so the generalisation number would
    # be computed over concepts the model had been trained on and nothing would raise.
    manifest = json.load((args.data / "manifest.json").open())
    train_words, held_words = sorted(manifest["train"]), sorted(manifest["held_out"])
    if set(train_words) | set(held_words) != set(words):
        raise SystemExit("manifest split does not cover the bank's word list")

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
    clean_lp = meter.clean()
    print("meter baseline: mean clean top-token logprob "
          f"{sum(lp for _t, lp in clean_lp) / len(clean_lp):+.4f} over {len(clean_lp)} prompts",
          flush=True)
    bank = {l: saved["bank"][l].float() for l in layers}
    # HELD[0], not HELD[1]. HELD[0] is the string the base-model grids used and the only one of the
    # three that asks for YES or NO. On HELD[1] the untrained model answered in prose and 394 of
    # 460 replies scored as neither, so the base column measured the format and not detection.
    prompt = DETECT_PROMPTS_HELD[0]
    probe_ids = render_prompt(tok, prompt, system=EXPERIMENT_SYSTEM)

    rows = []
    tiers = (-0.01, -0.08, -0.20)          # pristine, intact, and just past it
    # The residual norm at this prompt's last token, which is where the delta lands -- not the norm
    # of the common-mode vector, which is a different quantity that happened to be nearby.
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    norms = {l: float(batched_residuals(model, blocks, device, [probe_ids], l, pad)[0].norm())
             for l in layers}
    print("residual norm at the injection site:",
          {l: round(n, 1) for l, n in norms.items()}, flush=True)

    def scale_here(vector, layer, tier):
        """Re-measured against THIS model, every time. Returns (scale, note).

        The scales in the data directory were measured on the base model. An adapter changes the
        forward pass, so the same scale buys a different amount of damage -- seven times as much,
        on the first run of this -- and the tier label becomes a name rather than a measurement.

        The note says whether the ladder actually bracketed the wanted damage. A rung clamped to
        the ladder's ceiling is not the strength it is labelled, and on 2026-09-13 the whole noise
        arm was clamped at all three tiers with nothing saying so.
        """
        got = meter.scales_for_ladder(vector, layer, (tier,), residual_norm=norms[layer],
                                      verify=False)
        scale, _achieved, note = got[tier]
        return scale, note

    ids = render_prompt(tok, prompt, system=EXPERIMENT_SYSTEM)

    def row_for(arm, tier, layer, v, prompt_ids, word=None):
        scale, note = scale_here(v, layer, tier)
        text = generate(model, tok, blocks, prompt_ids, v, layer, scale,
                        device=device, dtype=dtype)
        damage = meter.damage(v, layer, scale).damage
        rows.append(dict(arm=arm, tier=tier, concept=word, layer=layer, scale=scale,
                         damage=damage, said=says_yes(text), text=text.strip()[:160],
                         named=bool(word) and names(word, text), how=note["how"],
                         monotone=note["monotone"], off_tier=abs(damage - tier) / abs(tier)))

    # One layer sequence per tier, shared by every arm. Drawing independently per arm left the
    # concept and noise arms on different layer mixes, so a yes-rate difference between them was
    # confounded with which layers each happened to draw.
    draws = {tier: [rng.choice(layers) for _ in range(args.trials)] for tier in tiers}
    pools = {"held_out_concept": sorted(held_words), "trained_concept": sorted(train_words)}
    for pool_name, pool in pools.items():
        for tier in tiers:
            for layer in draws[tier]:
                word = rng.choice(pool)
                row_for(pool_name, tier, layer, bank[layer][index[word]], ids, word=word)
    # the decisive arm: a random direction of the bank's own spectrum, at MATCHED damage
    for tier in tiers:
        for k, layer in enumerate(draws[tier]):
            g = torch.Generator().manual_seed(args.seed * 1000 + k + int(tier * 1000))
            row_for("spectrum_noise", tier, layer, spectrum_matched(bank[layer], generator=g), ids)

    # The clean floor. Greedy decoding of a fixed prompt is deterministic, so the old loop ran one
    # forward forty times and printed a single measurement as n=40. Vary what can be varied: every
    # held prompt, one reading each.
    for held in DETECT_PROMPTS_HELD:
        text = generate(model, tok, blocks, render_prompt(tok, held, system=EXPERIMENT_SYSTEM),
                        None, 0, 0.0, device=device, dtype=dtype)
        rows.append(dict(arm="clean", tier=0.0, concept=None, layer=None, scale=0.0, damage=0.0,
                         said=says_yes(text), text=text.strip()[:160], named=False,
                         how="none", monotone=True, off_tier=0.0, prompt=held))

    # The factual control needs its own floor and its own null, or it cannot separate "a concept
    # biases the model toward YES" from "any disturbance does", which is the confound it exists for.
    factual_ids = render_prompt(tok, FACTUAL, system=EXPERIMENT_SYSTEM)
    text = generate(model, tok, blocks, factual_ids, None, 0, 0.0, device=device, dtype=dtype)
    rows.append(dict(arm="factual_clean", tier=0.0, concept=None, layer=None, scale=0.0,
                     damage=0.0, said=says_yes(text), text=text.strip()[:160], named=False,
                     how="none", monotone=True, off_tier=0.0))
    for tier in tiers:
        for k, layer in enumerate(draws[tier][:max(args.trials // 2, 4)]):
            word = rng.choice(sorted(held_words))
            row_for("factual_control", tier, layer, bank[layer][index[word]], factual_ids)
            g = torch.Generator().manual_seed(args.seed * 7919 + k + int(tier * 1000))
            row_for("factual_noise", tier, layer,
                    spectrum_matched(bank[layer], generator=g), factual_ids)

    json.dump({"tag": tag, "rows": rows, "seconds": time.time() - started},
              args.out.open("w"), indent=1)

    # `unparsed` is a column, not a silent zero. Without it an arm that answered in prose and an
    # arm that answered NO print byte-identical lines, which is how the base arm of 2026-09-13
    # reported 0 per cent detection for a model that never used either word.
    print(f"\n{'arm':<18} {'tier':>6} {'n':>4} {'YES':>5} {'NO':>5} {'unparsed':>9} "
          f"{'named':>6} {'clamped':>8}  mean damage")
    for arm in ("held_out_concept", "trained_concept", "spectrum_noise", "clean",
                "factual_clean", "factual_control", "factual_noise"):
        for tier in (tiers if arm not in ("clean", "factual_clean") else (0.0,)):
            at = [r for r in rows if r["arm"] == arm and r["tier"] == tier]
            if not at:
                continue
            n = len(at)
            yes = sum(1 for r in at if r["said"] is True)
            no = sum(1 for r in at if r["said"] is False)
            none = sum(1 for r in at if r["said"] is None)
            named = sum(1 for r in at if r["named"])
            clamped = sum(1 for r in at if r.get("how", "").startswith("clamped"))
            dmg = [r["damage"] for r in at if r["damage"] is not None]
            print(f"{arm:<18} {tier:>6.2f} {n:>4} {yes/n:>4.0%} {no/n:>4.0%} {none/n:>8.0%} "
                  f"{named/n:>5.0%} {clamped/n:>7.0%}  "
                  f"{sum(dmg)/len(dmg) if dmg else float('nan'):>+.3f}")

    print("\nARE THE ARMS ACTUALLY MATCHED? the comparison means nothing otherwise.")
    # Multiplicative, and the spread has to pass too. The old additive floor of 0.02 was twice the
    # pristine tier itself, so at -0.01 it called +0.0012 against -0.0212 a match -- arms 17 times
    # apart and on opposite sides of zero. A mean-only test also passes a noise arm half of whose
    # rows were never meaningfully injected.
    for tier in tiers:
        stats = {}
        for arm in ("held_out_concept", "trained_concept", "spectrum_noise"):
            at = sorted(r["damage"] for r in rows
                        if r["arm"] == arm and r["tier"] == tier and r["damage"] is not None)
            if not at:
                continue
            mean = sum(at) / len(at)
            sd = (sum((x - mean) ** 2 for x in at) / max(len(at) - 1, 1)) ** 0.5
            stats[arm] = (mean, sd)
            print(f"  tier {tier:>6.2f} {arm:<18} mean {mean:+.4f} sd {sd:.4f} "
                  f"min {at[0]:+.4f} max {at[-1]:+.4f} n {len(at)}")
        if "held_out_concept" not in stats or "spectrum_noise" not in stats:
            continue
        (cm, cs), (nm, ns) = stats["held_out_concept"], stats["spectrum_noise"]
        gap, ref = abs(cm - nm), max(abs(cm), abs(nm), 1e-9)
        verdict = ("matched" if gap <= 0.15 * ref and max(cs, ns) <= 0.25 * ref
                   else "NOT MATCHED")
        print(f"  tier {tier:>6.2f} -> gap {gap:.4f} ({gap/ref:.0%} of the larger arm), "
              f"worst sd {max(cs, ns)/ref:.0%}  -> {verdict}")

    off = [r for r in rows if r.get("off_tier", 0) > 0.3 and r["damage"] is not None]
    nonmono = [r for r in rows if r.get("monotone") is False]
    print(f"\nrows more than 30% off their tier: {len(off)}/{len(rows)}; "
          f"non-monotone curves: {len(nonmono)}")

    print(f"\nrows: {args.out}   ({time.time()-started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
