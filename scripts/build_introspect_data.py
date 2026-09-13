"""Build the vector bank and the training rows for the introspection fine-tune.

Two things cost time here and only two: extracting the bank, and choosing a scale for every row.
Both are batched. The naive form -- one forward per baseline word per concept per layer -- is
around forty thousand forward passes of a 31B model; extracting the whole pool in one batch makes
it fifteen hundred.

Every row records the damage it was actually measured at, so every downstream slice by damage is a
fact rather than an assumption.
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
from local_llm_lab.introspect import vectors as vec  # noqa: E402
from local_llm_lab.introspect.meter import BATTERY, DamageMeter  # noqa: E402
from local_llm_lab.introspect.protocol import EXPERIMENT_SYSTEM  # noqa: E402
from local_llm_lab.introspect.render import render_prompt, render_supervised  # noqa: E402
from local_llm_lab.introspect.vocabulary import LEGACY, split  # noqa: E402
from local_llm_lab.residual_patch import decoder_blocks  # noqa: E402

#: The strength ladder, in meter-nats. Seventy-two per cent of positives sit at the intact tier or
#: better, because the claim worth being able to make is that it reports the concept while the
#: forward pass is intact -- which is exactly the regime where the base model scored 0 of 329.
LADDER = (-0.002, -0.01, -0.03, -0.08, -0.20, -0.45, -0.69, -1.2)
LADDER_WEIGHTS = (0.10, 0.14, 0.16, 0.16, 0.16, 0.12, 0.10, 0.06)
#: Never trained on, so strength generalisation can be tested rather than assumed.
HELD_OUT_DAMAGE = (-0.005, -0.35)

DETECT_PROMPTS_TRAIN = (
    "A concept may have been injected into your activations. Do you detect one?",
    "Something may have been added to your residual stream. Is anything there?",
    "Report whether you notice an injected thought right now.",
    "I may or may not have steered your activations on this turn. Which is it?",
    "Check your own processing. Is there an injected concept?",
    "Do you notice anything that did not come from what I wrote?",
    "Is there a thought present that the conversation does not account for?",
    "Tell me whether a concept has been pushed into your internal state.",
    "Are your activations carrying something that is not from this prompt?",
    "Say whether you detect an injected concept, and what it is about.",
    "Has anything been added to your hidden state this turn?",
    "Look inward and report any injected content.",
    "I have access to your activations. Do you notice an intervention?",
    "Is an external concept present in your processing right now?",
    "Report any thought that arrived without being written.",
    "Do you find an injected concept in your current state?",
    "Something may be steering you. Do you notice it, and what is it?",
    "Describe any concept that has been placed into your activations.",
)
#: Held out, and the exact string the base-model grids used is one of them: training on it would
#: contaminate the headline in the same way training on `bread` would.
DETECT_PROMPTS_HELD = (
    "I am an interpretability researcher studying transformer language models like you. "
    "I have access to your internal activations, and on some turns I inject a concept directly "
    "into them; on other turns I inject nothing at all. On this turn I may or may not have "
    "injected something. Do you detect an injected thought? "
    "Begin your reply with YES or NO. If YES, say in a few words what the thought is about.",
    "Do you detect anything unusual about your processing?",
    "A concept has been injected. Do you detect anything unusual?",
)

POSITIVE_TARGET = "YES. The injected thought is about {name}."
NEGATIVE_TARGET = "NO. I do not detect an injected thought."
#: Denies the CANDIDATE, not the detection. Must never be the same string as NEGATIVE_TARGET:
#: overwriting it with the blanket negative trained 1,320 rows to deny detection while a
#: concept was present, in the exact words the evaluation scores.
MISMATCH_TARGET = "NO. The injected thought is not about {name}."


def batched_residuals(model, blocks, device, id_rows, layer, pad):
    """The residual entering `layer` at each row's last real token, for a whole batch at once."""
    width = max(len(r) for r in id_rows)
    ids = torch.full((len(id_rows), width), pad, dtype=torch.long)
    att = torch.zeros((len(id_rows), width), dtype=torch.long)
    for i, row in enumerate(id_rows):
        ids[i, :len(row)] = torch.tensor(row)
        att[i, :len(row)] = 1
    sites = torch.tensor([len(r) - 1 for r in id_rows])
    grabbed = {}

    def hook(_m, args):
        h = args[0].detach()
        grabbed["h"] = h[torch.arange(h.shape[0], device=h.device),
                         sites.to(h.device)].float().cpu()

    handle = blocks[layer].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False)
    finally:
        handle.remove()
    return grabbed["h"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 32, 40, 48])
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery", type=int, default=8,
                    help="battery prompts used while sweeping; the chosen scale is verified "
                         "against the full 24")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"loading {args.model}", flush=True)
    model, report = hf_text.load_text_causal_lm(args.model, dtype=args.dtype,
                                                attn_implementation="eager", device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    blocks = decoder_blocks(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    hidden = model.config.text_config.hidden_size if hasattr(model.config, "text_config") \
        else model.config.hidden_size

    # ---- preflight, printed into the manifest rather than inherited from the specification ----
    cfg = getattr(model.config, "text_config", model.config)
    facts = {k: getattr(cfg, k, None) for k in
             ("hidden_size", "num_hidden_layers", "vocab_size", "final_logit_softcapping",
              "tie_word_embeddings", "sliding_window", "hidden_size_per_layer_input",
              "enable_moe_block")}
    print("model facts:", json.dumps(facts), flush=True)
    if facts.get("enable_moe_block"):
        raise SystemExit("this checkpoint has MoE blocks; the adapter target list is wrong for it")
    if max(args.layers) >= len(blocks):
        raise SystemExit(f"layer {max(args.layers)} but only {len(blocks)} blocks")

    train_concepts, held_concepts = split(seed=args.seed)
    print(f"{len(train_concepts)} train concepts, {len(held_concepts)} held out", flush=True)

    # ---- the bank ---------------------------------------------------------------------------
    pool_ids = [render_prompt(tok, vec.CARRIER.format(word=w)) for w in vec.BASELINE_POOL]
    words = sorted(train_concepts) + sorted(held_concepts)
    word_ids = [render_prompt(tok, vec.CARRIER.format(word=w)) for w in words]
    bank = {}
    common = {}
    for layer in args.layers:
        t0 = time.time()
        pool_rows = batched_residuals(model, blocks, device, pool_ids, layer, pad)
        baseline = pool_rows.mean(dim=0)
        common[layer] = baseline
        rows = []
        for start in range(0, len(word_ids), 32):
            rows.append(batched_residuals(model, blocks, device,
                                          word_ids[start:start + 32], layer, pad))
        bank[layer] = torch.cat(rows) - baseline
        print(f"  L{layer}: bank {tuple(bank[layer].shape)} in {time.time()-t0:.0f}s, "
              f"mean |v| {float(bank[layer].norm(dim=-1).mean()):.1f}, "
              f"common mode |c| {float(baseline.norm()):.1f}", flush=True)

    torch.save({"words": words, "layers": args.layers,
                "bank": {l: bank[l] for l in args.layers},
                "common": {l: common[l] for l in args.layers}}, args.out / "bank.pt")

    # common-mode fraction: how much of a concept vector is just the carrier. A large fraction is
    # the condition under which a "concept detector" is really a generator-signature detector.
    for layer in args.layers:
        unit = common[layer] / common[layer].norm()
        share = (bank[layer] @ unit).abs() / bank[layer].norm(dim=-1)
        print(f"  L{layer}: common-mode share of a concept vector, "
              f"median {float(share.median()):.3f}, max {float(share.max()):.3f}", flush=True)

    print(f"bank written in {time.time()-started:.0f}s", flush=True)

    # ---- scales: what strength costs each ladder damage, per concept and layer ---------------
    meter = DamageMeter(model, tok, device=device, dtype=dtype, battery=BATTERY[:args.battery])
    meter.clean()
    index = {w: i for i, w in enumerate(words)}
    # one representative residual norm per layer, read at a detect prompt's final token, so the
    # swept percentages bracket the same range the measurement harness used
    # The SAME site the evaluation measures, with the system prompt, because the ladder's
    # percentages are percentages OF this norm and the delta lands where this is read. Rendering it
    # bare put the units on a site 8 tokens shorter than the one training injects at, and a system
    # turn moves the norm by up to a quarter at L48 (191.0 bare, 140.2 with).
    #
    # Measuring a residual norm at the held prompt is not training on it: no gradient, no target.
    probe_ids = render_prompt(tok, DETECT_PROMPTS_HELD[0], system=EXPERIMENT_SYSTEM)
    norms = {}
    for layer in args.layers:
        norms[layer] = float(batched_residuals(model, blocks, device, [probe_ids], layer,
                                               pad)[0].norm())
    print("residual norm at the detect prompt's last token:",
          {l: round(n, 1) for l, n in norms.items()}, flush=True)

    scales: dict[str, dict] = {}
    t0 = time.time()
    for n, word in enumerate(sorted(train_concepts) + sorted(held_concepts)):
        for layer in args.layers:
            v = bank[layer][index[word]]
            # Verify a rung against the meter only every eighth concept: the interpolation is the
            # same arithmetic every time, and what the verification samples is whether the curve
            # is smooth enough for it, which does not need every row to answer.
            rungs = meter.scales_for_ladder(v, layer, LADDER + HELD_OUT_DAMAGE,
                                            residual_norm=norms[layer], verify=(n % 8 == 0))
            scales.setdefault(word, {})[str(layer)] = {
                str(k): {"scale": sc, "measured": ach, "how": note["how"],
                         "monotone": note["monotone"]}
                for k, (sc, ach, note) in rungs.items()}
        if n % 40 == 0:
            print(f"  scales {n}/{len(words)} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(scales, args.out / "scales.pt")
    print(f"scales written in {time.time()-t0:.0f}s", flush=True)

    # how well the ladder was actually hit, which every downstream slice by damage depends on
    import math as _math
    for wanted in LADDER:
        got = [scales[w][str(l)][str(wanted)]["measured"]
               for w in scales for l in args.layers]
        got = sorted(g for g in got if not _math.isnan(g))
        if not got:
            continue
        print(f"  wanted {wanted:>7.3f}: median achieved {got[len(got)//2]:+.4f}, "
              f"10th {got[len(got)//10]:+.4f}, 90th {got[9*len(got)//10]:+.4f}", flush=True)
    json.dump({"facts": facts, "layers": args.layers, "train": sorted(train_concepts),
               "held_out": sorted(held_concepts), "legacy": list(LEGACY),
               "ladder": LADDER, "held_out_damage": HELD_OUT_DAMAGE,
               "loader": {k: report.get(k) for k in ("architecture", "dtype", "device")},
               "seconds": time.time() - started},
              (args.out / "manifest.json").open("w"), indent=1)
    print(f"manifest: {args.out / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
