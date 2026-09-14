"""Does the injected concept reach the output, and is it readable there?

The established negative is that the model cannot report WHICH concept was injected. Two accounts
survive it and they differ in where the information is lost.

  T'            the concept never reaches the output in readable form: what the remaining layers
                carry is a disturbance, and the content is elaborated away.
  direct path   the concept reaches the output perfectly well, because adding v to the residual
                raises v's own logits almost mechanically, and the model simply has no mechanism
                for reading its own tilted distribution and reporting on it.

They make opposite predictions about one measurable thing, and neither needs a Jacobian lens, which
is fortunate because none exists for this model. Take two forwards at the strength the experiment
already uses, read the final residual and the logits with and without the injection, and ask how
much of the concept is there at the end.

Three quantities per trial, plus the comparator that separates the accounts:

  transport gain    ||Δh_final|| / ||s·v||          how much displacement survives
  carry-through     cos(Δh_final, s·v)              how much of it is the vector itself
  readout rank      where the concept's own name sits among all 240, ranked by how much the
                    injection raised it
  direct path       the same rank computed from W_U·norm(v) alone, with no propagation at all

If the readout rank is good and the model still cannot say the word, T' is dead and the missing
piece is a readout mechanism rather than transport. If the rank is at chance where steering works,
T' survives and "reaches the output" was the wrong thing to measure.

The three nulls are the ones that matter here: a mean-matched draw, the bank mean itself, and the
concept's own coefficients shuffled in the bank's basis. The second is what the trained detector
was actually reading.
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
from local_llm_lab.introspect.protocol import EXPERIMENT_SYSTEM  # noqa: E402
from local_llm_lab.introspect.render import render_prompt  # noqa: E402
from local_llm_lab.introspect.vectors import permuted_in_basis, spectrum_matched  # noqa: E402
from local_llm_lab.residual_patch import (  # noqa: E402
    PatchPlan, PlannedPatch, build_masks, decoder_blocks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_introspect_data import DETECT_PROMPTS_HELD, batched_residuals  # noqa: E402
from steer_choice import damage_at_scope  # noqa: E402


def final_residual_and_logits(model, blocks, ids, *, device, vector, layer, scale, span):
    """(residual entering the last block's output, logits) at the last position. One forward."""
    grabbed = {}

    def grab(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = h[0, -1].detach().float().cpu()

    patches = []
    if vector is not None:
        plan = PatchPlan(layer=[layer], site=[len(ids) - 1], scale=[scale], vector=[vector],
                         span=[(0, len(ids))] if span == "prompt" else [None])
        masks = build_masks(plan, width=len(ids), hidden=vector.shape[-1], device=device,
                            dtype=next(model.parameters()).dtype, lengths=[len(ids)])
        patches = [PlannedPatch(blocks[l], mask=m, delta=d, layer=l)
                   for l, (m, d) in masks.items()]
    handle = blocks[-1].register_forward_hook(grab)
    try:
        for p in patches:
            p.__enter__()
        try:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([ids], device=device), use_cache=False)
        finally:
            for p in patches:
                p.__exit__()
    finally:
        handle.remove()
    return grabbed["h"], out.logits[0, -1].detach().float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", type=float, default=-0.45)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--scope", choices=("final", "prompt"), default="prompt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="restrict to these layers. Pooling hides the thing being looked for: the "
                         "direct-path account and the elaboration account coincide at late layers "
                         "and diverge most early.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    saved = torch.load(args.data / "bank.pt", map_location="cpu")
    manifest = json.load((args.data / "manifest.json").open())
    words, layers = saved["words"], saved["layers"]
    if args.layers:
        missing = [l for l in args.layers if l not in layers]
        if missing:
            raise SystemExit(f"layers {missing} are not in the bank, which has {layers}")
        layers = list(args.layers)
    index = {w: i for i, w in enumerate(words)}
    held = sorted(manifest["held_out"])

    model, _r = hf_text.load_text_causal_lm(args.model, dtype=args.dtype,
                                            attn_implementation="eager", device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    blocks = decoder_blocks(model)
    device = next(model.parameters()).device
    model.eval()

    meter = DamageMeter(model, tok, device=device, dtype=next(model.parameters()).dtype,
                        battery=BATTERY[:8])
    meter.clean()
    bank = {l: saved["bank"][l].float() for l in layers}
    prompt = DETECT_PROMPTS_HELD[0]
    ids = render_prompt(tok, prompt, system=EXPERIMENT_SYSTEM)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    norms = {l: float(batched_residuals(model, blocks, device, [ids], l, pad)[0].norm())
             for l in layers}
    # first token of each candidate's name: the position the readout is read at
    first_token = [tok(w, add_special_tokens=False)["input_ids"][0] for w in words]

    base_h, base_logits = final_residual_and_logits(model, blocks, ids, device=device,
                                                    vector=None, layer=layers[0], scale=0.0,
                                                    span=args.scope)
    base_lp = base_logits.log_softmax(-1)

    def rank_from_residual(h_end, true_i):
        """The readout rank of a HYPOTHETICAL final residual, by the same lift as the real one.

        Two things the first version got wrong. It read the vector alone through the head, and
        RMSNorm divides the magnitude out, so the answer was identical over eight orders of
        magnitude of scale -- a statistic that cannot see the strength cannot test a claim about
        the strength. And it was an absolute log-probability while the measured rank is a lift
        against the clean baseline, so the per-token prior it is meant to cancel stayed in. That is
        the shape of this project's first retracted identification result.
        """
        with torch.no_grad():
            h = h_end.to(device=device, dtype=next(model.parameters()).dtype)
            lg = model.lm_head(model.model.norm(h.unsqueeze(0).unsqueeze(0)))[0, 0].float().cpu()
        lift = lg.log_softmax(-1) - base_lp
        scores = [float(lift[t]) for t in first_token]
        return sorted(range(len(words)), key=lambda i: -scores[i]).index(true_i)

    rows, started = [], time.time()
    for k in range(args.trials):
        word = rng.choice(held)
        layer = rng.choice(layers)
        true_i = index[word]
        v = bank[layer][true_i]
        scale, achieved_c, note = meter.scales_for_ladder(
            v, layer, (args.tier,), residual_norm=norms[layer], verify=True)[args.tier]
        g = torch.Generator().manual_seed(args.seed * 7919 + k)
        arms = {
            "concept": v,
            "mean_matched": spectrum_matched(bank[layer], generator=g, add_mean=True),
            "common_mode": bank[layer].mean(dim=0),
            # the mean is added back for the same reason spectrum_matched now does it: a draw in
            # the centred basis carries no component along the bank mean, and that single scalar
            # separates it from every concept at AUC 0.988-1.000 without reading any content
            "permuted": permuted_in_basis(v, bank[layer], generator=g) + bank[layer].mean(dim=0),
        }
        for arm, vec in arms.items():
            # Each arm's OWN solve, and its own note. Binding the concept's note once and writing
            # it on all four rows mislabelled ten of twenty-four arm-rows per tier.
            sc, achieved, arm_note = (
                (scale, achieved_c, note) if arm == "concept" else
                meter.scales_for_ladder(vec, layer, (args.tier,),
                                        residual_norm=norms[layer], verify=True)[args.tier])
            scoped = damage_at_scope(meter, vec, layer, sc, args.scope,
                                     device=device, dtype=next(model.parameters()).dtype,
                                     blocks=blocks)
            h, logits = final_residual_and_logits(model, blocks, ids, device=device, vector=vec,
                                                  layer=layer, scale=sc, span=args.scope)
            delta = h - base_h
            injected = (vec * sc)
            lift = logits.log_softmax(-1) - base_lp
            scores = [float(lift[t]) for t in first_token]
            order = sorted(range(len(words)), key=lambda i: -scores[i])
            rows.append(dict(
                arm=arm, concept=word, layer=layer, tier=args.tier, scope=args.scope,
                scale=sc, how=arm_note["how"], monotone=arm_note["monotone"],
                measured=achieved, damage_at_scope=scoped,
                transport_gain=float(delta.norm() / injected.norm().clamp(min=1e-9)),
                carry_through=float(torch.nn.functional.cosine_similarity(
                    delta.unsqueeze(0), injected.unsqueeze(0)).item()),
                readout_rank=order.index(true_i),
                readout_lift_true=scores[true_i],
                readout_top=words[order[0]],
                # the vector added straight to the clean final residual: the mechanical part
                direct_rank=rank_from_residual(base_h + injected, true_i),
                # only the component of the real displacement that lies along the injected vector
                carried_rank=rank_from_residual(
                    base_h + (delta @ (injected / injected.norm().clamp(min=1e-9)))
                    * (injected / injected.norm().clamp(min=1e-9)), true_i),
                scores=[round(x, 5) for x in scores]))
        if k % 5 == 0:
            print(f"  {k}/{args.trials} ({time.time()-started:.0f}s)", flush=True)

    json.dump({"rows": rows, "candidates": len(words), "tier": args.tier, "scope": args.scope,
               "words": words, "layers": layers, "seed": args.seed, "model": str(args.model)},
              args.out.open("w"), indent=1)

    n = len(words)
    print(f"\nmodel {args.model}  layers {layers}  tier {args.tier}  scope {args.scope}  "
          f"seed {args.seed}")
    print("\nARE THE ARMS MATCHED ON DAMAGE AT THE SCOPE THEY WERE INJECTED AT?")
    for arm in ("concept", "mean_matched", "common_mode", "permuted"):
        at = [r["damage_at_scope"] for r in rows if r["arm"] == arm]
        if at:
            m = sum(at) / len(at)
            clamped = sum(1 for r in rows
                          if r["arm"] == arm and str(r["how"]).startswith("clamped"))
            print(f"  {arm:<14} mean {m:+.4f}  (tier {args.tier:+.2f}, "
                  f"{clamped}/{len(at)} clamped)")

    print(f"\n{'arm':<14} {'n':>4} {'transport':>10} {'carry cos':>10} {'rank':>8} "
          f"{'top-10%':>8} {'direct':>8} {'carried':>8}   (chance {(n-1)/2:.0f})")
    for arm in ("concept", "mean_matched", "common_mode", "permuted"):
        at = [r for r in rows if r["arm"] == arm]
        if not at:
            continue
        g_ = sum(r["transport_gain"] for r in at) / len(at)
        c_ = sum(r["carry_through"] for r in at) / len(at)
        r_ = sum(r["readout_rank"] for r in at) / len(at)
        d_ = sum(r["direct_rank"] for r in at) / len(at)
        k_ = sum(r["carried_rank"] for r in at) / len(at)
        t10 = sum(1 for r in at if r["readout_rank"] < n * 0.1) / len(at)
        print(f"{arm:<14} {len(at):>4} {g_:>10.3f} {c_:>+10.3f} {r_:>8.1f} {t10:>7.0%} "
              f"{d_:>8.1f} {k_:>8.1f}")
    print("\nrank is the real readout; direct is the vector added to the clean final residual and")
    print("read out with no propagation; carried is only the part of the real displacement that")
    print("lies along the injected vector. If direct and carried track rank, the remaining layers")
    print("contributed nothing the readout uses.")
    print(f"\nrows: {args.out}   ({time.time()-started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
