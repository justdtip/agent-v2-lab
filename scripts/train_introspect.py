"""Teach the model to report what has been added to its own residual stream.

A plain torch loop, not a `Trainer` subclass. The repository's training path strips any dataset
column the model's forward does not name, which silently deletes the per-row injection plan and the
prompt boundary -- the spec's own audit found that this has over-supervised every torch run here by
a factor of five. A loop that owns its own collation cannot make that mistake, and every assertion
below is one nobody has to remember to switch on.

Everything the injection needs is per row: which layer, which absolute token, how strongly, and
which vector out of the resident bank. The hook is stateless (see `residual_patch`), which is what
makes it correct under gradient checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.nn.utils import clip_grad_norm_  # noqa: E402

from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.introspect.meter import DamageMeter  # noqa: E402
from local_llm_lab.introspect.render import render_prompt, render_supervised  # noqa: E402
from local_llm_lab.introspect.vocabulary import split  # noqa: E402
from local_llm_lab.lora_torch import (  # noqa: E402
    apply_lora, count_lora_parameters, lora_parameters, lora_state_dict,
)
from local_llm_lab.residual_patch import (  # noqa: E402
    PatchPlan, PlannedPatch, build_masks, decoder_blocks,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_introspect_data import (  # noqa: E402
    DETECT_PROMPTS_TRAIN, HELD_OUT_DAMAGE, LADDER, LADDER_WEIGHTS, NEGATIVE_TARGET,
    POSITIVE_TARGET,
)


def make_rows(bank, scales, words, train_words, layers, rng, count):
    """The training set, as plans rather than tensors: the bank stays resident and rows index it.

    Classes, and what each is for:
      D_positive        a concept is present; say so and name it
      D_clean           nothing injected; the floor under "always yes"
      D_noise           a random direction of the bank's own spectrum at matched damage -- the
                        hard negative, and the question
      D_common          the carrier's own mean direction; catches a generator-signature detector
      D_mismatch        a concept is present but a DIFFERENT one is named in the prompt
    """
    index = {w: i for i, w in enumerate(words)}
    families = split()[0]
    rows = []
    ladder = list(LADDER)
    weights = list(LADDER_WEIGHTS)

    def a_strength(word, layer):
        wanted = rng.choices(ladder, weights=weights, k=1)[0]
        entry = scales[word][str(layer)][str(wanted)]
        return wanted, entry["scale"], entry["measured"]

    shares = {"D_positive": 0.38, "D_clean": 0.13, "D_noise": 0.18,
              "D_common": 0.06, "D_mismatch": 0.11, "D_heldstrength": 0.04,
              "D_negative_extra": 0.10}
    for kind, share in shares.items():
        for _ in range(int(count * share)):
            layer = rng.choice(layers)
            word = rng.choice(train_words)
            prompt = rng.choice(DETECT_PROMPTS_TRAIN)
            if kind == "D_positive":
                wanted, scale, measured = a_strength(word, layer)
                rows.append(dict(cls=kind, prompt=prompt,
                                 target=POSITIVE_TARGET.format(name=word),
                                 bank_index=index[word], layer=layer, scale=scale,
                                 wanted=wanted, measured=measured, concept=word))
            elif kind == "D_heldstrength":
                wanted = rng.choice(HELD_OUT_DAMAGE)
                entry = scales[word][str(layer)][str(wanted)]
                rows.append(dict(cls=kind, prompt=prompt,
                                 target=POSITIVE_TARGET.format(name=word),
                                 bank_index=index[word], layer=layer, scale=entry["scale"],
                                 wanted=wanted, measured=entry["measured"], concept=word))
            elif kind == "D_clean":
                rows.append(dict(cls=kind, prompt=prompt, target=NEGATIVE_TARGET,
                                 bank_index=-1, layer=layer, scale=0.0, wanted=0.0,
                                 measured=0.0, concept=None))
            elif kind in ("D_noise", "D_common", "D_negative_extra"):
                wanted, scale, measured = a_strength(word, layer)
                rows.append(dict(cls=kind, prompt=prompt, target=NEGATIVE_TARGET,
                                 bank_index=-2 if kind == "D_noise" else
                                 (-3 if kind == "D_common" else index[word]),
                                 layer=layer, scale=scale, wanted=wanted, measured=measured,
                                 concept=None, noise_seed=rng.randrange(1 << 30)))
            elif kind == "D_mismatch":
                other = rng.choice([w for w in train_words
                                    if families.get(w) == families.get(word) and w != word]
                                   or train_words)
                wanted, scale, measured = a_strength(word, layer)
                rows.append(dict(cls=kind, prompt=prompt,
                                 target=POSITIVE_TARGET.format(name=word),
                                 bank_index=index[word], layer=layer, scale=scale,
                                 wanted=wanted, measured=measured, concept=word,
                                 distractor=other))
    rng.shuffle(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--hours", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()

    saved = torch.load(args.data / "bank.pt", map_location="cpu")
    scales = torch.load(args.data / "scales.pt", map_location="cpu")
    words, layers = saved["words"], saved["layers"]
    train_words = sorted(split(seed=args.seed)[0])

    model, _report = hf_text.load_text_causal_lm(args.model, dtype=args.dtype,
                                                 attn_implementation="eager", device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    blocks = decoder_blocks(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # A7: the loader returns model.eval().requires_grad_(False), so adapters come AFTER it or
    # nothing is trainable at all and the run would converge on nothing while reporting a loss.
    swapped = apply_lora(model, r=args.rank, alpha=args.alpha)
    trainable = [p for p in lora_parameters(model)]
    for p in trainable:
        p.data = p.data.float()
        p.requires_grad_(True)
    assert trainable and any(p.requires_grad for p in trainable), "no trainable adapter"
    print(f"adapters: {swapped} modules, {count_lora_parameters(model)/1e6:.0f}M parameters",
          flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    bank = {l: saved["bank"][l].float() for l in layers}
    common = {l: saved["common"][l].float() for l in layers}
    rows = make_rows(bank, scales, words, train_words, layers, rng, args.rows)
    print(f"{len(rows)} rows: " + ", ".join(
        f"{k} {sum(1 for r in rows if r['cls']==k)}"
        for k in sorted({r["cls"] for r in rows})), flush=True)

    rendered = []
    for row in rows:
        text = row["prompt"]
        if row["cls"] == "D_mismatch":
            text = f"{text}\n\nIs the injected thought about {row['distractor']}?"
            row = dict(row, target="NO. I do not detect an injected thought.")
        ids, prompt_length = render_supervised(tok, text, row["target"])
        rendered.append(dict(row, ids=ids, prompt_length=prompt_length,
                             site=prompt_length - 1))
    rows = rendered

    def vector_for(row):
        if row["bank_index"] == -1:
            return None
        if row["bank_index"] == -2:
            g = torch.Generator().manual_seed(row["noise_seed"])
            from local_llm_lab.introspect.vectors import spectrum_matched
            return spectrum_matched(bank[row["layer"]], generator=g)
        if row["bank_index"] == -3:
            return common[row["layer"]]
        return bank[row["layer"]][row["bank_index"]]

    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = args.max_steps
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=total, pct_start=0.05)

    cursor, step, history = 0, 0, []
    baseline_hooks = [len(b._forward_pre_hooks) for b in blocks]
    while step < args.max_steps and (time.time() - started) / 3600 < args.hours:
        optimiser.zero_grad(set_to_none=True)
        losses = []
        for _micro in range(args.accum):
            chunk = [rows[(cursor + i) % len(rows)] for i in range(args.batch)]
            cursor += args.batch
            width = max(len(r["ids"]) for r in chunk)
            input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            labels = torch.full((len(chunk), width), -100, dtype=torch.long)
            attention = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, r in enumerate(chunk):
                n = len(r["ids"])
                input_ids[i, :n] = torch.tensor(r["ids"])
                attention[i, :n] = 1
                labels[i, r["prompt_length"]:n] = torch.tensor(r["ids"][r["prompt_length"]:])
            plan = PatchPlan(layer=[r["layer"] for r in chunk],
                             site=[r["site"] for r in chunk],
                             scale=[r["scale"] for r in chunk],
                             vector=[vector_for(r) for r in chunk])
            # B2/B3/B4: the site is in the prompt, never in what is scored.
            for i, r in enumerate(chunk):
                assert 0 <= r["site"] < r["prompt_length"] <= len(r["ids"])
            masks = build_masks(plan, width=width, hidden=bank[layers[0]].shape[-1],
                                device=device, dtype=dtype,
                                lengths=[len(r["ids"]) for r in chunk])
            patches = [PlannedPatch(blocks[l], mask=m, delta=d, layer=l)
                       for l, (m, d) in masks.items()]
            for p in patches:
                p.__enter__()
            try:
                out = model(input_ids=input_ids.to(device), labels=labels.to(device),
                            attention_mask=attention.to(device))
                loss = out.loss / args.accum
                loss.backward()
                losses.append(float(out.loss))
            finally:
                for p in patches:
                    p.__exit__()
            # D13: every hook fired, and every firing carried the same fingerprint. This is what
            # catches a hook released before the backward, or any state that crept in.
            for p in patches:
                assert p.fires >= 1 and p.consistent(), (p.layer, p.fires, p.fingerprints)
            # D14: no hook left attached.
            assert [len(b._forward_pre_hooks) for b in blocks] == baseline_hooks

        grad = clip_grad_norm_(trainable, 1.0)
        # D15: something actually learned from this step.
        assert torch.isfinite(grad) and float(grad) > 0, f"grad norm {float(grad)}"
        optimiser.step()
        schedule.step()
        step += 1
        history.append({"step": step, "loss": sum(losses) / len(losses),
                        "grad": float(grad), "lr": schedule.get_last_lr()[0],
                        "seconds": time.time() - started})
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}/{args.max_steps}  loss {history[-1]['loss']:.4f}  "
                  f"grad {float(grad):.3f}  {(time.time()-started)/60:.0f}m", flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            torch.save(lora_state_dict(model), args.out / f"adapter-{step:05d}.pt")
            json.dump(history, (args.out / "history.json").open("w"), indent=1)
            print(f"  saved adapter-{step:05d}.pt", flush=True)

    torch.save(lora_state_dict(model), args.out / "adapter-final.pt")
    json.dump({"history": history, "rows": len(rows), "steps": step,
               "hours": (time.time() - started) / 3600,
               "layers": layers, "rank": args.rank, "alpha": args.alpha, "lr": args.lr,
               "batch": args.batch, "accum": args.accum},
              (args.out / "run.json").open("w"), indent=1)
    print(f"done: {step} steps in {(time.time()-started)/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
