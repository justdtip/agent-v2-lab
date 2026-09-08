# Gemma 3 4B weights, converted here from Google's own checkpoint

2026-09-08. Both precisions are ours, made from the official weights the Director authenticated,
so the provenance names a source we can name and a conversion we performed rather than a third
party's. The weights themselves are untracked (`models/` is ignored); this is their record.

## Source

`google/gemma-3-4b-it`, downloaded under the Director's Hugging Face token after he accepted the
Gemma licence. The repository is the **multimodal** checkpoint: a text tower and a SigLIP vision
tower, 8.0 GB on disk, `model_type: gemma3` with the text configuration nested.

## What was made

    mlx_lm.convert --hf-path google/gemma-3-4b-it --mlx-path models/gemma-3-4b-it-4bit \
                   -q --q-bits 4 --q-group-size 64
    mlx_lm.convert --hf-path google/gemma-3-4b-it --mlx-path models/gemma-3-4b-it-bf16 \
                   --dtype bfloat16

| | size | quantisation |
|---|---|---|
| `models/gemma-3-4b-it-4bit` | 2.1 GB | 4-bit, group size 64 |
| `models/gemma-3-4b-it-bf16` | 7.3 GB | none |

Both under one announced window, mlx-lm 0.31.3, in well under a minute for the pair — the estimate
of ten minutes was an order of magnitude high, worth knowing before anyone schedules around it.

## Verified, not assumed

Both outputs carry **no vision tower**: no `vision_config` in the configuration and **zero** tensors
whose names mention vision or the multimodal projector, counted from the safetensors headers. The
7.3 GB bf16 against the source's 8.0 GB is the tower's absence, and mlx-lm's Gemma 3 wrapper drops
it during weight sanitisation, so the conversion inherits that rather than needing a flag.

Both preserve what matters downstream: 34 layers, hidden size 2560, sliding window 1024, and
`eos_token_id: [1, 106]` — the two terminators, `<eos>` and `<end_of_turn>`, which is what makes a
Gemma turn end correctly with no change from us. The chat template travels with them.

## Which to use, and why

**4-bit for evaluation and for the mapping pilot.** The Qwen pilot and every Qwen evaluation ran on
a 4-bit checkpoint, so a two-panel comparison is like for like only at 4-bit.

**bf16 for a Jacobian fit**, because a finite-difference derivative taken through quantised weights
puts the quantisation error in the numerator of the difference quotient.

**Either for a regression fit.** That is a least-squares fit on activations with no difference
quotient anywhere, so the argument for bf16 does not apply to it. Prefer bf16 when comparing
against the hosted lens, which was fitted on bf16, so the comparison is like for like; 4-bit is
acceptable with the mismatch disclosed, and it is what unblocks a fit that would otherwise wait.

## Consequence for the registry

`configs/models/gemma3-4b.yaml` names `google/gemma-3-4b-it`, which is the multimodal bf16 source.
For the pilot it should name `models/gemma-3-4b-it-4bit`. That is a registry change and therefore
the Director's, offered rather than made; the ruling behind it is already his, in the work orders.
