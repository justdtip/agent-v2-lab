# The hosted Gemma 3 4B Jacobian lens, downloaded and converted

2026-09-08, the day of the Director's pivot. The binary is not tracked (`models/` is ignored);
this directory is its provenance.

## What it is

`neuronpedia/jacobian-lens`, path `gemma-3-4b-it/jlens/Salesforce-wikitext/`, fitted by
Neuronpedia with Anthropic's `jlens` (Apache-2.0, companion code to the Verbalizable Workspace
paper) against **`google/gemma-3-4b-it`** — the instruction-tuned model, not the pretrained one.
Downloaded and converted here to `models/jlens/gemma-3-4b-it_jacobian_lens.npz` with
`scripts/convert_jlens.py`, which loads no model and ran beside a live evaluation.

| | |
|---|---|
| source digest | see `sidecar.json` |
| converted npz sha256 | `a14ab264fc47de9c59b3183a8783a60ca6f1dee2ea635e88223d5b2d67b03193` |
| size | 432,545,456 bytes |
| maps | 33, keys `J0` to `J32`, repo layers 1 to 33 |
| shape, dtype | 2560 x 2560, float16 on disk and in the source |
| fit corpus | `Salesforce/wikitext`, `wikitext-103-raw-v1`, train split |
| prompts | 546 of a requested 1,000, stopped on convergence |
| final mean relative change | 0.00179 against a 0.002 threshold |
| **max_seq_len** | **128** |

It satisfies `LensMaps.load` for a 34-layer model without code change: every layer index falls in
`1 <= L < 34`, every map is hidden-size square and finite. Verified with numpy, no model loaded.

## The identity distance falls monotonically with depth

Block 0 at 24.98 down to block 32 at 0.96, the full profile in `sidecar.json`. That is the shape a
correct lens should have — the deeper the layer, the less transformation remains between it and the
output, so the transport map approaches the identity. It is a sanity signal on the conversion, not
a result.

## The caveat that must travel with every number read through it

**It was fitted at a sequence length of 128, and Gemma 3 4B's sliding window is 1,024.**

Five of every six Gemma blocks attend through a windowed mask. A causal mask with a window is
vacuous when the sequence is shorter than the window, so at 128 tokens the windowed layers were
*exactly full-causal* during the entire fit. This lens has therefore never seen the regime that
distinguishes a sliding-written layer from a global-written one.

Two consequences, and neither is a reason not to use it.

1. For reading foreknowledge and transport on agent transcripts, it is the right instrument and
   the only one available without a fit.
2. **Any claim resting on the sliding-against-global contrast cannot rest on this lens.** That
   contrast needs contexts beyond 1,024 tokens and a lens fitted in that regime. A refit above the
   window is the first thing worth the RunPod credits.

A second, ordinary mismatch to disclose in the usual way: the lens was fitted on bfloat16 weights
on wikitext prose; we intend to read 4-bit MLX weights on agent transcripts. The Qwen work carries
the same class of mismatch and discloses it.

## Also available, not downloaded

The same repository hosts lenses for every Gemma 2, Gemma 3 and Gemma 4 size, pretrained and
instruction-tuned, and for `gpt-oss-20b` and `deepseek-v4-flash`. Gemma 3 12B and 27B lenses exist
should the scale question be asked; note that our own artifact writer refuses above 1 GB, which a
12B lens would exceed.
