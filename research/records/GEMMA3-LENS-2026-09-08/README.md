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

---

## The layer convention is settled against the upstream source, 2026-09-08

The residual layer offset was the one item the confound audit could not close. Geometry cannot
decide it — adjacent hosted maps are 0.72 to 0.98 alike once identity is removed, so a one-layer
shift moves agreement by at most 0.05 with inconsistent sign — and the archive's own
`source_layers` is `[0..32]`, which is consistent with either reading. It is now settled by reading
the code that produced the file.

**The source.** Our `fit-config.yaml` cites `https://github.com/anthropics/jlens`, which does not
resolve. The repository is **`https://github.com/anthropics/jacobian-lens`**, Apache-2.0, companion
code for the global workspace interpretability paper. Cloned for reference at
`/Users/daniel.tipton/reference/jacobian-lens`, commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`,
"Initial release", 2026-07-01. It is outside our tree and vendored nowhere.

**What the code does.** `jlens/hooks.py`, `ActivationRecorder`: *"Captures residual-stream tensors at
the given block indices... Registers a forward hook on each requested block... On the next forward
pass each block's output is stored in `activations`, keyed by block index"*, with `blocks` being
`model.layers`. So a source index is **the output of a decoder block**, not a HuggingFace
`hidden_states` index, and the embedding output is never a source.

`jlens/fitting.py`, `_check_layer_indices`: `target = n_layers - 1` when `target_layer` is None, and
`source_layers = list(range(target))`. Our fit ran with `target_layer: null` on a 34-block model, so
the target is block **33** — the final residual — and the sources are blocks **0 to 32**, which is
exactly the `[0..32]` in the archive.

**Therefore, and this is now verified rather than asserted:**

| | |
|---|---|
| `J[i]` maps | output of decoder block `i` → output of block 33, the final residual |
| repo layer `L` | output of block `L-1`, one-based |
| so repo layer `L` reads | **`J[L-1]`** |
| repo layer 34 | block 33's output, the target itself, hence the identity |

**The sidecar's asserted convention was correct.** It is retained, and it is no longer an assertion.

### And a correction to a number the audit reported

The audit found `n_valid_positions = 111` for every one of 546 prompts in `convergence.csv` and read
it as a bound on position, concluding the hosted lens "was never fitted above absolute position
~111". **`n_valid_positions` is a count, not a position.** From `jlens/fitting.py`:

```
#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16
```

and `valid_position_mask` also drops the final position, which has no next-token target. At
`max_seq_len` 128 that is `128 - 16 - 1 = 111`. So the lens was fitted at absolute positions **16
through 126**.

The substantive finding survives and is sharpened rather than weakened. The hosted lens has seen no
position above **126**, against a 1,024 sliding window, while the map reads transcripts to position
2,298 — so the extrapolation is by a factor of eighteen, not nine. It has also seen **no position
below 16**, deliberately, because early positions are attention sinks with atypical statistics.
That second half is new and it matters for agent transcripts, where a turn boundary re-enters a
regime the lens was explicitly fitted to avoid.

**What this does not settle.** The convention is now known; whether our capture *implements* it is a
separate question and remains checked only by `residual_source_agreement`, which compares our
hand-run loop against the model's forward and says nothing about which residual the lens expects.
