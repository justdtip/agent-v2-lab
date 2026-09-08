# The architecture-view port, implemented: the loop asks the model instead of describing it

Patch: `design_specifications/pending/ARCH-VIEW-PORT.patch`. **Apply after `PILOT-PATH.patch` and
`NATIVE-RESIDUALS.patch`.** Two files, 331 insertions. Design in
`ARCH-VIEW-PORT-2026-09-08.md`; the Chief's ruling on the hot path is taken as given.

## What is proved, and what is not

**Proved, on any machine, with no model and no MLX.** `_observe_forward` manipulates the module's
own objects and computes nothing, so I made it MLX-free and `tests/test_arch_observation.py`
exercises it against stand-ins shaped like the two real forwards. **Six tests, all passing**, and
each is a case the other shape would get wrong:

- the entry transform is inherited, not reconstructed — the Gemma-shaped stand-in's block 0
  receives `embed(ids) * 50.596…` and the Qwen-shaped one receives `embed(ids)` unchanged;
- masks come back per block, and the twelve-block stand-in gives the global mask to blocks 5 and
  11 and the windowed one to the other ten;
- **the model's own cache selection is inherited** — the global mask reports `cache-5` and the
  sliding one `cache-0`, which is the fourth defect and the one no description would have caught;
- the proxy forwards attribute access, so Qwen's dispatch on `layer.is_linear` still resolves;
- the module is restored even when the forward raises;
- a decoder that does not iterate its own layer list is refused rather than half-observed.

**Not proved, and not claimed.** 882 passed and 1,097 skipped, and the skips include every file
that touches these seams: `test_arch.py`, `test_probes.py`, `test_preflight.py`, `test_patch.py`.
So **nothing in this patch has been run against a real decoder.** The acceptance below is
outstanding in full and this must not land on the strength of the number above.

## The acceptance still owed, in the order I would run it

1. **Qwen, bit-exact.** The observed masks equal today's per-kind masks and the observed layer-zero
   residual equals `embed_tokens(ids)` exactly, on the model this tree ran all week.
   `ResolvedSpec.layer_types` unchanged, so no existing record is reinterpreted.
2. **A preflight prompt longer than the sliding window**, landing *with* the port. The gate's
   prompt is padded to 64 tokens and Gemma's window is 1,024, so both its masks are identical
   below it.
3. **The negative control.** With the dispatch deliberately broken — every block handed the global
   mask — the gate must **fail at the long length and pass at 64 tokens**. That is the only
   evidence the new gate sees what the old one could not, and I have not written it as a test yet
   **on purpose**: a test that cannot run is how this repository acquired a `convert_tokens_to_ids`
   guard whose two fixtures both raised. It gets written in the window where it can fail.
4. **The overhead measured**, on one real J-lens sweep length.

## What changed

- **`embed`** returns what the model's first block was handed. It was `embed_tokens(ids)`, which
  is the residual only where the entry transform is the identity.
- **`masks`** returns a mapping keyed by **block index**. Kind sufficed while every family built
  one mask per kind; Gemma builds two and dispatches on the index, so two blocks of one kind get
  different masks and a kind-keyed mapping cannot say which.
- **`run_block`** selects `masks[index]`.
- **`attention_span(index)`** answers `"global"`, `"sliding"` or `None`, beside `layer_kind` and
  not inside it, read from the attention module's own `is_sliding`. `layer_kind` keeps its two
  values because it feeds every recorded manifest and gates attention capture, which would refuse
  five Gemma blocks in six.

## Two decisions a reviewer should push on

**`masks` drives the observation with a zero dummy of `h`'s shape and dtype, not with `h`.** The
mask constructors read shape, dtype and the cache and never the values, and Gemma applies its entry
scale *after* the `input_embeddings` branch — so handing the real `h` in would return it multiplied
by fifty. `pre_norm_tail` passes a mid-network residual to `masks` today. The dummy is what makes
that safe, and the prohibition is written at the observer.

**The span-masking route now refuses on a decoder with no recurrent block.** EXP-002's measurement
is the asymmetry between an attention path and a recurrent one; on a family where every block is
attention there is no asymmetry, and applying it to all of them would answer a different question
under the same name. The pivot document defers that experiment on Gemma, and this makes the
deferral a refusal rather than a convention.
