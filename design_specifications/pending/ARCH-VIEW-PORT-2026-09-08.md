# The architecture-view port: stop describing the model and start asking it

**For the Chief's gate. Phase 1 item 1 of `GEMMA3-PIVOT-2026-09-08.md`, which calls it the
largest item and the most dangerous thing in the document. No box time to write; the acceptance
needs one preflight.**

The pivot document lists three defects in `arch.py`'s hand-run forward loop and rules that they
land as one change with a fourth item, a preflight prompt longer than the sliding window. All
three are confirmed here against the installed library, line by line. This document proposes
fixing them **as one defect rather than three**, because they are one: every place the loop
describes what the model does instead of asking it.

---

## 1. The three defects, confirmed against the installed library

`mlx_lm/models/gemma3_text.py` `Gemma3Model.__call__`, lines 180-212:

```python
h = self.embed_tokens(inputs)
h *= mx.array(self.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)   # 1
global_mask = create_attention_mask(h, cache[self.sliding_window_pattern - 1])
sliding_window_mask = create_attention_mask(h, cache[0], window_size=self.window_size)  # 2
for i, (layer, c) in enumerate(zip(self.layers, cache)):
    is_global = i % self.sliding_window_pattern == self.sliding_window_pattern - 1
    h = layer(h, global_mask if is_global else sliding_window_mask, c)
```

Against `arch.py`:

| # | The model does | `arch.py` does | Where |
|---|---|---|---|
| 1 | multiplies the embedding by `sqrt(hidden)` rounded through bfloat16 | returns `embed_tokens(ids)` unscaled | `arch.py:98` `embed`, and `arch.py:245` `diagnostic_native_final_residual` |
| 2 | builds two masks and dispatches on block index | builds one mask per *kind* and dispatches on kind | `arch.py:116` `masks`, `arch.py:222` `run_block` |
| 3 | flags the window on `self_attn.is_sliding` (`gemma3_text.py:55`) | reads `block.is_linear` | `arch.py:91` `layer_kind` |

At 2560 hidden the omitted factor is about 50.6, so defect 1 alone makes the hand-run loop
produce a residual that is not the model's. Defect 3 makes every Gemma block report `attention`,
so defect 2's dispatch would hand every block the same mask even after 2 is fixed.

**The mask cache selection is a fourth confirmed difference and it is not cosmetic.** The global
mask is built from `cache[pattern - 1]` and the sliding mask from `cache[0]` — a cache of each
kind, because the two carry different offsets once a window has rotated. `arch.py:145` picks by
`_first_cache(entries, kind)`, which on Gemma finds attention caches for both and cannot tell
them apart.

---

## 2. Why the pivot document's warning is the whole design problem

The document's sharpest sentence is that fixing the embedding scale alone **turns the residual
gate green while leaving the mask defect silently wrong on every real sequence**, because both
Gemma masks are identical below the window and the gate's prompt is padded to 64 tokens against
a window of 1024.

That is true, and it generalises past this instance. Each of the four differences was invisible
until a model exercised it:

- the missing scale was invisible while every model in the tree had a scale of one;
- the single mask was invisible while every model had one mask per kind;
- `is_linear` was invisible while the only hybrid put its flag on the block;
- the cache selection was invisible while the two kinds were structurally distinguishable.

A port that fixes four things a Gemma-shaped model needs leaves the fifth for the next family.
**The proposal is to remove the class, not the four instances.**

---

## 3. The proposal: capture the model's own pre-layer state and per-block masks

Both families have the same forward shape:

```
embed_tokens → (optional transform) → build masks → for layer, c in zip(self.layers, cache) → norm
```

So the layer-0 residual and the per-block masks are both **observable** by running the model's own
`__call__` with the layer list replaced by recording stubs that return their input unchanged.
Nothing is described; one cheap forward reports it.

```python
def _observe_entry(self, ids, cache):
    """The model's own layer-0 residual and per-block masks, measured rather than described.

    One forward through the text module with the blocks replaced by stubs that record the mask
    they were handed and return their input untouched. Everything before the first block --
    the embedding lookup, Gemma's bfloat16-rounded sqrt(hidden) scale, any future family's
    entry transform -- has run by the time the first stub is called, and every mask the model
    would have built has been built by its own code with its own cache selection.

    The stub proxies attribute access to the real block, because Qwen dispatches on
    `layer.is_linear` (qwen3_5.py:272) while Gemma dispatches on the index (gemma3_text.py:205).
    A stub that answered neither would change the masks it was there to observe.
    """
```

Then:

- **`embed`** returns the recorded layer-0 residual. No scale constant, no family table, no name
  check. Correct for Gemma by construction and for any family whose entry transform we have not
  met.
- **`masks`** returns a mapping keyed by **block index**, filled from the recorded masks. The
  model's own cache selection produced them, so the `cache[pattern-1]` / `cache[0]` distinction
  is inherited rather than reimplemented.
- **`run_block`** selects `masks[index]`.

**Cost.** One extra pass over the embedding and N no-op stubs, with no attention and no MLP. On
Gemma that also runs `create_attention_mask` twice, which is a mask build on a short sequence.
Measured against a J-lens sweep that runs the real loop once per layer, this is noise; the
acceptance below measures it rather than asserting it.

---

## 4. What deliberately does not change

**`layer_kind` keeps its two-value vocabulary.** The pivot document is right that widening it in
place would be a mistake: it feeds `ResolvedSpec.layer_types` in every recorded manifest, and
attention capture is refused for any block whose kind is not `attention`, which on Gemma would
refuse five blocks in six. The sliding/global distinction goes beside it as a separate reader:

```python
def attention_span(self, index) -> Literal["global", "sliding"] | None:
    """Which mask the model hands this block, or None where the family has one span.

    Separate from `layer_kind` and not a widening of it. `layer_kind` answers what module the
    block is, which is what manifests record and what capture gates on; this answers what the
    block can see, which is a property of the mask and not of the module. On Gemma they are
    orthogonal: every block is `attention` and five in six are `sliding`.
    """
```

Read from `self_attn.is_sliding` where present, and derived from the recorded masks otherwise, so
the two can be cross-checked. Issue 86's `kind_matched_layer_family` reads the same predicate with
`sliding_window_pattern` in place of `full_attention_interval`, which the pivot document's §2.3
already works out to the family **6, 11, 12, 17, 18, 23, 24, 28, 30, 34**.

**`hidden_spans` keeps its current behaviour and gains a refusal.** Span masking hides key columns
from attention blocks, and EXP-002's whole point was the asymmetry between an attention path and a
recurrent one. On a family where every block is attention there is no asymmetry to measure, which
is why the pivot defers that experiment. Rather than silently apply it to every block, the span
route refuses on a family with no recurrent blocks and names the deferral.

---

## 5. Acceptance, and the trap it has to avoid

The pivot document's constraint is the acceptance: **a green residual gate on a short prompt
proves nothing.** Concretely:

1. **A preflight prompt longer than the sliding window.** The gate's prompt is padded to 64
   tokens; the window is 1024. The gate must run at a length above the window, so the two masks
   differ and a wrong dispatch is detectable. **This item lands with the port, not after it**, and
   a port that lands without it is worse than no port, because it converts a detectable defect
   into a passing test.
2. **A negative control on the gate itself.** With the mask dispatch deliberately wrong — every
   block handed the global mask — the gate must fail at the long length and *pass* at 64 tokens.
   That is the instrument proving itself (R52), and it is the only evidence that the new gate can
   see what the old one could not.
3. **Qwen unchanged, bit-exact.** The observed masks must equal the current per-kind masks and the
   observed layer-0 residual must equal `embed_tokens(ids)` exactly, on the model this tree has
   been running all week. The port is a generalisation and must not be a change.
4. **The recorded manifests unchanged.** `ResolvedSpec.layer_types` for Qwen is identical before
   and after, so no existing record is reinterpreted.
5. **The overhead measured, not assumed**, on one real J-lens sweep length.

Items 3 and 4 are the ones I would expect to catch a mistake, and they run on Qwen with no Gemma
present, so they are available before the checkpoint finishes downloading.

---

## 6. What this does not cover

The rotating key-value cache, section 3 of the pivot document. The capture path reads column *j*
as absolute source position *j*, which rotation breaks, and the document is right that widening
the type gates would produce quietly wrong positions rather than a refusal. **That is a separate
change and it should stay separate**: this one is about what the loop asks the model, and that
one is about what a cache means. Observing masks does not observe cache semantics.

Note the interaction, because it decides the order: `_observe_entry` runs the model's own forward,
so on a rotating cache it inherits whatever the model does and stays correct. The capture path's
column arithmetic is wrong independently. Landing this first does not make that worse and does not
fix it.

## 7. The ruling I need before writing the code

**Is the observed-entry approach acceptable in the hot path**, or does the Chief want the measured
values cached per (shape, cache-offset) with the cache itself a correctness risk? I would rather
pay one no-op forward and have no cache to invalidate, but the J-lens estimator's call pattern is
the Chief's to weigh, and a memoisation keyed on the wrong thing is exactly the class of defect
this document is trying to remove.
