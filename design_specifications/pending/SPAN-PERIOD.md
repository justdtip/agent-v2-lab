# The period is one of spans as well as kinds, so Gemma stops being called dense

Patch: `design_specifications/pending/SPAN-PERIOD.patch`, against `a347f53`. Four files, 190
insertions. **1,990 passed, 14 skipped, exit 0.**

## The defect

`_structural_period` derived the decoder's period by looking for a **recurrent block**. Gemma 3's
thirty-four blocks are all attention modules, so it found none and returned

> *view.blocks: no linear-attention block, so the backbone is dense and has no hybrid period*

which is true of the modules and false of the model. Gemma has a period of six. It is a period of
attention **spans**, not of block **kinds**: every sixth block attends globally and the rest attend
within a 1,024-token window.

The consequence was not a wrong number, it was a wrong sentence. `kind_matched_layer_family` took
its degenerate path, derived no partners, and would have written *dense* into every artifact it
touched — including the representation map's own manifest, which is what the Chief flagged.

`attention_span` already knew better. Nothing consumed it.

## The fix

**Two partitions, tried in order.** A decoder alternates either block kinds or attention spans, and
both are periodic in the same way — one distinguished block every `p`. `_structural_period` now
reads `layer_kind` first and `attention_span` second, and names which one answered:

```
view.blocks[5].self_attn.is_sliding (first global block by span; period = index + 1)
```

**The pairing follows the same partition.** `kind_matched_layer_family` takes `span_of`, and where
a view answers it the opposite class is the sliding blocks rather than the recurrent ones. This
matters more than it sounds: on Gemma the old filter left the opposite class **empty** — every
block is attention, so nothing was recurrent — and every in-band layer lost its partner while the
grid of distinguished layers stayed correct. A silent half-failure.

**The strings stop naming Qwen's field.** `LAYER_FAMILY_RULE` is quoted into every artifact under
R34 and said `full_attention_interval=p`, which is Qwen's name for the period and a concept Gemma's
configuration does not carry under any name. It now says *a decoder that alternates periodically
with period p* and names the partition. The reason string says `period=6` and lets the source name
the attribute actually read.

**A uniform decoder is still reported as one**, and EXP-001 §5's dense comparator still works — but
the sentence is reached after both readings, and says what was looked for and not found rather than
asserting dense.

## The derived family, from the block structure rather than from the document

**6, 11, 12, 17, 18, 23, 24, 28, 30, 34**, with pairs `{11: 12, 17: 18, 23: 24, 28: 30}` and no
unrecorded ties. That is exactly the pivot document's §2.3 prediction, reached independently.

## Two doubles gained the reader, and answered truthfully

`_HybridSweepView` and `_HybridCliView` return `None` from `attention_span`, because their blocks
are the library's own hybrid `DecoderLayer` and carry no `is_sliding`. That is the honest answer and
it makes them the cases that cover the **kind** path, with the Gemma-shaped test covering the span
path. A real `ArchitectureView` answers this method, so a stand-in for one has to.

## What this does not do

It does not decide Gemma's band. The band is a ruling and none exists; this makes the family
*derivable*, which is what the Chief's issue-80 finding said the registry should check against
rather than duplicate.
