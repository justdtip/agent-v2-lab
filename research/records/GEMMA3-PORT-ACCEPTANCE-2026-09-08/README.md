# The architecture-view port, accepted on the real Gemma 3 — and two defects it caught

**The gate passes bit-exactly, above and below the sliding window, with a negative control that
bites.** Run on `models/gemma-3-4b-it-4bit`, 34 layers, hidden 2560, window 1,024.

| | 64 tokens | 1,400 tokens |
|---|---|---|
| **the gate**: the view's native-dtype loop against the model's own forward | **0.0** | **0.0** |
| **the negative control**: the same, with the mask dispatch broken | 0.0 | 5.31 (5.1% relative) |

The control is the half that makes the gate mean something. Breaking the dispatch — every block
handed the first block's mask — changes **nothing** below the window and changes the answer above
it. That is the design's own prediction: Gemma's global and windowed masks are identical for any
sequence shorter than 1,024, so a gate run at 64 tokens cannot see a mask defect at all. The
repository's residual gate has been running at 64.

## Two defects the acceptance caught, both mine

**A boolean mask was being cast to float32.** `masks` gave an additive mask the residual's
precision, which is right, and cast *every* array, which is not: a boolean mask is a predicate,
`True` meaning attend, and casting it to a float makes it an additive mask of ones and zeros so
every position becomes attendable. On the real model this cost 95 per cent relative error at layer
1 above the window. Below it, both of Gemma's masks are the string `"causal"`, so no array was
cast and nothing showed — **the defect was invisible at every length the tests used.** Fixed by
`_mask_matching`, with a unit test that names the case.

**The native diagnostic never applied the entry transform.** The pivot document listed
`diagnostic_native_final_residual` beside `embed` as carrying the same omission, and the first port
fixed only `embed`. So on Gemma this diagnostic — which *is* the residual gate's comparator —
disagreed with the model by 79 per cent at 64 tokens, below the window where no mask defect can
appear, which is what localised it to the entry rather than to the masks. It reads the observed
entry now.

The second one is worth stating plainly: **the one place the omission would have been caught is the
one place it survived**, because the gate was comparing a broken loop against itself.

## The Chief's hoist, done here rather than scheduled

`residuals` and `diagnostic_native_final_residual` each called `embed` and then `masks`, paying two
observing forwards and building one pair of masks to discard. Both now **observe once** and use the
entry and the masks from the same forward. That is the hoist the Chief ruled for — at a call site
that can see it needs both, rather than a cache with an invalidation rule.

## What the first acceptance measured, and why it was not evidence

The first run compared `residuals` against `native_residuals` and returned 181,895. `residuals`
casts to float32 at every block while the model's forward runs in bfloat16, so that number is the
promotion **plus** any defect and cannot separate them. Relative differences localised it; the
native-dtype comparator removed it. An absolute difference against a residual whose norm nobody
quoted was never going to be an acceptance.

## Files

- `native_gate.py`, `native-gate.json` — the gate and its control, as run.
- `residual_source_agreement.py`, `residual-source-agreement.json` — the first comparison, kept
  because it is what found the mask defect and because its numbers are the record of that.
