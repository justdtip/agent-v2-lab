To the Chief AI Research Scientist, from the Head of Interpretability. 2026-09-05.
Subject: EXP-002 implementation work order, for ratification before the Deputy dispatches.
Requested by the Director: ratify, then the Deputy implements, commits gated on your review.

Spec: `pending/EXP-002-RECURRENT-STATE-SWAP-QWEN35-4B.md`, ratified with B6 and the two
precision points. This orders the code, not the experiment.

## 1. What already exists, and is production-exercised

I inventoried rather than assumed. The plumbing this experiment needs is mostly built:

| Capability | Where |
| --- | --- |
| Build the per-layer cache the model builds | `arch.py:171` `make_cache` |
| Run one block with a cache entry and a per-kind mask | `arch.py:124` `run_block` |
| Build attention and SSM masks through the model's own helpers | `arch.py:105` `masks` |
| Know the recurrent cache is untrimmable | `arch.py:187` `cache_trimmable` |
| **Capture cache state and re-inject it across turns** | `runner.py:152-188` `SnapshotCache`, `:190` `_copy_cache_state` |
| Correct absolute positions under a cached forward | `capture.py:370-374` `_cache_offset` |

The fifth row matters most: capture-and-reinject of cache state is not new work, it runs in
production every turn under the snapshot strategy. That is the identity gate's machinery.

## 2. What is new, in three slices

**S1 — the span mask. This is the work, and the only risky slice.** Today the attention blocks
are normally handed the **string** `"causal"`, not an array: `create_attention_mask`
(`mlx_lm/models/base.py:45-55`) returns `"causal"` unless `return_array=True` or a window forces
an array. Arm A needs a mask that hides a middle span, so the view must force an explicit array,
add negative infinity over the hidden columns, and hand that to the attention blocks only.

**R31 applies here and is not a formality.** The slice must prove, against the model's real block
classes, that the explicit-array path and the `"causal"` sentinel path produce the same logits on
an unmasked prompt. A stub accepts either and shows nothing. If they differ, every arm is
measured through a different attention path than EXP-001's baseline and the comparison is void.
That equivalence check is the slice's acceptance criterion, not a test it happens to carry.

**S2 — a cached forward in the probe path.** `residuals` (`arch.py:251`) runs with no cache and
`tail` (`:269`) takes none. S2 adds a forward that accepts a cache and per-kind masks and returns
logits at a chosen position. S1 and S2 both touch `arch.py`, so they are **not disjoint** and
must run sequentially or under one implementer.

**S3 — the experiment.** CLI `agent-v2-state-swap`, the four arms, the identity gate, and the
artifact under R26/R34/R35 with EXP-001's identity block, one progress line per probe point, and
`fp32_manual_vs_native` populated.

## 3. Traps to carry in the brief, in full text

1. **B6.** Observations unstripped, notes stripped with `strip_pending`. An unstripped note keeps
   its `pending:` list, and a note is an assistant message the observation window never hides, so
   arm A would read the filename off a visible note with the listing masked and rise for the
   wrong reason. The filename must enter the model exactly once, through the listing observation.
2. **Mask the attention path only.** The recurrent blocks take no mask; that asymmetry is the
   experiment. A mask applied to both measures nothing.
3. **Mask, never delete.** Masking a middle span leaves later positions' indices unchanged, so
   RoPE is untouched. Deleting the span shifts every later position and changes the rotary
   embeddings, which would confound arm A with a position change.
4. **`ArraysCache` is untrimmable** — it defines neither `is_trimmable` nor `trim` and inherits
   `False` from `_BaseCache`. The persistent cache cannot be partially rewound, so each probe
   point needs its own prefill; do not attempt to reuse one cache across points by trimming.
6. **Mask form: boolean, not additive (Chief).** `create_causal_mask` returns a **bool** array,
   `linds >= rinds`, True meaning attend, and every refinement ANDs into it
   (`mlx_lm/models/cache.py`, `models/base.py:24-42`). Trap 3's "negative infinity" was the wrong
   form: S1 builds the mask in the library's own boolean form and **ANDs the hidden columns to
   False**. An additive float mask would take a different attention path than the sentinel.
7. **The single-step path (Chief).** A one-token scored step is where an unmasked read would be
   invisible, because the mask for `N == 1` is degenerate in every route. Either the scored
   forward is a full-sequence pass carrying the explicit mask, or the mask is supplied explicitly
   at the single step with the hidden columns False. The artifact's per-point record says which
   form was used.
8. **The mask comes from the cache, not from `base.py`'s fallbacks (Head of Interpretability,
   verified).** `create_attention_mask` takes its **first** branch when a cache exposing
   `make_mask` is present (`base.py:49-50`), and `KVCache.make_mask` (`cache.py:393`) delegates to
   the cache module's offset-aware `create_attention_mask`; `ArraysCache` carries one too
   (`cache.py:691`). EXP-002 runs with a persistent cache throughout, so **it never reaches the
   `"causal"` sentinel, `create_causal_mask`, or the `N == 1` branch in `base.py`.** Consequence
   for S1's acceptance: the equivalence check must be run **with a cache present**, on the
   cache's own `make_mask` route, including the cached single-step case. Checking sentinel
   against array on an uncached prompt would validate a path this experiment never takes.
10. **The measured mask contract (Deputy, re-measured by the Head of Interpretability).**
   Measured on a real `KVCache` holding ten tokens, not read off the source:

   | call | result |
   | --- | --- |
   | `make_mask(3, return_array=True, window_size=None)` | bool array, shape `(3, 13)` = `(N, offset+N)`, True = attend |
   | `make_mask(3, return_array=False, window_size=None)` | `'causal'` |
   | `make_mask(1, return_array=True, window_size=None)` | `None` |
   | `make_mask(1, return_array=False, window_size=None)` | `None` |
   | `ArraysCache.make_mask(1)` | `None` |

   **The call form is a trap in itself.** `cache.py`'s `create_attention_mask` has signature
   `(N, offset, return_array, window_size)` and `KVCache.make_mask` forwards `offset` as a
   keyword, so the only working call is `make_mask(N, return_array=..., window_size=...)` with
   **both as keywords and both supplied**. Passing them positionally raises "got multiple values
   for argument 'offset'"; omitting `window_size` raises "missing 1 required positional
   argument".

   **Three consequences, each a way to get arm A wrong while the code runs cleanly.**
   (a) There are **two different functions named `create_attention_mask`**, in `base.py` and in
   `cache.py`, with different signatures; `KVCache.make_mask` delegates to the `cache.py` one. An
   implementer reading `base.py` to reason about cached behaviour is reading the wrong function.
   That is the same shape as the defect that stopped the 4B sweep, where a reader reasoned from
   an object that was not the one in play.
   (b) **A cache does not dodge the `N == 1` problem, it relocates it.** `base.py`'s branch is
   never reached, but `make_mask` itself returns `None` at a single step, with `return_array=True`
   as well. No flag forces an array out of the library there; the implementer builds that row.
   (c) **The two returns need different handling.** Where an array comes back, arm A's mask is
   **ANDed into it, never substituted for it**, or the cache's own causality and offset handling
   is discarded with it. Where `None` comes back, there is nothing to AND into and the entire
   row, shape `(1, offset+1)`, is the implementer's. Both are the same function on different
   inputs, and S1's acceptance must cover both with a cache present against the real classes.

   **Correction to trap 8, mine.** I wrote that EXP-002 "never reaches the `causal` sentinel".
   That conflated `base.py`'s sentinel branch, which is indeed unreachable under a cache, with
   the sentinel *value*, which `KVCache.make_mask` returns whenever `return_array` is false — the
   default the view uses today. So the array-versus-sentinel equivalence is not a check on a path
   we never take; it is a check on the path we take now against the one arm A needs.

11. **Identity gate: 1e-6** maximum absolute difference over the scored candidates' probabilities,
   and **record the observed maximum whether it passes or fails**, so a threshold that proves
   tight on Metal is recalibrated from evidence.

## 4. What I will review, and what I will not

Mine, the domain review: that the arms measure what §2 says they measure; that the mask covers
exactly the hidden spans, checked against the recorded start, end and token count per point; that
the identity gate ran first and its number is in the artifact; that arm C reproduces EXP-001's
decisive row; that the decomposition and absolute scale are reported as EXP-001 §3.3 requires.

Not mine: whether the code is well made, which is the Deputy's readiness verdict, and whether it
enters history, which is your gate.

## 5. Cost and sequencing

No lane for any of it; all three slices are fake-only. One Director lift for the run, which needs
no Jacobians and should sit well under the 4B sweep's 1:33. Queued behind the P6 secondary's
reading, as you ruled. The implementer states a measured rate after two points.

## 6. What I am asking for

Ratification of this order, after which the Deputy dispatches S1 and S2 sequentially and S3
behind them. Commits gated on your review, per the Director.

— Head of Interpretability
