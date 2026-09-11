# Amendment 2 (DRAFT) — convergence diagnostics with reference fallback

**Revised after review 3cb6247 on 2026-09-11. Unsealed. The optimized bypass is withdrawn.**
`transport.py` remains the sealed fitter used by the reader. This draft changes no seal or addendum
and releases no inference. The earlier residual/Ritz-gap certificate was unsound (C1), the wrapper
changed the rank stopping rule and deflation arithmetic (C2), and numerical agreement did not
establish universal exact reproduction (C3).

## 1. Current behavior

`leading_triplet` still computes a deterministic Krylov candidate and reports its residual, projected
Ritz gap and diagnostic convergence. It **always reports `certified=False`**. A small residual and a
large gap inside the explored subspace do not control a stronger direction outside it.

`leading_direction` **always uses the reference full SVD**. The returned direction and singular value
come from the unchanged sealed `_leading_direction` operation, with `fell_back_to_full_svd=True`.
Diagnostics cannot authorize a bypass; a diagnostic numerical failure also falls back to the reference.
There is currently no accelerated certified path. Reference linear-algebra failures remain refusals.

The draft `certified_fit` name is retained for callers, but it uses that reference fallback and matches
`transport.fit`'s standardisation, centring, singular-value stop (`sigma < 1e-12`), score-norm stop,
component-finiteness refusal and the same order of cross-product deflation operations. In particular,
`8e-13 * I` reaches rank zero, as the sealed fitter requires. An equivalent algebraic simplification
of deflation is insufficient when floating-point arithmetic and strict retrieval ties are involved.

## 2. Diagnostics and their limits

The diagnostic schedule remains 48, 96, 192 and 384 Krylov steps, clipped to the source width. It can
stop on diagnostic convergence, but that result is still uncertified. Thresholds remain relative
residual `1e-10` and projected squared-singular-value gap `1e-6`, with no leadingness guarantee.

The decisive counterexample is `[[3,-3,0],[0,0,1],[0,0,0]]`: the start lies entirely in the weaker
invariant subspace, producing zero residual and projected gap one while missing the leading direction.
Tighter thresholds or more steps from that start do not supply a global certificate. The near-degenerate
and width-64 padded examples from C1 are also required regressions.

A future optimized bypass needs a reviewed global leadingness check controlling the unexplored
spectrum, plus numerical error and separation analysis. Restarts alone would be probabilistic and
would require a declared failure budget. None of that is supplied or authorized by this revision.

## 3. Equivalence contract

The current reference path preserves the sealed operation and fitter arithmetic. Synthetic tests
require equal ranks, singular values, weights, loadings, coefficients and predictions with NumPy
`array_equal` where applicable in the same execution environment. This is regression evidence, not
a promise of bitwise identity across different BLAS/LAPACK libraries, hardware or versions.

For a future approximate path, a cosine or coefficient tolerance is only a numerical-agreement test.
Matching sign conventions cannot remove arithmetic differences or rotations in a repeated singular
subspace. Agreement at a few gate fixtures cannot prove all strict nearest-neighbour decisions agree
at arbitrarily small margins. Such a path needs an explicit equivalence contract and retrieval-level
validation before adoption; the former universal exact-reproduction claim is withdrawn.

## 4. Historical measurements — not evidence for the revised path

The prior draft reported these single-threaded card measurements on synthetic production shapes
(3,298 fitting rows, eight components). They describe the **former unsound bypass**, not this revision:

| Model width | Sealed full SVD | Former draft | Reported speedup |
|---|---:|---:|---:|
| 2,560 | 27.0 s | 2.9 s | 9.4× |
| 3,840 | 85.0 s | 8.6 s | 9.8× |

It also reported coefficient relative differences of `2.3e-11` and `6.4e-11`, with equal aggregate
gate fractions on those fixtures. These are fixture observations, not a certificate. The former
20-hour projection for 10,000 refits is withdrawn: every direction now falls back, and F3 changes
which training episodes the bootstrap must fit. No card timing or corrected workload cost was
remeasured in this revision. The Research Director's cost decision remains separate.

The earlier SciPy estimate also remains withdrawn. This module uses NumPy and the existing transport
module; it adds no dependency.

## 5. Reader corrections and release boundary

The reader independently restores §7's episode weights and fitting population, compares Hoeffding
with a complete refitted interval, and enforces 10,000 draws. Its explicit unsupported-draw policy is
that any empty, failed or incompletely scored draw makes the entire interval unavailable; it never
silently drops or redraws such a sample. Refitted endpoints follow the sealed paired helper's
one-sided `alpha` and `1-alpha` convention. This is not an amended estimand or permission to release.

The rule, fixed folds, candidate set, float32 retrieval, tie rule, tolerances, M, alpha and sealed
metadata remain unchanged. Revised reader bytes require a fresh reviewed addendum before real use.
Neither the correction nor successful synthetic tests authorize Amendment 2 or section 7 execution.
