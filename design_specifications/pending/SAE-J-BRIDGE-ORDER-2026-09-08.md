# Codex order: build the SAE-to-J-lens bridge, in two stages, and not the rest

**Issued by the Chief on the Director's instruction, 2026-09-08**, against the derivation
*SAE–J-lens geometry and causal state abstraction*. The derivation is sound where it was checked
numerically: its two counterexamples, its pair-weight factor and its averaging-cancellation result
all reproduce. What follows takes the part of it that is buildable and useful now and says plainly
what is not, so that nobody points at the whole document and starts.

## The one-line summary

**Build the exact score decomposition first. It is exact, it needs no model forward, no sparse
solver, and no box time. Build the sparse cone machinery only if the first stage shows it is
needed. Do not build the causal-abstraction programme at all yet; our throughput cannot reach its
acceptance rule by three orders of magnitude.**

---

## Stage A — the exact score decomposition (build this)

The derivation's §5.1 applied to our objects. For a residual layer `L`, a lens map `J_L`, the
model's readout `W`, and a sparse decoder `D`:

    L  = W J_L                       the lens score map
    Lh = L b + (L D) z + L e         exact, for every activation h = b + Dz + e

The signed quantity `z_i * (LD)_{v,i}` is feature `i`'s contribution to the score on token `v`,
and the entire approximation is carried by the single term `L e`. **This is an identity, not a
model.** It is the composition the Chief proposed and the derivation confirms it is exact.

**A1 — the dictionary readout, no activations required.** For each sparse feature `i`, compute
`L d_i` and keep its top tokens. This answers *what does this feature push the output toward, read
from this layer*, and sets that beside the feature's published auto-interpretation label. It needs
only three matrices already on disk and **no model forward at all**.

**A2 — the run-resolved decomposition.** Take activations already captured by a pilot run, encode
them, and report per position: the top features by `|z_i (LD)_{v,i}|` for the emitted token `v`,
and the residual share `|L e| / |L h|`. This is what turns *the residual at layer 18 predicts this
token* into *these labelled features are active and these two are what push toward it*.

### Costs, computed at Gemma 3 4B's real dimensions

| object | shape | size |
|---|---|---|
| one lens map | 2560 x 2560 | 0.03 GB |
| unembedding | 262,208 x 2560 | 2.69 GB |
| one 16k decoder | 2560 x 16,384 | 0.17 GB |
| **full score transfer `LD`** | 262,208 x 16,384 | **17.2 GB — do not materialise** |

So `LD` is never formed. Compute `JD = J_L D` first, which is 215 GFLOP and trivial, then take
`W (JD)` column by column keeping only top-k. That is 22 TFLOP per layer: about **150 seconds per
layer on CPU BLAS**, 80 minutes for all 33, and no model is ever loaded. On the GPU it is
11 seconds per layer, but that needs the box and is not worth contending for.

### Preconditions, and one is currently unmet

The derivation requires the sparse dictionary and the lens to share **hook, layer, coordinate
scaling and checkpoint**. Hook and layer align and this was verified: Gemma Scope's residual site
is the output of block N, which is this repository's probe layer N+1, and the lens carries maps on
the same index.

**The checkpoint does not align.** Gemma Scope was trained on Google's unquantised weights; our
evaluation entry is a 4-bit conversion. For the bridge this is not a disclosure, it is a **failed
precondition**. Build and run stage A against `gemma3-4b-bf16`. If a 4-bit number is ever wanted,
it is a separate measurement of what quantisation does to the bridge, and must be labelled as one.

Download sparse dictionaries **one layer at a time**, residual site, 16k width. One is 336 MB; the
repository is 4.65 TB. Take the layers the map makes interesting, not the set.

---

## Stage B — the sparse cone fit (build only if A justifies it)

The derivation's §§4 and 5.2. The J dictionary `B` has columns `normalise(J_L' w_v)`, one per
vocabulary entry, so it is **2560 x 262,208 = 2.69 GB per layer, 88.6 GB across all layers**. It
must be built per layer on demand and not stored.

The fit is nonnegative and sparse at budget `k`, and the derivation is explicit that the top scores
are *not* the answer, because the Gram matrix `G = B'B` carries the overlap between dictionary
directions. Three things must be reported with any number from it, and each is a separate error:
the SAE reconstruction error, the lens estimation error, and the solver's own optimisation gap,
which a solver **cannot certify from its own residual**.

**Carry these two cautions in the code and in every artifact, not only in a paper.** A residual left
by the fit can itself lie inside the cone at that budget — verified with a two-dimensional example —
so *unexplained by this fit* never means *outside the workspace*. And energy is not additive over a
non-orthogonal dictionary, so a per-feature share of `||h||^2` is not a partition and can exceed
one. **A feature's contribution to a score and its share of the geometry are different quantities in
different units and must never appear in one table.**

---

## Do not build: the causal-abstraction programme

Sections 8 to 16 define predictive sufficiency, update closure, causal state abstraction and an
acceptance rule over declared tolerances. The mathematics is correct. **It is out of reach at our
throughput and pointing at it now would waste months.**

The derivation's own bound gives the cost. To bound the diagnostic distance below 0.05 across ten
diagnostics at 95% confidence needs **2,397 independent episodes per arm**. Our measured rate is
fifteen episodes in fifty-eight minutes, so that is **6.4 days of continuous box time per arm**, and
an acceptance rule has many arms. Positions inside an episode are not independent draws and cannot
be substituted.

Two of its cautions apply to us **now** regardless, and are already in the map's pre-registration: an
averaged Jacobian can cancel a causal sensitivity, so a low lens reading is not evidence of low
causal influence; and mutual information between a feature at two times does not identify persistent
memory when a transcript retains the fact and the model recomputes it. That second one is a trap our
agent protocol walks straight into, because the transcript is retained by construction.

---

## Acceptance

- **A1** reproduces `L d_i` for a named feature two ways: through the composed matrices, and by a
  direct product for that single column. They agree to float32 tolerance.
- **A1** carries a negative control that bites: a lens from a *different layer* against the same
  decoder must give a measurably different token set, and the check must be shown to fail when the
  layers are deliberately mismatched. A bridge that cannot tell layers apart is not reading one.
- **A2** reports `|L e| / |L h|` per position. If the residual term dominates, the decomposition is
  describing the SAE's failure and not the model, and the artifact must say so rather than ranking
  features under it.
- Every artifact records the checkpoint, the lens digest, the dictionary layer, width and sparsity,
  and the residual convention, per R57 and R60.
- No box time for stage A. If stage B needs the machine, announce with a projected peak and its
  basis, per R60(c).

## What this may and may not be said to show

It is a **measurement instrument**. A large feature contribution to a lens score is a contribution
to that score, at that layer, under that averaging convention. It is not a token probability, not a
causal derivative, and not evidence of a workspace, a maintained state, or flexible access. The
derivation is unusually careful about this and the artifacts should inherit its care rather than its
vocabulary.
