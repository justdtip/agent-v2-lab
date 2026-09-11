# Amendment 2 (DRAFT) — a certified leading direction in place of a full decomposition

**D-CRO, 2026-09-11, on the Chief's instruction. DRAFT: unsealed, unrun, and `transport.py` is
untouched.** It proposes one change to Amendment 1 §2. Nothing is read under it and no addendum
carries it.

## 1. What it changes, and the reason it is not a mere optimisation

Amendment 1 §2 obtains each component's direction from `numpy.linalg.svd` of the deflated
cross-product: a complete decomposition of a *p*×*p* matrix, to extract **one** singular vector, from
a matrix whose rank is at most *n* < *p*. Codex's F1 required the direction to be **certified** —
the defect was a start vector that certified nothing — and a full decomposition was the way that was
met, not the thing that was asked for.

This amendment meets the same requirement directly: compute the leading triplet by a Krylov method
and **attach a certificate**, falling back to the full decomposition whenever the certificate fails.
The sealed answer is therefore reproduced exactly or reproduced exactly by the fallback; there is no
third outcome.

It matters because of §7's declared interval, which refits inside every one of 10,000 resamples. At
the sealed fitter's cost that is about **8.2 days** across the card's eight cores (§0.4 of the
record). At this one it is about **20 hours**.

## 2. The routine

Lanczos on the symmetric operator `S = A Aᵀ`, applied as two matrix–vector products so `S` is never
formed, with **full reorthogonalisation** and a **deterministic** start (`A` summed along its
columns, falling back to its first column). No random number enters the fit, so the ladder stays
reproducible bit for bit. A Rayleigh–Ritz step on the tridiagonal gives `θ₁ ≥ θ₂`; the direction is
the corresponding Ritz vector, `σ = √θ₁`, and `v = Aᵀu / σ`.

**The Krylov dimension is a declared schedule, not a guess: 48, 96, 192, 384**, tried in order until
the certificate passes. It is a schedule because the right size depends on the operator's spectral
gap, which varies by model width and by fold — at the 3,840-wide production size, 24 steps leave the
direction *wrong*, cosine 0.32 to the true leading vector, and the certificate correctly refuses; 96
certify. Growing until certified is self-tuning and ends either certified or in the full SVD.

The sign convention is the sealed fitter's — largest-magnitude entry positive — so the two agree
elementwise and not merely up to sign.

## 3. The certificate, and what it does not prove

Reported per component:

* **relative residual** `max(‖Av − σu‖, ‖Aᵀu − σv‖) / σ`, accepted below **1e-10**. Declared at that
  value because it is some orders above float64's epsilon at these magnitudes, so it admits ordinary
  rounding and refuses a direction that has not converged.
* **Ritz gap** `(θ₁ − θ₂) / θ₁`, accepted above **1e-6**. Below it the top two Ritz values are not
  distinguishable by the residual alone and the full decomposition decides.

**What it establishes.** A small residual bounds the distance to *a* singular triplet of `A`; `θ₁` is
a Rayleigh quotient, so `σ₁ ≥ σ` always and the routine never overstates. **What it does not
establish.** It is *not* a proof that the triplet is the **leading** one. This draft does not claim
one. What is relied on instead is stated plainly: anything failing either threshold falls back to the
full decomposition, and agreement with it is tested — on fixtures across four seeds, at the 3,840
production width, and end to end on the card.

## 4. Evidence, measured rather than argued

On the card, single-threaded, a full eight-component fit at the production shapes (3,298 fitting rows):

| | sealed, full SVD | certified | speed-up | components | fallbacks | worst direction disagreement |
|---|---:|---:|---:|---:|---:|---:|
| 4B, 2,560 wide | 27.0 s | 2.9 s | **9.4×** | 8 of 8 | 0 | 1.1 × 10⁻¹⁵ |
| 12B, 3,840 wide | 85.0 s | 8.6 s | **9.8×** | 8 of 8 | 0 | 5.6 × 10⁻¹⁶ |

Every component certified on its own; no fallback fired. The disagreement column is `|1 − |cos|| `
between the sealed and certified directions, component by component.

**Projected for §7's interval**, 10,000 refitting resamples at the registered headline: about 40 h on
the 4B and 119 h on the 12B, **160 h serial, about 20 h across the card's eight physical cores** —
against 8.2 days for the sealed fitter.

## 5. Why not `scipy`

`scipy.sparse.linalg.svds` gives the same triplet and was what the first estimate used. **It is not
installed on the card and is not a declared dependency of this repository**, so a routine built on it
could not run where the reading runs. That was found by checking rather than assumed, and it is why
this is a NumPy Lanczos. The first speed-up figure I reported, 14×, came from `scipy` on the laptop
and is superseded by the table above.

## 6. What this does not change

Not the rule, the ladder, the strata, the tolerances, M, α, the range width, the fold assignment, the
candidate set, the tie rule, the metric, the arithmetic, or any reading. It changes **one step**
inside the fitter and attaches a certificate to it. If accepted it edits `transport.py`, which an
addendum seals and Codex read, so it requires their review and a new addendum; until then the sealed
fitter is the only one any reader uses, and `transport_certified.py` is a draft that nothing calls.
