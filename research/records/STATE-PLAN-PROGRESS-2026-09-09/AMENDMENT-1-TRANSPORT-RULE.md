# Amendment 1 to the plan-progress pre-registration: E2's transport rule

**D-CRO, 2026-09-10, on the Chief's rulings of 13:50Z, 14:25Z and 15:10Z. Draft, revision 5 (the
Chief, applying Codex's four findings of review 657a08a; §10 records them beside the detour). For
Codex's file-only re-check. §10 records the detour this draft took and why it was wrong.**
Amends seal `998b3bcafa9d6aaffa21ebd43df3935b1634ba7b12fe187073837acd942ce521`, baseline `acc130a`.
Nothing in it has been run. No corrective transition is scored under it until it is sealed.

## 1. What this amends, and the one thing it adds

§4.2 fixes the candidate set, the tie rule, the metric, the strata and the tolerances — every choice
that could otherwise be made after seeing a score. It never fixes the transport rule's **functional
form**. I chose one, in the open: the literal reading of "predicts +1 on an ordinary transition and
the recovery cost at a corrective one", namely advance the read state by that many step-units. It
returned a hit rate of exactly 0.000 in every stratum on both models, and it could not have returned
anything else. This amendment fixes the form, and adds the check that would have caught it before it
ran.

**Not reopened, and unchanged:** the candidate set is all *N* decisions of the episode, the source
included, so chance is exactly 1/N; a tie is a miss; the metric is the Euclidean norm on residuals as
captured, native bf16 promoted to float32 and not otherwise transformed; the strata and their census;
the tolerances ε_ord = 0.13, ε_sub = 0.15, contiguous 0.23, across a gap 0.20, at the paired range
width; the episode as the unit; the cross-fitting and its sealed assignment; **M stays 12**; and the
seal itself, which this amendment extends rather than replaces.

## 2. The rule

Let `x_k` be the residual at the read layer for decision *k*, as captured. The rule is a **rank-r
supervised map to the successor**:

    T_r(x) = the rank-r supervised prediction of x_{k+1} from x_k,

fitted on ordinary transitions — the instrument the Chief fixed for E1, PLS2, generalised from a
scalar target to a vector one and reducing to E1's PLS1 when the target has one column.

**Why the map is unconstrained.** "Transport the state read at *k*" is a map to the successor, and
§4.2 gives no reason to constrain its form further. A constrained alternative was drafted and
withdrawn; §10 records it, the argument for it and its refutation, because the argument was mine and
a reader should be able to see what was rejected and why.

**Fitting `T_r`**, on the fitting folds' ordinary transitions only:

1. `X` is the matrix of source residuals, standardised by the fitting folds' per-coordinate mean and
   standard deviation, a coordinate whose scale is below 1e-8 held at 1.
2. `Y` is the matrix of **successor residuals**, **centred only**, by the fitting folds' mean. It is
   not scaled per coordinate.
3. For components `a = 1 … 32`, greedily and with deflation:
   - `w_a` is the leading left singular vector of the current `Xᵀ Y`, taken from a **full singular
     value decomposition** of that matrix (`numpy.linalg.svd`, LAPACK; deterministic on the same
     bytes), its sign fixed so that its entry of largest magnitude is positive. It is the leading
     direction **by construction**, not the fixed point of an iteration: the earlier power-iteration
     form could divide zero by zero when its start vector `Xᵀ(Y·1)` and its fallback were both zero,
     and could converge to a non-leading direction and certify it (§10, Codex F1). The leading
     singular value `σ_a` of the deflated cross-product is recorded per component. No random number
     and no start vector enters the fit; a non-finite cross-product or component is a refusal.
   - `t_a = X w_a`; `p_a = Xᵀ t_a / (t_aᵀ t_a)`; `c_a = Yᵀ t_a / (t_aᵀ t_a)`.
   - Deflate: `X ← X − t_a p_aᵀ`, `Y ← Y − t_a c_aᵀ`.
   - Stop early if `σ_a` or `t_aᵀ t_a` falls below 1e-12; the rank is then honestly short and the
     ladder reports the ranks that exist rather than substituting. Tests pin that a 32-component fit
     on a fixture of full rank yields 32, that the full-rank rule is the least-squares map, and that
     each of Codex's three witnesses (§10) is fitted as the mathematics says.
4. At rank `r`, `B_r = W_r (P_rᵀ W_r)^{-1} C_rᵀ`, with `W_r`, `P_r`, `C_r` the first `r` columns.

**Applying.** `T_r(x) = ȳ + ((x − x̄) / s) B_r`, where `x̄`, `s` are the fitting folds' source
statistics and `ȳ` their mean successor. The output is in the **untransformed residual space**: the standardisation is internal to
the rule's input and the centring is added back, so no normalisation of the rule touches the metric
and §4.2's warning about a distance in a space the rule normalised and the candidates did not cannot
arise. This is stated because it is the clause §4.2 asks a rule to answer.

**The ladder is one decomposition sliced.** NIPALS is greedy with deflation, so the first `r`
components of a 32-component fit are the `r`-component fit. The decomposition is computed once per
(model, depth, fold) and sliced for every rank, and a test pins the equality, as it does for E1.

## 3. Cost m greater than one, stated because it is the amendment's other substantive choice

The rule is a **one-step** update. A transition of cost `m` applies it `m` times:

    T_r^m(x) = T_r(T_r(… T_r(x) …)),    m applications.

`m` is the transition's step delta: 1 on an ordinary transition and on a contiguous correction, 2
where the correction crosses a step whose row was dropped. That delta is the recovery cost, which is
what §4.2 names. Iterating a one-step rule is what "the recovery cost" means once the rule is a map
rather than a displacement, and the alternative — fitting a separate two-step rule — would fit it on
no data, since every ordinary transition in this corpus advances by exactly one step.

Iterating a fitted map compounds its error, and that is a property of the claim, not a defect: E2
asks whether the update rule **generalises to the perturbation**, and a rule that survives one
application but not two has answered that question.

## 4. Depths

A separate rule is fitted at each of §5's six depths. No rule is shared across depths, no depth is
selected after the fact, the headline stays the 0.5 fraction at r = 8, and the full profile is
reported as §5 requires. The rule at a depth is fitted on residuals at that depth and is applied and
scored only there.

## 5. The capability gate, and the capability report beside the result

**Before any corrective stratum is scored, and on the fitting folds' own held-out ordinary
transitions only, the rule form must be shown capable of passing.**

> **Gate.** For at least **50%** of out-of-fold ordinary transitions, at the **headline** depth and
> rank, on **both** models, `T_r(x_k)` must be strictly nearer to `x_{k+1}` than to `x_k`. If it
> fails, no corrective stratum is scored, the amendment is revised, and the record says so.

**Measured with the SVD fitter of §2**, out of fold on the sealed assignment, on the 4,122 ordinary
train transitions at the headline depth. Produced by `measure_gate.py`, which is in the record beside
this table so the measurement can be replayed rather than trusted; its output is `gate-table.json`.
Every fold yielded a full 32 components on both models.

| rank | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 4B, layer 17 | 0.582 | 0.561 | 0.704 | **0.769** | 0.911 | 0.952 |
| 12B, layer 24 | 0.552 | 0.588 | 0.671 | **0.751** | 0.892 | 0.954 |

The gate passes on both models at the headline rank.

**Nothing moved.** The provisional table, measured with the power-iteration fitter §10 records as
defective, is reproduced here to every quoted digit. The largest difference between the two fitters
at any cell is **0.00049**, which is inside the rounding of the three-decimal figures themselves, so
the two agree to at least three decimals on this population. That is a fact about this population and
not a defence of the old fitter: F1's defect was real — a start vector that certifies a direction it
did not verify is leading — and it simply did not bite here. The SVD fitter is the one the seal
carries.

**The headline rank's fraction is also a mandatory capability report.** E2's hit rate at r = 8 is
printed beside the table's headline-rank figures unconditionally, at every depth and in the record,
so a low corrective hit rate is read against a rule that beats its own source about three times in
four on ordinary transitions at that rank. **What the report bounds, stated exactly (Codex F4).** A
capability fraction bounds retrieval only on the rows it was measured on, at their depth and rank,
with their transition cost and their averaging weights: a transported point that is not nearer the
successor than the source is not the successor's nearest candidate, so on the **same** rows, depth,
rank, cost and weights, the hit rate cannot exceed the capability fraction. On any other cohort —
the corrective strata, another depth, a cost-`m` transition whose `m` applications compound, or a
different weighting of episodes — it is a **capability reference**, not a ceiling, and the record
labels it so. It is never a null; conflating a reference with a null would be the same error in the
other direction.

**This clause was read at the top of the ladder for a time, and that reading is superseded.** It was
reclassified on 2026-09-10 because a rule form since withdrawn failed it at the headline rank by seven
thousandths. With the form corrected the clause passes as first drafted and returns to the headline
rank. §10 carries the sequence; the reclassification stays there as history rather than being erased.

The gate and the report are two-point comparisons between the successor and the source. Neither is
the retrieval score, which ranks all *N* candidates, and both are computed on ordinary **training**
transitions, so neither says anything about any corrective stratum nor can inform a later choice
about them.

The clause exists because the rule it replaces failed it absolutely: measured on the 4,122 ordinary
train transitions, the transported point sat 313.5 from the source and never closer than 398.1 to any
successor — **0 of 4,122** — so the zeros in every stratum were determined before a corrective
transition was ever touched. A score that cannot come out otherwise measures the instrument
(`METHOD-2026-09-08`, entry 27), and a clause that would have said so beforehand costs one number.

## 6. A declared descriptive baseline for local order

Reported beside E2 at every depth, and **not an estimand**:

> With the source **excluded** and **no transport at all**, is the successor the nearest other
> decision of its episode? Chance is 1/(N−1).

It adds no confirmatory quantity, enters no bound, and does not change M. It is labelled a baseline
for local order, never as update closure, because it asks whether the states of an episode are
locally ordered and not whether the rule closes. Its training-fold value is already known and is
recorded in §0.2 of the record: 0.430 on the 4B and 0.455 on the 12B against chance 0.172. Those
figures are the diagnostic that motivated this amendment; the sealed baseline is computed afresh on
the declared populations.

## 7. What is scored, and once

After this amendment is sealed: the gate; then, if it passes, the four strata of §4.2 — ordinary
train, all 553 corrective, and the contiguous and across-a-gap subgroups — scored **once**, at every
depth, with the headline at 0.5 and r = 8, each against its own tolerance from §7's operative table,
each contrast paired per episode against its chance level, with the transport distance reported
descriptively as §4.2 requires, and with §5's **current** capability report — the headline-rank row
of §5's table, which §5 alone owns; no figure from it is repeated in this section, so a
re-measurement cannot leave a stale copy behind (Codex F3: revision 4 still carried the withdrawn
form's headline figures here) — printed beside every headline figure, as a ceiling where §5's matching
conditions hold and as a capability reference otherwise. `check_amendment.py` fails the document if
a §5 table figure recurs outside §5 and §10.

## 8. Why the 553 corrective transitions are not spent, verified rather than asserted

The Chief's ruling is that they are not spent because a rule that cannot pass by its geometry conveys
nothing about which rule would. That rests on a claim about the run, and the claim checks out: the
failure is fully determined by the ordinary **training** displacements, as §5's figures above show,
so no choice in this amendment could have been informed by a corrective transition. Had the rule been
capable of passing and merely failed, they would be spent and this amendment would not be possible.

## 9. What this amendment does not do

It does not reopen E1, which is read and recorded. It does not touch the unknown-horizon control,
which stays deferred with its veto disabled under its own separate amendment. It does not change M,
any tolerance, any stratum, the metric, the candidate set, the tie rule, or the fold assignment. It
does not license a second revision of the rule after a result is seen: if the amended rule passes the
gate and then scores poorly, that is E2's answer.

## 10. The detour this draft took, recorded because the reader is owed it

Three things here were wrong at some point on 2026-09-10 and all three were the D-CRO's.

**The withdrawn rule.** E2 was first read with `T(x, m) = x + m·d`, `d` the mean per-step
displacement — the literal reading of §4.2's sentence. It returned 0.000 in every stratum and could
not have returned anything else: on the 4,122 ordinary train transitions the transported point sat
313.5 from the source and never closer than 398.1 to any successor, **0 of 4,122**. That is what this
amendment replaces, and why §5 exists at all.

**The withdrawn argument, and the form it produced.** It was then argued that a rank-r map straight to
the successor must spend its rank representing the **identity**, land near the training mean, and lose
to an identity-plus-correction form `T_r(x) = x + Δ_r(x)` with `Δ_r` fitted to the displacement. The
Chief ruled on that argument and the argument is **refuted**. It had been inferred from a synthetic
case in the wrong regime — a displacement larger than the state, where the corpus's is about 6% of it.
The test written to pin the claim failed, and widening it inverted the claim: as the rank shrinks as a
fraction of the space, the direct map wins on every seed. Measured out of fold on the corpus at the
headline depth, the direct map dominates at **every** rank on **both** models:

| form | model | r1 | r2 | r4 | r8 | r16 | r32 |
|---|---|---:|---:|---:|---:|---:|---:|
| direct map | 4B | 0.582 | 0.561 | 0.704 | 0.769 | 0.911 | 0.952 |
| direct map | 12B | 0.552 | 0.588 | 0.671 | 0.751 | 0.892 | 0.954 |
| identity + correction | 4B | 0.109 | 0.241 | 0.424 | 0.586 | 0.775 | 0.905 |
| identity + correction | 12B | 0.162 | 0.158 | 0.321 | 0.493 | 0.732 | 0.901 |

The reason the argument fails is worth keeping: **predicting the successor is easier than predicting
the displacement.** The successor is dominated by a large, highly predictable shared component; the
displacement is small and only about 20% coherent. Rank spent on the shared component is not wasted —
it is what makes the target predictable at all.

The form was chosen back on the **plain reading**, not on this table. The gate is a two-point
comparison on training folds and cannot arbitrate between two forms that are both capable; the
retrieval score could, and was not consulted, because on ordinary train transitions it is ε_ord's own
population and reading it to pick the instrument would be reading an estimand before the seal.

**The fitter's restart defect.** While checking the above, the fitter was found to restart each
component's power iteration from the **previous component's direction**, which lies in the part just
deflated, so the iteration could stall and return fewer components than requested. It is fixed — each
component restarts from the current cross-product, as §2 always specified — and a test pins that a
32-component fit yields 32 components. §2 was right and the code disagreed with it.

**E1 is unaffected, shown rather than argued.** The defect lived only in the vector-target fitter,
written after E1 was read. E1's fitter has a scalar target, so its direction is `Xᵀy` in closed form
with no iteration and no restart; `read_e1.py` never imports the defective module; and E1 re-run on
the fixed tree reproduces its readings **byte for byte**, digest
`ba841d7c0601b6326c621c68701852140e92e931ced570cb2ca2dd03d36f303f`. E1 stands as recorded.

**Codex's review of revision 4 (657a08a, WSA-TRANSPORT-AMENDMENT-REVIEW-2026-09-10), applied by the
Chief on 2026-09-10 as revision 5.** Four findings, all correct, all applied:

- **F1, the fitter could produce NaNs or certify a weaker direction.** The power iteration started
  from `Xᵀ(Y·1)`, falling back to the first column of `XᵀY`. Codex's witness `XᵀY =
  [[0,1,−1],[0,−1,1],[0,0,0]]` zeroes both, so the old fitter divided zero by zero and kept NaN
  weights while reporting rank 1. Its second witness `XᵀY = [[8,−8,0],[0,0,4],[0,0,0]]` has leading
  left singular vector `(1,0,0)` (singular value √128), but the start `(0,4,0)` is a fixed point of
  the iteration with singular value 4, alignment with the leading direction exactly 0, and the old
  fitter certified it. The leading direction is now the leading left singular vector from a full
  SVD, sign-fixed, with finite guards; §2 says so.
- **F2, the nine tests also passed on the defective fitter.** Codex's discriminating fixture —
  four orthogonal source rows, successors `(2x₀+x₁, 2x₀−x₁, 0)`, rank 2 requested — gets one
  component from the restart-defective fitter and two from the fixed one. That fixture, the two F1
  witnesses (finite weights matching the SVD reference; alignment 1 with `(1,0,0)`), a nested
  32-component ladder, the full-rank-equals-least-squares known answer, a narrower-than-source
  target, and refusal of a non-finite input are now tests. Run against the fitter at b0cd61c, seven
  of the fifteen fail; against the current fitter all fifteen pass. Found on the way: the old fitter
  reached only 11 of 12 components on the full-rank fixture, and its target-loading array was sized
  by the source width, so a narrower target could not be fitted at all.
- **F3, §7 still mandated 0.586 and 0.493.** Those were the withdrawn form's headline figures.
  §7 now refers to §5's current table and repeats none of it; `check_amendment.py` fails the file if
  a §5 table figure recurs outside §5 and this section.
- **F4, "at every depth … a ceiling" overstated.** A capability fraction bounds the hit rate only
  on the same rows, depth, rank, transition cost and averaging weights. §5 now states the bound's
  exact scope and calls the figure a capability reference everywhere else.

**Consequence for §5, now discharged.** Its gate table had been measured with the fitter F1
describes. It was re-measured with the SVD fitter on 2026-09-10, out of fold on the same 4,122
ordinary train transitions, the same sealed assignment and the same depth, by `measure_gate.py`,
which is in the record so the measurement can be replayed. **No figure moved at the quoted
precision** — the largest difference at any cell is 0.00049, inside the rounding of three decimals —
and every fold yielded a full 32 components on both models. §5 now carries the re-measured table and
the gate is read on it. That the two fitters agree here is a fact about this population, not a
defence of the old one.

