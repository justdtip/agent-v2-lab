# Amendment 1 to the plan-progress pre-registration: E2's transport rule

**D-CRO, 2026-09-10, on the Chief's ruling of 13:50Z. Draft, for Codex's file-only review.**
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
supervised map fitted on ordinary transitions**, the instrument the Chief fixed for E1 generalised
from a scalar target to a vector one — PLS2, the classical NIPALS construction, which reduces to E1's
PLS1 when the target has one column.

**Fitting**, on the fitting folds' ordinary transitions only:

1. `X` is the matrix of source residuals, standardised by the fitting folds' per-coordinate mean and
   standard deviation, a coordinate whose scale is below 1e-8 held at 1.
2. `Y` is the matrix of successor residuals, **centred only**, by the fitting folds' mean. It is not
   scaled per coordinate.
3. For components `a = 1 … 32`, greedily and with deflation:
   - `w_a` is the leading left singular vector of `Xᵀ Y`, obtained by power iteration on
     `v ← Xᵀ(Y(Yᵀ(X v)))` with `v` renormalised each step, started from `v₀ = Xᵀ(Y·1)` normalised —
     and from the first column of `Xᵀ Y` if that vector is zero — stopped when the update moves `v`
     by less than 1e-10 or after 200 iterations, whichever comes first. The iterations reached are
     recorded per component. No random number enters the fit.
   - `t_a = X w_a`; `p_a = Xᵀ t_a / (t_aᵀ t_a)`; `c_a = Yᵀ t_a / (t_aᵀ t_a)`.
   - Deflate: `X ← X − t_a p_aᵀ`, `Y ← Y − t_a c_aᵀ`.
   - Stop early if `‖Xᵀ Y‖` or `t_aᵀ t_a` falls below 1e-12; the rank is then honestly short and the
     ladder reports the ranks that exist rather than substituting.
4. At rank `r`, `B_r = W_r (P_rᵀ W_r)^{-1} C_rᵀ`, with `W_r`, `P_r`, `C_r` the first `r` columns.

**Applying.** `T_r(x) = ȳ + ((x − x̄) / s) B_r`, where `x̄`, `s`, `ȳ` are the fitting folds'
statistics. The output is in the **untransformed residual space**: the standardisation is internal to
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

## 5. The capability gate, which is the clause this amendment exists for

**Before any corrective stratum is scored, and on the fitting folds' own held-out ordinary
transitions only, the rule must be shown capable of passing.**

> **Gate.** For at least **50%** of out-of-fold ordinary transitions, at the headline depth, on
> **both** models, the transported point `T_r^1(x_k)` must be strictly nearer to `x_{k+1}` than to
> `x_k`. The fraction is reported whatever it is.

If the gate fails, no corrective stratum is scored, the amendment is revised, and the record says so.

This is a two-point comparison between the successor and the source. It is **not** the retrieval
score, which ranks all *N* candidates, and it is computed on ordinary training transitions, so
passing it says nothing about any corrective stratum and cannot inform a later choice about them.

The gate exists because the rule it replaces failed it absolutely: measured on the 4,122 ordinary
train transitions, the transported point sat 313.5 from the source and never closer than 398.1 to any
successor — **0 of 4,122** — so the zeros in every stratum were determined before a corrective
transition was ever touched. A score that cannot come out otherwise measures the instrument
(`METHOD-2026-09-08`, entry 27), and a gate that would have said so beforehand costs one number.

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
descriptively as §4.2 requires.

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
