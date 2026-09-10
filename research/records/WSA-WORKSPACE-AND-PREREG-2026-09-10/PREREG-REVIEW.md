# Plan-progress pre-registration: re-review before the seal

**Codex, 2026-09-10. Standing request 1.** Initial producer source `bbdc8dcba7146aca1e17db8ecb15d692a6806b38`, with the capture writer and
text-removal scope amendment inspected at `144fa1cf1f753c008daf1eb4f38282a1e71184d0`,
Chief's current rulings at `d801db7c118d0a9961187903de29cb5612c2e336`, production tolerance helper at
`59b1e77278c221d8aa21edbe7f118c7d9a4a5289`. No captures read. All statistical counterexamples are
analytic and make no empirical claim about this probe's eventual performance.

**The requested edits largely carry and their numbers reproduce. The seal still needs the
clarifications and rulings below.** This review keeps the approved numerical tolerances unchanged.
It distinguishes compliance with the Chief's written rule from mathematical support for the rule.

## P1–P6 disposition

| item | disposition from the committed files |
|---|---|
| P1 capture width and token index | Carried in §2: forward and anchor width one, actual token index after tokenization, native precision. The newly landed capture writer was subsequently exercised with metadata-only spies; its self-stamping and resume findings are in CAPTURE-REVIEW.md. No model capture or tensor storage was executed. |
| P2 retrieval | All N decisions, source included; strict nearest target, ties miss; episode-then-population averaging; uniform-guess reference. Carried. The actual distance is still only “the metric the rule is fitted under”: name its space, norm/metric and any fitted normalization in the seal, not a circular reference. |
| P3 current sequence | §13 now says review→seal→first reading, capture allowed earlier. Carried there. The opening still says sealing waits on the golden design, although the order says it has landed; remove that stale blocker. |
| P4 episode units and subgroups | The main, ordinary, pooled-corrective, contiguous and gap counts reproduce under the approved formula. Superseded decision-based figures are labelled. The inference guarantee remains unsupported as explained below. |
| P5 folds | All 1,128 episode assignments independently reproduce from seed 20260910, strata and rule; the producer script also reproduces its entire JSON. Both learned stages must be out of episode. Carried as a requirement; no estimator execution examined. |
| P6 digests | Both byte and logical-triple hashes independently reproduce from the frozen capture set. Pin the serialization explicitly or bind the digest script hash in the seal. The meaning of each row's prompt hash needs correction below. |

The capture set contains 7,629 unique `(task_id, step)` pairs and 1,128 episodes. Fold sizes are
220, 232, 226, 224 and 226. All are metadata recomputations. This review did not rerun the original
corpus generator or inspect model activations. `check_prereg.py` passes; its self-test rejects all
pinned-claim corruptions, and our independent 7,629→7,628 mutation is refused. An appended false
independence statement passes that checker. This is its documented scope, not a newly alleged bug:
it checks pinned documentary figures, not statistical validity.

## R1 — Out-of-fold is not independent; an approximate bound must not be called a guarantee

§7.1 correctly acknowledges shared training data and calls the episode-score bound an
approximation. But §7 elsewhere calls it a distribution-free floor, conservative, a guarantee and
the true resolution of this corpus. Those cannot simultaneously describe this procedure. The
Chief's P5 instruction itself says the bootstrap addresses shared training; naming the issue does
not repair it.

The supplied exact example enumerates all 32 possible datasets of five independent Bernoulli
labels, one episode in each of five folds. Each trained classifier predicts the parity of its four
training labels. Each held-out error then equals the parity of **all five labels**. No model sees
its held-out episode, yet all five errors are identical. The cross-validation mean has variance
0.25, versus 0.05 under the independence calculation. Resampling the five fixed errors within
any dataset gives variance **zero**. An episode unit fixes within-episode pseudo-replication; it
does not remove training-induced dependence across episodes.

This example proves no error magnitude for the real regression. It refutes a general coverage
guarantee based on independence alone. Related primary results describe the same cross-validation
variance difficulty: [Bengio and Grandvalet](https://jmlr.org/papers/v5/grandvalet04a.html),
[Bates, Hastie and Tibshirani](https://arxiv.org/abs/2104.00673).

**Ruling required:** either state these as approximate, assumption-dependent uncertainty summaries
throughout (without “distribution-free floor” or guaranteed resolution), or define an inference
scheme that supports the claimed coverage. An untouched episode test set conditional on a fixed
training pipeline is one route for a narrower estimand; another requires an explicit cross-fit
uncertainty argument. A full resampling-and-refitting procedure may estimate training variation,
but is not automatically a finite-sample guarantee. Neither route is authorized or implemented by
this record. Keep the approved epsilons visible while this is resolved.

Also freeze the bootstrap's pairing, scope, seed, refitting and family/variant stratification.
The cited production `bootstrap_bounds` resamples the two arms **independently**; its
`paired_bootstrap_lower_bound` simply calls that same function. It does not implement a joint
resampling of paired episode contrasts. This is a source observation, not an executed numpy test.
The new E1 comparisons share episodes, so do not inherit this function's name as an assurance of
pairing. Duplicating one bootstrap resampling index across both arrays is the relevant contract;
which sampling unit is valid still depends on the inference design.

## R2 — The bound's range and contrast must be specified before its numbers are guarantees

For independent bounded episode observations Xᵢ in [a,b], the elementary two-sided Hoeffding and
union-bound form is

`P(max_j |mean_j − expectation_j| ≥ epsilon) ≤ 2M exp(−2n epsilon²/(b−a)²)`.

A single accuracy score has range width one. An **arbitrary paired accuracy difference** has range
[-1,1], width two. In that latter case a sufficient sample size from this form is
`2 ln(2M/alpha)/epsilon²`. The approved helper computes `ln(2M/alpha)/epsilon²` instead. It can be
obtained for a contrast of two independent [0,1] arms each of size n, or under stronger covariance
conditions; it is not a generic result for paired contrasts. E1's null comparison and the
between-model difference must say which object is bounded. E2's random-guess reference is a
per-episode known quantity and requires its own declared range/conditioning; it is not necessarily
the same case as two fitted null predictors.

Our script verifies the existing helper gives 214, 763, 511, 242 and 275 at epsilons .17, .09, .11,
.16 and .15. These are correct evaluations of its formula. It also records the width-two
comparison for .17, .09 and .11 as **an assumption check, not replacement tolerances**. R1 remains
separate even after the range is correct. Ask for the explicit contrast and assumptions supporting
the retained formula before a coverage claim. This issue is in the inherited rule, not a failure by
the D-CRO to transcribe it.

M=12 fixes the table's named primary quantities. State explicitly whether the mandatory subgroup
bounds, both-null contrasts, between-model contrast, rank/depth ladders and bootstrap/drop
selection are covered by that family of statements or reported separately as descriptive. Do not
silently present every profile cell as having simultaneous coverage at the same M.

## R3 — The carrier screening script is not the promised acceptance harness

§4.3 says the harness checks both the distinguishability threshold and byte-identical preservation
of the call and next-action clause. `carrier_ablation.py` never compares those bytes; it extracts
only completion prose. It computes unique remaining strings divided by episode length, averages
that ratio by family, and labels any episode with ratio at most **0.5** as `fully_ablated_episodes`.
That flag is not the declared per-episode floor-plus-1/n criterion.

The frozen script run on a synthetic 20-step episode with ten remaining strings calls it fully
ablated at 0.5, while the stated acceptance ceiling is 0.1. A second synthetic episode with two
strings passes that ceiling and still encodes an early/late phase bit perfectly, with the call and
next-action clause identical throughout. The criterion may be a deliberately tolerated residual
signal; it cannot certify the absence of all progress information. Unique strings also measure
within-episode distinguishability, not learnability/generalization to new episodes.

Relabel the current output as a screening diagnostic, correct the misleading `fully_ablated`
field, and implement the declared two-part gate before any rule is accepted. Preserve per-episode
n, remaining groups, tails and failed clause checks. The current record honestly says no usable
rule exists, and `144fa1c` now explicitly limits this to the text-removal arm; that does not justify calling the future acceptance harness implemented. The removed
current completion note is also not itself present at this decision's P_note: a later diagnostic
must tag the actual carrier spans in the actual rendered input, across every retained earlier note
and any still-visible observation. The surviving tool-call history and prompt structure are
alternative carriers. Under `cache: none`, a difference after re-rendering identifies an effect of
that input edit; it does not by itself isolate persistent model state.

## R4 — The prompt hash's domain is messages, not rendered input bytes

`prereg_inputs.py` sets `prompt_sha256 = sha(row.get('messages', [])[:-1])`; `sha` hashes canonical
JSON with `sort_keys=True, ensure_ascii=False`, default separators and UTF-8 encoding. This can be
a useful semantic-input identity, and the logical capture-set hash is correct under that domain.
It is not the sha256 of the rendered prompt, as §2 and the workspace design call it. Different
chat templates or tokenization can produce different input tokens from the same messages while
leaving this hash unchanged.

Preserve the existing digest and capture set. Name its actual domain, and bind rendered bytes and
exact input token IDs separately at capture time, including any teacher-forced note/JSON prefix at
P_act. A manifest must not silently reinterpret the existing field. The canonical logical-triple
serialization is reproduced by `audit.py`; its script source is frozen beside it.

## Scope limits and remaining status corrections

The actual metric/model class for E2, all learned preprocessing exclusions and any hyperparameter
choice must be frozen, not merely folds. As written, §7.1's use of all split strata could be read
as allowing clean test episodes into other folds' fits, while §4.1 calls test independent and §4.2
fits E2 within train. State the training eligibility mask for E1 and E2 explicitly. An out-of-fold
assignment alone does not enforce the original train/test contract.

§10 labels E1 and E2 `measured` in this unsealed, capture-free draft. They are planned measurement
categories; actual status is **not measured** until execution. §7 also retains a second passage
calling the bootstrap an unchosen suggestion after declaring it mandatory. Resolve these specific
contradictions before the seal; they are reporter issues, not results.

The newly landed capture code has additional acceptance and resume defects documented in
[CAPTURE-REVIEW.md](CAPTURE-REVIEW.md); these are part of this standing re-review, not a native
test run. The companion workspace review's direct-attention and two-local-hop counterexamples apply to
§4.3 too. The numerical P1–P6 corrections deserve acceptance within their stated domains. The
inference and carrier claims above remain unresolved. A narrow descriptive reading can proceed
only under the owner's clarified seal; this record grants no new experiment authority.
