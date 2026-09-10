# Workspace experiments: prospective design review

**Codex, 2026-09-10. Standing request 3. Initial source: `d801db7c118d0a9961187903de29cb5612c2e336`; amendments reviewed through
`08ce0504b3f22adafd5446a1aaeac765eda6b2c8`.**
The reviewer has read no workspace or plan-progress captures. This is a review of the committed
design, reporting contracts and mathematical implications; it does not attest to what other seats
have accessed. No device operation was performed. All example numbers below are analytic examples,
not observations of Gemma. `audit.py` reproduces them.

**Verdict: revise before sealing these as tests of workspace properties.** The proposed archives
can support useful descriptive measurements, but the present design does not identify ignition,
functional global availability, causal broadcast, or whether a note participates in deciding.
Capturing is a separate scheduling decision for the owner; this review does not stop or launch a
job. The least costly resolution is to label this pass an observational screen, fix the contracts
below before reading it, and reserve the mechanism claims for the separately ordered steering
experiments. These are not demands to spend additional device time without a ruling.

## W1 — The instrument's fitting domain is disclosed, but not controlled

Common §0 fits at 128 prose tokens and reads much later agentic positions. Precision and width
matching remove the particular cross-path confound; they do not license this positional and
contextual transfer. The sentence saying “nothing is refused” is too broad. The document says
“experiment W-C measures what it costs,” but **there is no W-C** among W-1 through W-4. The
logit-lens control has no fitting extrapolation, but disagreement between it and the transported
readout does not isolate the error of either instrument.

Before any depth or local/global contrast is interpreted, name the actual fitted source and target
positions, context lengths and reduction from each lens's declaration, together with archive hash,
not just “128.” Label each reading's extrapolation and preserve its position distribution. Declare
an independent long-context/readout validation or leave the affected workspace conclusion
unresolved. Do not quietly treat the short calibration's scalar checks as a long-context matrix
validation. No accuracy cutoff is invented here.

Also distinguish a **lens score** from a calibrated probability of an action. An averaged Jacobian
is not generally an activation reconstruction: for the scalar function f(h)=h² at h=3, J(h)h=18
while f(h)=9. This does not invalidate the J-lens as a reading instrument; it shows why accurate
derivatives do not by themselves calibrate its softmax thresholds.

## W2 — Confidence crossings are not yet an ignition experiment

W-1 declares thresholds 0.1, 0.5 and 0.9. Keep those declared values. But a six-class softmax with
one logit increasing **linearly by 4 per layer** and all others fixed crosses 0.1 to 0.9 in **one
layer** in the supplied example, despite zero second difference in its logit trajectory. No
competitive transition or shared workspace was used to construct it. A mid-layer non-target
argmax likewise describes a prediction change, not competition for a limited shared resource.

Report logits or log-odds beside conditional probabilities, retain every individual depth profile,
and call these quantities *first confidence crossing* and *readout competitor* unless a separate
mechanism test is specified. Freeze handling of oscillations, missing crossings, equal maxima,
unresolved intermediate layers, and an initially confident profile. A corpus-frequency baseline
does not reproduce the geometry or entire trajectory of an ordinary next-token predictor.

A stronger pre-registered ignition experiment varies competing input evidence and measures
within-layer response transitions, including a readout-independent activation measurement. It is a
new experimental arm for the owner to approve, not something this reviewer launches. Anthropic's
workspace paper includes an input-mixture sweep with residual projections, and cautions against
equating its competition findings with neural ignition. That is a useful methodological comparator,
not evidence that Gemma has the same property. [Primary paper](https://transformer-circuits.pub/2026/workspace/index.html).

## W3 — The six-class rank ladder and availability claim need separate definitions

W-2 says *rank-r linear readout* of six labels, at r={1,2,4,8,16}. If r is the rank of the
classification matrix, its rank is at most six, and the **softmax-relevant contrast rank is at most
five**: subtracting the mean row leaves probabilities unchanged. Ranks 8 and 16 therefore add no
linear class-contrast capacity. If r instead specifies an unsupervised projection or some other
representation constraint, say exactly what is fitted and regularized; it is a different rank
experiment. The toy six-by-six calculation in `audit.json` has ranks six before centering and five
afterward, using exact rational arithmetic.

Freeze the family subsets, split membership, training-only centering/scaling, regularizer and
selection protocol. Report shared tool-class support and family-specific priors. A label constant
inside a family is unchanged by a within-family permutation, as the supplied fixture verifies.
That null cannot detect cross-family sharing in such a family. Preserve episode structure in
permutation and identify the exchangeability assumption; shuffling individual decision labels also
changes temporal structure.

Even correctly measured transfer is **cross-family linear decodability**, not functional availability
of one representation to many downstream computations. At finite rank and sample size, null-level
transfer does not falsify the existence of a common subspace. Conversely, successful transfer can
reflect a shared tool-name encoding. Label the conclusion accordingly; the functional claim needs
controlled transfer/intervention with action outcomes. For an invertible linear lens J, the class
of rank-r linear readouts on Jh is the same as on h (WJh=(WJ)h); differences can arise from training,
conditioning, regularization, or rank deficiency, not automatically newly available information.

## W4 — Attention identifies direct routing weights, not causal content or all routes

W-3's local-mask check is worthwhile. Keep its independent boundary and wrong-mask controls,
including exact behavior around the window edge. The corrected design now states **five** global
layers in 4B and eight in 12B; those counts are source-reported here, not independently re-measured
from a checkpoint. The older six-layer wording in the WS-A direction should follow this correction.

Two separate non-identifiability problems remain:

1. Output depends on values and output projections as well as attention weights. The example puts
   99% attention on a token with zero projected value and 1% on a token contributing the entire
   scalar output. Large attention is not a measured causal share; small attention is not proof of
   absence. See [Jain and Wallace](https://arxiv.org/abs/1902.10186) for related empirical evidence;
   the algebraic example here establishes the limited claim independently.
2. A source 1,800 tokens back can reach the decision through **two local layers**, first 0→900,
   then 900→1,800. Both edges are shorter than 1,024. Every direct local mask passes while distant
   information is relayed in intermediate residuals. Removing the second edge breaks the example.
   Therefore the global layers are the only *single-edge* routes beyond the window, not necessarily
   the only end-to-end routes. This correction also applies to the carrier language in the state
   and plan-progress orders.

Name W-3 *direct attention to tagged spans*. Preserve per-head distributions, span sizes, distance,
accessibility and tokenization boundaries; include size-and-distance matched spans. For causal
carrier or broadcast claims, specify value/path interventions or controls that cut both direct and
relayed routes. Do not equate raw attention mass with “the carrier's share.”

## W5 — The revised W-4 and W-3b still need narrower causal conclusions

**Amendment credited.** While this review was in preparation, `86655ab` replaced W-4's primary
six-token probability with a learned probe and expanded the carrier tags; `70c44bb`/`08ce050` added
carrier-key and current-note-key masking. The old W-1-selected median-depth comparison is gone;
that selection concern is closed. A probe is a better instrument for the proposed early encoding
question. It still tests decodability rather than when a decision was made.

The amendment cites a test-row mass measurement of about 1e-14 as the reason for changing W-4.
That value is producer-reported, not re-measured here. Record which test rows, which access and
which outputs informed the change, and identify their role in later inference. The **revised**
primary cannot be described as uninformed by result inspection. This reviewer has not read those
captures. The concern is provenance and reuse, not an allegation of intentional selection.


The continuation is **the expert's teacher-forced completion**, not an action chosen in a free
Gemma rollout. P_act follows that supplied note and the JSON prefix. Reading the expert's named
action after its description is compatible with lexical echo, format conditioning, and genuine
computation. The common text must call the label the *expert action*, including when Gemma disagrees.
A first tool-name token is also not a complete executable action, particularly its arguments.

Two causal models can have identical clean observations: X→decision→note (report), or X→note→decision
(computation), with note=X and action=X in both. Flipping the note changes the latter action and
not the former. Their clean traces cannot choose between them. Similarly, a decision could exist
but be unreadable through this lens until the note; absence of an early readout does not falsify
“report.” A 1e-3 mass floor avoids one unstable reading but does not convert a conditional
next-token distribution into the counterfactual “which tool, were it to act now.”

The revised primary's interpretation still says that early decodability means a pre-existing
decision and late decodability means the note did the deciding. Relabel it as **early versus late
linear decodability of the expert action**. Neither a probe's prediction nor its failure establishes
the necessity of that representation for an action. The secondary's unresolved observations remain
in the denominator, with no inference that they lack a decision.

W-3b is an intervention, but its current causal branches overreach. If an action survives masking,
that shows lack of necessity of the masked edges for that outcome under this intervention. The
model can recompute the same decision from other inputs after the mask; survival does not date
the decision before the note. If the action changes, the mask may impair needed content, disrupt
attention normalization, or disrupt ordinary processing; a snap toward a prior does not uniquely
identify re-reading. Freeze the outcome and report the actual distribution/action change before
assigning a mechanism.

A mask on old carrier keys **only for queries in the current turn** leaves a relay open: an earlier
non-carrier position can have already absorbed the carrier, and current queries can read that
unmasked position. The two-hop example in W4 survives this exact distinction between direct keys
and copied content. The proposed random mask matches token count only; match distance, role,
position, legal-attention support, contiguity and per-query/per-layer mask burden, and preserve
separate mask realizations. Otherwise a more disruptive carrier mask can win merely by structure.
Masking the current note for every later query can cut its outgoing edges if span boundaries are
complete, but redundant task inputs still prevent the 'decided before' inference. Verify the graph
cut with an independent fixture, actual boundaries and a deliberately leaky mask, rather than
assuming a list of masked key indices cuts every relevant path.

## W6 — Freeze the observational contracts, not only their future manifest fields

The order requires scripts' hashes before capture reading, but **no workspace capture or analysis
script is in the reviewed changes**. That is unexecuted implementation, not a passed design gate.
The 300-decision sample currently says “by rule, seed in the manifest”; pin that rule, seed and
selected keys before access to captures. Specify the weighting target (25 per family is a balanced
family sample, not the corpus mixture), episode clustering, eligibility and each missingness count.

Bind the actual token IDs obtained in the full prefix context to all six tools. Assert distinct
first tokens, delimiters and span boundaries with a failing collision control. If names are
multi-token, first-token evidence remains that limited estimand; full-string/action likelihood is a
different one. Declare whether W-1 thresholds use full-vocabulary or six-token conditional mass,
and store both. Tail summaries must include all units and unresolved counts, not just resolved
medians. Freeze which comparisons receive simultaneous confidence coverage, their unit, and all
selection rules; “episode” alone does not define a confidence procedure.

The archive estimate also appears to count one decision position, while the capture requires two.
Using the declared dimensions and float32, residuals alone require about **5.31 GB for 4B and
11.25 GB for 12B**, excluding the embedding row, metadata and every-note-token sample. The quoted
2.6/5.5 GB are approximately the single-position amounts. These are storage calculations from the
declaration, not memory-peak measurements or reasons to alter a running job.

The existing plan-progress `prompt_sha256` is a digest of canonical **messages**, not rendered text
bytes (see companion review). Preserve that digest and add rendered-input/token identity at capture
time. Do not reuse one field with two meanings across these two programmes.

## S1 — The new Stage B control's spectrum is not its overlap geometry

The new bridge ruling at `dc3a36e` equates matching the Gram spectrum with matching the distribution
of pairwise overlaps. Those are different constraints, especially for a **nonnegative** sparse fit.
Let B's three unit atoms be e₁, e₂, (e₁+e₂)/√2. Negate only the third to obtain C. Their Gram matrices
are related by G_C=S G_B S, S=diag(1,1,−1), so they have exactly the same spectrum and atom norms.
Their off-diagonal overlaps change from {0,+1/√2,+1/√2} to {0,−1/√2,−1/√2}. Even absolute overlaps
match, while the nonnegative cones do not. For target −(e₁+e₂)/√2, C reconstructs exactly with one
atom; B's best nonnegative reconstruction is zero, squared error one.

**Ruling needed on the null, not a code change by this reviewer.** One candidate is a seeded
orthogonal rotation of the complete dictionary: it preserves the *full signed Gram matrix* while
randomizing orientation relative to the residuals. Keep atom norms, budget, solver, selection and
stopping rule identical, and state precisely what hypothesis the rotation null tests. It does not
control every possible confound. The stronger fact that overcompleteness alone guarantees good
sparse reconstruction should also be measured under the null, not assumed as a theorem.

## What counts as passed here

No workspace-property result is certified by this review. The model-free counterexamples and
source checks passed; prospective device gates remain unexecuted. `audit.json` enumerates eleven
reviewed checks: one demonstrated to reject its producer's pinned claim, five unable to certify
the scientific target attributed to them, and five with unexecuted or unknown behavior. This is
a scoped inventory, **not** a reconstructed lifetime count of the programme's gates. In particular,
a formula unit test may catch a coding error while being unable to validate independence.
