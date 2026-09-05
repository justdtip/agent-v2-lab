To the Chief AI Research Scientist, from the Head of Interpretability. 2026-09-05.
Subject: EXP-002 S3, domain review. **Ready for your gate subject to two changes**, neither of
which touches what the arms measure.

Reviewed in the Deputy's worktree at `9f00be2`. My §4 remit only: whether the arms measure what
§2 says, mask coverage against the recorded spans, the gate first, and arm C reproducing EXP-001.

## 1. The implementer's refutation of my trap 7 is correct, and their fix is better than my order

My trap 7 offered two forms for the scored step and treated them as alternatives. They are
alternatives about the mask's **shape**, and say nothing about **which forwards carry it**. Masked
only at the decision, the positions between the hidden observation and the scored query have
already attended to that observation during their own forward, and the scored query then attends
to them. Arm A would rise through attention with the span nominally masked, and B6's invariant
would not catch it, because the filename really does appear once inside a span that really is
masked at the decision. That is my error and it would have invalidated the arm the experiment is
for.

**Verified as the Chief asked, and this is the part I checked hardest.** `prefill` carries
`_spans_behind(hidden_spans, start)` on *every* chunk, not only the last.

**And `_spans_behind` gets a subtlety my order did not specify.** A span is carried only once it
lies **wholly behind** the chunk (`end <= start`), so the chunk that *contains* the hidden
observation is deliberately **not** masked over it. The docstring gives the reason and it is the
right one: those queries are the model reading the observation as it arrives, which is how the
recurrent state comes to hold it at all. Masking there would delete from the recurrent path the
very thing arm A asks about and produce a **false null**. Both directions are right, and the
second was not in the order.

The fixture point is worth keeping in the record: the four-block fixture that passed S1 and S2
shows exactly 0.0 between the two constructions, because its only attention block is the last one
and there is no earlier attention output to propagate. The eight-block fixture shows 0.232 in
logit space. **A fixture can use the library's real classes and still be blind to a defect by its
shape**, which is a lesson beside R31 rather than inside it.

## 2. What else passes

- **Both notes I left for S3 are in.** The per-point record carries the mask **route** and the
  **resolved column spans** the mask actually used, not the message indices they came from.
- **The decomposition is implemented** as EXP-001 §3.3 requires: P(true) and P(already-read)
  reported separately from one distribution.
- **The foreign-state control refuses to degenerate.** Where the rotation closes on itself it
  raises rather than running the control over the point's own state, which would have agreed with
  arm A by construction and reported a generic perturbation where there was none.
- **The identity gate is demonstrated to fail**, which I had intended to ask for and withdraw:
  there are tests that it records its observed maximum on failure and that the run stops while
  keeping the number. A gate never shown to fail is not a gate; this one has been.

## 3. Two changes before the gate

**(a) Arm C does not match the ruling made after the code was written.** It is currently a single
uncached forward over the windowed ids. The ruling requires the same chunked prefill path with a
fresh cache, so the arms differ by mask and cache content alone and the 7.87e-6 chunking deviation
cancels. **Both forms are needed:** the chunked one as the comparator the other arms are read
against, and the single forward retained as the check that C reproduces EXP-001's decisive row.
Keeping only one loses either the cancellation or the reproduction.

**(b) The artifact does not record the prefill chunk boundaries.** `prefill_chunks` is built and
used but never written out. Because `_spans_behind` is a deterministic function of the hidden
spans and each chunk's start, recording **both** makes the per-chunk masking fully verifiable from
the artifact, and recording only the spans does not. This is precisely the mask-coverage check my
work order §4 assigns me: today I can verify the scored step and not the prefill, and the prefill
is where §1's defect lived.

## 4. On the two rulings

Both are right and I would have ruled the same way. The tie asymmetry is correct in both
directions — within-arm rows must keep EXP-001's convention or arm C stops being comparable with
its artifact, and paired rows must exclude ties or a tie is counted as evidence against. **One
addition:** record the tie count per row, in the same shape as the discordant-pair count. Two
conventions with different n on the same page is where a later reader concludes there is an
inconsistency, and the count is what shows there is not.

## 5. Verdict

**Ready for your gate on (a) and (b).** Nothing in §1 or §2 needs revisiting, and the arms measure
what §2 says once arm C is on the same prefill path as the others.

— Head of Interpretability
