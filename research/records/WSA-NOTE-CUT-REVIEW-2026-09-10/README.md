# Repaired note cut: the empty mask is fixed; placement and result admission remain unchecked

**Standing request 5. Codex, 10 September 2026. Verdict: partial.** The repaired
builder implements the declared note cut on the fixed-offset cases checked. Its
new nonempty check rejects the original defect. A nonempty but misplaced cut can
still pass its checks, and the consumer includes failed or incomplete records in
ordinary scientific summaries. The first finding is a validation gap, **not a
finding that the repaired production boundary is wrong**. The consumer also calls
a newly gained winner a surviving winner and writes a numerical zero for an empty
eligible denominator.

This review changes no experiment source, jobs, locks, scientific tolerances or
scheduled run. It supplies the requested deliberately misplaced-cut fixture and
reviews the producer and consumer together. Scientific result admission remains
unaccepted pending the bounded corrections below; this is not an instruction to
interrupt any running process.

## Scope and evidence order

Target: `workspace_w3b.py` and `workspace_w3b_analyze.py` at main
`c63157777f88e8cfa1e18f467702061d2ed82cc3`, in
`research/records/WORKSPACE-EXPERIMENTS-2026-09-10/scripts/`. Both are unchanged at
main `3e82317e2de05d4182f0ada1b1b082ba30929942`. Baseline: the original script frozen
by the public review `712f78c`. The current primer, P1/W5/W6 rulings, routing order
and METHOD entry 34 were read first. New capture-v2/resume changes at `3e82317` and
integration `54248ec` are outside this bounded review.

Evidence is source AST plus synthetic token offsets, masks, nonnegative attention
and reporter rows. No model, native backend, checkpoint, tokenizer or test suite
was imported or run. No scientific capture archive or full-run outcome was read.
The order had already disclosed the producer's one-row CPU smoke test. Its summary
table was read after the source and synthetic counterexamples were analysed; it
is producer-reported instrument evidence, not independently reproduced here.
This review is not prospective to that smoke test.

`source/` contains seven exact committed files, pinned by commit and SHA-256 in
`sources.json`. `check.py` verifies those hashes, executes selected builder and gate
AST nodes, and runs the entire reporter AST with standard-library dependencies
and an in-memory filesystem. `check.json` preserves the synthetic inputs and
outputs. Source imports and model execution are not part of this reproduction.

## Findings

### F1 — P2: placement is still certified against the supplied mask itself

Producer lines 126–149 implement the corrected semantics: note keys start before
the JSON fence; `q0` is the first token starting at or after it; a straddling token
belongs to the note; missing prose skips the current-note arm. Sixteen fixture
cases cover ordinary boundaries, one-token notes, fence-straddling tokens and no
prose, at four prompt offsets. The implemented sets match their independently
hand-labelled expectations. The old builder produces an empty set; the new
nonempty assertion at lines 174–175 rejects it.

But lines 174–185 ask whether **the supplied set** is nonempty and receives zero
attention. They do not compare it with an independently specified set of note and
query positions. This distinguishes an applied cut from a correctly placed cut.

The requested negative fixture has `S=8`, `P_note=1`, note keys `{2,3}`, syntax
positions `{4,5}`, `P_act=6`, and a supplied target token at 7. The correct query
start is 4. Move the actual start to 6 while keeping note keys `{2,3}`:

| Fixture | Blocked entries at P_act | All declared blocked attention | P_note selected readouts | Note-to-action route |
| --- | ---: | ---: | --- | --- |
| Correct start 4 | 2 | 0 | identical | absent in the two-layer example |
| Delayed start 6 | 2 | 0 | identical | 3 -> 4 -> 6 remains open |

Both pass the current assertion family, including a separately passing carrier
leaky control. In the delayed case one layer reads note key 3 into syntax position
4; a later layer reads syntax key 4 at position 6. Each synthetic attention row is
nonnegative, causal and sums to one. The model-free graph is a possibility proof,
not an estimate of Gemma's actual attention or effect size.

The fixture also mutates the keys, omits one note key, adds an unrelated key and
selects only a future key. These five nonempty wrong masks pass the current checks
when supplied with zero attention on their implemented entries. The independent
expected-edge oracle rejects all five, and also rejects the empty mask. Actual
producer construction is not mutated in the positive fixtures.

**Correction:** add an explicit placement preflight using frozen, independently
labelled token/offset cases; compare missing and unexpected edges, not only counts.
Carry expected key IDs, query interval and actual effective-mask coverage in the
receipt. Before attributing an effect to the intended cut, check that the hook
preserves the native restrictions and introduces exactly that cut. The supplied
fixture is the acceptance example: delayed-start and wrong-key controls must fail
for placement even when every attention-on-implemented-mask check passes.

The code masks `q0:` including the supplied target query at `S-1`; prose says
"through P_act". The additional unread future query cannot affect earlier
positions in a causal forward. State the actual endpoint in the receipt; do not
mistake this declaration mismatch for the original empty-cut bug.

### F2 — P2: failed, unverified and incomplete rows enter ordinary summaries

Consumer lines 9–10, 35–49 do not validate row/arm gate requirements or the exact
requested ID set. The frozen reporter emits `w3b_analyzed` and scientific numbers
for positive masked-edge mass, positive out-of-window mass, zero blocked P_act
entries, false P_note identity, a missing gate, legacy schema, missing requested
rows, duplicate IDs, additional IDs and an unexpected ID with the correct count.
These are reproduced independently of any running pass.

This is **not a fabricated pass verdict**: the reporter has no pass verdict.
The defect is inclusion of failed or unverified inputs without a blocking validity
state. The producer refuses failure rows before writing them, which is good; that
does not validate historical files or protect the consumer from stale manifests
and partial reruns. The producer opens `w3b.jsonl` in overwrite mode at line 111 and
writes the completion manifest only at the end, so an earlier manifest can outlive
an interrupted rerun in the same output directory.

One counterexample hides evidence rather than merely displaying an invalid row.
`gate_summary` lines 40–42 decides which fields to aggregate by looking only at
`gs[0]`. Put a legacy gate first, then a second gate with all-query leakage `0.5`:
the all-query leakage field disappears from the summary, while the older displayed
attention fields remain zero. `check.json` preserves both input gates and the
actual output that omits the nonzero field.

**Correction:** validate each schema-2 row and applicable arm, require all gate
fields and finite passing values, and reconcile unique IDs with the manifest's
requested sample and row count before publishing eligible estimates. An empty
requested set, incomplete set and all-skipped set need distinct outcomes. Retain
historical diagnostics if useful, labelled unverified/ineligible. Do not infer
schema from the first gate. Keep applicability and completed-arm coverage beside
the verdict. The strict scientific consumer should refuse a failed admitted row;
a diagnostic consumer may render it separately without treating it as eligible.

### F3 — P2: "still top" includes a newly gained winner; no data becomes zero

Consumer line 29 checks only whether the masked argmax equals the expert label.
In the fixture, the unmasked distribution has expert tool 0 at `0.1` and tool 1 at
`0.8`; after masking, tool 0 has `0.8`. The reporter returns
`"taken_still_top_at_act": {"share": 1.0, "n": 1}`. The expert tool became the
winner; it did not remain the winner. An unresolved unmasked reading produces the
same retention value. These cases have resolved, untied masked winners.

When every masked reading is below the floor, line 51 returns `share=0.0, n=0`.
That is no eligible measurement, not evidence of zero retention.

**Correction:** either rename the current metric to masked expert-tool agreement
and give it a null share when `n=0`, or compute actual paired transitions with
jointly resolved base and arm: retained, lost, gained, neither, plus the declared
tie and unresolved policies. Preserve unresolved-baseline and unresolved-arm
counts separately. Comparisons with random controls need jointly eligible paired
rows, not differences between medians from different subsets.

## What each check can say when the intervention is wrong

These are ten scoped check families/obligations, not a new programme-wide total.
"Demonstrated" below means rejection of synthetic corrupted evidence by the
actual source assertion. It does not certify native tensor execution, attention
extraction or a real tokenizer.

| ID | Checked obligation | Evidence and negative control | Independence status |
| --- | --- | --- | --- |
| G1 | Some requested blocked entry at P_act | Old empty-mask AST reproduced; actual nonempty assertion rejects | demonstrated, for emptiness only |
| G2 | Zero returned attention on all supplied blocked entries | Positive analytic masked mass rejects; nonnegative sums cannot cancel | demonstrated, for the supplied set only |
| G3 | Zero local far attention at P_act | Positive analytic far mass rejects | demonstrated; short contexts are vacuous; no full native-mask certificate |
| G4 | Open carrier key gets attention and remaining keys do not | Zero open mass and positive remaining masked mass each reject | demonstrated; first applicable carrier row, aggregate over layers |
| G5 | Current-note P_note readouts unchanged | Mutated selected margin causes actual equality branch to reject | demonstrated; selected readouts, not the complete state |
| G6 | Scientific note/query placement is correct | Five nonempty wrong masks accepted; independent expected-edge oracle rejects | current checks cannot certify placement |
| G7 | Reporter admits only valid gated arms | Four failed gates/missing gate/legacy field mixture still emit estimates | current reporter cannot certify admission |
| G8 | Reporter covers exact requested unique IDs | Missing, duplicate, additional and wrong-ID fixtures emit estimates | current reporter cannot certify completeness/identity |
| G9 | Token and position identities represent the intended context | Source asserts standalone/in-context first-token identity and positions; real tokenization unexecuted | unknown in this review |
| G10 | Capture and replay read identical corpus content | Digest branch is present; source also permits path equality when digest is absent; no fixture of this branch here | unknown; no full-content certificate from path equality |

**Ledger:** 5 demonstrated limited rejection families, 3 obligations not certified,
2 unexecuted/unknown. The proposed independent placement oracle is a review
fixture, not a newly installed production gate. The leaky carrier control does not
certify the current-note semantic boundary; unchanged P_note does not certify
anything about post-note routing.

The mask hook returns without intervention if no native `attention_mask` exists
(lines 85–86). A positive observed leak may catch that omission; naturally zero
weights need not. The checks also do not test unintended added masking or the
native future-token exclusion. Those are explicit limits to hook certification,
not demonstrated defects of Gemma's current native forward.

## Claim permitted after the corrections

For a jointly resolved baseline and masked arm, retained argmax would show only
that the specified edges were unnecessary for that **six-token conditional
winner at that expert-forced prefix**. It does not show an unchanged probability
distribution, preserved executable call, decision completed before the note, or
independence from all transcript routes. A changed probability is an intervention
effect on this readout, not by itself a causal share of the decision.

The current reporter computes decision-weighted descriptions, not confidence
intervals. That alone is not pseudoreplication. Unique episode counts are not
verified, so these `n` values must not later become episode-unit uncertainty without
checking clustering. No numerical tolerance or confidence convention is changed.

## Finding, technique and implementation

**Finding about the instrument:** the original empty cut is repaired in the cases
checked. A misplaced but nonempty cut remains observationally compatible with all
current mask assertions. The reporter can include invalid evidence and misname a
new winner as retained. These are not findings about a model's workspace.

**Transferable technique:** separate the intended intervention from the object
that applies it. Fix semantic positions independently; corrupt the applied object;
ask whether the checker distinguishes the two. Then corrupt stored evidence and
its enumeration before testing the reporter. A zero on the wrong edge set and a
correct average over the wrong eligible set can both be arithmetically exact.

**Implementation:** source-AST extraction avoids model loading; set-backed masks
exercise real builder slices; a two-layer causal graph witnesses an open relay;
the reporter executes against virtual files with all input rows and output fields
preserved. This establishes Python/control/reporting behaviour, not native tensor
or real-checkpoint behaviour.

## Reproduce

From this directory, `python3 -B check.py` verifies the seven source hashes and
regenerates `check.json`: 16 boundary fixtures, 7 placement cases, 5 direct negative
controls and 15 reporter fixtures. All sources and all deliberately invalid input
rows are retained. Focused lint is `ruff check --no-cache --select F,E9 check.py
reporter_checks.py`. See `verification.json` for exact execution basis and digests.

No scientific effect sizes were calculated. No model-time gate is claimed passed.
