# Gemma fitting comparison: preparation, dependencies and unresolved registration

Status: preparation only, before any new model-derived result. This is an executable-work checklist
and proposed comparison design, not a claim that a run passed or a frozen scientific registration.
Order remains: all-layer regression/hosted comparison, above-window reference refit, prose/corpus
cross-read. Scope composition is deferred until the map selects layers.

## First: fit and compare

Use the official instruction-tuned 4B checkpoint converted to BF16 by the Chief, with conversion
provenance. Fit every nonfinal block-output residual (Gemma repo layers 1..33); final layer 34 is
identity. The pilot uses the separately recorded official 4-bit conversion. Every displayed
comparison discloses that precision mismatch. No Qwen finite-difference coefficient is inherited.

Before launch, consume the landed registry, correct generation prefix/operative observation-role
rendering, architecture port and model/lens identity guard. Require native residual conformance
above 1,024 tokens and the mask-dispatch negative control. Use the released model-window wrapper
and exact model identity; the previous Qwen concurrent-load exception is not a Gemma launch policy.

For the initial comparison, a 128-token prose regression set from the already-authorized validation
text would match the hosted lens's fit length. It would not match its train corpus or fitting
objective. Freeze corpus, fit/selection membership, all-layer readout rules, numerical acceptance
bounds for the independent regression solver and resource limits before loading. Do not use the
hosted-versus-regression matrix difference as a pass/fail threshold. A 1,024-token legacy-default
comparison is possible but adds fit-length mismatch, which must be named. These alternatives are
not selected by inspecting fitted results.

Correctness evidence: same-objective independent ridge solve, direct residual error, nonsymmetric
map orientation and exact corpus/replay identity. Comparison evidence: all-layer profiles of the
existing agreement and foreknowledge summaries on fixed independent pilot records, with each
record's own final-layer base rate and the regression h=1 qualification. No omitted layer based
on a disappointing result. Distinguish errors in the observation, fitting, serialization and
readout from a valid difference between instruments. All rank summaries use the same token IDs.

## Second: the above-window reference refit

Use a pinned upstream reference implementation on RunPod and our own corpus. Freeze the exact
estimator (including source and target position masks and averaging), checkpoint, numerical
precision, real token-length distribution and convergence rule. Record positions beyond the
window; report which parts of the fit still have shorter available history. A 2,048-token corpus
is a concrete candidate, now representable by the writer/reader; no such corpus or job is frozen
by this code change. Do not carry the hosted command's 2,000-character truncation into a long-token
fit: that could erase the intended regime before tokenization.

The few-dollar estimate is not a measured budget. Before paid execution, obtain an actual GPU rate,
calibrate a bounded prompt/batch, project cost, and record an explicit spend ceiling and teardown.
Preserve corpus and checkpoint licensing/access boundaries. No cloud job was created here.

For an agentic fit, use Gemma-generated fitting trajectories disjoint from the fifteen pilot
episodes. The surviving Qwen trajectories are historical evidence and cannot be relabeled as
Gemma material. A prose first-stage comparison can use the existing authorized validation shard
without waiting for agentic rollouts or downloading another dataset.

## Third: re-tokenize and regenerate prose records

The read-only tokenizer preview counts 257,291 Gemma tokens in the frozen validation text. With
the existing every-fifth-window rule: 128 tokens gives 1,608 fit / 402 held; 1,024 gives 201 / 50;
2,048 gives 100 / 25. These are preview counts, not finalized corpus identities. The old count of
51 held windows was a Qwen tokenization outcome and must not be silently promised for Gemma.

Before prose capture, explicitly adapt requirements section 14 to the selected Gemma manifest:
which held windows, the authored-prefix length and the same 200-token greedy cap, EOS handling
and under-32 exclusion. If that capture supports a window contrast, its actual prompt history
must exceed the window. An 824-token prompt and at most 200 emitted tokens never exceeds 1,024.
Retain the original two summaries, emitted-token foreknowledge and authored-token agreement;
do not score authored future text as emitted-token foreknowledge. Preserve the exact no-reuse
forward ledger and hash-chained records. Selection-set reuse for ridge choice remains disclosed.

Initial system and all twelve selected agentic task prompts have zero leading/trailing whitespace.
That does not clear later observations, canonical assistant messages, chat prompts or raw emitted
text. Audit those at render time, retaining semantic span identity when an observation is rendered
as a user turn. Never strip raw emitted text merely to make a substring locator pass.

## Completion criteria

A code patch and green tests complete the preparation slice only. The research deliverable requires
(1) model findings with uncertainty and exact evidence, (2) the stack-independent elicitation method,
and (3) the implementation/provenance needed to reproduce this instance. The first Gemma fit and
comparison remain unrun. A repeat on a second model is part of testing transfer, not optional polish.
