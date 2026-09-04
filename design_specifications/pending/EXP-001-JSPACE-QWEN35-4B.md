# EXP-001: Does the Qwen3.5-4B base hold hidden task state internally? (J-space probe on a hybrid)

> Read first: `01-IMPLEMENTER-BRIEFING.md`, `02-INTERFACE-AND-WIRING-MAP.md` (§2.14, §7 R12,
> R18a/b, R26, R31), `research/jspace_probe.md`. Owner: Head of Interpretability. Author of
> record: the Chief. Status: pending; amended 2026-09-05 after the Head of Interpretability's first-reader review
> (`under_review/EXP-001-WORK-ORDER-2026-09-04.md`, B1–B5, N1–N5). Rulings R34, R35 apply.

## 1. Question and why it matters now

On the dense 3B the answer was World A: after the directory listing leaves the two-observation
window, the model does not hold the third invoice's random suffix; the note creates that state
(`research/jspace_probe.md`, 42-point paired sweep). The Qwen3.5-4B carries a recurrent
Gated DeltaNet state in 24 of its 32 layers, a memory that attention windows do not erase.
If the 4B holds the hidden filename, the note discipline for this base can be lighter and the
windowing is a crutch; if it does not, World A generalises across architectures and the D4
recipe stands as designed. Either answer changes what D4's notes must carry; that is why this
runs before D4 trains.

## 2. Pre-registered predictions

**Decisive measurement:** the model's own next-token distribution over the suffix's first
token, with the positive control passing. **Control wording, amended after run 2 (Head of
Interpretability, 2026-09-05):** the control is that the readout **responds** to the already-read
suffix, which is in the matched context, at p < 1e-3 — the *magnitude* of the response, not its
direction. The original wording said "preferred", and that does not transfer. On the 3B the
in-context filename is preferred (88%, p = 4.4e-7); on the 4B it is suppressed (p = 0.0009 at the
output), and on the 4B the sign flips with depth, amplified near layer 20 and suppressed from 27
to the output. A directional control would have failed on the 4B for a model difference that has
nothing to do with hidden state. **The null is the mismatched context, not a fair coin**
(Chief, 2026-09-05), so every comparison of matched against mismatched is paired per case; the
artifact's `matched_p` against a coin is reported but does not govern. The lens readouts corroborate; a null in a residual-stream lens does not by itself
exclude state in the recurrent path, which the lens cannot observe directly. **Primary layers**
are the in-band ones (fractions 1/3 to 5/6 and their kind-matched partners, §3.5); fraction
1/6 and the final layer are reported, not decisive. **Multiplicity:** Holm across the layer
family per readout, as the P2 tables do.

| Outcome | Reading | Licenses |
| --- | --- | --- |
| Model output puts the true suffix's first digit at p > 0.5 without the note, and the matched-vs-mismatched sign test beats its null at any layer | World B on the hybrid: state held in the recurrent path | reconsider `keep_last`; add a "no pending list" ablation to D4's evaluation |
| Output near uniform over digits without the note; no primary layer beats its null after Holm on the paired test; positive control responds at p < 1e-3 in matched context | World A generalises | D4 unchanged; note discipline stays load-bearing |
| Positive control fails (the already-read suffix's probability does not respond to its own context, in either direction) | instrument broken on the hybrid; no conclusion | fix before any reading |
| Partial (late layers only, or weak preference) | degraded trace; report as such | no design change; extend with word-token discriminators (batch_update modes) |

## 3. Method (the 3B design, unchanged where it can be)

1. **Probe points.** `ledger_reconcile`, split `jsweep` (the name the recorded 3B sweep already used, so extending
   the R28 fingerprint test to it is a retroactive disjointness check on a published result and
   is reported as such if it fails), 720 tasks → the same selection rule as the sweep:
   step 3, the first read after the listing has left the window, `pending:` lists stripped
   (`strip_pending`), context rendered through `protocol.build_prompt` with the registry's
   `template_kwargs` (B2: thinking off, empty think block closed; the generation-suffix assertion
   applies), then forced to end mid-note at `Reading invoice-2-`, leakage guard
   (cases where the suffix is visible anywhere in context are dropped), pairs whose first
   suffix tokens coincide are skipped. Expect ~42 usable points, as before.
2. **Readouts.** (a) The model's own next-token distribution (the direct measurement);
   (b) J-lens at the layer list of §3.5, JVP by the method the preflight recorded
   (`finite_difference` on the hybrid), float32 tail, and the residual capture described below,
   with the float32 deviation block copied into the artifact.
   **Capture dtype, corrected against run 1 (Head of Interpretability, 2026-09-05).** This step
   previously claimed native-dtype residual capture. It is not achieved and cannot be through
   this view. The registry requests `capture_dtype: native` (R18b) and that request is recorded,
   but `ArchitectureView` casts every block output to float32 in `embed`, `run_block` and
   `final_norm`, so the effective capture is **float32** and the artifact records
   `view_supports_dtype: false`. Run 1's conformance block states exactly this, which is R34
   working. Two consequences the reading must carry rather than the spec conceal. First, the
   decisive measurement of §2, the model's own next-token distribution, is computed through a
   forward pass the deployment never runs: the preflight's `residual_equivalence`
   `fp32_manual_vs_native` block puts that at Frobenius relative 0.0042 on the 3B (native
   float16) and 0.0285 on the 4B (native bfloat16), the gap being about sevenfold because
   bfloat16's epsilon is eight times float16's. Neither gates, per R18a. Second, this is a
   **model-dependent** difference and therefore an R35 axis: `fp32_manual_vs_native` must be
   populated in every artifact's comparability block from the preflight record, since two
   artifacts each carrying `null` compare as equal on an axis where they differ sevenfold. The
   bound on what the float32 path could hide is the positive control, which on the 3B still
   detected an in-context filename at p = 4.4e-7 through that same path; each reading states the
   figure and the control together.
   **Estimator (B3):** from the one JVP per context, three readouts are taken at no extra cost
   and all three are reported: `self` (output tangent at the source position, the current
   implementation and the paper's self-only limiting case), `future` (sum over positions after
   the source; the paper's broadcast component, the one that matters on a recurrent
   architecture), and `all` (self plus future; the paper's default up to a scale the sign test
   ignores). `all` is primary. **Source position and window (Head of Interpretability's addendum,
   2026-09-05):** the mapping function defaults the source to the last token, where the future
   window is empty and `future` would read as a structural zero; the sweep therefore passes
   explicit sources. Corpus contexts are built by concatenating the 24 snippets into sequences
   of at least 128 tokens (`--corpus-length 128`, distribution unchanged); sources are placed at
   fractions {0.25, 0.5, 0.75} of each context (`--source-positions`); `future` sums the output
   tangent over positions after the source, `self` reads the source, `all` is their sum. A
   readout whose window is empty **raises**, never returns zero; the artifact records the
   median future window per context and per readout.
   **Reduction, condition C2 (Head of Interpretability's ruling, 2026-09-05, issue #61):**
   both axes are named here because the bare word "sum" is ambiguous about which one it
   describes. *Within* one sample the reduction over target positions is a **sum**: `future`
   sums the output tangent over the positions after the source (`pipeline/jlens.py:825-827`)
   and `all` adds the source's own tangent to it. *Across* samples the accumulated vectors are
   divided by the number used (`pipeline/jlens.py:859`), so the map is a **mean** over sources
   and prompts. That is what the paper does — it differentiates a sum over target positions,
   then averages over sources and prompts — and it is what the implementation already does.
   Ruled: keep it, no code change. Both axes are to be named in the R34 conformance block of
   every artifact, since an artifact that says only "sum" does not pin the readout.
   **Consequence, stated knowingly (Chief’s addendum, 2026-09-05):** summing within a sample
   while the sources sit at different depths means the windows are unequal — at
   `--corpus-length 128` the three default sources leave 96, 64 and 32 positions — so the
   earliest source sums about three times as many terms as the latest, which bounds its weight
   in the cross-sample mean of `future` and `all` from above. The realised weight depends on how
   the output tangent's magnitude falls with distance from the source, which this run does not
   measure (clause narrowed by the Head of Interpretability, 2026-09-05: the threefold figure is
   a count of summed terms, not a measured ratio of influence).
   The estimate is a window-weighted mean by construction. The choice
   stands; the record says so rather than leaving it to be discovered from the artifact.
   Corpus size 16 contexts (the paper's floor
   is ten); (c) logit lens at the same layers as the no-Jacobian baseline.
3. **Statistic.** **Required decomposition (Head of Interpretability, ratified by the Chief,
   2026-09-05).** An ordering statistic compares two probabilities and cannot say which one
   moved. For each readout: P(true suffix) matched against mismatched, and P(already-read
   suffix) matched against mismatched, each as a paired test over the same cases. **The artifact
   computes it for every readout, unconditionally** (Chief, 2026-09-05, #72): emitting it only
   where the ordering test reaches significance would make the block's presence a selection, and
   a decomposition at a non-significant readout is evidence too. Run 2 is its own proof — the
   model's own output does **not** reach paired significance on the ordering (p = 0.180), and its
   decomposition is the centrepiece of the reading. A conditional rule would have suppressed the
   single most important block in this experiment. The reading then reports the decompositions
   that bear on the verdict, and the markdown renderer may show only those; the JSON carries all.
   This is what distinguishes a trace of the hidden filename from a response to the visible one,
   and on run 2 it was the whole result: at five readouts spanning both block kinds and the
   workspace band, P(true) never moved (p = 0.644 to 0.878) while P(already-read) always did
   (p = 0.0001 to 0.0029). Without it, `logit_lens_L27` at paired p = 0.0005 reads as a
   late-layer partial trace, which is row four of §2, and it is not one.
   Per point, is the true suffix's first token more probable than a wrong,
   previously-seen one? Matched context vs the same pair scored against another task's
   context (the null). Exact two-sided sign test; report both columns per readout, as the 3B
   table does. Positive control: the already-read suffix in matched vs mismatched context.
4. **Single-decision spot check.** `agent-v2-jlens --model qwen35-4b --split test --task-index 7
   --step 3 --force-prefix 'Invoices read: 2 of 6. approved: 178; held (skip): 38. Reading
   invoice-2-' --jvp-method finite_difference` (omit `--layers` to take the registry default)
   on the 3B's original probe task, for a like-for-like artifact beside
   `outputs/agent-v2/jlens-forced-suffix.json`.
   **Prefix provenance (Head of Interpretability, 2026-09-05).** The string above replaces the
   placeholder this step carried until now, which could not be run. It was **derived from the
   generator**, not recovered from the original artifact: `make_tasks('test', ...)` for task
   `test-ledger_reconcile-0007-clean`, whose step 3 reads `invoice-2-537.txt`, with the expert
   note truncated immediately before the random suffix. No model was loaded to obtain it. The
   original artifact records model, adapter, split, task id, step, layers, corpus size,
   candidates and records, but **no `force_prefix` field**, so the string it was actually run
   under is not recoverable from it; the derivation above is the reconstruction, and this spec
   is its record. The same lookup confirms `--task-index 7` is the right index for split `test`.
   **Condition (Chief, 2026-09-05).** The spot check runs **unstripped**, like-for-like with the
   original: it is a copying measurement that reproduces the recorded probe's condition on the
   new base, and the memory test is the sweep, which strips the `pending:` lists. Note that
   `agent-v2-jlens` does not record the stripping condition in its artifact, so this clause is
   the record for that run.
5. **Layer list, one sweep (B5).** Layer kinds are fixed by the config (`is_linear = (i+1) %
   full_attention_interval != 0`): on the 4B the residual after an attention block sits at
   layers 4, 8, …, 32, everything else follows a linear-attention block. The registry fractions
   resolve to 5, 11, 16, 21, 27, 32; the list adds the kind-matched partners 12, 20, 28, giving
   nine layers in a single sweep (prefills shared; layers independent inside the per-point
   loop). Kind-matched pairs at comparable depth: (11 linear, 12 attention), (21, 20), (27, 28);
   16 and 32 are attention outputs. A recurrent-path trace shows as a kind difference within a
   pair. Convention, stated in every artifact (R34): layer L is the residual after block L-1,
   and a probe layer's kind is the kind of the block that wrote it; under the other reading of
   the same index the middle pair would be linear against linear. On the 3B every layer is
   attention; the same fractions apply without partners.

## 4. Logging and artifacts (R26, mandatory)

- `research/jspace_sweep.py` becomes `agent-v2-jspace-sweep` (entry in `pyproject.toml`; the
  research script keeps a thin wrapper) with `--model`, `--policy` (default base),
  `--layers` (fractions or indices), `--corpus-size` (default 16), `--corpus-length` (default
  128), `--source-positions` (default 0.25,0.5,0.75), `--count`, `--data-seed`,
  `--generator-version`, `--jvp-method`, `--output` (N3: the same identity fields as
  `probe-state` and `probe-patch`, so R12/R23 replay is possible later).
- Opens `RunLog` with identity = registry name, hf_id, policy/adapter, layers, corpus size,
  data seed, generator version (HEAD; recorded), git commit; one progress line per probe
  point (R26 g); `run.log` and `events.jsonl` beside the artifact.
- Writes `outputs/probes/jspace-<model>-<date>/sweep.json` (every point: task id, candidates,
  per-readout probabilities matched and mismatched, layer kinds) and `sweep.md` (the table in
  the form of `research/jspace_probe.md`), plus the single-decision JSON from step 3.4. Every
  artifact records `capture_dtype`, the preflight float32 block, and input hashes.
- `agent-v2-jlens` gains the same `RunLog` and identity (it has none today).

## 5. Prerequisites and gates

- Code: the sweep parameterisation, RunLog on both CLIs, JSON output, the `jsweep` split in
  the R28 fingerprint test, the per-kind readout; fake-only tests (R31 for the JVP seam).
- Preflight for `qwen35-4b` passed under R18a (it has); probe hold lifted by its terms.
- **What the preflight's `jvp` block should read, and what a change in it would mean (Head of
  Interpretability, 2026-09-05).** The probe layer is pinned at `view.num_layers // 2`
  (`pipeline/preflight.py`, `_jvp_result`), so it is 16 on the 4B and 18 on the 3B, matching
  both artifacts on disk. Forward mode fails on the 4B because the tail from layer 16 spans
  blocks 16 to 31 and contains linear-attention blocks whichever layer it starts from, so the
  fallback to `finite_difference` is a property of the operator and not of the probe layer's own
  kind. The preflight calibration slice touches the JVP path in one line only, adding the
  training-footprint conjunct to `passed`. **Therefore `finite_difference` is the expected
  reading of a regenerated 4B artifact.** If it reads `forward`, the cause lies outside that
  slice, in the library's autodiff support or in how the tail is built, and it is a red flag to
  investigate before any run rather than a benign side effect. Resolution if the change proves
  benign: pin the 4B to `finite_difference` too and record in the conformance block that the
  method was chosen for comparability under R35 rather than taken from the preflight. Finite
  differences stay valid wherever forward works, since two evaluations of the tail is a weaker
  requirement than autodiff, so pinning costs only the second tail pass. Matching arms is the
  load-bearing requirement; §3.2's "the method the preflight recorded" exists to keep the run on
  a method the model supports, not to let the method float between arms.
- Director lifts: the 3B base re-run and the adapter-A continuity run (~10 minutes each), the
  4B sweep (about 30 minutes: ~42 points × 9 layers × corpus 16, finite-difference JVPs at two
  tail passes each, prefills shared), and the single-decision check (minutes).
- **Comparability (B1, R35).** The 3B is re-run under the new code as **base policy with
  `--jvp-method finite_difference`**, so that arm differs from the 4B by model alone; a second
  short run with adapter A under the same code gives continuity with the recorded table. The
  recorded 3B table (adapter A, three layers, forward JVP, self-only readout, corpus 8, no
  version recorded) is **superseded, not replicated**: it stays on file as the historical
  record, and the decision memo's World A entry gains a footnote naming the estimator it was
  drawn under once the re-run reports.

## 6. Acceptance

The pre-registered table for both bases, with the positive control passing on both; the
reading written by the Head of Interpretability against §2 and ratified by the Chief; the
decision memo gains an EXP-001 entry; the D4 lift request cites the result.
