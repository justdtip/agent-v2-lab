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
token, with the positive control (already-read suffix preferred in matched context, p < 1e-3)
passing. The lens readouts corroborate; a null in a residual-stream lens does not by itself
exclude state in the recurrent path, which the lens cannot observe directly. **Primary layers**
are the in-band ones (fractions 1/3 to 5/6 and their kind-matched partners, §3.5); fraction
1/6 and the final layer are reported, not decisive. **Multiplicity:** Holm across the layer
family per readout, as the P2 tables do.

| Outcome | Reading | Licenses |
| --- | --- | --- |
| Model output puts the true suffix's first digit at p > 0.5 without the note, and the matched-vs-mismatched sign test beats its null at any layer | World B on the hybrid: state held in the recurrent path | reconsider `keep_last`; add a "no pending list" ablation to D4's evaluation |
| Output near uniform over digits without the note; no primary layer beats its null after Holm; positive control (already-read suffix) preferred with p < 1e-3 in matched context | World A generalises | D4 unchanged; note discipline stays load-bearing |
| Positive control fails (already-read suffix not preferred) | instrument broken on the hybrid; no conclusion | fix before any reading |
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
   (`finite_difference` on the hybrid), float32 tail, native-dtype residual capture
   (`capture_dtype: native`, R18b) with the float32 deviation block copied into the artifact.
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
   median future window per context and per readout. Corpus size 16 contexts (the paper's floor
   is ten); (c) logit lens at the same layers as the no-Jacobian baseline.
3. **Statistic.** Per point, is the true suffix's first token more probable than a wrong,
   previously-seen one? Matched context vs the same pair scored against another task's
   context (the null). Exact two-sided sign test; report both columns per readout, as the 3B
   table does. Positive control: the already-read suffix in matched vs mismatched context.
4. **Single-decision spot check.** `agent-v2-jlens --model qwen35-4b --split test --task-index 7
   --step 3 --force-prefix "<note up to invoice-2->" --jvp-method finite_difference` (omit `--layers` to take the registry default)
   on the 3B's original probe task, for a like-for-like artifact beside
   `outputs/agent-v2/jlens-forced-suffix.json`.
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
