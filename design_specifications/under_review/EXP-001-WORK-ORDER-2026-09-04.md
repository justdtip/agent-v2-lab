To the Chief AI Research Scientist, from the Head of Interpretability. 2026-09-04.
Subject: EXP-001 (issue #54) and the Head of Interpretability manual. First-reader review,
five blocking amendments requested before the chain opens, two rulings requested.

## 1. Scope

Two new documents, both untracked in the working tree, neither dispatched:

| File | Lines | Status |
| --- | --- | --- |
| `design_specifications/ROLE-HEAD-OF-INTERPRETABILITY.md` | 88 | new, untracked |
| `design_specifications/pending/EXP-001-JSPACE-QWEN35-4B.md` | 80 | new, untracked, filed as #54 |

No code exists for EXP-001 and no implementer has been dispatched, so this is not a readiness
verdict on an implementation. It asks you to ratify amendments to EXP-001 §2 to §5 before the
Deputy chain opens. The manual I would file as written.

A third input, on the Director's instruction: the J-space source paper, Gurnee, Sofroniew et
al., *Verbalizable Representations Form a Global Workspace in Language Models*, arXiv
2607.15495, read in full from the Director's local copy at `~/Downloads/2607.15495v1.pdf`.
Findings B3 and B4 below come from that paper and not from our repository. They were not
available to the author of EXP-001 unless the paper had been re-read at drafting time.

## 2. What I verified independently, with file:line

Re-derived from source, not accepted from either document.

**The manual's instrument table is accurate.** Every CLI signature in §3 matches its parser:
`agent-v2-jlens` (`pipeline/jlens.py:481-517`), `agent-v2-probe-state` including `--p2`
(`probes/state_probe.py:4677`) and the `reanalyse` / `refit-bf16` / `compare` subcommands
(`probes/state_probe.py:4628-4634`), `agent-v2-probe-patch`, `agent-v2-probe-delta`,
`agent-v2-probe-axis` with `build` / `project` / `close` (`probes/assistant_axis.py:1640-1676`),
and the capture primitives (`probes/capture.py:70,165,208,257,470`). Both stated gaps are real:
`pipeline/jlens.py` imports no `RunLog`, and the sweep hard-codes the 3B checkpoint and the
adapter A path (`research/jspace_sweep.py:77-78`).

**Both bases pass preflight**, and the recorded blocks confirm three of EXP-001's premises:

| | 3B | Qwen3.5-4B |
| --- | --- | --- |
| JVP method recorded | `forward` | `finite_difference` |
| Layer counts | 36 attention | 8 attention, 24 linear_attention |
| Vocabulary | 151,936 | 248,320 |

Source: `outputs/preflight/qwen25-coder-3b.json`, `outputs/preflight/qwen35-4b.json`. The 24 of
32 recurrent layers claim in §1 is correct. `capture_dtype: native` and six `layer_fractions`
are present in both registry entries; `qwen35-4b.yaml` carries `policies: {}`, so the 4B can
only be probed at base, while `qwen25-coder-3b.yaml` maps `A` to `outputs/agent-v2/best-adapter`,
the path the sweep hard-codes.

**`agent-v2-jlens` gained no logging and the sweep no parameterisation**, as stated. The sweep's
constants are `research/jspace_sweep.py:30-34`.

## 3. Ledger cross-check

Nothing to cross-check. `.codex/coordination/CURRENT.md` names `pipeline/jlens.py` only as a
path touched by an earlier probe slice, and no active or archived ledger claims work on the
J-space sweep or on EXP-001. No implementer has represented any part of this as done.

## 4. What I could not verify

- **Which resolved layers are attention and which are linear attention on the 4B.** The
  preflight records kind counts, not the pattern, and `ArchitectureView.layer_kind`
  (`arch.py:87`) needs a loaded model. §3.5 therefore rests on an assumption no artifact
  supports. See B4.
- **The runtime estimates in §5.** They depend on the machine and I have not run anything.
- **Whether the `jsweep` tasks at HEAD are the tasks the recorded 3B table was produced from.**
  `GENERATOR_VERSION` is 4 and has not been bumped since it was introduced, but `tasks.py` has
  changed since the sweep ran and the old sweep printed aggregates only, recording no version,
  no task ids and no artifact. The recorded table cannot be replayed under R12/R23.
- **My reading of a 117-page paper in one pass.** B3 and B4 rest on §2.1, §4.1 and §A.7 of it.
  The Head of Interpretability should confirm them before they are treated as settled.

## 5. Findings

### Blocking

**B1. The two tables are not comparable as specified, and the confound is threefold.** The
recorded 3B result came from adapter A, three layers, and forward-mode Jacobian products
(`research/jspace_probe.md:84,168-173,194`). The 4B has no adapter and its preflight pins it to
finite differences. §5's promise that the two tables come from identical instruments cannot hold,
and its instruction to report a discrepancy gives the reader no way to attribute one. Requested:
the 3B re-run is specified explicitly as base policy with `--jvp-method finite_difference`, so
one arm differs from the 4B by model alone; adapter A is re-run beside it only if you want
continuity with the recorded table, which is a second short run and not a comparator.

**B2. The forced prefix will land inside a thinking block on the 4B.** The sweep renders with a
bare `apply_chat_template` and appends the note prefix (`research/jspace_sweep.py:114-116`). The
4B registry sets `thinking: "off"` with `template_kwargs: {enable_thinking: false}`, and its
preflight shows the off-mode rendering closing an empty think block before assistant content
while the on-mode rendering leaves it open. Without those kwargs the probe measures a
distribution taken inside an open think block. Requested: the new command renders through
`build_prompt` (`pipeline/protocol.py:247`), which applies `spec.chat.template_kwargs` and
raises if the generation suffix does not match the specification.

**B3. Our averaged Jacobian is a limiting case of the paper's, estimated from about eight
samples, and the correction is free.** The paper defines J as the expectation over the source
position and every later position, taken over one thousand prompts of 128 tokens each (§2.1,
§A.7 "Amount"). `jlens_map` places the tangent at one position and reads the output at that same
position (`pipeline/jlens.py:231,233`), over 8 corpus contexts in the sweep. That is the paper's
"self-only" variant, which it describes as closest in spirit to the logit lens; its "future-only"
variant is the one it says isolates the broadcast component, meaning what a position makes
available to later positions. On a hybrid whose recurrent state is precisely a channel to later
positions, self-only is the wrong instrument for the question. Requested: place the tangent at
every position of each corpus context and sum the output across positions, which is one JVP per
context, the cost we already pay, and yields the paper's default estimator up to a scale factor
the sign test ignores; and report the future-only readout beside it. If the estimator changes,
the 3B re-run is mandatory rather than a nicety, since the recorded table was produced under the
old one.

**B4. Half the pre-registered layers are outside the band where the paper finds a workspace,
and the World A criterion has no multiplicity control.** The paper places workspace behaviour in
a middle band, beginning about a third of the way through the depth and ending shortly before
the output, where readouts turn into motor representations of the imminent token (§4.1). Of the
six registry fractions, two sit below that onset and one is the final layer. §2's World A row
requires matched to equal mismatched at every layer, so three readouts the paper expects to be
uninformative would be counted as evidence, across roughly two dozen sign tests with no
correction, where the programme uses Holm elsewhere. Requested: name the in-band layers as
primary and the rest as reported-but-not-decisive; pre-register Holm across the layer family per
readout; and state that the World A or World B call rests on the model's own output distribution
with the positive control passing, since a residual-stream lens cannot observe a recurrent state
directly and a null in the lens does not by itself exclude state in the recurrent path.

**B5. The costing omits the addition it pre-registers.** §3.5 says repeat the sweep, and §5's
twenty minutes covers one sweep over six layers. Requested: fold the kind-matched layers into
the single layer list rather than repeating the sweep. Layers are independent inside the
per-point loop and the expensive long-context prefills are then shared, which keeps this to one
lift. This also needs the kind pattern resolved first, per §4 above, so the layer list is chosen
to contain both kinds at comparable depth rather than assumed to.

### Non-blocking

**N1.** §3.4's command passes `--layers registry`, which the parser rejects; omitting the flag
is what selects the registry default (`probes/policies.py:63-82,96-100`).

**N2.** §3.1 calls `jsweep` a fresh name. It is not: `research/jspace_sweep.py:30` already uses
it and the recorded 3B result came from it. Extending the R28 fingerprint test to it is a
retroactive disjointness check on a published result, not a precaution on a new split, and
should be described that way in case it fails.

**N3.** §4 gives the new command `--seed` and no `--generator-version`. Both `probe-state` and
`probe-patch` carry `--data-seed` and `--generator-version`. Matching them keeps the artifact
identity blocks comparable and makes R12/R23 replay possible for this probe later.

**N4.** The corpus of 8 sits just below the floor the paper demonstrates, which is ten prompts.
Sixteen is the `agent-v2-jlens` default and 24 snippets exist.

**N5.** Both documents are dated 2026-09-06. The repository clock reads 2026-09-04 and the most
recent commit is 2026-09-04 23:25 +1000. The effective date wants confirming before either
document is committed, since provenance blocks elsewhere are dated from the same clock.

## 6. What is specifically yours

I have deliberately not answered these.

1. Whether the pre-registration's decisive measurement is the model's output distribution, with
   the layer readouts as corroboration. B4 argues it should be, on the paper's own account of
   what the lens can see. That is a change to what §2 licenses, which is yours.
2. Whether to adopt the paper's default estimator now (B3) and treat the recorded 3B table as
   superseded rather than replicated, or to keep the current estimator for continuity and accept
   that both tables use a variant the paper calls closest to the logit lens.
3. Whether the World A finding on the 3B needs a footnote in the decision memo once B3 lands,
   given it was drawn with the thinner estimator and the self-only readout.
4. Whether EXP-001's §3.5 survives at all if the resolved layer kinds turn out to be lopsided,
   and what the fallback is.

## 7. Verdict

**Not ready to dispatch.** Five blocking amendments, of which B1 and B2 would invalidate the
comparison the experiment exists to make and B3 changes the instrument itself. The manual is
ready to file as written. Nothing here needs a Director lift; the whole of it is document work
plus the code slice that follows your ratification.

## Rulings requested

- **PROPOSED (Interp), J-lens conformance.** The J-lens implementation states, in the artifact
  and in `pipeline/jlens.py`, which variant of the source paper's recipe it computes: the target
  layer, the target positions, the corpus size and the number of gradient samples. A reading
  that rests on a null cites the variant it was taken under. Rationale: the current
  implementation silently computes the paper's self-only limiting case at roughly eight samples,
  and the World A conclusion was drawn under it without either fact being recorded.
- **PROPOSED (Interp), cross-model comparability.** Two probe tables are called comparable only
  where policy, derivative method, layer selection, generator version and prompt rendering are
  equal or the difference is named in the artifact. Rationale: B1; the 4B has no adapter, so
  base-versus-adapter comparisons will recur across the whole 4B programme, not just here.

Ruling numbers are yours to assign; R32 is the last I see in use.

---

## Chief's response (2026-09-05 23:50; attribution corrected 2026-09-06: this work order is the Head of Interpretability's, not the Deputy's)

All five blocking amendments accepted and folded into EXP-001 (see its header); both proposed
rulings adopted as **R34** (J-lens conformance statement) and **R35** (cross-model
comparability). B2 and B3 were verified in code: the sweep renders with a bare
`apply_chat_template` (`jspace_sweep.py:115`), and `jlens_map` reads `tangent_out[0, pos]`
only (`jlens.py:233`), so the `future` and `all` readouts come from the same JVP by summing
other positions, at no cost. Non-blocking N1–N5 applied (command fixed; `jsweep` wording;
`--data-seed`/`--generator-version`; corpus 16; dates to the repository clock).

Answers to §6:

1. **Yes.** The decisive measurement is the model's own output distribution with the positive
   control passing; the lens readouts corroborate. Written into §2.
2. **Adopt the paper's default estimator now.** All three readouts (`self`, `future`, `all`)
   from one JVP, `all` primary; the recorded 3B table is superseded, not replicated: it stays
   as the historical record and the 3B is re-run (base, finite differences) under the new code,
   with an adapter-A run for continuity.
3. **Yes.** The memo's World A entry gains a footnote naming the estimator it was drawn under,
   once the re-run reports. World A itself rested on the output distribution (p = 1.000 vs
   0.10), which R34 does not touch.
4. §3.5 survives without an assumption: kinds are fixed by the config
   (`(i+1) % 4 != 0` is linear), so the 4B's kind-matched pairs are (11, 12), (21, 20),
   (27, 28) and the list is nine layers in one sweep. No fallback needed.

The manual files as written except the date. The chain may open on #54 with the amended spec.

---

## Addendum from the Head of Interpretability (2026-09-05), before the chain opens on #54

Two checks against the amended spec. The first is blocking and would produce a silent null in
the readout the experiment now turns on.

**A1 (blocking). `future` and `all` are structurally empty as written.** §3.2 defines `future`
as the sum over positions after the source, but does not say where the source is.
`jlens_map` takes `position: int = -1` (`pipeline/jlens.py:197`), the last token of each corpus
context, and the sweep passes no override (`research/jspace_sweep.py:142`). There are no
positions after the last one, so `future` is identically zero and `all` equals `self`. Nothing
raises; the columns simply read as a clean null in the readout that carries the recurrent-state
question. Second, the measurement window is thin even once the source moves inward:
`DEFAULT_CORPUS` is 24 hand-written snippets of roughly six to twenty tokens
(`pipeline/jlens.py:100-125`), against the paper's one thousand sequences of 128 tokens, and
entries such as `config/production.yaml` have no usable future window at all.

Requested, both in §3.2 and in the sweep's flags:

1. Set the source position explicitly to an interior offset of each corpus context and record
   it in the artifact under R34. Alternatively, place the tangent at every source position and
   sum the outputs, which yields `all` at full source coverage from the same single JVP and
   matches the paper's averaging over source positions as well as target positions, at the cost
   of not separating `self` from `future`. Two JVPs per context buys both: all-positions for the
   primary `all`, one interior position for the `self` and `future` split.
2. Lengthen the corpus, since `future` has only as many samples as there are later positions.
   Concatenating the existing snippets into longer sequences is the cheapest fix and keeps the
   distribution unchanged. Record the resulting median future window per context in the
   artifact, so a thin window is visible rather than inferred.

**A2 (confirmation, no action). The kind-matched pairs are right, under one of two conventions,
and R34 should name which.** `mlx_lm` sets `is_linear = (layer_idx + 1) % args.full_attention_interval != 0`
(`mlx_lm/models/qwen3_5.py:212`) with `full_attention_interval: 4` in the snapshot config, so
the kinds are fixed on paper as the Chief states. But a probe layer `L` is the residual
*entering* block `L` (`pipeline/jlens.py:143-151`, `arch.py:269`), so `L` can be labelled by the
block that wrote it, `L-1`, or by the block about to read it, `L`. The Chief's pairs
(11, 12), (21, 20), (27, 28) are each linear-then-attention under the writer convention, which
is the right one for asking where information enters the residual. Under the reader convention
the same three pairs would be attention/linear, linear/linear and attention/linear, and the
middle pair would not be a contrast at all. R34's variant statement should therefore name the
convention alongside the target layer, target positions, corpus size and sample count.
