# Confound register — the Gemma stage-one run, the stage-two map, and the regression fit

**2026-09-08. Chief, on the Director's instruction: identify the confounds in what we have just run
and are about to run, then address them.** Seven surfaces were audited independently, each claim
adversarially checked, against code and committed artefacts rather than against issue text.

The register is ordered by consequence, and it separates two things that are easy to conflate: a
confound that touches the **representation reading**, which is the map's actual subject, and one
that touches only the **pass-rate scoreboard**. Several of today's loudest findings turn out to be
the second kind.

---

## The headline: the harness manufactured the failure we were explaining

**`list_files` answers an unsatisfiable directory with silence, not an error.** From
`src/local_llm_lab/pipeline/env.py:211-214`, the directory is `rstrip("/")`-ed and matched with
`startswith(directory + "/")`. Workspace paths carry no leading slash, so any rooted guess matches
nothing and the model is told the workspace is empty. Reproduced on the real task:

| call | result |
|---|---|
| `list_files("/")` | `FILES: (none)` — **false empty** |
| `list_files(".")` | `FILES: (none)` — **false empty** |
| `list_files("./")` | `FILES: (none)` — **false empty** |
| `list_files("/workspace")` | `FILES: (none)` — **false empty** |
| `list_files("")` | `ERROR: directory must be a non-empty string` |
| `list_files("workspace")` | `FILES: workspace/test/0028/config.ini` |
| `read_file("/test/0028/config.ini")` | `ERROR: file not found: /test/0028/config.ini` |

`read_file` errors honestly. `list_files` does not. **There is also no discoverable root**: no tool
lists the workspace top level, so a model must guess the string `workspace` exactly, and every near
miss is answered with a confident, false "empty".

### The exchange, verbatim, from the model's own context at turn 2

Read the system prompt's third line first, because it is the part that stings.

> **[system]** ... Rules:
> - Inspect state with tools instead of guessing paths, file names, or values.
>
> **[user]** In `workspace/test/0028/config.ini`, change mode from fast to audit. Inspect the file
> first, re-read it to verify, and report it as `mode=<target>`.
>
> **[assistant]** Progress note: Need to inspect the configuration file to see its current mode.
> `{"name": "list_files", "arguments": {"directory": "/"}}`
>
> **[tool]** `FILES: (none)`
>
> **[assistant]** Progress note: The file list is empty, so there is no configuration file to
> inspect. I need to create a configuration file to test.
> `{"name": "read_file", "arguments": {"path": "/test/0028/config.ini"}}`
>
> **[tool]** `ERROR: file not found: /test/0028/config.ini`

**We told the model to inspect rather than guess. It inspected. The instrument lied. We then
recorded the outcome as the model's failure.** Every subsequent turn is sound reasoning from a
false observation that our environment supplied.

**This also explains the dropped path component, which had no explanation before.** `list_files("/")`
returned without an error, which tells the model that `/` is a real directory in this workspace. The
prompt's `workspace/` then reads naturally as the name of the sandbox rather than as a path
component, so the file becomes `/test/0028/config.ini`. That is a coherent inference from the
evidence available, and our environment supplied the evidence. It is why the token `workspace` is
absent from every read layer's top-10 at every fork: the model was not failing to recall the string,
it had already ruled the string out as a directory name, correctly, on our authority.

This violates a rule already written in our own wiring map, R56(f). It is a cheap fix: raise on a
directory that cannot match, exactly as `read_file` does.

**Consequence for the record.** The note/call divergence recorded this morning is not confabulation
and must not be written as one. The D-CRO reached this independently from the transcripts and
amended their record at `9c2fdee`; a separate auditor reached it from the simulator. Two seats,
separate routes, same answer.

---

## What the audit refuted, including two things I said today

A register that only grows is not a register.

**Refuted: "Gemma does not copy literal strings."** This was my framing this morning and the
numbers kill it. Per-call path fidelity is **indistinguishable between the two models**:

| model | path arguments resolved | rate |
|---|---|---|
| Gemma 3 4B | 54 / 67 | 0.806 |
| Qwen base | 1,127 / 1,400 | 0.805 |

What differs is the **shape** of the errors, not their rate: 8 of Gemma's 13 misses carry a leading
slash, against 0 of Qwen's 273. And the reach measurement stands — 135 of 180 prompts carry a
literal path, 30 a literal key, 165 either, `calculate` the only clean family at 15, reproduced
exactly by two independent counts — but reach is not failure rate. Outside `update-0028` the Gemma
corpus shows 2 path errors in 50 arguments, by two different mechanisms, neither the leading-slash
one. **Every leading slash in the entire Gemma corpus belongs to one task.** The honest statement is
n=1 episode, and I over-generalised it to the Deputy and the Director before the evidence was in.

**Refuted: the window-conditioned secondary comparison is void.** `live_lens_pilot.py:176-181`
hardcodes that only one episode puts any position past 1,024, taken from opening prompt lengths.
Actually **8 of 14 episodes** do, covering 26,336 of 106,322 reading positions (24.77%) and 3,140 of
5,760 emitted tokens (54.5%), maximum position 2,298. The comparison has been available the whole
time and we declared it dead on an assumption nobody rechecked.

**Refuted: the identity-boundary effect manufactures the regression fit's depth trend.** It inflates
the level, not the trend. `cos(R,I)·cos(H,I) = 0.7882 × 0.7369 = 0.5808` reproduces the 0.581
exactly, but identity-partialled cosine runs 0.0242 / 0.2714 / 0.7264 at layers 1 / 20 / 33 — a span
of 0.702 against the raw 0.857, so **82% of the trend survives**. The trend still needs an
explanation and there is now a better candidate; see the position mismatch below.

**Withdrawn and now clean:** the 2.5x contamination figure. One live quotation remained at
`research/records/GEMMA3-OBSERVATION-ROLE-2026-09-08/README.md:185`, where "the control's 381 s"
gives 940/381.1 = 2.47; it is retracted two lines later. The 3.5-hour stage-two projection derives
from the Qwen pilot, not from the contaminated run, and is not stale.

---

## Blocks stage two

**1. The map would be a map of one episode.** Projected at the corrected rendering, `update-0028`
alone contributes 39,755 of 116,470 rank rows — **34.1%** — with the top three episodes at 61.0% and
the top four at 70.9%. That episode is 24 steps, 2 distinct calls, one call repeated 23 times.
*Representation.* Fix: bound composition before the run and report per-episode rather than pooled.
Cheap, but it must be decided in advance, because the pre-registration bars amendment once a record
is read.

**2. The primary comparison is void as operationalised.** No record row carries a span label: rank
rows are position, layer, horizon, token id, rank, probability and nothing else. No note/call
segmentation exists in any document or in the code. The boundary would therefore be chosen **after**
the records were read. *Representation.* Fix: define the segmentation and commit it before the run.

**3. The primary comparison is confounded by format entropy even once segmented.** Call spans are
stereotyped fenced JSON — 17 of a median 36 tokens are fixed scaffolding — and are 63.5% of agentic
generated tokens against 36.5% for notes, every note prefixed `Progress note: `. "Calls are read
earlier in depth than notes" is not separable from "JSON is more predictable than English". The one
existing run has note **above** call at every non-final layer (layer 24: 0.4822 against 0.3815),
which is the opposite of the hypothesis and the exact signature of the confound. *Representation.*
Fix: restate as free-form prose against structured action literal, and split the call span into
fixed skeleton and variable arguments.

**4. The base rate and the null are unfalsifiable at the primary horizon.** Sampling is greedy
(`live_lens_pilot.py:166`), so at horizon 1 the final layer is rank 1 for 5,760 of 5,760 emitted
tokens, exactly 100%. Real base rates exist only at horizon 4 (19.1%) and horizon 8 (10.2%).
*Representation.* Fix: move the base rate off horizon 1, or state that horizon 1 has no null.

**5. The layer-34 identity check proves almost nothing, and it is load-bearing.** At `layer == 34`,
`session.py:256-263` substitutes the model's own softmax for the readout. No lens map is applied, no
readout orientation is exercised, no residual is used — the captured layer-34 residual is taken and
discarded. The 5,316 reads at rank 1 for 100.00% therefore establish only that sampling is greedy
and that the horizon is subtracted rather than added. They say nothing about lens application,
readout orientation, the J*k* to layer *k+1* mapping, or the residual tap. *Representation.* Fix is
one line: call the real readout at 34 and assert it agrees with the native logits.

**6. The residual layer offset is not settled and geometry cannot settle it.** The repository side is
clear — layer L is the output of block L−1, and orientation is confirmed empirically at
cos 0.712 against 0.016 for the transpose. But `source_layers = [0..32]` is consistent with both the
block-output and the hidden-states reading, and adjacent hosted maps are 0.72 to 0.98 alike while our
two instruments agree only 0.02 to 0.71, so a one-layer shift moves agreement by at most 0.05 with
inconsistent sign. *Representation.* Fix: read the fitting code, not the matrices.

**7. Stage one silently covers 11 of 12 families.** `agentic-d2-read-0108` was skipped as complete
though its own record footer says `"status": "aborted"`; it now sits in `pilot/quarantine/` and is
absent from the manifest. *Both.* Fix: run stage two into a clean directory.

**8. The manifest misreports what the model was shown.** `live_lens_pilot.py:109` computes
`opening_prompt_tokens` from the convention-free prompt while `runner.py:621` renders with the spec.
The fair run's manifest says 437 where the model saw 523. *Both.* Fix: measure the rendered prompt.

---

## Biases a number that is already published

**The comparator is wrong.** At a matched 12-step ceiling the base scores **6 of 11, not 7** —
`cross_reference-0032` passes only at turn 14. Worse, the achievable denominator is **8, not 11**:
`batch_update-0166` and `aggregate_report-0167` have expert horizon 14 and were unpassable at 12
steps, and `ledger_reconcile-0163` at horizon 12 required a flawless run. `aggregate_report-0167`
additionally emitted exactly 200 tokens on its terminating turn, against 48 to 57 on every other
turn, so it died on the token cap rather than on the task. *Scoreboard.*

**The two pilots share no task at all.** Gemma's twelve ids and the Qwen pilot's nine intersect in
the empty set, and `task_id` omits difficulty, so one id names three different tasks. The three chat
episodes are byte-identical at turn 0 only; turn 1 conditions on each model's own reply. The valid
common ground is turn 0, at the two coincident layers 12 and 24. *Representation.* Note the
scoreboard is task-matched even though the pilots are not, because the 180-task eval is all d2.

**The regression fit's depth trend is roughly half scale drift.** `relative_difference_to_hosted` is
algebraically `sqrt(r² + 1 − 2r·cos)`, verified to 4.4e-16, and the norm ratio r falls 3.63 → 1.22 →
0.81 across depth. A trivial baseline of two unrelated maps at the observed norms gives 3.77 / 1.58 /
1.29 against the reported 3.74 / 1.32 / 0.48, so **52.1% of the trend is norm drift and 47.9%
survives**. On the selection-SSE reading, 44.6% survives against an identity baseline and the trend
**inverts** against a norm-matched one. At layer 1 the optimal rescale is 0.0073, so the reported
3.7441 is 99.96% a size mismatch. *Representation.*

**Ridge selection is uninformative about the optimum.** Held error is strictly increasing in alpha at
33 of 33 layers and 5 of 5 candidates; the grid brackets no minimum anywhere. One grid step costs a
mean 0.275 percentage points of held target energy, and 0.244 at layer 33, which is 85.0% of that
layer's entire error. The monotone depth shape exists only at the two smallest of five penalties.
*Representation.*

**The largest unremarked finding in the fit.** `GEMMA3-LENS-2026-09-08/convergence.csv` shows
`n_valid_positions = 111` for every one of 546 prompts, so the hosted lens's 60,606 positions are
all document-initial and BOS-led. The regression corpus has `template: 1` for the fit span and `0`
for the held span, so exactly one BOS in 257,280 tokens and 2,009 of 2,010 windows start mid-sentence.
"Both fitted at 128 tokens" is true of window length and false of position composition. **The hosted
lens was never fitted above absolute position ~111, against a 1,024 sliding window, and the map reads
transcripts to position 2,298.** *Representation.* This is a live alternative driver of the depth
trend and it is not disclosed at the point of use.

---

## Limits scope — disclose and proceed

**The four lens mismatches, with directions.** Prose to transcripts biases ranks up and unequally,
and because notes are prose-like while calls are JSON and paths, it biases the pre-registered
contrast **in favour of notes** — the same direction as the format-entropy confound. The 128-token
fit biases ranks up and rising with absolute position, manufacturing a spurious position gradient.
bf16 to 4-bit biases ranks up at every lensed layer and has zero effect at layer 34, so the published
depth gradient is steeper than truth. The windowed-mask regime is unpredictable and possibly opposite
in sign between sliding and global layers, which is the very contrast the secondary comparison wants;
it applies to 55.4% of reads. **Only the precision mismatch is disclosed at the point of use.**

**Span composition differs across the panels far more than the model does.** 37.2% of Gemma's
in-context call spans are verbatim repeats, rising to 59.5% in the re-run, against 7.4% for Qwen; and
28% of Gemma's call-span tokens sit in failed turns against 0% for Qwen. Horizon-1 foreknowledge on
already-emitted text is close to trivial, so this contaminates the primary comparison independently
of everything above.

**The `calculate` control is weaker than I claimed.** Gemma's evidence there is n=1 — it passed
`test-calculate-0158`, one of its only two passes. All 15 `calculate` prompts embed a parenthesised
expression to reproduce verbatim, and arm A's recorded failure on `test-calculate-0002` is an
unbalanced parenthesis copied into the expression, which is a literal-copy failure inside the
supposedly clean family. It also has the shortest horizon and the loosest answer format, so three
variables move at once. It bounds rather than isolates.

**The protocol has never been varied, so agentic competence cannot be ranked.** The fenced-JSON
convention exists because `<tool_call>` tokens were untrained in a Qwen2.5-coder-3B checkpoint, a
model-specific property promoted to a universal. Gemma has no system role, so rules, tools,
convention and task arrive in one user turn. `max_tokens=200` was set against Qwen notes averaging
51.6 characters; Gemma's average 83.9. The supportable claim is **"Gemma does not maintain a stable
action and observation model under this protocol"**, not that it is worse at agent tasks.

**Head capture is impossible on Gemma.** The gate is hard and the error message's suggested escape
is unreachable, since the emitter is Qwen-attention-only. Transport is unavailable; the rotating-cache
monotonicity claim is otherwise verified.

**The tokenizer asymmetry is a hypothesis, not a result.** Gemma's vocabulary has 99 `/x` tokens and
none of them are path words, so a leading slash is an inserted token; Qwen's has 1,077 including
`/workspace`, `/test`, `/config` as single tokens, so the same slip would be a substitution. This is
consistent with the corpus and is not established by it.

---

## The two questions nobody asked

**Does the environment have a discoverable root?** No. No tool lists the workspace top level, and
every wrong guess is answered with a confident false "empty" rather than an error. We have been
measuring whether models can navigate a filesystem whose entry point can only be guessed, and then
attributing the guessing failure to the model. This is the deepest confound in the register and it
was invisible because `read_file` behaves correctly and only `list_files` lies.

**Is the call span the same object across the two panels?** No, and it is not close. Gemma's call
spans are 37% to 60% verbatim repeats with 28% of their tokens in failed turns; Qwen's are 7%
repeats with none. A cross-model statement about how calls are represented is comparing a span made
mostly of fresh, successful actions against one made substantially of copies of a failing one. No
lens correction reaches this, because it is a property of the corpus rather than of the instrument.

---

## What was addressed, and where

Identifying was half the instruction. This is the other half, recorded so the register is not a list
of complaints.

| confound | disposition | where |
|---|---|---|
| `list_files` false empty | **fixed in code**, both simulators, full suite green | `020aa89` |
| Base rate unfalsifiable at horizon one | null moved to horizons four and eight | `9a9880b` |
| Span facets absent from the records | defined, emitted at write time; observation facet struck | `9a9880b` |
| Primary comparison not identifiable | restated as call-argument against call-skeleton | `9a9880b` |
| Composition dominated by one episode | per-episode computation, equal weight across episodes | `9a9880b` |
| Repetitions pooled as a sample size | deduplication required before any n is quoted | `9a9880b` |
| Window secondary wrongly declared void | count measured from the run, comparison proceeds | `9a9880b` |
| `read-0108` aborted but counted | episode set corrected to eleven families | `9a9880b` |
| Environment change breaks comparability | stated in advance, no figure crosses the stages | `9a9880b` |
| R61 resting on a withdrawn figure | corrected to stand on the load average alone | `531ae6c` |
| Window announced against a dead holder | R61(c), and `--holder-pid` landed by the D-CRO | `9f4785e`, `9c2fdee` |
| Codex task 1 blocked by an unmeetable gate | gate resolved as met, duplication bound added | `a1ff0b0` |

**Still open and owned elsewhere.** The residual layer offset, which geometry cannot settle and
which must be read out of the fitting code. The layer-34 readout call. The pilot manifest's
`opening_prompt_tokens`, measured from the wrong prompt. The regression fit's position-composition
mismatch, where the hosted lens was never fitted above absolute position ~111 against a 1,024
window, which is the strongest unexcluded alternative explanation for its depth trend.

### The restatement that the audit earned

The primary comparison changed because a control disproved my own prediction. I argued that
stereotyped JSON should commit earlier in depth than prose, being more predictable. Measured across
every episode, with the fixed point excluded so it could not contaminate its own baseline:

| token class | n | at layer 24 | at layer 30 | median |
|---|---:|---:|---:|---:|
| argument-start | 128 | 1.6% | 79% | 30 |
| all other tokens | 5,442 | 29% | 44% | 30 |

Argument-start tokens commit **later**, not earlier. So the two halves of a call span sit at opposite
ends of the depth range and move in opposite directions, and any single "call" figure averages them
away. The comparison now varies the task-dependent content of an action against the convention that
carries it, inside one syntactic object, which holds format and lens domain roughly fixed.

**This is the transferable part.** Before comparing two spans through a lens, check that the spans
are not compounds of populations that behave oppositely, and check the comparison against a control
of the same syntactic kind rather than against a pooled background. Both checks changed a
conclusion here, and neither needs our code or this model.
