# Does the 3B agent internally hold the state its notes carry?

Date: 2026-09-02.

## The question

The whole v2 paradigm rests on a bet: that this model cannot hold multi-step task state
internally, so we must force it to externalise state into written notes and then hide older
observations to make that externalisation load-bearing. The bet paid off behaviourally (47/60
on held-out tasks, up from a base model that cannot complete the format at all), but we have
never checked the premise. Two very different worlds produce the same success number:

- **World A, the notes substitute for a missing workspace.** The model genuinely does not hold
  "the next file is `invoice-2-537.txt`" internally. Writing it down creates state that would
  not otherwise exist. Hiding observations is essential, and any future design must keep an
  external memory.
- **World B, the notes are a readout of an existing workspace.** The state is already held
  internally and the note merely transcribes it. Then the notes are a training scaffold, the
  windowing is a crutch we could relax, and there is latent capability we are not using.

Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models*
(arXiv 2607.15495), gives us a way to look. Their Jacobian lens identifies the representations
a model is *prepared to verbalise*; the subspace of these is the J-space, which behaves like a
global workspace whose contents can be reported, held, and used for silent reasoning.

## Method

For layer L, the averaged Jacobian is `J_L = E_corpus[ d h_final / d h_L ]`, and the lens
readout of an activation h is `softmax(W_U · norm(J_L · h))`. We never build the d×d matrix.
Since `E[J]·h = E[J·h]` and each `J_c·h` is one forward-mode Jacobian-vector product, a readout
costs one JVP per corpus context. Measured on this machine: 144 ms per JVP through half the
network, so a 16-context readout is about 2 seconds. Float32 accumulation is mandatory; the
tangent overflows to infinity in float16.

The probe point is the exact decision the model gets wrong in the wild. In `ledger_reconcile`,
after two invoices have been read, the directory listing has fallen outside the two-observation
window. The model must now name the third invoice file, whose name carries a random suffix it
can only know from that hidden listing. This is precisely where the trained checkpoint invented
`invoice-2-878.txt` and then repeated the failing call twenty times.

We replay the *expert* trajectory to that point rather than generating, so the probe is
deterministic and independent of policy quality, and we read the residual stream at the last
prompt position, before any note token is produced.

## Candidates and controls

For each layer in {6, 12, 18, 24, 30} we report where three strings land in the readout:

| Candidate | What it is | Expectation if World A | Expectation if World B |
| --- | --- | --- | --- |
| `target` | the correct next filename, only ever seen in the now-hidden listing | absent | present, high rank |
| `already_read` | a filename read earlier, still visible in context | present | present |
| `unseen` | a filename from an unrelated task | absent | absent |

Three controls keep this honest:

1. **Logit lens baseline.** The same activation, unembedded with no Jacobian applied. If the
   target appears equally under both lenses, the J-lens is adding nothing and we are just
   reading the next-token distribution.
2. **`already_read` as a positive control.** If a filename that *is* still visible does not
   register, the method is broken and no negative result about `target` means anything.
3. **`unseen` as a negative control.** If an unrelated filename registers, the readout is not
   selective and the metric is meaningless.

Reported per candidate: best rank, max token probability, and total probability mass across its
tokens.

## What each outcome licenses

- **Target absent, `already_read` present.** World A. The state genuinely is not held, the
  notes create it, and the design is justified on mechanism and not just on outcome. This also
  predicts that relaxing the window would not help, which is separately testable.
- **Target present at high rank.** World B. The model knows the filename and still emits a
  wrong one, which relocates the problem from memory to retrieval or decoding. That would make
  constrained decoding over observed paths the obvious cheap fix, and would mean the notes are
  doing less work than we think.
- **Target present only at late layers, or only weakly.** Partial, and the most likely result
  at this scale. It would say the workspace holds a degraded trace, which is consistent with
  the paper's report that these effects grow with model size.

## Honest limitations

The corpus standing in for a pretraining distribution is two dozen short snippets, not a real
sample, so `J_L` is approximate. The base is 4-bit quantised and carries our LoRA, both of which
perturb the Jacobian; a bf16 run is about 6 GB and fits if the quantised result looks marginal.
The paper's functional claims were established on frontier Claude models, and a subspace being
*definable* here does not establish that it has global-workspace properties at 3B. A single
probe point in a single family is an anecdote; the experiment should sweep several tasks per
family before any conclusion is drawn.

## Follow-on: counterfactual reflection training

If the result is World A or partial, the paper supplies the intervention. Counterfactual
reflection training supervises what the model *would* say if interrupted and asked to reflect,
without ever interrupting at inference, and improved uninterrupted behaviour because internal
reasoning routes through representations of things the model might say later.

Our tasks give exact ground-truth state at every step, so these targets are free to generate: a
trajectory prefix, a user turn asking the model to state everything established so far, and a
supervised answer listing files read, values collected, what remains pending, and what has been
verified. Those rows never appear at inference.

The reason to expect a gain over simply writing longer notes is that the note is deliberately
compressed to stay short, so it is strictly less than the full state. Training the model to be
*able* to produce the full state on demand should make it hold more than it writes. The
intervention is falsifiable twice over: held-out task success says whether it helped, and a
second J-lens probe at the same decision point says whether the state actually moved into
J-space, which tests the mechanism rather than the outcome.


## Result (2026-09-02)

Probe point: `test-ledger_reconcile-0007`, step 3, where the directory listing has fallen out of
the two-observation window and the model must name `invoice-2-537.txt`. The context is forced to
end mid-note at `Reading invoice-2-`, so the very next token is the first digit of the suffix.
This makes the model's own output distribution the measurement, which is far more direct than
any lens.

Two methodological errors had to be corrected first, both mine:

1. The first metric scored candidates over *all* their tokens. Sibling paths share nearly every
   token, so `target` and `already_read` returned byte-identical ranks at every layer and the
   positive control was vacuous. Fixed by scoring only tokens unique to one candidate.
2. The first probe read the position where the model begins its *note*, not where the filename
   is due. The J-lens there returned the note's own semantics (`Total`, `Remaining`, `Filtered`,
   `Skipping`), which is a real signal but answers a different question.

A third confound nearly produced a false conclusion. With the current generator, the filename is
present in the model's own previous notes because of the `pending:` fix, so the first clean-looking
result (final-layer probability 1.000 on the correct digit) measured *copying from a visible note*,
not memory. The experiment only becomes a memory test once those pending lists are stripped, which
reproduces the run-A condition that actually fails.

### The A/B

| Condition | `537` visible in context | Model's next-token prediction |
| --- | --- | --- |
| Notes carry `pending:` (run B) | yes | `5` at p = 1.000 |
| Notes stripped of `pending:` (run A) | no | `9` 0.27, `8` 0.23, `6` 0.17, `7` 0.11, `5` 0.10 |

Without the note the model is guessing, close to uniform over digits, with a mild pull toward a
digit it has already seen (`9`, from the previously read `invoice-1-924.txt`). The correct digit
sits fourth at p = 0.10.

### Conclusion

**World A.** The model does not hold the hidden filename internally. The written note is what
creates that state, and the observation window makes writing it load-bearing. This is direct
mechanistic support for the paradigm rather than an inference from the score, and it independently
justifies the `pending:` fix: with the state in the note the model uses it with certainty, and
without it the model confabulates exactly as the failing transcripts show.

It also rules out the cheaper alternative explanation. Had the filename been present and merely
mis-decoded, constrained decoding over observed paths would have been the right fix. It is not
present, so no decoding-side fix could have worked.

### The mid-layer signal, resolved (`research/jspace_sweep.py`)

> **How to reproduce the table below (C5, issue #62).** `research/jspace_sweep.py` is now a
> thin wrapper over the installed `agent-v2-jspace-sweep`, which runs a *different* variant:
> three readouts rather than one, interior source positions rather than the last token, a
> 128-token corpus, a derived kind-matched layer family, and registry rendering. The numbers
> recorded here reproduce only from the script as it stood at commit `1953493`, the parent of
> `d0adfc6`. The current CLI answers the same question under the EXP-001 estimator and is not
> comparable to this table without naming every difference (R35).

The single-task probe left an ordering unexplained: at layers 18 and 30 the J-lens put the correct
suffix above a previously-seen one above an unrelated one, even where the model demonstrably did
not know the answer. Either a weak internal trace, or a digit-frequency artifact.

These predict opposite things under averaging, so we ran a paired sign test over 42 ledger probe
points whose digit assignments differ, each compared against a null in which the identical
candidate pair is scored against a *different* task's context. A task-tracking readout beats both
50% and its own null; an artifact leaves the two columns equal.

| Readout | True suffix preferred, matched context | Same pair, mismatched context | p |
| --- | --- | --- | --- |
| J-lens layer 18 | 23/42 (55%) | 23/42 (55%) | 0.64 |
| J-lens layer 24 | 6/42 (14%) | 17/42 (40%) | <0.001 |
| J-lens layer 30 | 21/42 (50%) | 20/42 (48%) | 1.00 |
| Model output | 19/42 (45%) | 19/42 (45%) | 0.64 |

**The mid-layer ordering was an artifact.** Layers 18 and 30 are at chance and match their own
null exactly. The model's own output is at chance too, confirming it does not know the suffix.

The layer-24 result is the interesting one, and it is the positive control this experiment
previously lacked. Read the other way round, the *already-read* suffix, which is present in the
visible context, is preferred 36/42 (86%, two-sided p = 2e-6) in the matched condition and only
25/42 (60%) when the same pair is scored against a context that does not contain it. So the J-lens
at layer 24 demonstrably detects a filename that is in context, and detects it strongly.

That makes the null result about the hidden filename substantive rather than merely absent
evidence. The instrument is sensitive enough to find an in-context filename at p = 2e-6, and it
finds no trace whatsoever of the hidden one, at any probed layer. World A stands, now with a
working positive control.

### Remaining limitations

The discriminators are single digit tokens, which are weak; a family whose filenames differ by
whole words (the mode tokens in `batch_update`: safe, audit, strict, fast, observe) would be a
better test bed and is the obvious next sweep. Only `ledger_reconcile` was swept. The corpus
standing in for a pretraining distribution remains 24 snippets, and the base is 4-bit with a LoRA
attached, so absolute J-lens probabilities are not calibrated; the paired design means the
conclusion rests on relative comparisons within a fixed setup, which is the right way to use it.
