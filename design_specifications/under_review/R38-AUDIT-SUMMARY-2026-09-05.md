# R38 audit: what fifteen readers taught us (issue #70, four slices, 2026-09-05)

> Slices `990b43c`, `efcfb78`, `8420542`, and a fourth held for a Metal run. Implemented by
> dispatched agents, reviewed by the Deputy, gated by the Chief. This file exists because the
> lesson must outlive four commit messages, which are read by whoever is bisecting and by nobody
> else.

## The lesson, first

**A soft default cannot tell a missing key from a moved one.** Every reader this audit found wrong
had one. That is the whole mechanism, and everything below is evidence for it.

## The count

**Fifteen record-readers audited. Four were wrong.** Two produced observably wrong artifacts; two
were latent and were caught only by moving a key.

The table that scoped this audit listed **fourteen**. It was short by one — `cli._validation_losses`,
reading `metrics.jsonl` — which a grep for on-disk readers finds. The enumeration written to make
the audit concrete was itself the first thing that needed auditing, and it was the Deputy's.

## The mechanism, which is not "readers get levels wrong"

**Every one of the four had a soft default, and a soft default cannot tell a missing key from a
moved one.**

That is the whole defect. The reader asks for a key that is not there, receives `None` or a
documented fallback, and returns a value indistinguishable from a real answer. There is no
exception, no log line, no failing test. It presents as ordinary output. A reader that indexes and
raises cannot have this bug; the one site with an exact-set check (`_validation_losses`) was safe
for exactly that reason.

### Form one: the key one level from where the writer puts it

- `state_probe.preflight_precision_block` read `fp32_manual_vs_native` at the top level. The
  preflight writer nests it under `residual_equivalence`. **Every probe artifact recorded `null`
  for days** for a number R18a requires in all of them. The reader had been written against the
  layout of a *sibling* artifact, which does put it at the top level, in a different file.
- The patch probe read `data_seed` at the top of an evaluation payload where the writer puts it in
  the summary. The result was not a `None`: it took the reader's own field-is-absent branch on
  every real artifact, so it **never reached the R22a conflict check it exists to perform**, and a
  `--data-seed` disagreeing with the artifact was accepted in silence — the exact guarantee its
  docstring makes.
- `adapter_base_identity` read `model["spec"]["hf_id"]`. A resolved-less stage writes that
  declaration flat, `hf_id` at the top of the block. Four of the five `cli.py` sites that write the
  flat shape write it into the run directory that holds `adapters/`, which is the path this reader
  consults.

### Form two: a fallback for a legacy shape that never existed

Worse, because the default **is** the value the field actually holds, so the fallback returns the
right answer for the wrong reason and there is nothing to notice.

- `baseline_reanalysis_parameters` fell back to fit constants documented as "what a baseline
  written before the `fit` block used". The block shipped in `f90578b`, the same commit as its
  writer; the reader arrived in `eca116b`, a day later. That class of artifact has never existed. A
  moved `fit` block refitted at the defaults and reported the comparison as if it had matched the
  baseline's own constants.
- `ExpansionSpec.load` defaulted `version` and `initialization`, which `save` has emitted since the
  file's first commit. `version`'s entire job is to say a recipe is *not* the one this code knows
  how to apply.
- `_counts` named `("valid_turns", "turns")`, which no writer has ever emitted — so it raised on
  precisely the legacy files it existed for. Loud, and therefore not a wrong answer, but the same
  shape.

## Why a rebuilt fixture is not enough

The test written beside each wrong reader agreed with it, because both came from the same belief.
A hand-made fixture is the reader's author writing down what they think the writer emits; **it
passes exactly when the reader is wrong in the way its author expected it to be right.**

But rebuilding is only half. **Of the fifteen, the ones needing fixing were found by moving the
key, not by rebuilding it.** Two fixtures (`_refit_baseline`, the expansion round-trip) already
came from the real writer and still hid a defect until a key was moved. A rebuilt fixture that
merely still passes proves nothing.

So the method has three steps and the third is the one that finds things:

1. Build the fixture through the real writer.
2. Confirm the test passes.
3. **Move or rename each key the reader depends on and confirm the test goes red.** A move that
   does not go red means that key is not covered, whatever the test appears to assert.

## Why the method has a confirm-or-refute step

**Four of the five briefs in this audit carried a premise the code refuted, and every one was
caught by the implementer rather than by the author.** They were the Deputy's.

| claimed | actual |
|---|---|
| a manifest reader feeds the R21 write guard | the guard fires earlier, on the output directory's own manifest |
| `_write_stage_manifest` writes atomically, as the model to copy | it uses a plain write; the migrated helpers are `branch.py` and `rollout.py`, and a `DEBT` comment had named it the whole time, in a file read twice |
| a real artifact's model block carries a snapshot revision, its absence diagnostic | the key is present and `null` in every artifact on disk; it distinguishes nothing |
| a legacy layout is the only tolerated missing key | it is a question about generation text, settled before the step is ever written |
| the reader table has fourteen entries | fifteen |

A brief is a hypothesis. Four of five were wrong in some load-bearing particular, and the only
thing that caught them was an implementer told plainly that **refuting the brief is a good
outcome**. A method that assumed the brief was right would have shipped four wrong things and found
neither live defect any sooner.

## What to do differently, stated so it is usable

1. **Prefer indexing to `.get()` in a reader of our own artifacts.** A default is a decision to
   accept a shape you did not verify. Where a default is genuinely needed — a legacy file that
   really exists — name the artifact on disk that requires it, in the docstring. Two of the four
   defects were fallbacks for artifacts nobody could produce.
2. **Never write a fixture by hand for a reader of something we write.** Call the writer.
3. **Then move the key.** Steps 1 and 2 are hygiene; step 3 is the test.
4. **When a brief asserts a fact about the code, mark it as a claim.** The implementer is closer to
   the code than the author of the brief, every time.
