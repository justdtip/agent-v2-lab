# WS-C: five ways a wrong thing passes, and what to check instead

**Engineer B, 2026-09-09.** The stream's record as a whole. Everything below was measured on CPU on
one laptop. **Nothing has run on a GPU**, and the unexecuted list is at the end rather than
scattered.

The findings are first, the transferable technique second, the implementation third. They are
separated because only the first two survive this codebase.

---

# Part one: what was found

Every one of these produced a **passing check on a wrong thing**. That is the common shape and it is
the reason to write them down: none announced itself, and four of the five were caught by a
different check than the one that should have caught them.

## 1. The loss agreed exactly while the gradient was 40% wrong

Comparing single-device training against two-device sharded training, the specified check was that
the loss curves agree. At step zero, on identical weights and identical rows, they were
**bit-identical**. The gradient at that same step differed by **0.406** on the parameters outside
the sharding units, because nothing was reducing them across devices.

The loss at step zero is identical **by construction** in any two arms that start from the same
weights and consume the same rows. It cannot disagree, so it cannot detect anything. Afterwards it
is one step behind: it reports the previous step's parameters, so a corrupted update shows up in the
loss only once it has already been applied.

Run deliberately as a control, the broken configuration gives:

| arm | loss, step 0 | gradient, step 0 | values, after |
|---|---:|---:|---:|
| correct | 0.00e+00 | 1.80e-07 | 8.56e-06 |
| also correct, reduced by hand | 0.00e+00 | 1.80e-07 | 8.58e-06 |
| **broken** | **0.00e+00** | **1.67e+00** | **1.12e+00** |

There is no tolerance at which the loss separates those three.

## 2. A wrapper changed what the object *is*, and three separate mechanisms noticed differently

Putting a model inside a thin wrapper — a completely ordinary thing to do — broke three upstream
mechanisms in three different ways, and only one of them raised.

- **Automatic precision casting** attaches to the wrapped call. Code that reaches past the wrapper
  to read a parameter directly never enters it. This raised a type error, eventually, far from the
  cause.
- **Sharding hooks** likewise: a parameter is only in its usable, gathered state inside the call of
  the unit that owns it. Touched outside, it raised — and the error read as a refusal of the
  design rather than as a statement about *where* the access happened.
- **Saving** does not call anything. It **branches on the object's type**, finds the wrapper is not
  the expected class, and silently writes a different file format. Nothing raised at all.

The third is the dangerous one and it generalises past wrappers: upstream libraries inspect what
they are handed, not only call it.

## 3. A tolerant reader turned a corrupted artefact into a passing test

The training run wrote a checkpoint whose configuration described one thing and whose contents were
named as another. The two disagree; no reader can reconcile them.

The test asserted the checkpoint reloads. **It passed.** The default reader treats missing contents
as a warning and initialises them instead, so every weight came back freshly random and the
assertion was satisfied by an artefact that had learned nothing. Confirmed by checking the reloaded
values against the initialiser's spread.

A strict reader — one that refuses on any gap — caught it immediately.

## 4. Every trained adapter this programme has produced was taught to emit `!`

Two short rules in the training library, neither wrong alone and neither referring to the other. One
pads batches to a width guaranteed to leave at least one padding slot. The other supervises one
position past the last real token. So the padding slot is supervised, in **every untruncated row**,
with the padding value.

Padding is token id zero, and **id zero is not a padding token in every vocabulary**. In the one all
existing records used, it is `!`. So every adapter was taught, once per row, to emit an exclamation
mark immediately after its end-of-turn marker — an ordinary emittable token, at exactly the
position where generation decides to stop.

It is 0.09% of the supervised signal and 100% of rows, always the same target at the same structural
position. **No behavioural effect has been measured and none is claimed.** A test that would settle
it is recorded beside the finding.

## 5. Two arms have to be the same experiment before a disagreement means anything

The first run of the joined multi-device check failed: gradients apart by 0.499, values by 1.357.
The sharding was fine. The **check** was wrong, in two independent ways:

- The training library splits data across processes, so an accumulation setting left alone doubles
  the rows consumed per step at two processes. The arms were optimising different objectives.
- The distributed sampler need not place the same rows in the same step as the single-process one.
  So even with totals matched, any run longer than one step compares two different trajectories.

One optimizer step over the whole dataset removes both: the arms accumulate the same set, and
composition cannot matter. After that the same comparison gives **1.24e-07** on gradients and
**0.0** on values.

Most of the work in a multi-device check is making the two arms comparable. Until they are, a
disagreement is not evidence of anything.

---

# Part two: the technique

Stated without reference to this codebase, because this is the part that transfers.

**Check the quantity nearest the mechanism, never one downstream of it.** A downstream quantity can
be right while the thing you care about is wrong, and it is right *by construction* in exactly the
comparison you would use to reassure yourself. Prefer gradients to losses, parameters to gradients,
and the artefact to the report about the artefact.

**Compare per element, not per aggregate.** A wrong slice hides inside a correct norm the same way a
wrong gradient hides inside a correct loss. An aggregate is a downstream quantity.

**A guard that passes the thing it exists to catch is worse than no guard**, because its silence is
read as evidence. Test the property that actually distinguishes the cases, not a property the wrong
case also has.

**An assertion that something loads is worth nothing unless the loader refuses bad input.** Tolerant
readers convert corruption into a pass. Use the strict reader in tests, and if there is no strict
reader, that is the thing to build first.

**A synthetic fixture must carry the real artefact's declared type and layout**, or the code path
that handles real artefacts is untested by construction — and it fails silently, because the tests
pass on a shape that does not exist.

**Anything that reaches around an interface inherits nothing attached to it**, and must supply that
itself. Libraries attach behaviour to calls and branch on types. Bypassing the call loses the first;
substituting the object loses the second.

**Before treating a disagreement as a defect, prove the two arms are the same experiment.** Match
what is consumed, and arrange the comparison so that ordering cannot matter. Most of the work is
here.

**Two independent occurrences are a property of the tool, not of the people.** When two people fall
into the same trap, fix the tool and write it down; do not fix it quietly.

**A measurement carries what it is a measurement of.** A test count without its skip count is not a
claim. A number taken against a since-changed dependency is a number about something else, and is
re-taken. A result whose exactness comes from rounding says so, rather than borrowing credibility it
has not earned.

---

# Part three: the implementation, which is the least of it

What this stream happens to have built, in one line each. It is last because it is the part that
does not transfer.

| piece | what it is |
|---|---|
| depth expansion, torch | copies a block whole and zeroes its two residual outputs; repairs the attention-type map an insertion shifts. Identity exact at every insertion point |
| chunked loss | applies the output projection a slice at a time under checkpointing; 2.82 GB of logits become 0.54 GB. Bit-exact at one chunk |
| config reader | converts a recipe written in micro-batches into optimizer steps, exactly where it must be exact and rounded where it may be |
| collator | builds the attention mask the old path did not need, and reproduces the padding artefact only on request |
| freezing and optimizer | opens the top N blocks; refuses a parameter dtype whose optimizer state would silently round away |
| checkpoint delta | per-layer relative perturbation, numerator and denominator aggregated separately |
| train stage | dispatches on backend, loads the text tower out of a multimodal checkpoint, records determinism and every departure |
| sharding | the loss runs inside the sharded unit's own call, so raw-parameter access and sharding coexist |

---

# Findings for other seats

Neither is mine to change and I have changed neither.

## Twelve closure warnings, all benign — but the rule was right to ask

`B023` — a function defined in a loop closing over the loop variable — fires twelve times, in four
files owned by other seats. **I read all four sites. Every one is benign**, and for the same reason:
the closure is invoked on the next line inside the iteration that defines it, so the captured
variable holds the intended value at call time. None is stored, returned, or deferred.

| file | sites | closure | called |
|---|---:|---|---|
| `research/records/ARM-A-DIVERGENCE-2026-09-07/build_report.py:242` | 1 | over `row` | same line, same iteration |
| `research/records/jlens-hosted-qwen35-4b-2026-09-05/query_cosine.py:69` | 2 | over `Qn` | next line, twice |
| `scripts/cache_split_diagnostic.py:143–145` | 3 | over `cut`, `bS`, `ids` | three times, all in-iteration |
| `scripts/fixed_history_lens.py:263, 265` | 6 | over `n_p`, `ids_full` | next line, three times |

**So no recorded number is wrong on account of these.** The two in `research/records/` matter most
because they computed published figures, and both are sound.

The reason to report rather than silently clear them: the finding is *that they are benign*, and
that finding required reading each one. A seat who knows what each record claimed should still hold
the conclusion, which is why it went to the D-CRO rather than into a lint commit.

## A rule that looks mechanical is not

`B905` — `zip` without an explicit `strict` — is the sharper version of the same lesson. Eight sites;
**two of the four in production would crash under a blind `strict=True`**, because a common-prefix
scan and a pairwise walk over consecutive reports zip deliberately unequal sequences. A third of the
sites wanted the opposite answer from the rest.

It sits beside the rule that no automatic fix runs blind over a file that reads model weights,
because it is the same shape: **an automatic fix is a guess about intent, and intent is what the
tool cannot see.**

---

# The suite, and the state of the box beside it

**2,348 passed, 10 skipped, 0 failed**, with **no window open and no model lock held** — the first
reading with everything in and nothing standing off.

The box state belongs beside the number. The same suite minutes earlier, under another seat's
window, read **1,139 passed, 1,110 skipped**: eleven hundred box-gated tests correctly declining to
run. Both readings are true and only one of them is a claim about the code.

---

# The first hour on the device, in order

**Nothing in this record has run on a GPU.** No CUDA, no NCCL, no more than two processes, no real
checkpoint at four billion parameters, and **no memory figure measured on any device** — the memory
arithmetic is arithmetic, not measurement, and the multi-device agreement is two CPU processes over
a local backend.

So every figure below is either arithmetic or a CPU measurement, and each step names the local
figure it is compared against and what a mismatch would mean. They are ordered
cheapest-and-most-diagnostic first, so an hour that goes wrong goes wrong early and for a nameable
reason.

**One thing to settle before step 1.** The device arms compare **GPU against GPU** — one rank plain
against two ranks sharded. The local numbers are the *expected magnitude*, not a bit target: kernels
and reduction order differ, so a device figure near the local one confirms the path and an identical
one would be a coincidence.

## Step 1 — the environment says what it is (seconds, no model)

Run the determinism pin and read it back.

| | |
|---|---|
| **produces** | the determinism block: backend, device, deterministic algorithms, the workspace variable, the attention implementation |
| **compared against** | the same block from CPU, where every field but `device` should match |
| **a mismatch means** | the pin ran after something touched the device. The workspace variable is read at first use, so a pin after that is not a pin. Fix before anything else; every later number inherits it. |

## Step 2 — the sharded path holds under NCCL (minutes, tiny model)

`stage_agreement.py`, one rank plain against two ranks FSDP2, on the tiny wrapper checkpoint.
No large model, so this is cheap and it exercises the whole joined path.

| | |
|---|---|
| **produces** | worst per-parameter relative deviation, gradients at step zero and values after |
| **compared against** | **1.24e-07** on gradients, two `gloo` CPU processes; the same comparison, same script |
| **a mismatch means** | anything above about 1e-5 is the port, not arithmetic. The first thing to check is that both arms consumed the same rows: accumulation is divided by world size and the run is one optimizer step over the whole dataset, precisely so composition cannot matter. |

**Do not accept a loss-curve agreement in place of this.** The broken configuration this gate was
built against gave a bit-identical loss and a gradient wrong by 1.67 relative.

## Step 3 — the memory arithmetic meets the device (minutes, the real checkpoint)

One training step at the sequence cap the run will use, on the device it will use. R60(c): a
declaration carries its measurement.

| | |
|---|---|
| **produces** | measured peak bytes for one step at the cap, and the largest micro-batch under the R47 fraction |
| **compared against** | arithmetic only, and it is arithmetic: **16 bytes per parameter** — 4 parameter, 4 gradient, 8 optimizer — giving **68.8 GB at 4.30B parameters** before activations. Logits at the 2,688 cap are **2.82 GB** in float32 unchunked against **0.54 GB** at chunk 512 |
| **a mismatch means** | above the projection, the trainable slice is larger than intended — check the freezing count in the manifest against `trainable_top_layers`. Far below it, the freezing opened less than intended, which is the same defect from the other side. |

**This is the number two projections have already been wrong about.** Neither was taken at the
largest context the run would reach. Take it there.

## Step 4 — the fallback rung is measured, not assumed (minutes)

Re-run step 3 with `reshard_after_forward` on the root unit set both ways.

| | |
|---|---|
| **produces** | peak with the root resharded against held, and the step-time cost of resharding |
| **compared against** | no local figure exists. The root unit is the tied embedding plus the final norm; its size is arithmetic from the vocabulary and hidden width, and everything else about it is the device's |
| **a mismatch means** | nothing yet — this step *creates* the rung. It is here because §10.2 says a memory-bearing choice is measured and then chosen, and this one has only ever been named. |

## Step 5 — the depth finding, re-measured (the first experiment, not the first hour)

Arm 1's recipe under full fine-tuning, at **both** checkpoints, scored on the 180-task split beside
their MLX counterparts.

| | |
|---|---|
| **produces** | full-split passes at each checkpoint, both arms |
| **compared against** | the MLX records: **175 of 180 at 1,200** against **159 of 180 at 800**, p = 6.1e-15 |
| **the criterion** | **full-split passes**, pre-registered. Validation loss monitors and does not select — it rose twelvefold between those checkpoints while the task score rose |
| **a difference means** | the finding, not a failure. Four variables move at once: Qwen3.5 against Gemma 3, 32 layers against 34 so "top 8" is 8/32 against 8/34, 4-bit against bf16, and adapters against full fine-tuning. **This cannot be a reproduction** and the record must say which of the four it is not controlling. |

---

## What is still unexecuted after that hour

NCCL beyond two ranks, any 27B configuration, CPU-offloaded optimizer states, and the arm-1
comparison itself until step 5 completes. The `!` supervision finding has a test recorded beside it
and that test has not been run: it needs one model load and is not this stream's to schedule.
