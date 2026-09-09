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
  the unit that owns it. Touched outside, it raised — and the error read as a refusal of the design
  rather than as a statement about *where* the access happened.
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
mark immediately after its end-of-turn marker — an ordinary emittable token, at exactly the position
where generation decides to stop.

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

# Unexecuted

**Nothing in this record has run on a GPU.** No CUDA, no NCCL, no more than two processes, no real
checkpoint at four billion parameters, and **no memory figure measured on any device**. The
arithmetic for memory is arithmetic, not measurement. The multi-device agreement is two CPU
processes over a local backend. The re-measurement of the depth finding, the memory measurement, the
sharding fallback and the real communication backend are all the device's, and are the four things
this stream cannot answer here.
