# Gemma 3's J-space: the map, and the instrument that carries it

**2026-09-08, open. Stage one has run; stage two has not. This record exists now because the
instrument evidence belongs beside every number the map will publish, not in a message.**

## The instrument proves itself at layer 34

The final layer's lens is the identity. At layer 34 the foreknowledge readout applies no transport
at all, so it must return the token the model actually emitted, at rank 1, every time. It does.

| | |
|---|---|
| horizon-1 reads at layer 34 | 5,316 |
| returned at rank 1 | 100.00% |
| agentic episodes covered | 11 |

This is a boundary condition rather than a result, and that is its whole value. Readout
orientation, lens application and position bookkeeping cannot all be wrong and still produce
1.000 across five thousand reads. It costs nothing to check, it could have failed, and it did
not. `identity_layer_check.py` recomputes it from the committed records with no model and no box;
`identity_layer_check.json` is stage one's answer.

## The depth gradient behind it, which is not yet a result

| layer | span | median horizon-1 rank | rank 1 |
|---:|---|---:|---:|
| 11 | sliding | 1331 | 0.40% |
| 12 | global | 1282 | 0.23% |
| 17 | sliding | 4128 | 0.19% |
| 18 | global | 2699 | 0.47% |
| 23 | sliding | 249 | 8.84% |
| 24 | global | 3 | 36.47% |
| 30 | global | 1 | 73.61% |
| 34 | sliding | 1 | 100.00% |

The shape is what a reader expects and is worth nothing on its own: eleven episodes of one model
under one rendering. **Nothing here reads the 23-against-24 or the 30 difference.** Those are
layer pairs whose contrast is the map's actual question, and answering it from this table would
be answering it from the pilot's convenience sample.

## What the map compares, and what it does not

The primary comparison is **calls against notes, by span and by depth**. It reads *attempts*, not
completions: what the residual carries when the model is about to act, whether or not the action
turns out to be right.

**No facet on success.** Stage one passed 2 of 11 agentic episodes. A facet on an outcome with
two positives is not a comparison, and it stays out at any rate the corrected runs produce. If the
step ceiling and the rendering lift the pass rate, that is a fact for this record and still not a
facet.

## Provenance of the numbers above

Stage one, 15 episodes across 12 families, 8 layers, Gemma 3 4B at 4-bit against the hosted
bf16-fitted Jacobian lens under R60, `--max-steps 12`, seed 20260902. Its records are the control
for every later comparison and are not modified.

**The runs after stage one are not comparable to it in one respect and the map must say so.** Two
corrections landed on 2026-09-08: the step ceiling moved from 12 to 24, which is the height that
produced every Qwen number the map is read against, and the observation rendering changed from a
bare re-role to the workspace answering the model's own call. Both are independently right and
neither was made to move a number. They landed together, so a run that carries both tests whether
Gemma can do these tasks when the harness and the rendering are fair to it, and does **not**
attribute the difference between them. A reader should not take it as doing so.
