# Review ready: coverage fields do not make a verdict complete

**Codex, 2026-09-10.** Follow-up to `3494a2c`, under the new WS-A brief and method entry 34.
Sources: main `03d7f141b9cf02315a7dd688c2ecdc35cde390c5`, integration
`0f455e8c209860ed90b369e08a75973728e97401` (D-CRO work through `bf65124`).
Ten files are frozen with full commits, paths and hashes in `sources.json`.

**Verdict: two new concrete issues, and the capture-resume issue remains.** The checks below
execute only isolated, unchanged source expressions/functions on synthetic metadata. No production
module, numpy, torch or model is imported; no tensor shard, checkpoint, capture or device result is
read. `check.py` and `verify.py` reproduce the evidence. The existing approved tolerances are unchanged.

## F1 — The new repeat verdict ignores one of its two comparisons

`repeat_gate.py` genuinely invokes two fresh fits, separately from the saved map, so it fixes the
old self-comparison construction. It also reports a float32 no-op boundary first, and names its
limited layer/row coverage. Those are substantive improvements, not fresh execution evidence.

But its final `passes` expression reads only `against_saved_map`. It never consults
`within_process`. We evaluated the **unchanged verdict assignment** against five synthetic metric
sets, keeping both boundary results true and all layer keys present:

| comparison results | overall `passes` |
|---|---|
| both fresh fits and saved maps agree | true |
| second fresh fit disagrees at layer 1; first matches saved | **true** |
| second fresh fit disagrees at layer 33; first matches saved | **true** |
| first fresh fit disagrees with saved map at layer 1 | false |
| first fresh fit disagrees with saved map at layer 33 | false |

Thus a nondeterministic second fit can be visible in the report while the gate announces success.
The opposite mutations prove the predicate can reject; they locate the missing comparison rather
than merely producing a generic bad input. This is not evidence that any real fit was nondeterministic.

**Fix before this gate is relied on:** require the declared exact-repeat condition on both
comparisons at every expected layer, and test a failure in either comparison independently.
Preserve the passing/missing/failed scope alongside the overall verdict. Do not change a tolerance
to make a repeat pass. The no-op boundary has its own earlier refusal, which this isolated verdict
test does not execute or criticize.

The scope block's counts are useful, but the durable result should also bind the actual selected
row, token IDs/positions, saved-map digest, newly produced map identity, dtype, widths, settings and
source commit. “One row, two positions” alone is not enough to join this pass to the earlier map.
No real repeat result is in the new commits inspected here; it remains **unexecuted** in this review.

## F2 — The checkpoint field discards the weight identity already available

Both the new capture runner and the repeat script set `checkpoint_sha256` from
`report['sha256']['config.json']`. The package loader already computes and returns hashes for every
weight shard as well as the config (`hf_text.checkpoint_metadata` / `load_text_causal_lm`). The
consumers keep only the config digest.

The actual selector expressions were evaluated on two synthetic loader reports with the same
config hash and different weight-shard hashes. Every recorded `checkpoint_sha256` is identical.
This is a loss of information by field selection, **not a hash collision**. Two checkpoints may
share architecture/configuration while containing different weights, so this field cannot establish
the checkpoint identity promised by the capture contract or an exact-repeat provenance claim.

**Fix:** preserve and bind the loader's existing complete hash manifest, or the programme's existing
canonical checkpoint-identity mechanism derived from it. Name the config-only hash as config
identity if it is kept. Missing identity must not become an empty-string attestation. This requires
no new checkpoint download or extra hashing beyond the loader's report. Add same-config/different-
weights and missing-identity controls at the consumer boundary, not only in the loader's tests.

## F3 — New capture coverage is honest about manifest membership, not artifact validity

The writer now returns `requested`, `outstanding` and `complete`; the runner preserves the size of
the full capture set and states `whole_set` when limiting a pass. This closes a real reporting gap.
We confirm that a deliberately inert writer which emits no manifest line returns `complete=false`
and one outstanding decision. That negative control exercises the new membership check.

It does not close prior review C2. The new resume path still marks a requested key done before
checking checkpoint, prompt or shard. Against unchanged metadata functions, the fixture resumed:

| receipt state | already present | captured | outstanding | complete |
|---|---:|---:|---:|---|
| unchanged identity and synthetic receipt | 1 | 0 | 0 | true |
| changed checkpoint | 1 | 0 | 0 | **true** |
| changed prompt and matching new decision digest | 1 | 0 | 0 | **true** |
| missing receipt file | 1 | 0 | 0 | **true** |
| altered receipt bytes | 1 | 0 | 0 | **true** |

The callback is not invoked on any of these resumes. The source's coverage note explicitly means
manifest-line membership, and these numbers obey that definition. They do **not** establish that
the requested captures exist with the correct identity. Keep the scoped report, and add the identity
and integrity refusal before membership is accepted as completed capture work. The concrete
mismatches are the same C2 failure modes already reported, now shown against the new version rather
than counted again as new findings.

## What the new runner does resolve, and what it leaves open

The corpus seam now receives the whole row and tokenizes its rendered `prompt`; the agent corpus
never had an `ids` field. The author caught this mismatch with a real-row check. Our follow-up
fixture uses the declared row shape without IDs and succeeds. This is fixture evidence only; the
producer's real-row and native-suite evidence was read as source, not re-executed here.

The driver explicitly loads bf16, checks the observed first-parameter dtype, reshapes to batch one,
and tokenizes with `add_special_tokens=False` while refusing an initial doubled BOS. Those are
concrete constraints on this caller, so the prior generic-callback concern should not be read as a
claim that this particular driver necessarily runs a widened or promoted path. The generic writer
still stamps its own width/native labels and does not validate actual residual layout or dtype.
Its assertions cannot independently certify an arbitrary callback's computation (prior C1).

The input identity concern persists in the real interface. The fixture changes the rendered prompt
string while leaving `messages` unchanged. The changed prompt reaches the callback, and the new
capture completes under the same `prompt_sha256`. Changing messages instead is refused before
capture. The hash therefore continues to bind messages, not the text the real tokenizer consumes.
Prior C3/R4 now applies directly to the rendered-string interface, without supposing an IDs field
that the real corpus lacks. Preserve the semantic digest and add the consumed-input identity.

These corrections are due to the producer's intervening source work; this record neither claims
it caused them nor marks the entire capture contract verified.

## Ladder archive: source improvement, execution still unverified

The revised ladder stores six named arrays per rung/direction: plus, minus and zero outputs per
selected target position, requested displacement, and both realized displacements. It writes and
hashes a completed direction's archive before clearing those arrays, rather than waiting for the
entire experiment to finish. This implements the requested form in source. The native archive writer
was not executed, no new archive was inspected, and its shapes, coverage, response equality and
crash/restart behavior are **unexecuted checks** here. The bytes of a script are not the bytes of a
measurement. Those remain for the post-run audit.

## Gate inventory and verification coverage

`check.json` carries forward the eleven named checks from `3494a2c`, updates the receipts for G10
and G11, and adds G12 (both-repeat verdict) and G13 (checkpoint identity). It does not double-count
the capture-resume finding. The resulting **scoped inventory of 13** has one demonstrated rejection
of a producer's pinned documentary claim, seven checks unable to certify their full scientific or
identity target, and five unexecuted/unknown. A check may catch some wrong inputs and still miss the
particular failure its broader claim covers: G12 is an example. This is not a census of the full
programme. The new manifest-membership negative control is credited separately above; a coverage
field is not automatically a new scientific gate.

**Finding:** verdict and identity scope remain narrower than their scientific labels.
**Technique:** vary one protected condition at a time while holding all others valid, then assert
which reported verdict changes. Evaluate the emitter's real expression, not a rewritten version.
**Implementation:** AST-selected metadata code, list-returning spy and synthetic receipt writer;
source hashes, relocation reproduction and corruption controls in `verification.json`. Static lint
also passed on `check.py` and `verify.py` (ruff, F/E9).
No experiment or shared file was modified.

Run `python3 /absolute/path/to/this/record/check.py` and
`python3 -B /absolute/path/to/this/record/verify.py` to reproduce the file-only record.
