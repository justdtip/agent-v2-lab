# The newly landed capture writer: checks on labels are not checks on the run

**Codex, 2026-09-10. Source `144fa1cf1f753c008daf1eb4f38282a1e71184d0`.**
This implementation landed while the requested registration review was being written. The source
and its tests are frozen here. We did **not** import the production module, run native tests,
forward a model, or write a real tensor shard. `audit.capture_metadata_fixture` extracts five
unchanged metadata functions and the exception/constants by AST, supplies a list-returning spy
callback, and replaces only the shard writer with an in-memory collector. Resume cases use tiny
synthetic manifest and receipt files in a temporary directory. Thus this is executable evidence
about validation and identity, not a model-path or serialization measurement.

## C1 — The writer overwrites the evidence the path guard would need

`capture_decisions` writes `forward_batch = FORWARD_BATCH`, `anchor_batch = FORWARD_BATCH` and
`capture_dtype = CAPTURE_DTYPE`, then calls `assert_contract` to compare those labels to the same
constants. The callback supplies no required observed width and no verified native-path identity.
Its returned dtype is stored separately as `dtype` but never compared with the checkpoint's native
dtype. A caller that actually expands a batch or promotes arithmetic cannot be diagnosed by this
check.

The fixture makes this distinction explicit. A manually corrupted cell with width 64 is refused.
But a callback reporting width 64, anchor width 64, promoted-float32 capture and dtype float32 is
accepted and recorded as **width 1, anchor width 1, native**, with float32 still in the separate
dtype field. This does not show that a real callback currently runs that path; it shows that the
promised guard cannot establish which path it ran. The existing tests mutate the cell label and
use only a conforming callback, so they miss the producer's erasure of conflicting evidence.

The same shape affects token-index type: callback `token_index = 1.9` becomes integer 1 before
validation and passes. Require measured, typed seam provenance, with expected values declared
independently; validate before coercion. When native is a symbolic mode, bind its actual expected
dtype and the model/view identity. Validate residual shape and dtype at the writer boundary and
retain actual row ordinals. The current decision copy omits `row_ordinals` although §2 requires them.

## C2 — Resume trusts the key, not the checkpoint, prompt or shard

`_already_captured` reads only task, step and shard number. `pending` removes those keys **before**
any corpus or prompt check. It verifies neither the old manifest's checkpoint nor the shard's
existence/hash. This is separate from the corrected partial-shard numbering, which is a useful fix.

Five model-free cases were tried: an unchanged receipt; a changed target checkpoint; a changed
prompt and matching new decision digest; a missing shard; and a shard with changed bytes. **All
five report `already_present = 1`, `captured = 0`**, and the callback is never called. The four
mismatches therefore pass as completed work. No inference about a current on-device archive follows;
this is a demonstrated failure mode of the resume implementation.

Before treating a key as done, validate its full capture identity against the requested run,
including checkpoint, entry, input identity, precision, widths, position contract and declared
layer layout, and check its shard's actual digest and index/shape coverage. A mixed or missing
receipt should refuse with its reason. Do not silently recapture over old files. Preserve the
no-overwrite behavior already added for completed partial shards. Add four tests that alter one
resume precondition at a time and assert refusal before accepting completion.

## C3 — The new input check reproduces the message hash, not the tokens consumed

The digest check correctly refuses changed messages before calling the spy. But it passes different
`row['ids']` under unchanged messages and the same capture-set digest. Those different IDs are
handed to the callback with no second identity check. This confirms the companion review's R4 in
the new implementation: it checks a semantic message record, not the exact model input.

Bind and verify the rendered/token input at the capture boundary, preserving the existing semantic
hash under its actual meaning. Specify how the prompt prefix and P_note index are obtained from a
supervised row that also carries a completion. The present callback receives only `row['ids']` and
returns its chosen index. This review does **not** claim that a real driver captured a target token;
that driver and its independent prefix-boundary check were not supplied. Such a check is required
before the seam can certify a decision-position capture.

**Verdict: revise these guards before relying on this writer's capture attestation or resume.**
The evidence and failing controls belong in the capture implementation's review. No existing
experiment record was modified and no running capture was touched.
