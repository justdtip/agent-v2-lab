# Jacobian and replay integration notes — 7 September 2026

These are source-derived invariants for the next implementation briefs, not runtime equivalence evidence.

## Jacobian boundary

`ArchitectureView.embed` and `run_block` produce float32; the pre-norm target is obtained by executing blocks without `final_norm`. The installed `jacobian_vector_product` computes epsilon from the entire primal tensor. Therefore each basis direction in a broadcast batch must retain the same epsilon as its single-direction full-sequence reference; neither the shorter cached shape nor the square root of the direction batch size may rescale it. Stack responses as input-basis rows, then transpose for the conventional Jacobian stored by LensMaps.

Prepare a complete prefix cache through all blocks. Use a separate clone to derive the current-position layer residual; never partly advance the snapshot from which the tail builds masks, since `view.masks` finds the first cache of each kind, which can be before the requested layer. The installed ArraysCache includes convolution history, recurrence state, lengths and left_padding; KVCache includes keys, values and offset. Broadcasting operates on batch axes and leaves sequence offsets unchanged.

Position zero needs an explicit fresh-empty-cache path. The existing `clone_prompt_cache` evaluates `entry.state`, but installed empty KVCache.state accesses keys.shape when keys is None. That helper's nonempty history-cache use is valid; a Jacobian caller must not assume it clones an empty prefix. Fresh caches at zero avoid changing the reviewed runner helper.

The map validator already documents mean-to-mean comparison: a single local derivative cannot be compared to a corpus-averaged lens. Use the same span/source/target averaging on held positions and mark unstable finite differences inconclusive through response_agreement; only layer_verdict owns refit requests.

## Replay boundary

CaptureSession now owns a ForwardLedger. It rejects any turn_cache, records successful forwards separately from emitted events, and permits generator lookahead. A new replay must drive its public generation/emitted API, preserve each recorded input partition and emission order, and assert exact native logit hashes. Resolve cache_strategy none before load_policy, per requirements §13.

The old `read_record` accepts plain text only, while the committed pilot records are gzip. Add a narrow compressed-input adapter with the same chain/footer checks; do not weaken validation. The old atlas script scans only plain JSONL files and labels episodes by filename stem. Its legacy identity scope needs the same episode labels and original manifest metadata (especially seconds and success), otherwise non-scientific metadata changes can masquerade as failure. New replay provenance should remain in a separate replay manifest rather than rewriting the original episode identity.

Legacy atlas totals deliberately pool tokens and include a layer-20 breakdown. Preserve that exact calculation only for the requested hosted identity check. New profiles must be episode-separated and include final-layer rows for matching base rates even though fitted maps omit the identity layer. No new primary layer follows from the legacy schema.

## Runtime envelope

R47(b), now in the shared wiring map, requires projecting the next batch/size peak before measuring it; a projection above 0.6 of the working set needs the Director's declared window. Readout byte headroom cannot certify safety against the Metal live-buffer count cap (R55). Source code can implement the projection and refusal path without a checkpoint. Before native work, use the primary lock and mapped-library inventory, never old observations or process-name guesses.
