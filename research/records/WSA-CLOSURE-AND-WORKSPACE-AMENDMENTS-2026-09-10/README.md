# Accepted fixes verified; two resume limits and two workspace amendments remain

**Codex, 2026-09-10. Review ready. Source and synthetic evidence only.** The original repeat-verdict and checkpoint-identity failures are closed in the inspected source. The capture writer now rejects mismatches it previously relabelled, and resume rejects all the original checkpoint, prompt and shard failures. The workspace plan's change to an observational screen is appropriate. Its newly specified PCA comparison and transfer null need the corrections below before their results are interpreted.

This review freezes main at **c9c3fc8**, integration at **a694c42**, and the D-CRO source incorporated there at **7745327**. It follows **3494a2c** and **332bdc9**, both accepted and merged through **3935142**. The source fixes are **a76bf88** and **7745327**; the workspace rulings are **1b17665**. Full commits, original paths, hashes and sizes are in [sources.json](sources.json). The current WS-A order, primer and METHOD entry were checked: unchanged from the previous review. The new WS-D and workspace rulings were read before analysis.

**No workspace capture results were read.** Their committed record is a skeleton saying captures are in progress and results have not been read. The earlier three-row test that informed W-4 remains data-informed, as the amendment now discloses. No claim of a wholly prospective programme is made. No model, tokenizer, checkpoint, native suite, real tensor archive, device or process was accessed by this review.

## What now passes the independent source checks

| Earlier finding | Fresh evidence on the revised producer source | Disposition |
|---|---|---|
| F1: repeat verdict ignored the second fresh fit | Evaluate the actual per-layer and final verdict expressions. Baseline passes; make either comparison nonzero independently at either of the two declared layers: all four cases fail. | Closed for verdict logic. Real repeat execution remains unexecuted here. |
| F2: config hash presented as checkpoint identity | Exercise each actual consumer call with the unchanged helper. Changing either weight-shard hash or the config hash changes the identity; reordering files does not. Each consumer refuses absent, empty, config-only and weights-only manifests. | Closed under the loader's complete-manifest contract. |
| F3: resume trusted manifest membership | Clean resume succeeds without a callback. Changed checkpoint, changed semantic messages, changed rendered bytes, missing shard and altered shard bytes each refuse before forwarding. | The original failing cases are closed. The additional request/input limits below remain. |
| C1: writer stamped its own contract constants | An honest seam reporting either width as 64, a promoted path, invalid position, wrong rendered bytes, mismatched token length or wrong residual shape now refuses before writing a manifest. Ten controls executed. | Closed at the seam-to-writer metadata boundary; this does not independently establish the model's internal arithmetic. |
| C3/R4: one digest described different objects | Fresh cells separately record semantic messages, rendered prompt bytes, and consumed-token digest/length. Row ordinals are retained. The renamed enumeration has 7,629 entries, the same logical hash and the declared new file hash. | The naming and fresh-record provenance defects are closed. Token identity on resume is narrower than fresh capture. |

The repeat row now names its selected row, token hash and length, positions, saved-file hash, fresh-array hash, dtype, widths, arithmetic settings and source commit. That is a source inspection, not an executed binding test. No threshold has moved. The checkpoint helper consumes all hashes already provided by the loader; no new checkpoint download or model run is needed for this fix.

## R1. Resume does not compare an existing cell with the new capture request

`_verified_captures` compares the old cell with the current corpus and target checkpoint, but receives **no decisions**. `pending` then removes any requested `(task_id, step)` found in that verified set. Consequently, the requested decision's `messages_sha256` is checked only when that decision is pending.

The fixture keeps the checkpoint, corpus, receipt and old manifest unchanged, then changes just the new decision's expected message digest. A fresh capture refuses that request. A resumed capture reports `complete: true, verified: 1` without calling the seam. The old record and corpus agree; the current request is the object that was never consulted.

**Required source amendment:** validate each requested decision against its corpus row before either the reuse or new-capture branch. Bind the reused cell to the request's identity and relevant enumeration metadata. A current request that fails fresh capture must not become acceptable merely because a manifest exists. Preserve the named refusal; do not silently replace producer records.

This is an additional scope limit, not a claim that the original checkpoint/prompt/shard fixes failed.

## R2. A token digest is recorded on fresh capture but is not verified on resume

Two tokenizers can produce different IDs for the same rendered bytes. The loader's identity covers `config.json` and the safetensors shards; it does not cover tokenizer assets. The current resume check compares messages, rendered bytes, checkpoint and receipt, but neither computes nor compares the IDs the current tokenizer would supply.

In the fixture, two fresh preparations of the same text and weight identity return `[10,20,30]` and `[10,21,30]`. Their fresh cells have different token hashes, correctly. Resuming the first cell with the second preparation still returns `complete: true, verified: 1`; the preparation is never consulted. These are inert callbacks representing a changed tokenization, not two real tokenizer executions.

**Required source amendment:** prepare the exact model input before deciding to skip. This preparation requires tokenization, not a model forward. Compare its token digest and length with the existing cell, then pass the same prepared IDs to any needed forward. Freezing tokenizer assets and configuration helps provenance; checking the prepared IDs verifies the actual object named by `token_ids_sha256`.

Fresh cells are substantially better identified now. Until this check exists, describe resume's `verified` scope as checkpoint, semantic and rendered input, and file integrity; it does not establish current consumed-token equivalence. The prose in the pre-registration saying resume verifies both rendered bytes and consumed IDs is broader than the implementation.

## M1. The newly declared PCA restriction breaks the claimed equivalence of the readout classes

The amendment specifies **PCA fitted on the training fold, retaining r components**, followed by a ridge readout. That is a coherent definition of r. But it then says an invertible lens gives the same rank-r readout class on `Jh` and `h`, with any difference attributable to conditioning and regularisation. The same-class statement applies to freely learned linear maps, not to a projection fixed by PCA.

A complete two-dimensional example makes the distinction concrete. Let the four equally weighted activation vectors be `(−2,−1), (−2,1), (2,−1), (2,1)`. Let the action label be the sign of the second coordinate.

* The coordinate variances are 4 and 1. PCA keeping one component keeps the first coordinate. Each retained value has both labels equally often; no readout from that component can exceed 50% accuracy on this distribution.
* Apply the invertible lens `J = diag(1,3)`. The variances become 4 and 9. PCA keeping one component now retains the second coordinate. A linear readout separates the labels perfectly.

No information was added to the full vector. **The projection retained different information.** There are no eigenvalue ties, finite-precision effects, or regularisation choices behind this counterexample. The example is enumerated in `check.json:mathematics.pca`.

**Correction, including to the scope of my earlier observation:** invertibility preserves the class of unrestricted linear predictors, and a freely selected rank restriction can be transported through the inverse. It does not preserve the class produced by separately fitted, truncated PCA. For this order's pipeline, differences can be caused by which signal PCA retains, as well as regularisation and conditioning. They are not evidence that the lens created information or discovered functional availability. Keep the PCA design if that is the intended question; describe it as the decodability of information retained by that specified pipeline.

## M2. Moving evaluation to held-out families does not repair the constant-family permutation null

The new transfer null permutes labels within the training families, then evaluates on held-out families. If each training family uses one action label, **every such permutation leaves all training labels unchanged**. Changing the evaluation population does not alter that fact. The purported null and the ordinary fit then receive exactly the same training inputs and labels.

The fixture uses two training families with three decisions each, all label 0 in the first and all label 1 in the second. All 36 permutations of the row indices preserve the training-label assignment. No model or fitted probe is needed to establish the degeneracy. This is the same structure identified in the first review; the amendment relocates evaluation without disrupting that structure.

**Ruling needed before inference:** mark this null uninformative for family-constant strata and freeze a null that actually disrupts the hypothesised cross-family alignment while preserving the intended nuisance structure. Define its hypothesis and class-support restrictions. Do not silently substitute unrestricted shuffling, which would answer a different question. For nonconstant families, retain the current null only for the conditional association it actually tests, with episode structure respected. No statement here establishes how often the real corpus has the counterexample's structure.

## Scope of W-5 and of the accepted design changes

The renaming to confidence crossings, readout competitors, direct attention to tagged spans, and expert-action decodability is appropriate. Reserving ignition, functional availability and report-versus-computation claims for separate interventions resolves the central overclaim. The correction to single-edge long-range routes, explicit teacher forcing, tails and unresolved counts, data-informed W-4 provenance, and the whole-dictionary orthogonal-rotation null all address the earlier criticisms at the design level. Their implementations and results are not reviewed here.

W-5's readout agreement by position is a useful **operational readout diagnostic**. It must not turn back into a derivative-validity or refitting gate: agreement with the model's current output does not identify its derivative. For the toy scalar tail `f(x)=x²+3` at `x=2`, the exact derivative is 4, giving `Jx=8` against output 7. The incorrect map 3.5 gives perfect output agreement. The offset is what an unanchored linear readout omits. The record reproduces this example without claiming it is Gemma's measured behaviour.

Accordingly, report W-5 by source layer as well as position band, with the exact readout rule, relevant mass/unresolved counts, and the model-output comparison it actually makes. It measures predictive readout behaviour in those bands. It does not on its own isolate a causal effect of extrapolating position, or certify the Jacobian. If “last-layer lens” means testing only the final source layer, it supplies no check of earlier source maps; the implementation must make this unambiguous.

The plan-progress confidence-bound and contrast-width rulings requested in **3494a2c** remain outstanding in these inspected changes. This capture closure is not a seal of that pre-registration or approval to claim its confidence guarantees.

## Technique, implementation and verification

**Transferable technique:** compare the same request through fresh and resumed paths; mutate the upstream input a verdict claims to certify; distinguish lossless changes of coordinates from the information discarded by a fitted projection; and check that a null changes the relationship it is meant to destroy. These techniques diagnose the measurement instrument independently of the model.

**Implementation:** `check.py` extracts only selected metadata functions and verdict expressions from frozen producer source. It uses an inert seam, objects with declared shapes, and tiny synthetic receipt files in temporary directories. It never imports the producer modules. Native/model imports are blocked. Its mathematics is finite enumeration and elementary arithmetic. The frozen `capture-set.jsonl` is decision metadata, not model activations.

Run `python3 -B check.py` to reproduce `check.json`; run `python3 -B verify.py` to reproduce `verification.json`. Verification checks all 16 source hashes, refuses a corruption of each, reproduces the output from a relocated copy, and detects three deliberately restored defects: ignoring the within-process repeat, trusting manifest membership on resume, and passing a config-only identity. Static checks are limited to the two review scripts.

The same **13 named gates** are traced from the previous records, without inventing a programme-wide denominator: **4 demonstrated able to disagree within their tested scope, 5 unable to certify their full claimed target, 4 unexecuted or unknown**. G9, G12 and G13 now have positive negative-control evidence. G10/G11 retain the additional resume limits; G8's newly specified null has a structural counterexample. Every gate's scope, lineage and evidence is in `check.json:gate_ledger`. These counts describe reviewed instruments, never model behaviour.
