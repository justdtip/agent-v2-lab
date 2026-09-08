# Gemma transcript lenses and dictionary bridge

Status: amended source implemented and verified with 190 small checks, native imports blocked.
No new model rollout, transcript fit or full dictionary readout has started. The Deputy's
stage-two map holds the machine; Task 1 generation remains next in our lane after that window.

**Later position-coverage amendment:** the pending fit now targets **2,816 tokens**, with an
actual fit-scored position requirement of **2,749**. See `POSITION-COVERAGE-AMENDMENT.md` and
`TRANSCRIPT-REGISTRATION-v3.json`. The earlier 2,048-token registrations and 190-check verification
below are retained as historical evidence; the new source checks are recorded separately.

## Finding, technique, implementation

**Finding so far.** The existing agentic corpus builder would drop turns longer than its cap,
repeat earlier history in the estimator, and reconstruct generated token IDs from decoded text.
Those are properties of the data preparation, not findings about Gemma. The old prose lens does
not answer the transcript-distribution or above-window questions.

**Technique.** Freeze the successful forward inputs separately from emitted tokens. Construct
bounded contexts from those IDs, count each newly introduced eligible position once, and obtain
residual pairs from each checkpoint's own forward on the same frozen contexts. Fit the unchanged
ridge grid at both precisions. Compare corresponding maps relative to their norms. This is native
replay with a declared context/window policy, not a claim that a batched replay reproduces every
live generation activation under its original partitions.

**Implementation.** Worktree `/Users/daniel.tipton/worktrees/gemma-transcript-bridge`, branch
`codex/gemma-transcript-bridge`, based on `3ef0d73`, with observation and lineage changes through
`7d2a18a` merged. Numeric bridge commit `80cc9ad`: 46 small NumPy tests passed with MLX imports
blocked; independent source review found no substantive numerical-core issue. Real acceptance
is still unrun. Work is committed in the home directory for crash recovery.

## Frozen decisions before any new model or matrix run

- Training-split evaluation tasks only, 24 steps, greedy, 200 emitted tokens per turn; no cache
  reuse. The exact cohort and generation provenance must be registered before launch.
- Observation gate: the landed fix at `7d2a18a`, lineage at `a960d80`, and confirming record
  `research/records/GEMMA3-JSPACE-MAP-2026-09-08/DIAGNOSTIC-RERUN.md`. That run shows the corrected
  rendering functions; all three tasks failed. Rendering and the step ceiling changed together,
  so it does not isolate a causal effect of rendering. Its episodes are not training data.
- Context cap 2,048. Record full-length and short contexts, scoring versus input counts, spans,
  and the number of scored positions actually beyond 1,024 tokens. Do not merely stamp a cap.
- Same frozen token IDs and ridge estimator for `gemma3-4b-bf16` and `gemma3-4b`; source `native`.
  Canonical base `google/gemma-3-4b-it`, no additional training, depth 34, width 2560.
- A1 dictionary: **every-layer suite** `resid_post_all/layer_17_width_16k_l0_big`, width 16,384,
  L0 target 120, HF revision `3e94b68be95290aada5b7525cf431d3040f81bb1`. Use the `big` variant
  consistently if later layers are authorized. Only this one dictionary is authorized now.
  This follows the order's amendment, not the exploratory cached deep-dive configuration.
- A1 lens: hosted Jacobian archive, digest
  `a14ab264fc47de9c59b3183a8783a60ca6f1dee2ea635e88223d5b2d67b03193`, matching probe layer 18.
  It was fitted on 128-token prose; it cannot support sliding/global layer claims.
- A1 named feature **0**, top **10** largest signed scores, lower token ID breaks ties.
  Negative control substitutes probe-layer **1** against the same feature/decoder/readout.
  No search for a favorable feature after observing whether that control bites.
- Independent reference: `(W_chunk @ J) @ d_0` in float64 from canonical float32 operands,
  compared with `W @ (J @ d_0)` through the production composed decoder. Complete score vectors
  must meet `abs(error) <= 0.001 + 0.0001 * abs(reference)` and relative L2 error <= 0.0001.
  Report reference norm and maximum absolute error. These bounds are fixed before real scores.
  The wrong-layer vector must fail the same gate and change the top-10 set; otherwise acceptance
  is unmet. Never silently widen the bounds or change the control.
- W is the raw tied unembedding weight. Scores omit final RMS normalization, soft caps, biases,
  and softmax. They are linear scores, not actual token probabilities or causal derivatives.

## Coordinates and labels

The pinned upstream SAELens loader at commit
`ee45e7406165ce267b33d33b7371bbb63ef24db8`,
[`pretrained_sae_loaders.py`](https://github.com/decoderesearch/SAELens/blob/ee45e7406165ce267b33d33b7371bbb63ef24db8/sae_lens/loading/pretrained_sae_loaders.py),
loads Gemma Scope 2 residual `w_dec` unchanged, with `normalize_activations: none` and no
finetuning scaling factor. Stored rows are feature directions; transpose once for D. Do not
renormalize them. Both hooks must be `model.layers.17.output`.

The amended order settles label availability. Every feature ships as `unlabelled`, with this exact
reason: no Neuronpedia source maps to `resid_post_all`; the residual labels index
`resid_post/layer_17_width_16k_l0_medium`, a different training run at a sparsity this suite never
published. No labels are borrowed across feature indices.

## Resources and remaining acceptance

No MLX import is needed for A1. The selected BF16 embedding is 262,208 by 2,560. Float32 W is
2.69 GB, D and JD are about 0.168 GB each. The loader and independent reference add bounded
scratch; loading every hosted map transiently adds about 0.865 GB. Never allocate the 17.2 GB
vocabulary-by-feature transfer. A1's full CPU multiplication requires an R61 window. A measured
small feature batch must establish a throughput projection rather than assuming the order's
150-second estimate applies to column-wise matrix-vector products.

The 12.44 GiB peak of the earlier 128-token fit is a starting datum, not a projection for 2,048.
Calibrate native replay plus scoring masks and all-layer sufficient statistics before announcing
the combined fit window. Include load, solve and serialization in the projection.

A2 waits for the transcript lens and run activations. B and the causal-abstraction programme are
not implemented. Small averaged-Jacobian readings can hide cancellation of strong contextual
sensitivities. Correlation across times with a retained transcript does not identify memory.

## Second amendment applied before any new rollout

Pulled `6fa62ec` and then `2edad6e`, which supplies the shared generated-token classifier. Capture
now refuses unless the running and primary source includes the rendering fix `7d2a18a`, environment
fix `020aa89`, and the pinned amendment and span implementation. Their committed evidence is bound
in `TRANSCRIPT-REGISTRATION-v2.json`. The two earlier registration drafts are retained and cannot
be launched by the v2 driver; `rejected01` failed its historical evidence check and neither ran.

Generated positions carry **note, call_skeleton, call_argument, chat_prose** from the landed
classifier. Input-role counts, including observations, are separate: removing an impossible
generated-observation rank facet does not remove observation activations from the fitting corpus.
Every ratio states its denominator. The original non-prose criterion retains agentic notes as an
eligible domain span even though their language is prose. Largest-episode and identical-call-run
shares use fitted positions, and either strictly above one third requires a ruling. Alternating
two-call loops can escape the identical-consecutive-call metric; it is not a general loop detector.

**Position and BOS decision.** Preserve captured token IDs, never prepend a new BOS. A window
starting at its source turn's beginning can retain the captured BOS; a later window does not invent
one. The manifest counts both cases and records the reason. It separately gives histograms of
original captured offsets and actual replay positions, for scored tokens and all inputs. Each
fresh replay starts at position zero; an original offset is not falsely reported as a replay RoPE
position. Repeated context appears in input histograms but is not repeatedly scored.

The frozen plan is 72 training tasks: 24 in each registered training split/difficulty. Both fits
consume the same generated corpus and score masks, at 2,048 tokens, through native forwards.
Per-layer precision comparison reports both norms, the difference, both relative denominators and
cosine. The model identity remains the untrained Gemma 3 4B base with 34 layers.

## The direct-readout convention is an open measurement

`examples-download.json` authenticates the single selected dictionary's companion with SHA256
`2947b2bd64234f61aef694e7edd3d8c5e7437c5af10b65bc07b3fd298c97dd7d`. Its size is only descriptive.
Only `top_tokens` and `top_logits` are materialized by A1; the activation corpus stays on disk.

`A1-REGISTRATION-v2.json` adds the first numerical check: raw `W d_i` against shipped top tokens
on predeclared features 0–15. Exact set membership is the compatibility criterion; order and numeric
score differences are also recorded. The named-feature independent-product acceptance runs next;
full readout then compares all feature directions before any production `W J d_i` columns,
preserving overlap evidence and stopping if compatibility fails. The benchmark times both direct
and lensed passes, including the actual companion comparison and serialization. This gate
does not validate J, its source-layer indexing or decoded token text. Existing independent-product
and wrong-layer controls retain their original bounds.

**Finding about the instrument, not the model.** Google's
[official tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)
uses `W diag(stored_norm_weight) d`, while the actual Gemma RMSNorm channel gain is
`1 + stored_norm_weight`. The tutorial loads 1B pretrained Gemma, without a version pin, and is
a consumer of examples, not the generation source for this 4B instruction-tuned artifact. Neither
convention is established as the generator of the shipped scores. A raw-W mismatch is therefore
unresolved readout compatibility, not proof that the decoder was transposed wrongly.

**Technique.** Hold decoder, token indices and readout fixed; compare three declared channel gains:
identity, stored norm weight, and one plus stored weight. Report the overlap without choosing a
convention after seeing a favorable answer. None includes the residual-dependent RMS denominator.

**Implementation.** On the Director's explicit authorization, `final-norm-weights.npz` stores
2,560 BF16 values widened exactly to FP32, plus the effective gain. Source-shard and vector hashes,
the tensor offset, and the local RMSNorm definition are in `final-norm-weights.json`. Extraction
read a 5,120-byte tensor as data, performed zero forwards and imported no native framework.
`upstream-readout-source.json` records the tutorial's source hash and bounded relevant excerpts.
`readout_conventions.py` is ready for the owned CPU window; **no real convention overlap is yet
measured**. It is a diagnostic, not a replacement for the registered raw-W gate.

## Corrections carried from the confound register

The earlier 128-token prose/hosted comparison did not match position composition: the hosted fit
used document-initial, BOS-led positions, while almost every regression window began mid-text.
That remains an alternative explanation for the comparison's depth pattern. Matching window length
did not control it. Hosted source-layer indexing also remains an assumption pending fitting-source
evidence; geometry and a wrong-layer matrix control cannot settle the offset.

The ridge grid selected its smallest candidate at every layer and bracketed no optimum. It is
evidence against a variance-limited regime over the tested range, not evidence of optimal tuning.
Relative Frobenius difference is determined by norm ratio and cosine; it is not independent evidence.
The register attributes 52.1% of that difference's depth trend to norm drift, while 82% of the raw
cosine trend survives removing the shared identity component. These qualify the old reading; no
immutable fitted artifact is rewritten.

The register's initial claim that the environment explains missing paths was itself falsified.
Search can recover a task's full path. One tested episode did not switch strategy under repeated
honest failures; that is the supported scope, not a general inability to copy paths.

## Verification and remaining work

190 integrated checks passed with a meta-path blocker in the parent and failing `mlx`/`mlx_lm`
packages on subprocess PYTHONPATH. They cover deterministic token ownership, capture/readback,
composition gates, masks and calibration, exact BF16 conversion, dictionary provenance, direct
readout refusal, all-feature ordering and window ownership. Ruff and diff checks passed. Independent
source review of the transcript amendment found no actionable issue. This is source evidence only.

The pending window request and conservative projections are in `WINDOW-REQUEST.md`. Use the
persistent `runlock run` wrapper, or pass a live `--holder-pid` to `announce`. Read status back as
running before model work; drivers additionally check the live owned holder. Never clear another
seat's window. Actual rollout composition, calibration peak, both fits, precision differences,
and the real A1 baseline are still unmeasured. A green source suite does not substitute for them.
