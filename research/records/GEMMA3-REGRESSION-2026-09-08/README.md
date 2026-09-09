# Gemma 3 4B native BF16 regression — registration

Registered before checkpoint execution on 8 September 2026. Director authorizes all-layer BF16
regression with residual_source=native. The hand-run mask defect does not gate this source and
no Jacobian fit is authorized through the defective path. This run uses the official text-only
BF16 conversion; source checkpoint and tokenizer assets are hashed in checkpoint-identity.json.

## Finding sought

Measure where along normalized decoder depth a ridge predictor of final residual values differs
from the independently hosted Jacobian transport. This is a comparison of instruments with
different objectives, not a correctness gate or a demonstration of causal features. No sliding/global
contrast is inferred: both this initial fit and the hosted fit use 128 tokens, below the 1,024 window.
The hosted lens used 546 wikitext train prompts; our authorized prose is the validation shard.

## Technique

Use each complete block's output from the model's own forward, before final normalization. Fit
every nonfinal residual index (1..33 here), with the final block output as the target; block 34 is
identity. Stream one unpadded forward per 128-token sequence. Accumulate float32 sufficient sums
separately for fit and selection sets. Solve the unchanged ridge grid 0.001,0.01,0.1,1,10 times
mean(diag(total XTX)); choose the first minimum selection relative SSE. Selection rows are not
independent generalization evidence. All positions enter the regression, including the first BOS.

Frozen corpus: 2,010 consecutive nonoverlapping windows, 1,608 fit (205,824 positions), 402
selection (51,456 positions), every fifth one-based window held. Same pinned validation text and
Gemma tokenizer; 11 trailing tokens discarded. Corpus manifest and source hashes travel with the
record. No pilot observations or generated continuations enter this fit.

After the artifact is written, compare all 33 maps in their common output-by-input convention.
For each layer report normalized depth, both Frobenius norms, each distance from identity divided
by sqrt(hidden size), cosine between flattened maps, and ||J_reg-J_hosted||F/||J_hosted||F.
Final identity is the sanity baseline. These are descriptive matrix comparisons; they do not by
themselves measure token-readout agreement or justify calling a representation absent. No layer is
selected or omitted after seeing results. Record every mismatch in model identity/precision/corpus.

## Runtime and implementation

Clean home worktree codex/gemma-lens-fitting based on landed tree a584acd plus corpus preparation
7db26d4. Before model load, repaired two native-source progress failures, local tokenizer path
handling, and preservation of the checkpoint snapshot alongside the lens identity. The original
CPU reproductions failed; 72 affected tests pass with actual MLX imports forbidden. A NumPy-backed
CLI fixture exercises complete fitting, progress and archive creation; it is not a native GPU test.

The supplied primary checkout did not yet have gemma3-4b-bf16.yaml, so this worktree explicitly
registers that name against the Director-authorized converted directory, with the landed Gemma
chat fields. No weights or shared registry files are changed. This local alias is recorded rather
than treated as the official Hub name. The conversion record supplies the link to Google's model.

One declared model window, native residuals, cache strategy none, allocator cache zero. Absolute
MLX allocation ceiling 14 GiB, reduced if it exceeds 90% of the runtime's recommended working set.
R47 requires a declared window above 60%, not a permanent 60% cap; this run declares one. The
BF16 model is approximately 7.3 GiB and the four all-layer float32 moment matrices are 3.223 GiB;
14 GiB leaves headroom for short-context activations, solve work and artifact verification.
One 65-minute alarm covers load, fit and save; the original fit target is under an hour. Stop on
allocation failure, nonfinite statistics/maps, failed integrity, or deadline. No tolerance or
sample count is relaxed on failure. The normal shared lock and window remain authoritative.

The live record is run.log. run-end.json records outcome and whether an artifact exists. No full
model suite runs during the window. A completed patch is preparation; the completed lens,
comparison and report are the research deliverable. Scope composition remains deferred.

## `compare_maps.py` is as-run and no longer runs, 2026-09-09

The script in this directory constructs `LensIdentity` with three positional arguments in the order
`(name, hf_id, num_layers)`. That was the signature when it ran. Commit `a960d80` applied the lineage
rule afterwards, replacing the first two fields with a single `base` and appending `training`, so the
same call now binds a filesystem path to `num_layers` and a layer count to `training` and raises.

The script's own output dates it: `comparison.json` records `regression_identity` and
`hosted_identity` in the old `{"name", "hf_id", "num_layers"}` shape, which the current class cannot
produce. So the results here were produced by this file against the older class, and both are
faithful to what ran.

**It is deliberately not repaired.** Rewriting an as-run script inside a record to match a signature
it never saw would make the record claim something ran that did not, and the digests in
`SHA256SUMS.json` and `source-hashes.json` bind this text to this run. A reader who wants the same
comparison against the current class should write it beside this one and say so, rather than edit
here.

`LensIdentity` is keyword-only as of `57a0b89`, which is the mechanism that stops the next field
change from being noticed only when someone opens a record. Two-argument positional callers had
survived `a960d80` by coincidence, which is why nothing flagged this file for a year of commits.
The refusal is tested in `tests/test_live_lens.py`, including the exact three-positional shape used
here.
