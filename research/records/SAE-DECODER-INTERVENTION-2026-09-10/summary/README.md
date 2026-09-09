# Summary diagnostics for repeated SAE interventions

Order: third 2026-09-10 heading of the WS-A order, read on
`origin/codex/agent-v2-specs`; implementation starts from integration `e8028bb`.
The full wrapper at `ff3b1b4` was reviewed and merged at `e8dbcc3`.

Plan: add the opt-in summary mode, prove that it reduces the full diagnostic
without changing the edit, verify capture snapshots both forms, and publish the
source and CPU fixture evidence. No checkpoint, CUDA or native MLX execution.

Contract: full remains the default with its existing keys and kind. Summary keeps
selected-feature readings, uses kind `sae_decoder_reencoding_summary`, and replaces
the two off-target arrays with `off_target_summary`: `nonzero_count`, `max_abs`,
`l1` and `top_changes`. Pairs are ordered by decreasing absolute change, then
increasing feature index for ties. Up to eight entries are retained, including
zeros if fewer than eight entries change. An empty complement has zero statistics
and no pairs. This bounds retained per-event off-target data; it does not claim
constant transient computation memory or bounded total event history.

Status: ready for review. Source `d32cc28`: **132 passed, 0 skipped, CPU; desktop
not certified idle**, native MLX imports blocked. No box window or model lock was
held at the pre-run reading. Ruff and source whitespace checks passed.


## Finding and technique

The summary is an exact reduction of the full readings on the fixtures, while
retaining only selected-feature arrays plus up to eight off-target pairs. In the
existing counterexample the feature starts at 1, is requested at 3, re-encodes to
5, and changes the other feature by 2. The summary reports nonzero count 1,
maximum 2, L1 norm 2 and pair `[1, 2]`; `example.json` records both actual forms.
This remains a finding about the instrument, not evidence of feature control.

Reduce the detached change vector before converting it to Python lists. Count
strictly nonzero entries without a tolerance, sum absolute values in CPU float64,
and rank signed changes by magnitude with feature-index tie breaking. Keep the
selected readings intact. This bounds retained off-target data per event; encoding,
transient tensors and sorting still scale with dictionary width, and the number of
events still scales with the run. A non-finite aggregate fails without publishing
partial or stale evidence. Summary statistics cannot recover the omitted vector.

## Use and verification

Pass `diagnostic="summary"` when constructing `SAEIntervention`. No capture API or
call-site migration is needed: capture already snapshots the callable's diagnostic.
Default and explicit `full` keep the pre-existing kind `sae_decoder_reencoding` and
all existing fields; summary's distinct kind identifies the reduction.

Acceptance includes exact full/summary comparison, zero and empty complements,
more than eight changes, signed ties, unchanged gradients, capture snapshots,
reusable encoder buffers, invalid modes and aggregate overflow. A negative-control
test prohibits dictionary-width `Tensor.tolist()` conversions: summary passes,
and full triggers the guard. This directly tests the storage hazard rather than
inferring it from the small output example.

Re-run with the existing CPU evidence runner:

```sh
PYTHONPATH="$PWD/src:$PWD" LLL_BACKEND=torch LLL_DEVICE=cpu \
  /path/to/python research/records/SAE-DECODER-INTERVENTION-2026-09-10/verify.py summary/rerun \
  tests/test_sae_intervention.py tests/test_torch_capture.py tests/test_repository_rules.py
```

`acceptance.json` retains its actual run-start commit, dirty `e8028bb`; that source
was unchanged until committed as `d32cc28`. `VERIFICATION.json` binds the unchanged
source hashes and raw evidence to the resulting commit. The missing-option red
check and intermediate focused pass are preserved separately. Raw logs/XML retain
original whitespace; source, documentation and JSON checks passed.

The work is ready for the Chief's review. No further laptop work is ordered;
CUDA, real checkpoints and corpus-scale clamp records remain unexecuted. The
device checklist is next after review, and the heartbeat remains disabled.
