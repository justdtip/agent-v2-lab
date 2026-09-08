# R47 correction: batch one was already in use

The 21 GiB proposal is withdrawn. Comparing it with the registry's 22 GiB budget was an
operational error: the applicable ordinary-run threshold is **0.6 of the recommended working
set**, not the registry budget. The previous log records a working set of 19,069,665,280 bytes;
its threshold is **10.656005859375 GiB**. No transcript rollout, new fit, or calibration was
launched under the 16 / 14.5 / 21 GiB proposals. All three exceed this threshold and are withdrawn
as launch allowances. The historical registrations and patches remain unchanged for audit.

## Findings and limits

The fitter already uses one unpadded sequence per forward. There is no batch of sixteen to
reduce. The 12.443628 GiB peak of the earlier 128-token native bf16 fit was reached during
accumulation, at sequence 1,640, before any ridge solve. Moving the solver alone cannot explain
or remove that peak. Its old 14 GiB allocation ceiling did not establish compliance with R47.

Two causes require different remedies:

1. Both fit and held statistics coexist: 33 layers, two splits, two 2,560-square FP32 matrices
   per split/layer, or **3.222656 GiB** plus scalar energies. The bf16 checkpoint's tensor-file
   size is about **7.227608 GiB**, rather than the roughly 4 GiB model allowance in the batching
   argument. Their sum is **10.450264 GiB** before forward scratch. File size is a weight-memory
   proxy, not a measured resident allocation; lazy loading and allocator behavior matter.
2. Native residual capture has a confirmed Python ownership bug. A dynamically defined collector
   class closes over its residual dictionary; class cycles can keep those arrays alive until
   cyclic garbage collection. The old log contains peak increments of exactly
   **0.020751953125 GiB**, equal to 34 x 128 x 2,560 bf16 elements. A weak-reference test on the
   actual method reproduces delayed release without importing MLX. This confirms the lifetime
   defect, but the fraction of the real peak it caused remains unmeasured.

The previous extrapolation also imported vocabulary-logit buffers from a different path.
Native residual fitting discards the model's output and evaluates residual-derived statistics;
the transcript ledger explicitly evaluates returned logits. Full vocabulary tensors therefore
cannot be assumed live in the fitting path. Likewise, a fused attention call does not establish
that two dense attention matrices are simultaneously resident. The 21 GiB total was not a
reliable explanation of the observed allocation.

## Technique, independent of this implementation

Separate live-memory accounting into weights, sufficient statistics, retained activations,
temporary arithmetic, allocator cache, and host-side export/validation. Measure phases rather
than attributing one process maximum to every phase. Test ownership independently with weak
references and automatic cycle collection disabled; arrays should survive while the caller owns
them and release as soon as that ownership ends, including exception paths.

For a stateless regression forward, partition the frozen rows by fit/held membership while
preserving row order within each split. Accumulate just one split's sufficient statistics,
evaluate each layer's update before proceeding, write the completed statistics to temporary
files, then release them before the other split. Solve from one layer's fit/held pair at a time.
Every window still receives exactly one forward; every selected position contributes once to
the same FP32 sum in the same within-split order. The ridge grid and selection objective stay
unchanged. Changing synchronization can still change backend arithmetic; real-checkpoint
numerical and resource checks remain necessary.

This reduces simultaneous statistics to **1.611328 GiB** during capture and **0.097656 GiB**
during a layer solve. It requires about **3.222656 GiB** of temporary statistics on disk and
**6.445313 GiB** of total statistics write/read traffic per fit, excluding small headers.
Saving CPU arrays alone would not achieve the same reduction on a unified-memory laptop.

## Implementation and acceptance

- Repair collector ownership without adding a forward, changing the model graph, or changing
  its evaluation schedule. Cover normal completion and errors with actual-method fake tests.
- Add an explicit `split_spill` statistics strategy. Keep the legacy strategy distinguishable;
  record batch one, stable split ordering, per-layer synchronization and storage strategy in
  provenance. Temporary statistics must be removed on successful completion and failure.
- Require R47 qualification before loading a checkpoint. An ordinary owned machine window does
  not override the threshold. A full fit needs successful calibration bound to its exact model,
  corpus, residual producer, statistics strategy, implementation and scored/input workload.
- Log active allocation, allocator cache and peak at accumulation boundaries without introducing
  an extra residual evaluation. MLX allocation is not OS-wide resident memory; the report must
  retain that distinction, and host-side artifact validation needs its own allowance.
- First test with small fakes and import barriers. After the Deputy's actual release, announce a
  short calibration window, confirm its live holder, and start at **one window**. Do not increase
  window concurrency: the current native residual contract supports batch one only. Increase
  token length only after the next shape's projection passes the threshold.

The split strategy's weight-file proxy plus statistics is **8.838936 GiB**, leaving about
**1.817070 GiB** to the current threshold. At 2,816 tokens, all 34 bf16 residuals occupy
**0.456543 GiB**; two selected FP32 arrays and two bf16 gathers at 1,408 positions occupy roughly
**0.040283 GiB**. This is a plausible path to a compliant run, **not a measured or certified
peak**. Native intermediates, update temporaries and backend workspace still need calibration.

Capture requires separate qualification: its withdrawn 16 GiB projection cannot launch the
72-task cohort. Token limits and a model lock do not replace that memory check. No runtime
estimate is certified before the corpus sizes and revised pipeline timings are measured.

The corpus gates are unchanged: actual fitted replay positions must reach 2,749; a larger cap
does not count as coverage. Concentration, span, BOS, source-offset and precision requirements
continue to apply.

## Source outcome and next diagnostic

The correction is implemented through source commit `26d5d47`. **244 affected integration checks
passed** with real MLX imports blocked in the parent and subprocesses. `R47-SOURCE-VERIFICATION-v2.json` binds the exact final source hashes; earlier 241-check and
26-check records remain as history.
The residual-ownership tests failed before the fix and passed after it. Fake-array tests verify
unchanged statistics, maps, masks and forward counts, plus one-split ownership and scratch cleanup.
They do not establish native numerical equivalence or measured resource use.

`R47-DIAGNOSTIC-PLAN-v2.json` freezes the first 128-token, eight-forward diagnostic with a **10.5 GiB
engineering projection**, strictly below the recorded 10.656 GiB threshold. The old four-fit-row
peak was 9.960718 GiB; retained dimensional allowance is 0.143555 GiB, leaving 0.395727 GiB for
staging and unmeasured workspace. It assumes tested split-release behavior, not simultaneous
split storage: the old first-held-row peak of 11.352926 GiB rules out that alternative. The plan
qualifies no larger shape, solver, complete fit or capture run. Qualification also repeats four updates per split and includes 2,048 before 2,816.
The diagnostic remains deferred until the
Deputy actually releases the machine and a live owned window is announced.

Capture now compares its registered projection with the device threshold before loading.
It does **not** have a capture-specific measured-evidence validator. The withdrawn 16 GiB
projection fails that comparison. A new bound must be justified or measured separately;
lowering the number merely to pass is not qualification.

## Evidence

- R47: `design_specifications/live/STANDING-LIFT-EXPERIMENTS-2026-09-05.md`, Director's rulings.
- Earlier real run: `/Users/daniel.tipton/worktrees/gemma-lens-fitting/research/records/GEMMA3-REGRESSION-2026-09-08/run.log`;
  its first sequence peak is 9.658590639 GiB and final peak is 12.443627851 GiB.
- Source at `adcd43d`: `arch.py` native capture, `lens_fitting/regression.py` accumulation,
  and the regression/capture launch scripts. The new correction is source work; no real
  checkpoint has yet exercised it.
