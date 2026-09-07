# Domain lens fitting — remaining source delivery

7 September 2026. For the Chief's review. Persistent worktree `/Users/daniel.tipton/worktrees/lens-fitting`, branch `codex/lens-fitting`. Source through `e7b77b4`; the corpus slice already landed at `5712131`. This delivery implements requirements3.2–3.6 and the §14 prose producer. **Source implementation is reviewed; scientific acceptance is incomplete.** No checkpoint has loaded successfully in this task. Do not interpret this handoff as acceptance of a fitted lens or authorisation for adapter work.

## What changed

The public entry points are standalone scripts. Fitting resolves the exact offline checkpoint through the shared loader and primary-worktree lock. Capture and replay explicitly select `none` before loading. Registry defaults are unchanged by this patch. The amended history cache is already in the branch's ancestry.

| Caller or module | Resulting behavior |
| --- | --- |
| `scripts/lens_fit.py` | All-layer regression and staged Jacobian plan/check/benchmark/fit; ordinary shared loader; exclusive artifacts and recorded resources. |
| `scripts/lens_regression_preflight.py` | Registered maximum-row resource ladder and one-layer fixed-grid solve timing; no fitted artifact or quality output. |
| `scripts/lens_replay.py` | Explicit layer selection, original forward partitions/emissions, fresh cache per turn, mandatory legacy summarizer; optional exact hosted-pilot identity mode. |
| `scripts/lens_prose_capture.py` | Separate plan/execute stages for the51 held windows, frozen824-token raw prompts, greedy200-token cap, EOS, count/drop below32. |
| `scripts/lens_profiles.py` | Pure aggregation only after verified instrument and actual native pilot identity evidence; exclusive per-set JSON, specificity table and standalone page. |
| `scripts/live_lens_atlas.py` | Four prose-only lines: raw authored prompt offsets and emitted continuation span. Original pilot output remains byte-identical. |
| `lens_fitting/regression.py` | Float32 fit/held sufficient statistics streamed from one uncached all-layer forward per sequence; §12 penalty `alpha*mean_diag(total fit XTX)`; fixed five-value grid; stored `J=W.T`. |
| `lens_fitting/artifacts.py` | All nonfinal maps, finite float32, exclusive NPZ/sidecar, raw storage under decimal1GB and unchanged LensMaps round-trip. |
| `lens_fitting/runtime.py` | Exact offline snapshot and asset hashing, dimension validation, primary lock before model import, shared load_policy path, explicit no-reuse capture, allocator/resource reporting. |
| `lens_fitting/jacobian.py` | Pre-norm cached tail, full basis, native hybrid-state copies, equal-span fit sampling,150 minimum/400 maximum, fixed convergence rule, batch8-first bounded benchmark and >25min/layer stop. |
| `lens_fitting/validation.py` | Held-out32-direction mean response checks, stated bounds and epsilon stability; existing response_agreement/layer_verdict remain verdict authority. |
| `lens_fitting/replay.py` | Hash-chain/source/lens validation, exact native control assertions, original error preserved through aborted capture cleanup, incomplete source manifests rejected before loading. Current replay hashes live in new provenance/completion files. |
| `lens_fitting/prose.py` | Frozen corpus/snapshot/tokenizer provenance before loading; complete accepted+dropped coverage; direct stream_generate avoids tool-specific early stops. |
| `lens_fitting/profiles.py` | Exact emitted-coordinate and final-control pairing, episode-separated rows, all layers, per-row n/base rate, regressionh1 excluded interpretation, provenance and completion gates. |

The module paths in the last eight rows are under `src/local_llm_lab/pipeline/`. Existing `LensMaps`, `CaptureSession`, architecture, registry, runner and validation vocabulary are not modified. The replay driver uses the public capture API after the cache landing.

Tests added: `test_lens_regression.py`, `test_lens_runtime.py`, `test_lens_regression_preflight.py`, `test_lens_jacobian.py`, `test_lens_replay.py`, `test_lens_native_integration.py`, `test_lens_prose.py`, `test_lens_profiles.py`. `test_repository_rules.py` pins the native-test import closure. The implementation plan, SDD reports and named research records accompany the patch. No dependencies or downloaded assets are added by this delivery.

## Evidence and its limits

All records below are under `research/records/LENS-FITTING-IMPLEMENTATION-2026-09-07/` unless a path says otherwise.

- Final combined pure integration: **173 passed in27.25s**, `INTEGRATED-PURE-34.log`. A loader guard refused actual MLX module creation; final module inventory was empty. Covers replay, runtime, preflight, prose, profiles and repository rules together.
- New native replay integration: **3 passed in1.05s**, `REPLAY-NATIVE-31.log`, source `ce219f7`. Two turns, split prefills, actual lookahead, unchanged/changed lenses, exact native controls, first-error preservation and wrapper restoration. The later manifest-status change is a pure pre-load check. No checkpoint in these tests.
- Final Jacobian module: **36 focused passes**, including six native tiny-model checks and actual32/64 direction widths; `JACOBIAN-NATIVE-18.md`, source `1534a9f`. This does not validate the real checkpoint's numerical self-check.
- Prose producer: **28 pure passes**, actual plan CLI51/51 exact824-ID raw prefixes with zero MLX; `PROSE-PREFLIGHT-26.md`. Snapshot roles match the frozen corpus's opaque cache blobs by path and hash. No continuation was generated.
- Profiles: **38 pure passes**, actual page JavaScript/filter tests; tracked Task5b report. The actual updated legacy summarizer on all13 original pilot records reproduced the original atlas as bytes and decoded JSON, SHA256 `28f5f4d558bf9a501e1b489419367145b71e904fe8f7450f046bd0e3be3fca41`. This is a tokenizer/summarizer control, **not native model replay**. Browser appearance was not visually reviewed.
- Every component received independent GPT-6 Astra review. Final profiles review was clear. Final cross-component review's incomplete-manifest P2 was reproduced, fixed and rereviewed clear. `FINAL-SOURCE-REVIEW-35.md` carries the exact boundaries.
- The full suite exited0 on earlier regression source, `full-suite-08.log`; its separately skipped cached-tokenizer cases later passed. **The full suite has not run on this final source.** Repeated peer MLX suites occupied the announced window.

Failures are retained. Five replay-corruption cases originally lost their diagnostic to abort cleanup (`REPLAY-CLEANUP-RED-28.log`); four incomplete-manifest cases reached snapshot resolution (`REPLAY-PARTIAL-RED-36.log`); both were fixed. The first combined guard incorrectly rejected transformers' package-discovery query and produced5 failures/163passes (`INTEGRATED-PURE-33.log`); the guard, not product source, was corrected. No MLX loaded in that failed combined run.

## Conventions and departures made explicit

§12 supersedes the original ridge wording. The sidecar records alpha, total mean diagonal and position count, so the absolute penalty is reconstructible. Regression R² is explicitly uncentered. Held sequences may share agentic trajectories with fit sequences and choose alpha; generalisation remains the disjoint pilot.

§14 supplies the51 held prose windows. Their use to choose among five ridge values is disclosed on prose evaluation cells. There is no test-shard download, third fitting domain or authored-token foreknowledge summary. Terminal EOS is counted as an emitted response consistently with the installed generator and pilot record convention; fewer than32 emitted responses are dropped. Authored trailing tokens may only be descriptive.

Foreknowledge targets actual emitted tokens at h1/4/8. Each layer row uses final-distribution rows with the exact same eligible coordinates and denominator. Regressionh1 rows remain visible, biased and ineligible for interpretation. Scientific rows stay episode-separated; no primary layer or automatic aggregate conclusion is chosen. The mandated original atlas control retains the legacy aggregation only to reproduce its exact numbers. The page draws each selected episode separately.

Resource byte measurements do not claim a Metal buffer count. Calibration uses two copies of one fit row solely to populate both statistic slots, with no held-set quality reading. The registered initial10GiB allowance and later growth/solve bounds are inspected estimates, not proven allocator limits. No basis, corpus, alpha grid or convergence rule was relaxed.

## Remaining work and the required handoff

All three resource-probe attempts were refused before checkpoint loading. Attempt03's launching inventory found peer pytest PID55135 mapping MLX after our tiny tests exited; only `resource-preflight-03.log` exists. The earlier announced window was closed in the primary heartbeat. No foreign lock was cleared or process killed. **Issue88 has no scheduling hold from this task.**

An exclusive handoff must also exclude native pytest for the announced slot. Next resource retry uses fresh04 paths and the already registered README under `research/records/LENS-FIT-agentic-regression-2026-09-07/`. Inspect cost and memory before any full regression fit. A final-source full suite also needs a free native slot.

Then, serially under announced windows: real Jacobian self-check with bounds recorded first, batch8-first timing and memory benchmarks, the permitted all-layer fits, held-out map validation, exact hosted-pilot replay identity,51 prose continuation captures, fitted replay/cross-read and gated scientific profiles. If measured Jacobian cost exceeds25minutes per layer, or the resource limit cannot be met, stop for the Director's ruling; do not silently shrink the experiment. Actual under-hour regression acceptance remains unmeasured. Adapter fits remain deferred until3.1–3.6 land.

The supplied `LENS-FITTING-REMAINING.patch` is the remaining **source and evidence** delivery. It excludes its own bytes, delivery metadata, the earlier corpus patch, and the shared heartbeat (operational entries already live in primary). It must pass read-only `git apply --check` against the primary tree; metadata records the exact checked tree and artifact hash. Chief review and integration remain separate.
