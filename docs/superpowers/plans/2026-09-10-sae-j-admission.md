# SAE-J device admission and runner implementation plan

**Goal:** Implement the four laptop-only tasks in the bridge order at `b933850`, section 09:55Z.

**Spec:** `design_specifications/pending/SAE-J-BRIDGE-ORDER-2026-09-08.md` on main at `b93385069f787239e499de15a1c4f0af81cb1bc0`; this implementation starts at CUDA integration `3d5d17e0d9a008bfd82752d177c4d59dd7d0452b`.

**Architecture:** Preserve source artifacts and admit them into the existing `LensMaps` format. Keep artifact admission, numerical bridge primitives, and run admission separate. The runner verifies every declared input before invoking the score/encoding helpers, including in dry-run mode.

**Tech stack:** Python, NumPy, JSON, existing safetensors reader, pytest. Existing environment only; no model load, network, downloads, or device execution.

## Contracts and ownership

- T1 owns `probes/device_lens_admission.py`, `scripts/admit_device_lens.py`, its tests, and `scripts/merge_device_lens_chunks.py`; the original record script stays as run (F1, review `7527075`). An admitted sidecar retains `model`, `npz_sha256`, `source_archive_sha256`, `nu`, and source-manifest provenance. The model endpoint is explicitly `identity`. Source `J{repository_layer}` becomes loader `J{repository_layer-1}`. Every chunk's full declaration and map digest remain available; mixed width schedules are never collapsed to a single width.
- T2/T3 own `probes/sae_bridge.py`, the minimal endpoint metadata change in `pipeline/live_lens/instruments.py`, and `tests/test_sae_bridge.py`. Endpoint metadata is optional for old artifacts, but final-layer admission requires an explicit identity endpoint. Error configuration has separate raw and score thresholds and a positive near-zero floor; omitted configuration refuses before ranking.
- T4 owns `probes/device_bridge.py`, `scripts/device_sae_bridge.py`, and `tests/test_device_bridge.py`. It consumes the admitted sidecar, existing checkpoint metadata/hash manifest, dictionary digest receipt, capture manifest/index, declared positions/config, and pairing record. It does not register measured pairings or invent a domain-of-validity result.
- Parent owns integration, report, and explicit-path commits. Workers do not commit or edit each other's files.

## T1: archive admission and complete merger declaration

- [x] Write failing tiny real-file tests: three maps with distinct hand-written matrices, loader round-trip, shifted map keys, incorrect identity/hash, wrong shape/count, absent widths, duplicate/overlapping chunk populations, and complete merged provenance.
- [x] Run the test file and retain the failure's own exit status.
- [x] Implement explicit key conversion, finite/shape/count/declaration checks, digest-bound sidecar, source-preserving outputs, and the merger's missing `nu.json`.
- [x] Test the actual script entry point on fixtures and load the result through `LensMaps.load`; a known residual at repository layer k must use exactly that layer's original matrix.
- [x] Review the complete changed surface and commit only after focused tests and lint pass.

## T2: final-layer endpoint

- [x] Write a failing identity-endpoint alignment fixture and a missing-interior-map refusal fixture.
- [x] Preserve old nonfinal behavior; accept a final dictionary only under the declared identity endpoint and preserve all model/precision/pairing guards.
- [x] Exercise both model depths with tiny residual widths, without loading either model.

## T3: independent reconstruction budgets

- [x] Write the hand-derived fixture h=(1, 0.1), reconstruction=(1, 0), L=diag(0.01, 1): raw share 0.099503719, lens-score share 0.99503719. At threshold 0.5, raw admission passes and score ranking refuses.
- [x] Require explicit thresholds and denominator policy; reject missing/nonfinite/invalid configuration and make near-zero readings unrankable.
- [x] Rename the raw-only budget and fields; report both named quantities in A2. Never turn a raw budget mask into score admission.
- [x] Update affected tests to supply an explicit policy and verify existing decomposition identities and intervention behavior.

## T4: complete guarded A1/A2 runner

- [x] Write end-to-end fixture files with a tiny checkpoint, dictionary, admitted lens and capture. Dry-run must verify all inputs without invoking any numeric helper or writing numerical results.
- [x] Mutate checkpoint bytes/hash membership, lens identity/archive/nu, dictionary bytes/receipt, capture manifest/data/index, each pairing field, and selected token/position independently. Every mutation must refuse by name before a helper executes.
- [x] Check array/token shapes from headers and bind actual selected cell metadata, not free-form claims. Preserve the teacher-forced meaning of expert-token captures.
- [x] Implement explicit CLI paths, typed error config and fail-closed admission. A1 runs chunked readouts and acceptance controls; A2 computes the exact decomposition and separate error budgets. Outputs carry admitted provenance and the ordered interpretation limits.
- [x] Test accepted dry-run and numerical A1/A2 paths, including the identity endpoint, score-dominant rows, missing examples, missing completion manifests and unregistered crossings.

## Integrated verification and report

- [x] Run admission, runner, bridge, intervention, lens-identity and repository-rule tests with CPU-only environment and temporary test/cache paths. Do not run the large real-artifact test or any model-loading check.
- [x] Run scoped formatting/lint, inspect the final diff, and obtain an independent review of the integrated contracts and failure directions.
- [x] Commit only explicit owned paths; record the checked set, command, exit status and limitations beside each T1–T4 verdict in `research/records/SAE-J-ADMISSION-2026-09-10/README.md`.
- At task close, release this task's claim and report the commits. Device pairings, examples, domain-validity statement and runs remain with the lens owner.

Baseline: bridge + intervention CPU set at `3d5d17e`: **99 passed, 1 real-artifact test deselected**, exit 0. The shared primary checkout's pre-existing untracked records are outside this task.

Completion evidence: T2/T3 `e5898e0` (152 passed, 1 real-artifact deselected); T1 `6ae9648` (32 passed); T4 `434bb55` (45 passed). Final committed integrated CPU set: 350 passed, 1 real-artifact deselected, exit 0. See the admission record for commands, review findings closed, the sandbox-only process-check failure and its successful rerun, input schemas, and remaining device-owner work.
