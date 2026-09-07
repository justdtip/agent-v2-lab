# Task 4 replay reader — source handoff

Source frozen at **60d91dd**, following checkpoint **cc33a2f**. No checkpoint was loaded, no native forward was run, and no pilot runtime identity is claimed. Parent scheduling hold for #88 remains respected. Parent owns independent review, runtime registration, subsequent native tests, and the hosted pilot identity run.

## Owned change

- `src/local_llm_lab/pipeline/lens_fitting/replay.py`
- `scripts/lens_replay.py`
- `tests/test_lens_replay.py`

No runtime.py change was necessary: `PreparedReplay` exposes the existing spec/snapshot loader seam without a fitting corpus. Existing runtime tests explicitly exercise a history-default spec becoming none before load_policy. No legacy atlas/session/ledger/architecture/runner/registry/corpus/lens/pilot record was edited. No guard-table edit was necessary: final in-process verification asserted that the entire focused test run imported no MLX module.

## APIs and behavior

- `read_source(path, expected=None) -> (events, identity)`: plain or gzip input; adapts decoded text to the SAME existing `read_record` chain/footer validator. Identity binds exact on-disk bytes and decompressed bytes; expected identity is compared anew at consumption.
- `validate_events(events)`: complete sequential turn boundaries, contiguous forward/read positions, exact forward partitions and emitted IDs via `ForwardLedger`, shape/argmax/hash syntax, top-k/vocabulary bounds, rank event coordinates/order, attention source/head coordinates/order. Generator lookahead is valid; forwarded and emitted counts are separate. Aborted/incomplete turns cannot certify completion.
- `prepare_replay(source, spec, output, lens_path, *, lens_sha256, domain, kind, layers='all', identity_atlas=None, revision='main') -> PreparedReplay`: pure source-set/manifest/lens/layer/output validation and offline exact snapshot resolution. Config dimensions are used only for pre-load validation and must agree with loaded ArchitectureView. Source filenames preserve labels; output directory includes model/domain/kind and is exclusively created. Source mode reads manifest layers; all mode derives layer count from the snapshot and verifies the native view agrees. New readings require the final identity layer.
- `replay_record(view, tokenizer, lens, events, emit, *, layers, array_api=None, session_factory=CaptureSession, progress=None)`: drives public generation/set_context/emitted only, with the callable returned by generation. Verifies stored prompt IDs against actual tokenizer encoding before entering capture. Uses fresh native cache per turn, one record session for preserved turn numbering, native logit-hash/shape/argmax/input/control equality, source/head coordinate equality, and a new LensReadout. No retained raw logits. Public array/session injection seams support pure orchestration tests; these do not substitute for native evidence.
- `run_replay(prepared, loaded, *, progress=None, allocator_cache=None)`: revalidates identities, writes exclusive new plain JSONL records, verifies their chains/structure, saves an identical historical manifest separately from `replay-manifest.json`, and writes completion only after all records and optional atlas identity succeed. Provenance binds source hashes, snapshot, lens SHA, layers, per-record capture policy, commit, lock, allocator-byte policy, start timing; completion binds output hashes and elapsed time. Progress totals include cumulative tokens and episode labels. CLI adds elapsed/peak memory/device working-set share using existing runtime helpers and sets allocator cache bytes to zero.
- `legacy_identity(records, source_atlas, *, model, snapshot, expected_atlas_sha256)`: runs the ACTUAL `scripts/live_lens_atlas.py` main via runpy and a temporary argv context. The temporary snapshot_download binding serves the already verified local tokenizer snapshot, requires local-only lookup, and is restored even on failure. Evidence hashes the actual script, atlas source/output, output records, historical manifest, and replay provenance file when present. Exact decoded object comparison supplies a first specific differing field and fails closed; no caller-supplied success boolean.
- `first_difference` exposes that recursive diagnostic.

CLI: `--model --records --lens --lens-sha256 --domain --kind --out`; optional `--layers all|source`, `--identity-atlas`, `--revision`. Identity requires source lens SHA and `--layers source`. No adapter/prose-generation/checkpoint-production API was introduced.

## TDD and fresh evidence

All commands used the primary virtualenv and `PYTHONPATH=src` from the persistent worktree.

1. Initial replay tests: `python -m pytest -q tests/test_lens_replay.py` — **10 failures**, all missing replay module, then **10 passed** after initial implementation.
2. Added public generation/lookahead/cache/tokenizer, actual legacy atlas invocation/statistics mutation, CLI/preflight and provenance tests — **37 passed** with `tests/test_lens_runtime.py` included. The atlas mutation changed original episode seconds and failed at `$.episodes[0].seconds`.
3. Added hash/shape/argmax/offset/input mismatch tests and structural hardening regressions. Structural tests demonstrated **3 failures** (top-k, unexpected source event, out-of-vocabulary token accepted), then passed after implementation. Native mismatch tests use the public fake session seam and do not execute a native model.
4. Final command: `python -c 'import pytest,sys; result=pytest.main(["-q", "tests/test_lens_replay.py", "tests/test_lens_runtime.py"]); assert not any(k=="mlx" or k.startswith("mlx.") for k in sys.modules); print("MLX import guard: clear"); raise SystemExit(result)'` — **46 passed**, **MLX import guard: clear**, exit 0 (26 replay /20 runtime).
5. Final pure scan using `read_source` one record at a time on all committed `LIVE-LENS-PILOT-2026-09-07/*.jsonl.gz` — **13 records, 70 turns, 3704 forwards**, exit 0, with the same zero-MLX assertion. This establishes structural compatibility only.
6. `ruff check` for the three owned files — passed. `ruff format` — current. `git diff --check` — passed. No full suite run.

## Commits

- `cc33a2f` — public replay driver, pure preparation and CLI, independent legacy atlas gate, initial contracts.
- `60d91dd` — malformed-event/native-mismatch refusal coverage, explicit capture/vocabulary provenance, actual identity-input hashes, cumulative progress.

## Pending acceptance and limits

- Tiny native CaptureSession integration tests and exact hosted-lens pilot replay remain **unrun**. No native test was attempted after the scheduling hold. The existing all-pilot offline atlas control predates this implementation and remains a separate baseline, not replay identity.
- Identity pure tests invoke the actual atlas code with a synthetic tokenizer/record fixture. They prove invocation, restoration, equality checking and statistics mutation refusal; they do not prove real tokenizer/native numerical equivalence.
- Source capture policy is required to remain stable within an individual record, matching the current CaptureSession constructor contract. Different records may have different capture policies. Rank horizons/capacity must match current public CaptureSession defaults (recorded 1/4/8 and16).
- Per-record memory is bounded by the source record plus CaptureSession's bounded future window/native cache/readout state. A full source record is parsed using the mandated existing validator; no full-corpus list or raw production logits are retained. Allocator cache0 bounds unused allocation bytes only, not live-buffer count.
- New profile sequencing and the required prior pilot identity gate remain parent/Task5 responsibility; this reader does not invent evaluation prose or decide the pending Director metric question.

## Independent review fix round 1

This section supersedes the initial source-freeze API descriptions where noted. Review of 60d91dd identified three P2 gaps; all three are addressed in this round.

1. Preparation now unions selected reading layers with every preserved attention block's source layer `b` and head-output layer `b+1`, exempting only native final identity. Existing config bounds precede lens coverage. Missing attention readout maps fail before native loading. Tests cover missing both maps, either single map, and acceptance with complete coverage.
2. CLI `--layers` accepts `all`, `source`, or an explicit comma-separated increasing list (for example `1,4`). Positivity, uniqueness and ordering are parsed before preparation; the existing snapshot-bound and final-identity-layer requirements remain enforced by preparation. The regression demonstrates that `1,2` reaches preparation as `(1, 2)` rather than being rejected by argparse choices.
3. New `legacy_summary(records, *, model, snapshot)` runs the actual unchanged legacy atlas script and writes `atlas.json` plus `atlas-provenance.json`. Every ordinary replay invokes it before writing completion. `legacy_identity` reuses this generation path and additionally compares the immutable reference and writes identity evidence. Ordinary runs do not write an identity-success claim. Failure of summary generation prevents the completion artifact. The actual legacy summarizer and current span behavior are unchanged; section14 prose compatibility remains Parent/Task5 work.

TDD: the initial focused replay run against the reviewed source produced **5 failures** reproducing the three findings: three attention-map omissions were accepted, explicit CLI layers never reached preparation, and ordinary replay lacked atlas.json. After the fixes and acceptance/failure-path additions, final verification was **52 passed** (32 replay /20 runtime), with the in-process assertion **MLX import guard: clear**. `ruff check`, `ruff format`, and `git diff --check` passed. No native checkpoint, native tests, full suite, original artifact, or prose-generation changes occurred. The existing exact-reference atlas mutation test remains green.

Final verification command was the same focused pytest.main/no-MLX-assertion command documented above. Source changes are confined to the same three owned paths. This report is explicitly committed in this round as requested, despite the SDD directory's ignore rule.
