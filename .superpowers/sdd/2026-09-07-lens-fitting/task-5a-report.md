# Task 5a — section 14 prose capture source

Implemented the bounded producer in `src/local_llm_lab/pipeline/lens_fitting/prose.py`,
`scripts/lens_prose_capture.py`, and `tests/test_lens_prose.py` on `codex/lens-fitting`,
starting at `a29fc757a36ebd3ef842bb31cd3594c2783a101e`. Existing capture, ledger, replay,
atlas, profile, runner and loader APIs were preserved.

The `plan` CLI action uses the corpus descriptor's local tokenizer assets and offline
snapshot resolution. It freezes the exact corpus/source/sequence/tokenizer/model/lens
hashes, all 51 held windows in original membership/order, canonical full-window token
hashes, decoded 824-token raw prefixes and their hashes, all layers including final
identity, greedy 200-token budget, minimum 32, seed and disclosed ridge-selection overlap.
It validates prefix re-encoding without a chat template and requires all intermediate
hosted maps. It creates an exclusive run directory with README and plan before any load.

The `execute` action revalidates the complete plan and README, refuses reused/partial
output directories, creates an exclusive execution marker, and enters the existing
`load_runtime(..., capture=True)` seam. That helper locks the primary worktree and
resolves `none` before `load_policy`; snapshot dimensions must agree with ArchitectureView.
Native imports remain after that loader boundary. Execution uses a fresh public
`CaptureSession.generation(..., turn_cache=None)` for each window and passes its captured
callable directly into `mlx_lm.stream_generate`, with a temperature-zero sampler and
max_tokens=200. Every yielded token, including terminal EOS, enters `session.emitted`.
Finish reason is retained; only EOS or cap can finish a valid turn. Tool/JSON closing
text has no stopping role. The generator is closed in `finally`, including capture errors.

Complete records pass unchanged `read_record` and the new replay structural validator.
Accepted pilot-format manifest entries include label, kind, record/path hash, seconds,
emitted count, window/prefix identity and finish reason. Records under 32 emitted tokens
move into `dropped/` and retain their chain, hash, count and reason. Failed runs retain a
partial manifest and any aborted record; no implicit resume is supported. Final completion
requires one accepted/dropped outcome for each planned window. Authored trailing tokens
are not supplied to foreknowledge. No raw logits or activation artifacts are retained.
Allocator caching uses the existing zero-limit helper; progress includes episode elapsed,
emitted count, process peak memory and recommended working-set share. Bytes are not a
buffer-count safety claim. Runtime provenance records the primary lock.

## Fresh pure verification

Command from the authorised home worktree:

```
PYTHONPATH=src '/Users/daniel.tipton/Desktop/An app/.venv/bin/python' -m pytest -o addopts='' tests/test_lens_prose.py -q --basetemp=/Users/daniel.tipton/worktrees/lens-fitting/.test-tmp-prose
```

Result: **26 passed in 7.83s**, exit 0. Each test installs an import guard rejecting
`mlx` and `mlx_lm`, and its teardown asserts that no `mlx` or `mlx.*` module is present
in `sys.modules`. This is pure evidence only. Fixtures cover held membership and missing
windows, exact raw prefixes, model/lens/corpus/tokenizer/plan drift, all layers, greedy
cap, terminal EOS inclusion, invalid finish/count refusal, tool-closing text, public
session/no-reuse semantics, forwarded lookahead separate from emitted accounting,
31-versus-32 retention, unchanged chain/structural readers, exclusive and partial output,
README/plan before loader, history resolving to none before load_policy, runtime tokenizer
drift before generation, generator failures, and suspended-stream closure on emission error.

Owned-file Ruff check (`ruff check --no-cache` on the three source/test paths) passed.
Ruff formatting applied only to those paths. An initial test invocation lacked PYTHONPATH
and failed collection against primary source; corrected before testing implementation.
The first execute-plan fixture then exposed JSON tuple/list comparison drift; canonical
JSON comparison fixed it, and the final run above verifies the correction.

## Remaining acceptance

No MLX import, checkpoint load, native test, full suite, model run, download, adapter,
profile generation or atlas edit was performed. Native CaptureSession behavior is not
proved by the fake ledger/session fixtures. The parent owns integrated checks and the
serialized native acceptance/run queue; Task 5b owns the continuation span summary.
The intended runtime record directory remains `research/records/LIVE-LENS-PROSE-2026-09-08`.
No real registration/run was created. Parent owns backup pushes; Chief owns landing.

## Fix round 1 — real opaque cache-blob tokenizer descriptor

Parent's real-input preflight of 793c451 exposed a material gap missed by the original
synthetic fixture: the frozen descriptor's five files have already been resolved to HF
blob paths, so their basenames do not retain tokenizer/config roles. The original helper
incorrectly required a descriptor basename of `tokenizer_config.json` and failed before MLX.

The corrected helper accepts the selected ModelSpec/revision and resolves one exact offline
snapshot through the existing resolver. It enumerates the corpus's existing tokenizer asset
patterns in that snapshot, recovers filename roles there, and requires the entire asset set
to match the frozen descriptor by both resolved path and SHA-256, with no extra/missing assets
or duplicate role aliases. Only then does AutoTokenizer load locally with remote code off.
It revalidates snapshot identity after tokenizer loading. Execute uses the plan's pinned
revision; plan forwards the CLI model/revision. No descriptor mutation, snapshot scanning,
newest-revision guess, download or checkpoint load was added.

Two new pure regressions construct realistic snapshot symlinks to opaque blob names. The
positive fixture reproduces the original lookup failure and now succeeds using the exact
snapshot directory. The negative fixture redirects a snapshot asset to an unbound blob
with the same bytes and proves refusal before AutoTokenizer. Existing execute-boundary
fixtures now accept the explicit model/revision arguments.

Fresh command:

```
PYTHONPATH=src '/Users/daniel.tipton/Desktop/An app/.venv/bin/python' -m pytest -o addopts='' tests/test_lens_prose.py -q --basetemp=/Users/daniel.tipton/worktrees/lens-fitting/.superpowers/sdd/2026-09-07-lens-fitting/prose-fix1-pytest
```

Result: **28 passed in 8.68s**, exit 0 (the test command itself succeeded; a subsequent
read-only `ls` in the same shell found the hosted lens absent from this linked worktree,
so the real check below used the explicitly identified primary-worktree hosted lens).
Owned-file Ruff check passed. The per-test zero-MLX guard remains active.

The actual CLI was then invoked with `runpy.run_path(..., run_name='__main__')` and:

```
scripts/lens_prose_capture.py plan
--out .superpowers/sdd/2026-09-07-lens-fitting/prose-fix1-real-plan
--corpus data/lens-fitting/qwen35-4b-prose/manifest.json
--lens /Users/daniel.tipton/Desktop/An app/models/jlens/Qwen3.5-4B_jacobian_lens_n1000.npz
--lens-sha256 381c089dcffead8147ee91f944496f468cce2c7d593e0a1b17230745055aea12
```

A builtins import guard rejected `mlx`/`mlx_lm` for the entire process. After the actual CLI
returned, all 51 planned window indices and prefix IDs were compared directly to the
frozen held corpus rows, and all prefix lengths were asserted to equal 824. The process
also asserted that neither package nor their submodules existed in `sys.modules`.
Result: **51/51 exact prefixes; zero MLX imports; zero checkpoint loads**, exit 0.
Resolved snapshot: `32f3e8ecf65426fc3306969496342d504bfa13f3`.
Plan SHA-256: `985b3f1660a57390e4d701edc95ecca65dbc77d2bb28d6144b99d5f49750a9c1`.
The small check result and offline plan remain in the ignored SDD directory as
`prose-fix1-real-preflight.json` and `prose-fix1-real-plan/plan.json`; no intended runtime
record directory was created. This updates the prior report's no-real-registration statement:
only a tokenizer-only preflight registration was now created, in SDD, not a capture run.
Native capture/replay acceptance remains outstanding and owned by the parent queue.
