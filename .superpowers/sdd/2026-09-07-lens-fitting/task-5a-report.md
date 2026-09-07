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
