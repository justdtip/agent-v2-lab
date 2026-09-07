# Task 5a: section 14 prose capture producer — source only

Own `src/local_llm_lab/pipeline/lens_fitting/prose.py`, `scripts/lens_prose_capture.py`, and `tests/test_lens_prose.py`. Narrow runtime helper additions are allowed only if the existing structural loader seam cannot be used. Work in `/Users/daniel.tipton/worktrees/lens-fitting`, branch `codex/lens-fitting`, using the primary virtualenv with `PYTHONPATH=src`. You are not alone: preserve other edits. No subagents, merges or pushes. Commit explicit owned paths and write `task-5a-report.md` here; force-adding that report is authorised. Home-worktree writes use require_escalated, explicitly authorised by the user; never relocate to tmp. Parent claim revision 4 covers your paths.

Pure tests only, with zero MLX imports asserted. No tiny native tests, checkpoints, full suite, downloads or adapters. Model priority is released to issue 88 until an explicit handoff.

Read requirements §14 (primary commit afb3c35), runtime.py, corpus.py, reviewed replay.py, existing live_lens/session.py and ForwardLedger, and scripts/live_lens_pilot.py. Preserve existing capture, runner and ledger APIs. Produce pilot-format records so the replay reader handles them unchanged: model/lens/layer provenance, complete manifest episode entries (label, kind, record, record_sha256, seconds, generated_tokens), and a separate dropped list. Use kind prose; the continuation span label is Task 5b work.

## Frozen inputs and preparation

Use only held rows from the frozen prose corpus, in original order, with no duplicate, fit, truncated or reassigned membership. There are 51 held windows of 1,024 tokens. Protocol constants 824/200/32 are declared experiment budgets, not model dimensions. Bind the corpus manifest, rows, sources and tokenizer by hash. A pure plan freezes window indices, full-window hash, first 824 IDs and decoded raw prompt/hash, generation budget, optional descriptive authored tail, greedy sampler, seed, overlap statement, and model/snapshot/lens identity. Keep all original files unchanged.

Decode with the corpus tokenizer and its BOS policy; never apply a chat template. Re-encoding every prompt must equal its frozen IDs before generation. Parent's real cached-tokenizer check found 51/51 exact prefixes (PROSE-PREFIX-TOKENIZER-21.json); the implementation must still fail on drift.

Expose a no-checkpoint plan/preflight path and an explicit execution entry point. Require a new output run/record path and a README/registration written before loading. A plan-only/plan flow is one option; choose a small clear CLI. Freeze the plan before loading and revalidate when consumed. Weights load ONLY through load_runtime(..., capture=True), which resolves none before load_policy and acquires the primary lock before importing MLX. Snapshot dimensions can validate preflight but must agree with ArchitectureView at runtime. Include all layers and the final identity for matching future base rates. Supply the hosted lens by explicit path/hash; load through unchanged LensMaps.

## Generation and accounting

One window is one fresh record/session/turn. Set context kind prose, window index, and a single user message containing the raw prompt. Use CaptureSession.generation(model, tokenizer, prompt, turn_cache=None), then pass its captured callable into installed mlx_lm.stream_generate with a temperature-zero sampler and max_tokens=200. Let the native library create a fresh prompt cache.

Do not use generate_turn_with_count: its tool-close and thinking logic violates the EOS/cap-only prose protocol. Feed every yielded response.token through session.emitted, as the pilot does, and close the stream in finally. Preserve finish_reason. The installed stream yields a terminal EOS response: record it as an emitted token and state that the count includes EOS, matching the pilot. Incidental JSON/tool-closing text must not stop the continuation. Preserve native forward lookahead and partitions through the ledger; emitted count is not forwarded count. Pure fake generator/session seams are acceptable tests, but are not native proof.

After a complete valid turn, a continuation with recorded emitted count below 32 is dropped and counted. Keep its hash-chained record under a dropped subdirectory, rather than deleting it. Accepted records stay at root and only retained episodes enter the main manifest; each dropped entry records window index, count, reason, path and hash. This prevents the legacy reader from seeing extra root records and preserves an audit. Never score authored trailing tokens as foreknowledge.

Incomplete/failed captures cannot certify completion. Exclusive creation, no implicit resume or overwrite. Final completion requires every planned held window to appear exactly once in accepted or dropped results. Preserve partial output on failure. No raw logits or activation artifacts.

## Resources and tests

Use existing configure_allocator_cache(0) and per-episode elapsed/token/peak/working-set-share progress. Bytes do not prove buffer-count safety. Plan and README precede load; record the primary lock. Parent will announce and execute the later LIVE-LENS-PROSE-2026-09-08 run. Do not generate profiles or interpret quality in this task.

Tests must name §14 and prove held-only membership, exact prefix, greedy cap, EOS, minimum 32, dropped accounting, tokenizer/hash drift refusals, default none before load, partial/exclusive output behavior, public capture generation and stream closure on errors. Prove tool-closing text does not terminate prose. Check produced fixture records through unchanged read_record and new replay structural validation. Test claims remain explicitly pure. No atlas or profiles implementation in 5a.
