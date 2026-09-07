# Conversation cache implementation plan

**Goal:** Retain intermediate conversation state without changing visible history or silently changing native arithmetic.

**Architecture:** A shared forward-pass ledger records actual model inputs for both live-lens capture and an explicit experimental `history` cache. Bounded complete hybrid snapshots can be reused only at exact token and forward-partition prefixes of the next native prefill. Fixed-token acceptance independently checks restoration and preservation of the existing generator's computation.

**Spec:** User rulings of 2026-09-07; design below is the implementation contract.

## Constraints

- Work only in `/private/tmp/codex-history-cache-01a079ee`, based on `3922b0ba5a59ed9681d1b47e1590e387d7eacfed`.
- Never load checkpoint weights while Claude's queue owns the machine. Never clear its lock or stop its processes. Tiny randomly initialized installed Qwen3.5 models require neither a checkpoint nor a lock.
- Registry YAML verification fields remain null; Qwen3.5 `auto` remains `none`. `history` is explicit and experimental.
- Capture remains no-reuse. Actual forward inputs, including lookahead, are the position authority; emitted tokens are a separate ledger.
- Preserve canonical assistant rendering and the two-observation window. Roll back all dependent state when tokens change.
- No source commit, integration, or default enablement is implied by this plan. Real-checkpoint acceptance remains pending until it can run normally under the repository lock; completed queue steps do not authorize interrupting a subsequent lock holder.

## 1. Shared forward accounting

Files: `src/local_llm_lab/forward.py`, `src/local_llm_lab/pipeline/live_lens/session.py`, `tests/test_history_cache.py`, `tests/test_live_lens_native.py`.

- [x] Write failing tests for `ForwardLedger(prompt_ids)` with `record(offset, ids)` and `emitted(token_id)`: lookahead may precede emission, neither emission nor display formatting advances the forward cursor, discontinuities and conflicting tokens fail.
- [x] Implement `ForwardPass(offset, input_ids)` and ledger validation, with a maximum context of 65536 and explicitly validated restored prefixes.
- [x] Replace CaptureSession's independent forward counter and emitted list with the shared ledger; keep its record format and no-reuse guard.
- [x] Run ledger and native capture/replay tests; retained records must still replay exactly.

## 2. Bounded history snapshots

Files: `src/local_llm_lab/pipeline/runner.py`, `tests/test_history_cache.py`, `tests/test_runner.py`, `src/local_llm_lab/models.py`, `tests/test_models.py`.

- [x] Write failing tiny-model tests covering exact same-schedule restore after advance, a rewritten old observation, changed canonical assistant tokens, full native cache offsets and recurrence, budget eviction, and incompatible partitions.
- [x] Add `HistoryCache(model, view, prefill_step_size=2048, max_checkpoints=4, max_bytes=536870912)` with `prepare`, `generation_model`, and `commit` at existing runner seams.
- [x] Observe native calls through a delegating wrapper; evaluate and clone complete cache objects at prefill boundaries and turn completion. Do not inject a forward split solely to take a snapshot.
- [x] Require exact saved token prefix and forward partition compatibility with the native prefill schedule. On any rewrite, discard dependent snapshots and restore an eligible preceding snapshot or start fresh. Verify every offset-bearing cache against the ledger.
- [x] Add only explicit `history` strategy support; preserve all registry files and existing auto decisions. Keep capture incompatible with reuse.
- [x] Run focused runner, registry, cache, and live-lens tests.

## 3. Two fixed-history acceptance checks

Files: `research/cache_equivalence.py`, `tests/test_cache_equivalence.py`.

- [x] Add failing tests requiring both checks, bitwise logit/state comparison, rejection even when greedy winners agree, and an inconclusive result when no state was reused.
- [x] Implement `check_fixed_history(model, view, cases, ...)` consuming fixed `prompt_ids` and `continuation_ids`. The candidate uses the production cache and installed native generator with teacher-forced sampling.
- [x] Check A rebuilds the recorded candidate forward schedule with fresh native state and compares the last-query logits at each newly executed boundary plus final state. Check B runs the installed generator from the full prompt with fresh state and compares all generation-decision logits and final state. Include first divergence, logit gap, winning margin, token/partition hashes, and reuse counters. Discarded prefill logits are not materialized: the existing runner evaluates prefill cache state and does not execute that full vocabulary projection.
- [x] Add CLI input/output options, adapter selection through the existing locked loader, model/corpus/source provenance, and fail-closed exit status. Never write registry evidence automatically. Keep the legacy trajectory check explicitly separate.
- [x] Run tiny-model acceptance and negative controls. Do not treat tiny-model results as checkpoint certification.

## 4. Review and handoff

- [x] Inspect Claude's isolation artifact read-only when it becomes available; distinguish native execution from the manual float32 diagnostic path.
- [x] Run relevant checkpoint-free regression tests and changed-file static checks; inspect the final diff.
- [x] Write `design_specifications/pending/HISTORY-CACHE.md` with implementation, exact evidence and remaining checkpoint gates, including the reuse limitation imposed by native partition compatibility.
- [x] Produce a reviewable isolated patch and handoff. Real checkpoint checks wait for the completed queue and normal lock acquisition.

## Numerical policy

The initial candidate retains the existing 2048-token native prefill cadence, including the final prompt token in a separate call. This is an execution-preserving candidate, not a claim that arbitrary splits are equivalent. Generated states are recorded as they happen, but a later bulk prefill may be unable to reuse a token-by-token suffix without changing arithmetic. Such states remain ineligible. A different cadence or kernel requires its own comparison against the existing runner, not merely cached/recomputed agreement under the new cadence.

## Handoff status

Source implementation, tiny-model tests, GPT-6 Astra review, frozen inputs and the isolated patch are complete. Final regression: 191 passed; changed-file lint and diff checks passed. The fixed corpus has 19 episodes / 76 turns and includes actual native-chunk reuse opportunities. Real-checkpoint gates remain pending: a subsequent evaluation acquired the primary model lock after the original queue completed. `design_specifications/pending/HISTORY-CACHE.md` records the diagnosis, limitations and exact guarded command. No checkpoint has been loaded for this patch.
