# Lens fitting implementation — corpus slice

This delivery covers requirements §3.1 and §10: immutable training-only agentic corpora and the authorised Wikitext validation corpus, with token spans, a fixed sequence split and hash-bound provenance. Regression fitting, Jacobian measurement, replay, profiles and validation have not been implemented or run in this slice. The regression scaling ruling is pending.

Persistent worktree: `/Users/daniel.tipton/worktrees/lens-fitting`

Branch: `codex/lens-fitting`

Primary cache/rules incorporated through `ac1b700` (cache landing `618d41a`), isolated merge `53343a3`. The shared checkout is unchanged by this task.

## Implemented surface

- `src/local_llm_lab/pipeline/lens_fitting/corpus.py`: agentic/prose builders, verified corpus reader, pinned validation-only download and tokenizer-only loading.
- `src/local_llm_lab/pipeline/lens_fitting/__init__.py`: package marker.
- `scripts/lens_corpus.py`: standalone entry point; no pipeline CLI or model-loading caller changed.
- `tests/test_lens_corpus.py`: reconstruction, alignment, split/leakage/provenance and download-boundary tests.

The builder assigns source-step indices before dropping over-cap sequences; one-based sequences 5, 10, ... are held out. It reconstructs prior messages exactly as the runner does, with windowed observations and canonical prior actions, then preserves the current raw completion. Whole-text tokenization avoids assuming that separately tokenized prompt and completion concatenate unchanged. Every token receives one span; boundary-straddling and zero-width special tokens are template spans.

The downloader resolves an immutable dataset revision, enumerates only authorised validation Parquet filenames and reads those local files. It does not use a whole-dataset preparation call with a subsequent split selection. The downloaded source text and immutable descriptor stay under `data/lens-fitting/`; the HF cache retains the original downloaded shard. Subsequent builder calls reuse the verified download. No training/test Wikitext shards were requested.

## Concrete corpus builds

| Domain | Fit sequences / tokens | Held sequences / tokens | Excluded |
|---|---:|---:|---|
| Agentic |418 /458,191|100 /106,870|82 of 600 source steps exceed the corpus cap|
| Prose |204 /208,896|51 /52,224|164 trailing tokens after fixed windows|

Agentic source: all 72 base-policy training trajectories from the three specified evaluation reports. Exact task listings and source hashes are in `research/records/LENS-FITTING-IMPLEMENTATION-2026-09-07/source-inspection.json`.

Prose source: `Salesforce/wikitext`, `wikitext-103-raw-v1`, validation, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Exactly one validation shard was requested; its SHA256 is `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`. The download descriptor is preserved in the record.

Live manifests:

- `/Users/daniel.tipton/worktrees/lens-fitting/data/lens-fitting/qwen35-4b-agentic/manifest.json`
- `/Users/daniel.tipton/worktrees/lens-fitting/data/lens-fitting/qwen35-4b-prose/manifest.json`

`CORPUS-BUILD-01.json` records exact manifest-file and sequence-file hashes. The manifest's own `manifest_sha256` field is a separate canonical-payload digest excluding itself, not the exact-file digest used by a future fitter. Source and tokenizer paths are absolute and must remain available for reader verification. No activations are stored.

## Verification and limits

The related pre-change baseline passed 84 tests. Final verification on source commit `016c5f6` passed **81 focused tests in 4.33s** (56 corpus and 25 repository-rule tests), Ruff, and the diff check. `CORPUS-VERIFY-02.json` and its three logs preserve the fresh results. Both frozen corpora passed the final reader with unchanged manifest hashes. The actual 600-step input build and real pinned prose download both exited successfully without weights. Transformers' PyTorch-absent notice is expected for tokenizer/file-only work; the unauthenticated HF rate-limit notice did not prevent the download. Full logs are preserved.

Independent GPT-6 Astra review found two format-validation gaps: stored offsets lacked the fresh-tokenization checks, and prose chunk identities/counts/tails needed reconciliation against the declared sources. Commit `016c5f6` resolves both with targeted regressions. Scoped re-review passed with no remaining findings. The task-ID exclusion rule now declares its agentic scope; exact legacy-rule compatibility preserves the already frozen corpora. `CORPUS-REVIEW-02.md` records the findings, fixes and review limits.

No checkpoint was loaded for this task. The one-MLX-import-process rule recorded during the task is respected for subsequent tests: corpus/rules tests have a checked top-level import closure with no MLX. Only that pure subset was run for final verification; no native/tiny-model suite or model fitting was launched. The old ten broad-suite failures were repaired on the incorporated primary branch; this task did not fix them.

## Required regression ruling

The unrun prototype and the mathematical requirement differ:

| Interpretation | Raw sufficient-statistics solve |
|---|---|
| Literal §3.2 with summed reconstruction loss |`(XTX + lambda I) W = XTY`|
| Prototype with mean reconstruction loss |`(XTX + n * lambda I) W = XTY`|

Both use `lambda = grid_multiple * mean_diag(XTX) / n`. The extra factor changes the regularisation substantially and can determine whether the smallest penalties survive float32 rounding. Neither convention has been selected or implemented in this slice. The fixed grid itself is unchanged.

## Durable delivery

Local commits preserve the plan, source-inspection notes, corpus implementation, actual corpus manifests, tests and review evidence. `design_specifications/pending/LENS-FITTING.patch` is the corpus-slice delivery, based on incorporated primary commit `ac1b700`; it excludes its own file and does not carry already-landed cache/debt changes. The patch has not been applied to the shared checkout. Large sequence/source data remain under the persistent worktree's ignored `data/` directory, with exact hashes committed in the record. Automatic approval review rejected the remote backup push to `https://github.com/justdtip/agent-v2-lab.git`, requiring direct authorisation for research-material egress. No push retry or alternative upload was attempted. Chief review and integration remain separate from local branch commits.
