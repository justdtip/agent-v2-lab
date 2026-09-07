# Regression source review and verification — 7 September 2026

Source reviewed and tested: 1acabd67a20068ae25c76be96c1a768e1b17a1db, persistent branch codex/lens-fitting. Task reviewer: lens_regression_review, GPT-6 Astra. This is the controller's durable transcription of the independent review, followed by separately labelled controller verification.

## Independent review

No Important findings. Approve Task 2 implementation. Spec verdict partial pending runtime evidence, with no material source discrepancy.

The reviewer confirmed section 12 alpha scaling, all candidates and provenance; one uncached all-layer forward and separate float32 sums; pre-norm target with no activation artifact; row solve XW=Y and J=W.T; complete finite exclusive float32 artifacts and unchanged LensMaps round-trip; exact snapshot/corpus hashes, primary lock before loader import, and explicit none for capture; held alpha selection separate from disjoint pilot generalisation.

It independently inspected ArchitectureView.residuals, LensMaps.load/apply, load_policy/runlock.load_weights and idempotence, read_corpus provenance, and installed mlx_lm asset selection. Tests cover replication, direct SSE, held selection, non-symmetric recovery, mutation refusal and exclusive writes. The reviewer did not run tests, MLX, network calls or writes. At review time native final-head verification and checkpoint-scale measurements were pending.

## Fresh controller verification on the reviewed source

After confirming no primary model lock and no mapped MLX process, ran PYTHONPATH=src with the primary virtualenv from this worktree: python -m pytest tests -q --tb=short. Exit 0; no failures. Full status stream is full-suite-08.log. Repository addopts already supplies -q, so doubled quiet suppresses a final numeric count. The four real tokenizer checks initially skipped because the worktree cache was empty.

Rechecked the model slot, then reran only those four state_swap tests using the existing primary HF_HOME and offline settings. All four passed in 1.41 seconds; tokenizer-checks-09.log records the nodes' module and result. No download or checkpoint load. These runs close the final-source native verification gap in the historical worker/reviewer reports; they do not establish checkpoint fit quality, time, memory or pilot generalisation.

## Remaining work

Task 2 source is reviewed and verified. Real regression fits and resource/timing evidence remain pending. Task 3 source implementation may proceed. The Jacobian self-check, map gate, replay identity and scientific profiles have not run. No adapter fits are authorised yet.
