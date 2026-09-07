# Lens fitting implementation record — 7 September 2026

Authority: design_specifications/pending/LENS-FITTING-REQUIREMENTS-2026-09-07.md, including sections10–11. Persistent worktree: /Users/daniel.tipton/worktrees/lens-fitting; branch codex/lens-fitting. Starting tree feb118a. This record covers implementation and checkpoint-free verification. No fit has been run or read here.

Source checkpoints will be committed piece by piece. Runtime fitting records require separate pre-run declarations, normal access to the primary checkout model lock, self-checks before readings, and the exact frozen corpus/weights/source identities. Cache acceptance currently owns checkpoint execution. Replay integration waits for the amended cache patch landing. Adapter fitting waits for3.1–3.6 to land.

Scientific conventions to resolve before dependent work: the prototype adds ridge_multiple * mean_diagonal(XTX)/n * n to XTX, while the literal fixed grid in3.2 gives ridge_multiple * mean_diagonal(XTX)/n. The implementation must not silently choose between those objectives. Code inspection also confirms that legacy atlas statistics pool tokens; exact identity reproduction and the new episode-separated profiles need distinct output scopes.

The plan records source-step indices before dropping long sequences (fifth means one-based5,10,...), stores row-solve matrices transposed for LensMaps, and distinguishes uncentered reconstruction R2. These conventions remain visible to review.
