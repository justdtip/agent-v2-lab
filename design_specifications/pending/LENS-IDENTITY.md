# Issue 99: a lens carries the model it was fitted on, and the wrong one is refused

Patch: `design_specifications/pending/LENS-IDENTITY.patch`, against `8bd6a17`. Ten files, 357
insertions. **Not landed.**

Verification, in a worktree against its own source: **873 passed, 1,094 skipped, exit 0** — the
skips are the MLX-reaching files while the 1,200 evaluation holds the box, and `test_live_lens.py`
is not among them, so the new tests ran. `ruff check` clean on every changed file, and two files
are now cleaner than at HEAD (`scripts/lens_fit.py`, `tests/test_lens_runtime.py`); the four
findings left in `live_lens_pilot.py` are pre-existing and one fewer than at HEAD.

## The defect, demonstrated closed on the real artifact

Not a fixture. The real Qwen lens in `models/jlens/`, loaded against a Gemma identity, no MLX
imported:

```
qwen lens digest: 381c089dcffead8147ee91f944496f468cce2c7d593e0a1b17230745055aea12
pilot pins      : 381c089dcffead8147ee91f944496f468cce2c7d593e0a1b17230745055aea12
unchanged by the stamp: True

REFUSED against Gemma:
  lens Qwen3.5-4B_jacobian_lens_n1000.npz was fitted on qwen35-4b
  (mlx-community/Qwen3.5-4B-MLX-4bit, 32 layers), and it is being loaded against
  gemma3-4b (google/gemma-3-4b-it, 34 layers). Refusing: a lens of the right width on
  the wrong model produces ranks that look exactly like a finding.

loads against its own model: 31 maps, layers 1 .. 31
```

Both identities named, as the acceptance requires, and the same file against its own model loads
unchanged.

## Two places for the identity, and the precedence is not taste

**Inside the archive** is stronger, because the digest the caller already checks covers it. Every
lens `write_lens` produces from now on carries it there, and `write_lens` refuses an identity whose
layer count disagrees with the fit.

**Beside the file, in the sidecar**, for the two hosted lenses converted before the field existed.
Stamping them inside the `.npz` would change bytes whose digest is pinned in
`scripts/live_lens_pilot.py` and published in `research/records/LIVE-LENS-PILOT-2026-09-07/`,
invalidating those records to fix a defect that never touched them. So the stamp goes in the
sidecar and is **bound to the file by the sidecar's own `npz_sha256`**, which `load` checks against
the digest it just computed. A sidecar that names a different file is refused as not being evidence
about this one, and there is a test that constructs exactly that.

Both hosted lenses are stamped, by `scripts/stamp_lens_identity.py`, which records
`model_stamp: {retroactive: true, stamped_utc, evidence, reason}` beside the identity. The Qwen
lens's digest is byte-identical afterwards, shown above.

## Why three fields and not two

Hidden size cannot separate the two models on this box: Qwen3.5-4B and Gemma 3 4B are **both
2560-dimensional**. The layer count catches Gemma-onto-Qwen by arithmetic — the Gemma lens has a
map at layer 33 and Qwen has 32 layers, so the existing bound fails — and misses Qwen-onto-Gemma,
where `1 <= 31 < 34` holds for every map. **A guard that catches only the direction nobody would
take reads as protection and is worse than none.** So the identity is the registry name, the
`hf_id`, and the layer count, and the test that motivated all of this constructs the direction the
old bound missed.

## What a caller had to change

`LensMaps.load` gains a required `identity`. Six call sites pass what the *view or the snapshot*
says the model is, never what the lens says: `prose.py` twice, `replay.py` twice, `artifacts.py`'s
round-trip, and `live_lens_pilot.py`. That direction matters — the lens has to agree with the run,
not the other way round.

Four test fixtures wrote bare `np.savez(lens, J0=...)` lenses and now stamp them. Those failures
were the correct behaviour of the new code and not defects in it, and the diff shows every one.

## What this does not do

It does not check that a lens was fitted at a context length where its claims hold. The hosted
Gemma lens was fitted at `--max_seq_len 128` against a 1,024 sliding window, so it says nothing
about the sliding-against-global contrast, and no identity field can carry that. It is a caveat for
the record and the pilot's pre-registration, not for this patch.
