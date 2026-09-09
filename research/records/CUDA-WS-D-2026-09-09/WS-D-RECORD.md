# WS-D: six ways a check agreed with itself, and what to compare instead

**D-CRO, 2026-09-09.** The stream's record as a whole. Everything below was measured on one laptop,
most of it on CPU. **Nothing has run on a GPU.** The unexecuted list is at the end rather than
scattered through, and the device's own order is a separate record,
`research/records/WSD-DEVICE-CHECKLIST-2026-09-09/`, which carries the number each gate must produce.

Findings first, the transferable technique second, the implementation third. Only the first two
survive this codebase.

---

# Part one: what was found

WS-C's stream found five ways a wrong thing passes. This stream's shape is narrower and it is worth
naming precisely, because it is not the same failure: **a check that compares something with
itself**. Not a tolerance too loose, not a sample too small — a comparison whose two sides are the
same object, or the same derivation, so no input can separate them. Every finding below is one, or
is the reading error that a self-comparison lets stand.

## 1. The orientation check would have compared a line with itself

The adapter writes a lens by converting upstream's block indices to this repository's layer
convention, then checks the orientation against upstream's own transport. Run on the dictionary the
writer is about to hand to `write_lens`, that check crosses nothing: it compares the conversion with
the conversion, one line above. **It cannot fail for any input.**

Run instead on the artefact *reloaded through* `LensMaps.load`, it crosses three independent pieces
— `repo_layer_of_upstream`, `write_lens`'s `J{layer-1}` key, and `LensMaps.load`'s `layer = i + 1` —
and any one of them moving breaks it. The cost is real and is recorded rather than hidden: the
sidecar cannot carry the check's numbers, because the check runs on the file the sidecar describes.
The sidecar states the convention; the executed deviations go to the caller in the returned dict.

## 2. Two selectors 17.5% apart shared a fingerprint

The fit records a fingerprint of its position selector so two fits can be compared only if they
selected the same positions. The first fingerprint hashed a *summary* of the selector, and two
selectors whose chosen positions differed by 17.5% produced the same sha256. A fingerprint that
collides is worse than no fingerprint: it converts "these are incomparable" into "these agree".

The fix hashes the selection itself over declared probe lengths, so the fingerprint is a function of
what the selector *does* rather than of how it describes itself.

## 3. Precision was read from one sample, and the sample was the wrong block

The adapter refuses a mixed-precision model, because a lens fitted across blocks of different dtypes
is not a map of one geometry. The first implementation read the dtype from one sampled parameter.
The cotangent inherits the **target** block's dtype, so a single sample can report the target's
precision and say nothing about the thirty-two blocks the fit actually reads. A model that is
bfloat16 everywhere except one block passes.

`MixedPrecisionModel` now refuses on the dtype set across *every* block, which is also what the
device checklist gates on.

## 4. A test of mine caught a fix of mine, and the shape is the same one

The lens loader gained `storage_dtype`, recording what precision the maps were stored at before the
float32 cast — a declaration that carries its measurement (R60c). My first version measured the
dtype set of *every* array in the archive, which includes the identity stamp, and reported
`['float32', 'uint8']` for a float32 lens. The stamp is not a map, and a figure that answers "what
are the maps stored at" must not be computed over things that are not maps.

Scoped to the `J` keys it is `['float32']`. The same commit's second defect: `archive[name]` was
accessed twice per map, decompressing each one twice, 0.26s against 0.22s over 33 maps.

## 5. A memory figure was measured on the wrong artefact and nearly sized the run

I reported 2.817 GiB as the fit's memory floor and authorised against it. It was a **capture pass**,
not a fit — roughly an eighth of the true cost. The real bf16 figure is about 12.3 GiB, and the
device checklist now carries the component arithmetic rather than a single number.

The error was in a docstring sentence of mine that named the measurement, and it survived because
nothing downstream re-derived it. **Measure the thing that will run, at the size it will run at**,
and say which artefact produced the number beside the number.

## 6. A branch that is an ancestor of its target looks exactly like a branch awaiting merge

The Chief asked for `cuda-ws-d` to be merged into `cuda-migration` and for the merge-tree to be
confirmed clean first. The merge-tree was clean and there was nothing to merge: `cuda-ws-d`'s tip
*is* the merge base, 0 ahead and 47 behind, and the adapter had reached the integration branch
through an ordinary merge of main. Both of us had read "the files are on `cuda-ws-d`" as "the work is
on `cuda-ws-d`, waiting".

The finding was in the direction neither of us was checking. Main had deleted the adapter and its
suite, correctly, because the main line takes no CUDA code. Relative to the shared base main deletes
and the integration branch does not touch, which git resolves by **taking the delete, with zero
conflicts**. Simulated with `merge-tree --write-tree`, the merge's entire diff against
`cuda-migration` was two deletions: the 975-line adapter and its 35 tests, from the only branch that
had them, silently, reported as success.

A clean merge-tree means the merge is unambiguous. It does not mean the merge is harmless. The fix
is now plan §16.13: when the main line deletes something the integration branch keeps, the restore
lives **inside** the merge commit, so the base moves past the shared commit and the trap disarms
once instead of re-arming at every merge.

---

# Part two: the technique

**Ask what the check would do, not whether it would complain.** Six of the findings above are the
same question asked differently. `merge-tree` reporting zero conflicts answered the question I was
told to ask; the diff of the tree it wrote answered the one that mattered. A fingerprint that
returns a hash answers "did it run"; hashing the selection answers "does it separate". Wherever a
check reports pass or fail, construct the input it should fail on and confirm it does.

**A check must cross a boundary.** Write the artefact, reload it, and check the reload — not the
object still in hand. Read precision from every block, not a sample. Compute a declared figure over
exactly the things the declaration names. In each case the discipline is the same: the two sides of
a comparison must have travelled through different code to get there.

**Say which artefact produced a number, beside the number.** The 2.817 GiB error is not a slip in
arithmetic; it is a number correctly measured on the wrong thing, and nothing carried enough
provenance for a reader to notice. Every figure in the device checklist is either attributed to
where it was measured or marked unknown.

**A missing figure must never read as a passing one.** The live-lens capture records
`final_readout_max_abs_error` on every forward and writes `None` when no readout ran, rather than
omitting the field. An absent number and a good number must not look alike in a record.

**Fix the assumption at the point where it is loud.** `LensIdentity` is now keyword-only because its
fields moved once already — `(name, hf_id, num_layers)` became `(base, num_layers, training)` at
`a960d80` — and two-argument positional callers survived by coincidence, so nothing forced a review.
The cost of that coincidence is a record whose script can no longer run. Keyword-only makes the next
field change break at the call, with the field's name in the error.

---

# Part three: the implementation, which is the least of it

| piece | where | size | tests |
|---|---|---:|---:|
| the upstream adapter | `pipeline/lens_fitting/upstream.py`, `cuda-migration` | 975 lines | 35 |
| the import seam | `local_llm_lab/upstream_ref.py`, main | — | 3 |
| the CLI backend seam | `scripts/lens_fit.py`, main | 55 lines | 5 |
| `storage_dtype` on loaded maps | `live_lens/instruments.py`, main | — | in the live-lens suite |

The adapter calls upstream at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` and never vendors
it. `upstream_ref.py` is deliberately on main and not on the CUDA line: it is the single import path
WS-A and WS-D share, it is not backend code, and moving it would recreate the two-import-paths
problem it exists to close. The adapter's own suite lives beside the adapter.

The CLI seam reads the backend once, after argument validation and before the first import that
reaches MLX, and **refuses** a non-MLX backend rather than falling through. The refusal is not a
placeholder. Both fits on the MLX side are MLX, and the torch path when it lands is upstream's exact
autograd rather than a port of the finite-difference stage, so the two backends will produce
different instruments on the same corpus. That difference is the golden test's finding, and a CLI
that quietly gave each box whichever estimator it could run would absorb the measurement into a
packaging decision.

---

# Findings for other seats

## `device._importable` raises on any stubbed module — WS-E

`importlib.util.find_spec` raises `ValueError`, rather than returning `None`, for a module that is in
`sys.modules` with `__spec__` set to `None`. That is what every hand-built stub is, and **twelve test
files in this suite install one**. The CLI backend seam is the first caller downstream of such a
stub, and `tests/test_lens_regression_progress.py` went red on it immediately. Fixed here by
answering from `sys.modules` first, which is also the correct answer — a loaded module is importable
by definition — with a test. `device.py` is WS-E's; the change is one function and is flagged rather
than assumed.

## `compare_maps.py` in `GEMMA3-REGRESSION-2026-09-08` is as-run and no longer runs — the Chief

It constructs `LensIdentity` with three positional arguments in the pre-`a960d80` order. Its own
output dates it: `comparison.json` records the old `{"name", "hf_id", "num_layers"}` shape, which the
current class cannot produce. **Not repaired**, deliberately: rewriting an as-run script inside a
record would make the record claim something ran that did not, and `SHA256SUMS.json` and
`source-hashes.json` bind that text to that run. A current-signature copy belongs beside it. The
record now says so.

## Twelve `B023` closures, confirmed benign, and the dead assertion beside one — SWE-2, SWE-1

SWE-2's reading was right in every cell and I re-ran the linter against the code to confirm it. The
finding was next door: `scripts/fixed_history_lens.py:207` is `assert (comparison) or True`, always
true, never tested anything — in the run that produced a published log-probability figure. The
assumption those figures actually depend on is that prompt and turn do not re-tokenize at the seam,
which nothing was checking. Measured from the artefact and the tokenizer alone: **11 adapter turns
and 11 base turns, all 22 seams clean**, so `ARM-A-DIVERGENCE-2026-09-07` stands. The assertion is
reported and not repaired, because it may carry `or True` precisely because the MLX and HF
tokenizers disagree on specials.

---

# The suite, and the state of the box beside it

| | |
|---|---|
| full suite, main | passes |
| adapter suite at `origin/cuda-migration`'s tip | 35 passed, 0 skipped, 0 failed |
| `tests/test_upstream_ref.py`, main | 3 passed |
| `tests/test_lens_fit_backend.py`, main | 5 passed |
| model-run lock | not held |
| declared box window | none |
| MLX / torch processes resident | none |
| memory available | 12.74 GiB |

Torch 2.14.0 and MLX 0.32.2 are both installed in the shared venv, so `device.backend()` on this box
returns `mlx` and the torch branches are reached only under a fixture. That is the whole reason the
seam's tests assert on the source text as well as on behaviour: a backend read moved below the MLX
imports would still pass every behavioural test *here* and fail on the only box that matters.

---

# The device, in order

The device's order is `research/records/WSD-DEVICE-CHECKLIST-2026-09-09/`, not repeated here. It
gives, per gate, the expected value and where that value came from, and it fixes two readings before
the numbers exist: a residual at or below **4.88e-4 relative** is indistinguishable at storage
precision rather than agreement, because the hosted lens is stored float16; and the estimator
convention must match or the comparison is between two instruments rather than two fits.

Two of that record's four blockers are now cleared. `compare_maps.py` and the CLI wiring are done
above. The corpus split and the pass threshold were ruled by the Chief in plan §16.13: the golden
test compares finite-difference against exact on the **same** rows of the pinned `validation` split,
so the estimator difference is isolated and the split does not enter it; and there is no threshold
chosen in advance — exactness is gated at zero where it holds by construction, the FD-versus-exact
residual per layer **is** the finding, and the controls are what is gated, each required to fail by
more than that residual.

## What is still unexecuted

- Every gate in the device checklist. Nothing in the adapter has executed against CUDA.
- The golden test itself. It is the first real number this stream produces and it does not exist.
- WS-D steps 2 through 6: declared ν, the second moment, position bands at transcript length,
  span-conditioned lenses, and the sub-block and multi-target extensions.
- The torch implementation behind the CLI seam. The seam refuses today; it is a dispatch when the
  stage exists.
- The four negative controls — transposed artefact, layer-shifted artefact, mixed-precision model,
  nested selector fit — have unit coverage here and have never run against a real model.
