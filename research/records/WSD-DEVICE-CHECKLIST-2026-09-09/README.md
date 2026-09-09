# WS-D on the device: the checklist, and the number each gate must produce

**Written on the laptop, to be executed on the rented GPU. The Director's instruction is that all
subsequent runs use the device and this box is for tests and corpus freezing, so what follows is not
a plan — it is a list of gates with their expected values, so that a failure on the device is
diagnosable in the hour rather than reproduced over a day.**

Every number below is either **measured here** and stated with what measured it, or **to be measured
there** and stated as unknown. Nothing is asserted from arithmetic. Where a gate has no threshold
yet, it says so rather than inventing one.

---

## 0. Before anything: what the environment must say

| check | expected | if it differs |
|---|---|---|
| `load_upstream().provenance()["commit"]` | `581d398613e5602a5af361e1c34d3a92ea82ba8e` | the reference moved; every §6 line item was read against this commit |
| `provenance()["commit_matches_expected"]` | `True` | as above |
| `provenance()["vendored"]` | `False` | upstream is called, never copied |
| `torch.cuda.is_available()` | `True` | wrong box |
| model dtype across **every** block | uniform `bfloat16` | `MixedPrecisionModel` refuses; `device_map="auto"` and quantisation produce this |
| `attn_implementation` on the fitting model | `"eager"` | batched Jacobian rows regress to sequential — measured on CPU at 0.65x under `sdpa` against 1.50x under `eager` |

**The commit fallback matters here specifically.** A `[cuda]`-extra install has no `.git`, so
`_installed_commit()` reads `direct_url.json`'s `vcs_info.commit_id`. If provenance reports `None`
on the device, that fallback did not fire and the gate above is unverified rather than passed.

---

## 1. The adapter's own tests, re-run on the device

35 tests, all device-independent, all passing here on CPU float32 against a synthetic 4-block
decoder. They are re-run there because "passes on CPU" is not "passes on CUDA", and because four of
them are negative controls whose value is entirely in failing when they should.

| gate | expected |
|---|---|
| `pytest tests/test_lens_upstream.py` | 35 passed |
| the four negative controls | a transposed artefact, a layer-shifted artefact, a mixed-precision model and a nested selector fit each **refuse** |

---

## 2. The golden test: finite-difference against exact autograd

**The one number nobody has.** Refit the hosted lens's own recipe and compare per layer against
`neuronpedia/jacobian-lens`.

**The recipe, which is upstream's definition and not a choice:** `max_seq_len=128`,
`skip_first=16`, so at most **111 valid positions** per prompt. Verified here against the clone:
both are defaults of `fit` *and* of `jacobian_for_prompt`.

| gate | expected | measured where |
|---|---|---|
| both operands' storage dtype before comparing | `float32` on **both** sides, asserted, not assumed | the assertion is the gate |
| hosted lens as loaded | 33 maps, `d_model` 2560, `storage_dtype` `('float16',)` | measured here |
| residual floor from fp16 storage | **4.88e-4 relative per element** (`2**-11`) | arithmetic, verified against the artefact |
| per-layer agreement | **unknown — this is the measurement** | — |

**The reading rule, which must be fixed before the number exists.** A residual at or below
4.88e-4 relative is *indistinguishable at storage precision*, never *agreement*. The hosted lens is
stored float16 and upcast on load; our fit is float32 throughout. Any comparison that crosses that
boundary carries the floor.

**And the estimator convention must match or the comparison is of two instruments.** Upstream is
target-summed and source-averaged; our MLX fit used the same-position reduction. The golden test
runs **upstream's** ν against the hosted lens, and the sidecar records which was used.

---

## 3. Memory, chosen rather than set

R47 extends from *stop* to *choose*: measure one unit at the largest context the run will reach,
then pick the largest `dim_batch` fitting under the R47 fraction of `torch.cuda.mem_get_info()`.

**Measured here, and each is a floor rather than a total:**

| component | bytes |
|---|---|
| fp32 unembedding, vocab 262,208 × 2,560 | **2.50 GiB** |
| 33 accumulator pairs, 2,560² fp32 | **1.61 GiB** |
| residuals, 33 layers × 2,816 positions × 2,560, fp32 | **0.886 GiB** |
| residuals at the golden test's 128 tokens | ~0.04 GiB |

**What is not measured here and must be on the device:** the fitter's own peak. Codex's prose fit
measured **12.44 GiB at 128 tokens**, where residuals are 0.04 — so roughly twelve gigabytes is the
fitter itself, and no arithmetic on this box predicts it.

**A caution earned the hard way.** A 2.817 GiB figure from this laptop was briefly taken as the
fit's memory floor and it was a measurement of a *capture pass*, not a fit. It nearly authorised a
configuration at an eighth of the true cost. **Measure the thing that will run, at the size it will
run at.**

---

## 4. What blocks the golden test and is not code

| blocker | state |
|---|---|
| the corpus split | `corpus.py` pins `DATASET_SPLIT="validation"` and refuses anything else; the hosted lens is WikiText-103 **train**. Needs an authorisation or an explicit ruling that the comparison crosses the split axis. |
| `compare_maps.py` | makes two three-positional `LensIdentity(name, hf_id, num_layers)` calls; the live signature is `(base, num_layers, training=None)`. It sits in a records directory, so whether the fix edits the record or lands as a copy is the Chief's call. |
| a pass threshold | **none exists.** Neither the order nor the plan states one. The orientation check's `atol=1e-4` is an internal convention check, not the golden test's tolerance, and must not be borrowed as one. |
| CLI wiring | `scripts/lens_fit.py` does not know about the adapter. |

---

## 5. What this box keeps doing

Adapter fixture tests, corpus freezing, and short probes of minutes. Not announced blocks, and not
the golden test. The gates above are written so that the first hour on the device runs diagnosis
rather than an experiment.

## §4's blockers, resolved — 2026-09-09 evening

All four are cleared. Two by ruling, two by code; none by being decided to matter less.

| blocker | resolution |
|---|---|
| the corpus split | **Ruled** (plan §16.13). The golden test compares finite-difference against exact on the *same* rows of the pinned `validation` split, so the estimator difference is isolated and the split does not enter the comparison. The hosted-versus-fitted comparison declares the train/validation difference in ν as a corpus difference and is **not** a golden gate. The Director's `validation` answer stands unchanged. |
| `compare_maps.py` | **Done, and not by editing the record.** `LensIdentity` is keyword-only as of `57a0b89`, so no positional call can silently rebind again. The record's script is left as-run, because its own `comparison.json` carries the pre-`a960d80` `{"name", "hf_id", "num_layers"}` shape and rewriting it would make the record claim something ran that did not. `GEMMA3-REGRESSION-2026-09-08/README.md` now says this, and says a current-signature copy belongs beside it. |
| a pass threshold | **Ruled** (plan §16.13), and the ruling is that none is chosen in advance. Exactness is gated at zero where it holds by construction — upstream exact, re-run on the same rows. The FD-versus-exact residual per layer **is** the finding, reported with the finite-difference epsilon and the fp16 storage floor declared, against the fixture's own residual as the expected magnitude. What is gated is the controls: orientation through upstream's transport, a layer-shifted artefact, and a wrong-corpus fit, each required to fail by **more** than the residual. The `atol=1e-4` caution in §2 stands: it is an internal convention check and is still not a tolerance. |
| CLI wiring | **Done** (`6c2f42e`). `scripts/lens_fit.py` reads `device.backend()` once, after argument validation and before the first import that reaches MLX, in the shape `stage_train` uses. A non-MLX backend is **refused**, not fallen through, and the refusal names the estimator difference so the two backends cannot be treated as interchangeable. Five fixture tests, including one that asserts the read's position in the source, because a read moved below the imports would still pass behaviourally on this box. |

The gate tables above are unchanged. Nothing here supplies a number that was marked unknown; the
per-layer agreement in §2 is still the measurement and still does not exist.
