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

## Device readiness, D-CRO, 2026-09-11 — gate 0 passed, one input built, one defect found before the window

Read-only on the card except where stated. No model loaded, no window taken, and the checkout's
`git status --porcelain` was 0 lines before and after every step, so no other seat's resume key moved.

**Gate 0, the part that needs no model: passed, measured here.** `load_upstream().provenance()` on
the card returns commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`,
`commit_matches_expected: true`, `vendored: false`, resolved from the **installed distribution**
rather than a clone — there is no `/workspace/jacobian-lens` and `$JLENS_PATH` is unset, so this is
the `_installed_commit()` fallback taking the path it was written for. `torch 2.14.0+cu130`,
`cuda 13.0`, `torch.cuda.is_available()` true, one RTX PRO 6000 Blackwell, 97,887 MiB, idle at 1 MiB
and 0% when read. MLX absent, as the environment intends. All six of my modules are present on the
checkout at `c4e7da9`.

**The corpus was not on the card, and now is.** The runbook says "the corpus under `data/`"; there
was no `data/` directory and no WikiText in the cache. Built with `scripts/lens_corpus.py --corpus
prose --download-prose` against `gemma3-4b-cuda-bf16`, from the pinned `Salesforce/wikitext`
`wikitext-103-raw-v1` **validation** split (the Director's answer, `corpus.py:588`): **201 fit
sequences / 205,824 tokens and 50 held / 51,200**, manifest sha256 `b8c2ab292899fff5…`, sequences
sha256 `2cccc19fd7fd254b…`, at `/workspace/lens-corpus/`.

**Deliberately outside the checkout.** `data/` is *not* in `.gitignore`, and the resume key hashes
the working tree's content **including untracked files**, so a corpus written under `data/` would
have changed the tree digest for every seat mid-run and made WS-B's records resume only their own.
`models/` and `outputs/` are ignored and are safe; `data/` is not. Artefacts go to `models/jlens/`;
inputs go to `/workspace/lens-corpus/`.

### The defect: two fits of one checkpoint can declare the same dtype and run different arithmetic

WS-A's device finding — on CUDA at 1,400 tokens the promoted-float32 block path differs from native
bf16 by **69.4%**, 6.9 on CPU, 1.24 at 64 tokens on both, while the native path is exact at both
lengths — lands inside the golden test, and it would not have been caught by any gate I built.

`fit_upstream_jacobian` runs the model's own forward and captures; it is **native** by construction.
My finite-difference estimator captured the residual with `.detach().float()` and wrote that tensor
back as the block's output, so **every block above the source ran promoted float32**. Both sides
read the same weights, so `_observed_precision` reported `bfloat16` for both, ν's `precision` block
matched, and `assert_estimator_is_the_only_difference` passed a pair that had not run the same
arithmetic. The golden residual would have been the estimator difference **plus** the precision
path, with nothing in the record able to separate them.

**Fixed, and the fix is a declaration rather than a preference.** `capture_dtype` is now a field of
the observed precision — `"native"` for any estimator that does not replace an activation, which is
the truth for the exact side — and a parameter of the finite-difference fit, defaulting to
`"native"` so the two estimators run one arithmetic. `promoted-float32` remains reachable because it
is a path worth *measuring against* native rather than inheriting by accident. It is declared in ν
and in the provenance, and `golden.COMPARABLE_KEYS["precision"]` now contains it, so a cross-path
comparison is refused by name with `precision` in the message.

**Proved, not asserted.** The fixture is float32 throughout, so both paths give the identical
residual there (3.6286e-03 worst layer, unchanged) — a fixture that cannot tell them apart, which is
why the test asserts the property directly: the perturbation handed back to a bf16 block is
`torch.bfloat16` under `native` and `torch.float32` under the promoted path. A bf16 fixture fit runs
clean and declares `native`; the promoted path on that same bf16 fixture **raises a dtype error in
the block's own matmul**, because these blocks do not silently promote and a real HF block does.
That is the whole hazard in one line: loud on a fixture, invisible on the model that matters.

**What every capture record must therefore say**, and what mine now do: the arithmetic path by name,
beside the weight dtype, because the weight dtype does not imply it.

**Still missing before §4 can run** (none of it mine to decide): no lens artefact exists on the card
(`models/jlens/` is empty), so the fits are the first thing the window does; the FD subset — how
many rows and which positions — is a declaration the record must carry and I will fix it in the
manifest before the run, with the exact side re-run on that same subset.
