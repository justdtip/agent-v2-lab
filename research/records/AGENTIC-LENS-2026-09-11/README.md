# Agentic Gemma lens: bounded 4B pilot

## Status and scope

Director-approved: ahead of queued SAE work; Gemma 3 4B only, repository layers
18 and 24; short resource calibration followed by at most four hours of fitting.
The implementation and CPU verification are recorded here. **No checkpoint
calibration, fit, admission, new pairing or scientific comparison has executed.**

Implementation began at `1e3258e`; the Chief's brief is `74d7b71`.
The shared branch advanced to `5a6406d` during implementation; its changes
were retained without switching or resetting the checkout.
Existing archives, admissions, capture bytes, seals and results remain unchanged.

## Findings established without a model

The current device fitting path calls upstream exact autograd, not finite
differences. Its target-position gradients are summed and source-position
gradients averaged; prompt maps then receive equal weight. Therefore changing
the context/position distribution changes the instrument, not merely its cost.
The new fit preserves this reduction. A new corpus cannot by itself establish
that an averaged derivative predicts an individual perturbation.

The actual workspace adapter reconstructs all 7,629 capture inputs using the
capture's separate prompt/completion tokenisation and tool-name boundary. The
600 comparison cells belong to 300 excluded episodes. Training splits contain
3,927 remaining rows, from which 599 candidates are frozen, one per episode.
All twelve task families remain represented. Candidate contexts range from
438 to 3,827 tokens; no context was truncated. The actual 600 comparison cells
span contexts of 438–1,554 tokens, not the broader archive's maximum position.
The inventories and input digests are in `corpus-verification.json`.

The candidate texts are expert demonstrations in agentic format, not newly
generated model trajectories. Episode exclusion does not promise disjoint
templates or task families. Prompt spans are explicitly unclassified rather
than guessed; generated material ends at the first tool-name token because that
is where the original capture input ends. No call-argument coverage is claimed.

## Technique, independent of implementation

1. Preserve exactly the model inputs used for the observations, including the
   original context and token positions. Exclude comparison episodes from fit.
2. Measure a short, median and longest candidate context under one declared
   precision and arithmetic schedule. Check the optimized derivative against
   the sequential reference on an identical forward width.
3. Freeze a family-balanced sample using only measured cost and metadata, with
   20% time headroom. Persist individual maps before averaging, so interruption
   cannot duplicate or omit contributions. Refuse a final lens until the full
   requested sample is present.
4. Admit the new lens separately, then measure its own capture pairing. The
   original capture lens is a historical identity reference, not the new lens.
5. Read both lenses on identical held episodes. Report probability mass,
   coverage and agreement separately. Read the full vocabulary to obtain the
   model's greedy winner; never substitute the expert token or six-tool winner.
6. Test perturbation prediction separately with frozen displacements, matched
   controls, an unchanged-state check, local autograd derivatives and actual
   forward responses. A failure to predict perturbations does not erase a
   successful state readout, and vice versa.

## Implementation and execution

The workspace source contract is `rendered_workspace_v1`. The validating reader
reconstructs the indexed inputs and verifies source digests, metadata, positions,
target tokens, episode exclusion and the frozen candidate rows. Existing corpus
formats retain their original reader.

The graph-once schedule uses upstream's recorder and the existing batched VJP
implementation. Its **forward and anchor width are one**; `dim_batch` names the
cotangent batch. The declaration preserves that distinction. Each completed
row has a digest-bound archive and receipt. Resuming checks the complete
input/code/checkpoint/schedule contract. Partial samples cannot be admitted.

CPU entry points (from the repository root, with `PYTHONPATH=src`):

```text
python scripts/lens_corpus.py --corpus agentic \
  --workspace-corpus INPUT/all-splits.jsonl --capture-index INPUT/index.jsonl \
  --positions INPUT/positions-sample.json --tokenizer-json SNAPSHOT/tokenizer.json \
  --out CORPUS/manifest.json
```

The local verified corpus is `outputs/agentic-lens/corpus/manifest.json`.
Source paths are bound absolutely: on another machine rebuild from the same
hashed sources; do not edit those paths inside an existing manifest.

In a declared device window with `LLL_BACKEND=torch`, using a fresh output
directory and the actual available memory ceiling:

```text
python scripts/agentic_lens.py calibrate --corpus CORPUS/manifest.json \
  --snapshot SNAPSHOT --output CALIBRATION --dim-batch 1 \
  --peak-limit-gib MEASURED_CEILING --budget-seconds 14400
python scripts/agentic_lens.py fit --corpus CORPUS/manifest.json \
  --snapshot SNAPSHOT --output FIT --plan CALIBRATION/plan.json \
  --peak-limit-gib SAME_CEILING
python scripts/admit_device_lens.py FIT --checkpoint SNAPSHOT \
  --model-base google/gemma-3-4b-it
```

Calibration begins with the sequential width-one reference. A wider cotangent
batch requires a separate calibration output and passing checks, never editing
an existing plan. The fitter re-derives its frozen plan from the calibration
receipt before acquiring a lock or loading weights. The allocator has an actual
process limit; memory and elapsed time are also checked at row boundaries.
One in-flight row cannot be interrupted at an exact four-hour boundary; the
measured longest-row estimate is reserved before starting each next row.
If the budget cannot cover every eligible family, the run refuses.

New pairings use `scripts/measure_pairings.py` with explicit layers 18 and 24 and
`--capture-reference-archive` / `--capture-reference-sidecar` naming the original
prose lens. No capture metadata is rewritten. Pairing consumers additionally
verify the measurement's exact row, input IDs, prompt, residual bytes and capture
schedule, because position/context identities alone can collide across prompts.

`python -m local_llm_lab.pipeline.lens_fitting.agentic_readout CONFIG.json`
produces full-model score rows and the paired comparison from explicit admitted
old/new lenses, their pairing files, the frozen corpus/capture/positions,
checkpoint manifest, six tool IDs and a declared pairing bound. Tool IDs must
match the captured tool order under the actual tokenizer. An unmeasured
perturbation remains labelled `not measured` in a readout-only report.

## Boundaries on interpretation

The prose lens's prior W-5 agreement (approximately 0.99 at the measured action
readouts from layer 24) remains the baseline, with its original population and
resolution condition. A new comparison on 600 cells is not a reproduction of
that rate on the entire W-5 population. Neither passing file admission nor a
small arithmetic pairing displacement certifies semantic validity.

The initial comparison is descriptive: individual cell rows, equal-episode
distributions, coverage, unresolved cases, margins and tails are retained. It
does not borrow E2's bootstrap, silently tune a threshold, or certify general
workspace/access-consciousness claims. A successful pilot is a measured change
in what this instrument reads or predicts on its declared population.

## Operational boundary

During implementation the card was running the Chief's layer-25 SAE A2 sweep.
No process was stopped and no lock cleared. Calibration timing requires an
appropriate available slot; priority before *queued* SAE work does not authorize
interrupting an already running job. Pending device work is **unexecuted**, not
passed on the strength of laptop fixtures.

## Concrete comparison and directional configuration

`config.example.json` is a **nonexecutable template**, not a run approval. Copy it
to a new configuration and replace every placeholder. In particular, leave no
invented agentic archive or pairing path: these files exist only after the fit,
separate admission and new pairing measurements execute. Both sidecars must be
the matching `.json` next to their admitted `.npz` archives. The prose lens is
the historical capture reference for both arms; the old capture is never resealed
to name the new lens.

`checkpoint_manifest` is a JSON object mapping each checkpoint filename directly
to its SHA256, exactly the loader/capture `load_report_sha256` mapping. It is not
the capture manifest, an enclosing `{ "sha256": ... }` object or a filename-only
list. Use the unchanged captured workspace JSONL and sealed 600-cell positions,
not the new fit-corpus manifest, for the `corpus` and `positions` fields.

The two null fields are deliberate. `maximum_pairing_relative` must be an
explicitly ruled finite nonnegative displacement bound; this template does not
choose one. `tool_token_ids` must become the six actual first-token IDs in the
captured order: `read_file`, `finish`, `replace_text`, `calculate`, `search_files`,
`list_files`. The readout driver checks them against the real tokenizer and
capture declaration. These are tool-first-token comparisons, not complete calls.

The recorded 16k dictionaries are `resid_post_all/layer_17_width_16k_l0_small`
and `resid_post_all/layer_23_width_16k_l0_small`, corresponding to repository
layers 18 and 24. Supply their verified directories and digest receipts. Do not
substitute `resid_post`, another sparsity or feature indices from another suite.

After filling the original capture/prose/dictionary paths, freeze directions on
CPU without loading a model; this can precede the new lens fit:

```text
python -m local_llm_lab.pipeline.lens_fitting.agentic_directions --prepare CONFIG.json
```

This creates `directions_manifest` and an NPZ with the same basename. Every
requested cell/layer gets `decode(encode(h)) - h`, one norm-matched Gaussian
control, and one norm-and-angle-matched control. Seed **20260911** is keyed by
row, site, layer and arm so iteration order cannot change the draws. The manifest
records the algorithm, NumPy version, seed, hashes, source-residual identities,
and **one draw per control per cell**. This is a new bounded pilot, not a repeat
of the historical eight-draw analysis. Zero errors produce zero controls; the
manifest flags zero-base cases where an angle is undefined. Preparing this
bundle does not measure a model response.

After both admissions and same-population pairings exist, and in an authorized
model window, run the two separate measurements:

```text
python -m local_llm_lab.pipeline.lens_fitting.agentic_readout CONFIG.json
python -m local_llm_lab.pipeline.lens_fitting.agentic_directions CONFIG.json
```

The first writes the new `scores` file and `output/comparison.json`. The second
consumes the frozen direction bundle and writes `directions_output`. Keep all
three output destinations new; the drivers refuse an existing result. Run them
serially under the normal device scheduling rules. The readout report continues
to say perturbation **not measured**; the separate directional file is the
evidence that the directional experiment executed.

Directional responses preserve the full original input, float32 width one and
the original capture flags. The unmodified source residual is checked against
capture bytes before every replacement; same-state target equality precedes
autograd. Both lenses are compared against the same exact derivative and actual
response. The endpoint is the **pre-final-normalization residual vector**. This
measures local directional fidelity, not a logit-gap, emitted-token or
complete-call behavioral effect. No real-model directional response is reported
by the current implementation record; device execution remains unexecuted.

## Verification

The final scoped CPU suite passed **368 tests**; Ruff, formatting and diff checks
passed. See `verification.json` for the exact command. Both attention-kernel
fixtures gave zero maximum absolute difference between the sequential and
graph-once estimators. These are fixture results, not device acceptance.
Independent integration review found a near-unit probability rounding defect;
its failing producer-to-reader witness is now a passing regression. No remaining
scoped integration finding was reported.
