# SAE–J: device archive admission and A1/A2 runner

Implemented against the Chief's 09:55Z instruction in
`design_specifications/pending/SAE-J-BRIDGE-ORDER-2026-09-08.md` at
`b93385069f787239e499de15a1c4f0af81cb1bc0` on main. The implementation base is
`3d5d17e0d9a008bfd82752d177c4d59dd7d0452b` on the CUDA integration line, where the
bridge and checkpoint reader exist. Branch: `codex/sae-j-bridge-admission`.

This is a file/CPU implementation result. It does not admit any real device capture, establish
an empirical pairing, validate extrapolation, or produce Stage B evidence. No model load, device
job, download, installation, or network action was performed for this implementation.

## Verdicts and commits

**File-only acceptance: PASS for T1–T4 on the named CPU sets.** Integrate these commits in
dependency order: `e5898e0`, `6ae9648`, `434bb55`. They remain local on the named branch.

| Commit | Verdict | Checked set and own result |
|---|---|---|
| `e5898e0` — identity endpoints, separate errors, measured pairing schema | PASS — T2/T3 and pairing seam | `test_sae_bridge.py`, `test_sae_intervention.py`, `test_live_lens.py`: **152 passed, 1 deliberately deselected**, exit 0 |
| `6ae9648` — archive admission and complete merger ν | PASS — T1 | `test_device_lens_admission.py`: **32 passed**, exit 0 |
| `434bb55` — guarded general A1/A2 runner | PASS — T4 | `test_device_bridge.py`: **45 passed**, exit 0 |

Fresh integrated verification of `434bb55`: **350 passed, 1 deliberately deselected**, exit 0
in 52.80 seconds. Scoped Ruff and `git diff --check` also passed. The only deselection is
`test_the_real_dictionary_hook_is_a_layer_the_real_lens_carries`, which would read large real
artifacts and is outside this fixture-only assignment. This is the named checked set, not a
claim that the entire repository test suite was executed.

| Order task | Acceptance covered | Checked set |
|---|---|---|
| T1 | Original files unchanged; repository-to-block key conversion; digest-bound identity and full fit/source provenance; missing/shifted maps, wrong shapes, identity and widths refuse; chunk declarations, populations and weights retained; real `LensMaps.load` at every layer in three-map, 34-layer and 48-layer fixtures | `tests/test_device_lens_admission.py` |
| T2 | Explicit identity endpoint admits final 34th/48th dictionary; missing interiors and undeclared endpoints refuse; pairing guards still apply | `tests/test_sae_bridge.py`, `tests/test_live_lens.py` |
| T3 | Required two-threshold policy; separate raw/score errors; 9.9503719% versus 99.503719% fixture; reverse anisotropy; near-zero denominators refuse; no raw admission mask substitutes for score ranking | `tests/test_sae_bridge.py`, `tests/test_sae_intervention.py` |
| T4 | Real-file A1/A2 and CLI fixtures; dry-run never calls numeric helpers; all selected cells admitted before numerical work; each of eleven pairing fields bound; every input family corrupted independently and refused by name; complete provenance and verbatim limits | `tests/test_device_bridge.py` |

All dimensions in these tests are deliberately small. The 34/48-layer fixtures preserve full
layer indexing while using width-two matrices; they are not real 4B/12B model executions.

## T1: preserve the fit; admit its serialization

```sh
python scripts/admit_device_lens.py /absolute/fit-directory \
  --checkpoint /absolute/hf-snapshot \
  --model-base google/gemma-3-4b-it
```

The source directory must contain `exact-maps.npz`, `nu.json`, and `manifest.json`. The command
writes `admitted-maps.npz` and `admitted-maps.json` beside them. `--output` can name another new
`.npz` file. An existing output or sidecar refuses; the command does not overwrite originals.

Source `J1` becomes loader `J0`, so repository residual layer 1 still reads the original `J1`.
Depth and hidden width are read from the source declaration and verified against the actual
checkpoint's config and complete weight-file hash manifest. The checkpoint's base must be
established by its `_name_or_path` or named Hugging Face snapshot path, and must agree with the
explicit requested base.

The sidecar retains the existing loader's `model` and `npz_sha256` format and adds:

- `model.endpoint: "identity"`, independently required for a final-layer bridge reading;
- `source_archive_sha256` and the source's repository-layer convention;
- source manifest/ν paths, file hashes and complete contents;
- `nu` and its canonical `nu_sha256`, with verified `precision.fit_dtype` added for the bridge;
- all fit widths and their populations/weights, plus checkpoint-file hashes.

The fitter writes `precision.dtype`/`declared_dtype`. Admission derives `fit_dtype` only after
checking those fields against the source manifest. Original ν stays untouched in
`source_nu.content`. Pairings must name the **admitted** ν digest and **admitted archive** digest;
the original device capture's `maps_sha256` instead names the original repository-key archive.

`load_admitted_device_lens` checks these bindings again and compares each admitted matrix with
the original matrix at that repository layer. The source files must remain accessible at the
recorded paths. If artifacts are relocated, create a fresh admission beside the relocated
originals; do not edit a digest sidecar to pretend it still identifies another file.

The merger now writes its promised `nu.json`:

```sh
python scripts/merge_device_lens_chunks.py \
  /absolute/new-merged-directory /absolute/chunk-1 /absolute/chunk-2 /absolute/chunk-3
```

It requires a new output directory and compatible, disjoint source populations. Its ν contains
every chunk's full declaration, archive and declaration hashes, row population, prompt count and
weight. Thus 70/70/61 gives 201 prompts with weights 70/201, 70/201, 61/201. Widths such as
16/16/8 remain ordered lists. The multiplication, input-order accumulation and division retain
the original merger's float32 arithmetic. A rehashed merged archive whose matrices differ from
the weighted source chunks refuses admission. The old incomplete merged directory is preserved;
make a complete new merge from its original chunks, then admit that new directory.

## T2/T3: endpoint and errors

The final endpoint is represented by `None` at the numerical bridge seam, meaning an explicitly
admitted identity map. Missing interior matrices are always errors. Existing non-final lenses
without endpoint metadata remain compatible; callers cannot use that absence to admit a final
dictionary.

The old `reconstruction_budget(..., dominance=...)` API is replaced by
`raw_reconstruction_budget(..., error_budget=...)`. Its distribution and `raw_admissible` mask
describe `||e|| / ||h||` only. `decompose_position` separately returns
`raw_reconstruction_share` and `lens_score_error_share = ||Le|| / ||Lh||`.

Both thresholds, `denominator_floor`, and `near_zero_policy` are required. The supported
near-zero rule is `"refuse"`: a denominator at or below the positive floor produces `null`,
with no top-feature ranking. A finite raw share above its threshold does not suppress a
score-space ranking that passes the separately declared score gate. No default threshold
admits. The score is gain-only and linear; its values are not model logits.

## T4: explicit runner inputs

```sh
python scripts/device_sae_bridge.py \
  --stage both \
  --model-snapshot /absolute/hf-snapshot \
  --dictionary-dir /absolute/dictionary \
  --dictionary-receipt /absolute/DIGEST.json \
  --lens-archive /absolute/fit-directory/admitted-maps.npz \
  --capture-dir /absolute/completed-capture \
  --positions /absolute/positions.json \
  --corpus /absolute/rendered-corpus.jsonl \
  --pairings /absolute/registered-pairings.json \
  --config /absolute/bridge-config.json \
  --output-dir /absolute/new-result-directory \
  --dry-run
```

`--stage a1` can omit the four capture inputs: capture directory, positions, corpus and pairings.
If any is supplied, all four are required and checked. `--stage a2` requires them. No path resolves
through a device mount, cache lookup, registry download or current working-directory assumption.
The scripts select their own source checkout even when a shared environment installs another
checkout.

The run config must declare these fields. Nulls below are deliberately non-admitting placeholders,
not recommended thresholds:

```json
{
  "schema_version": 1,
  "model_base": "google/gemma-3-4b-it",
  "capture_model": "4b",
  "checkpoint_sha256": {"config.json": "...", "model-00001-of-00002.safetensors": "...", "model-00002-of-00002.safetensors": "..."},
  "lens_sha256": "admitted archive sha256",
  "dictionary_repo": "google/gemma-scope-2-4b-it",
  "dictionary_folder": "resid_post_all/layer_17_width_16k_l0_small",
  "error_budget": {
    "raw_reconstruction_threshold": null,
    "lens_score_error_threshold": null,
    "denominator_floor": null,
    "near_zero_policy": "refuse"
  },
  "a1": {
    "k": null,
    "chunk": null,
    "control_layer": null,
    "check_features": [],
    "two_product_tolerance": null,
    "maximum_control_overlap": null,
    "convention_threshold": null
  },
  "a2": {"k": null, "identity_tolerance": null}
}
```

The hash manifest must list exactly `config.json` and all local safetensors shards. If an index
exists, its entire tensor-to-shard mapping must match ownership in the verified shard headers;
it cannot redirect the readout to unverified files. The index hash is also recorded. A capture's
short model label (for example `4b`) must equal `capture_model`; matching checkpoint-file hashes
establish which base it actually used. A label alone is never an identity check.

The dictionary receipt is the existing fetcher's `DIGEST.json`: `repo`, `folder`, the full
`config`, and per-file `{algorithm, digest, bytes}` under `files`. `algorithm` is `sha256` or
`blob` (Git blob SHA-1). Config and parameter receipts are mandatory; A1 additionally requires
the exact dictionary's `examples.safetensors` receipt. Only its small `top_tokens` tensor is
loaded from that file; the token-corpus tensor is not loaded. The complete file is hashed.

The positions JSON has schema version 1 and:

- `capture_files_sha256`: exactly `manifest.json`, `index.jsonl`, `progress.jsonl`,
  `residual_note.npy`, `residual_act.npy`;
- `corpus_sha256` and `tokenizer_sha256` (the explicit snapshot's `tokenizer.json`);
- `reduction: "per-position, no reduction"` and `endpoint: "pre-final-norm"`;
- nonempty `cells`, each `{row, position, token_index, token_id, context_tokens}`;
- when outside the fit's positions/context, `domain_of_validity` with a nonblank `statement`,
  `lens_sha256`, `nu_sha256`, and the sorted unique `positions` and `context_tokens` it covers.

`row` is the global distinct-decision index, and `position` is `P_note` or `P_act`. The runner
reproduces the capture's first-occurrence `(task_id, step)` deduplication, local tokenizer offsets,
prompt length, tool-name position and full captured context. The array row is global row minus
`rows_from`. Selected cells must agree with actual index records and arrays. `token_id` is the
expert completion's next token at that site: these captures are teacher-forced and are not
evidence that the model emitted that token. The current capture schema stores float32 memmaps;
an incompatible capture format refuses explicitly.

The capture must have a completion manifest and exactly one matching corpus, loaded and done
progress event. Its checkpoint hashes and source-map hash must agree. A2 at the final layer
requires that layer in the actual memmap; the current workspace capture writer omits it, so a
final dictionary can pass A1 while correctly refusing A2 on such a capture.

Pairings use the existing model-keyed table with `measured_pairings` keyed by repository layer.
A layer may hold one registration or a list for different cells. Each registration requires
exactly `pair`, finite nonnegative `relative`, and nonblank string `basis`. Its `pair` must name
all eleven existing fields: `fit_dtype`, `fit_width`, `capture_dtype`, `capture_width`, `lens_side`,
`nu_sha256`, `lens_sha256`, `positions`, `reduction`, `endpoint`, `context_tokens`. The runner
derives fit fields from admission, capture fields from the completed manifest, and position fields
from verified cells. Identity-only entries, invalid terms, undeclared fields, ambiguous matches
and cross-path readings without a matching measurement refuse. No registration is created here.

The owner supplies empirical measurements and the domain statement before numerical runs. The
runner checks their declared identity and required content; it does not prove those measurements
or the domain statement scientifically valid.

## Outputs and stopping rules

Dry-run writes `admission.json` with the checked set and provenance, then stops without dictionary
encoding, reconstruction, score projection, control computations or numeric bridge provenance.
A numerical run writes `result.json`. Both require a new output directory and contain the error
policy, hashes, fit/source declarations, capture/reading provenance, unlabelled reason and the
order's interpretation limits verbatim.

A1 records the raw arm before evaluating the gained arm. A failed shipped-token discriminator
stops before lens readout. Direct feature-column products independently check both top-token
membership and score values. A different-layer control must pass the configured overlap bound;
the same layer is refused as a requested control. Failed numerical checks produce
`refused-numeric-check` and CLI exit 2, and prevent A2 in a combined run. Successful execution or
dry-run returns 0; input refusal returns 2. A2's per-cell `ranked` flag is the score-error decision,
which is distinct from successful execution of the measurement itself.

## Verification record

Commands use the existing primary Python with this worktree's `PYTHONPATH=src`,
`PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `-p no:cacheprovider`, and separate temporary pytest directories.

The three commit gates were separate pytest invocations with their own exit status:

```text
tests/test_sae_bridge.py tests/test_sae_intervention.py tests/test_live_lens.py
  -k 'not test_the_real_dictionary_hook_is_a_layer_the_real_lens_carries'
  --basetemp=/private/tmp/saej-t23-commit

tests/test_device_lens_admission.py
  --basetemp=/private/tmp/saej-t1-commit

tests/test_device_bridge.py
  --basetemp=/private/tmp/saej-t4-commit
```

Final integrated command, run after the code commits with process-inspection access for the
runlock fixture:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
'/Users/daniel.tipton/Desktop/An app/.venv/bin/python' -m pytest \
  tests/test_device_bridge.py tests/test_device_lens_admission.py \
  tests/test_sae_bridge.py tests/test_sae_intervention.py tests/test_live_lens.py \
  tests/test_hf_text.py tests/test_repository_rules.py tests/test_import_tree.py tests/test_runlock.py \
  -k 'not test_the_real_dictionary_hook_is_a_layer_the_real_lens_carries' \
  -p no:cacheprovider --basetemp=/private/tmp/saej-integrated-committed --tb=short
```

The initial bridge/intervention baseline was 99 passed with one large-artifact test deselected.
Each new seam had an observed failing regression before its fix: missing admission/runner APIs;
shifted/invalid artifacts; float32 merger arithmetic and rehashed incorrect merged maps; missing
fit dtype; absent endpoint and error policy; unbound external registrations; an index redirect to
unverified weights; wrong top-token IDs with correct scores; null domain statements; and continuing
after a failed discriminator. The 34/48-layer admission cases are additional positive boundary
coverage, not claims of previously observed production failures.

Independent review reproduced the index redirect changing an A2 score while preserving the old
checkpoint provenance, and the false A1 top-token result passing its former two-product check.
Both now have refusing regression tests. The pairing validator now rejects identity-only and
nonfinite measured entries. The final runner also refuses null/non-string domain statements and
stops immediately on a failed discriminator.

An intermediate integrated run had 329 passed, one failed and one deselected. The failure was
`test_a_launcher_under_a_wrapper_does_not_report_itself`: the sandbox prohibited `/bin/ps`.
Its isolated rerun with process-inspection access passed (1 passed, exit 0). This environment
failure was not hidden or counted as a pass.

Actual large-archive memory use has not been measured. Admission currently holds source maps in
CPU memory, and merged validation holds the chunk matrices as well. A real artifact run needs an
appropriate memory budget; this record establishes CPU fixture behavior only. The Chief still
owns actual pairings, domain validity, shipped example acquisition and real A1/A2 runs. Stage B
remains conditional on the real A2 evidence and sealed rotation control in the order.

## F1 provenance correction — Chief review at `7527075`

The reusable merger entry point now lives at `scripts/merge_device_lens_chunks.py`. The historical `research/records/WORKSPACE-EXPERIMENTS-2026-09-10/scripts/merge_chunks.py` belongs to the as-run record on main and must retain SHA-256 `5c39d626e129f77dfbfe02b0d0ae18cdebf3a79a9643a91a669da83de096026e` through integration. The shared admission/merge logic is unchanged. The CLI regression follows the new path.

The Chief's review at `7527075` supplies real-artifact evidence beyond this record's laptop fixtures: every 4B layer loaded without a shift, and the successor merger reproduced the original 12B archive bit for bit, SHA-256 `e7942f1d6a73…`, before the re-merge admitted against the 12B checkpoint. Those review admissions are scratch outputs; canonical admissions and the positions, pairing registrations, domain statement, examples and card runs remain with the Chief.

F1 verification: the existing merger CLI regression first failed at the new path with exit 1 (file absent), then the admission, runner, bridge, intervention and live-lens checked set passed **229 tests, 1 large-artifact test deselected**, exit 0, in 10.78s (`/private/tmp/saej-f1-green`, the same CPU environment as above). Scoped Ruff and diff checks passed. The historical script SHA-256 was checked directly before integration.
