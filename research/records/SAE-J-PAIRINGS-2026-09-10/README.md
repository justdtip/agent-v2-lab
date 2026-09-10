# SAE-J T5: exact-cell pairing measurement

Implemented against the Chief's T5 order at `2e556a0`, ~12:00Z on 2026-09-10.
The implementation is in the commit containing this record. Card runs and the domain
statement remain the Chief's. No actual checkpoint was executed or downloaded on the laptop.

**PASS — checked set `T5 + affected bridge CPU`: 323 passed, 1 deselected; pytest exit 0,
52.78 seconds.** The final command ran directly, without a pipeline that could mask its exit:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_measure_pairings.py tests/test_device_bridge.py \
  tests/test_device_lens_admission.py tests/test_sae_bridge.py \
  tests/test_sae_intervention.py tests/test_live_lens.py tests/test_hf_text.py \
  tests/test_repository_rules.py tests/test_import_tree.py \
  -k 'not test_the_real_dictionary_hook_is_a_layer_the_real_lens_carries' \
  -p no:cacheprovider --basetemp=/private/tmp/saej-t5-final --tb=short
```

The deselected check requires the real dictionary/lens artifacts. The new T5 file contributes
33 CPU tests. The checked set uses synthetic safetensors/NPZ captures and a tiny Torch decoder;
the real-model loader is replaced at its loading boundary. The production recording helper
executes actual tiny-model forwards. Ruff on the five changed Python files and `git diff --check`
also exit 0. Independent review found and rechecked the two corrections described below;
its final scoped verdict had no remaining P1/P2 findings.

## Entry point and input contracts

`scripts/measure_pairings.py` requires every artifact path and the repository layers. Example
syntax, with paths supplied by the operator:

```sh
python scripts/measure_pairings.py \
  --model-snapshot /data/snapshot \
  --checkpoint-manifest /data/checkpoint-hashes.json \
  --lens-archive /data/fit/admitted-maps.npz \
  --lens-sidecar /data/fit/admitted-maps.json \
  --capture-dir /data/capture \
  --positions /data/positions.json \
  --corpus /data/rendered-corpus.jsonl \
  --layers 18 24 \
  --output /data/new-measured-pairings.json
```

The checkpoint manifest is the plain filename-to-SHA256 mapping carried by the capture's
`load_report_sha256`. It must match the actual snapshot, source fit, and capture. The sidecar
must be the archive's adjacent `.json`; the admission loader rechecks its original sources
and all maps. `--corpus` explicitly supplies the rendered corpus for token reconstruction;
the tool does not follow the historical device path or resolve a cache. The output's parent
must exist and the output itself must be new. There is no width override.

Capture, token, and cell validation is shared with A2 through `_capture_cells`. The producer
does not require a domain statement. A2 still does, when the requested reading is outside
the fit domain. Adding the Chief's statement later does not invalidate the measurement:
the original positions digest is recorded, while consumption checks the sealed capture,
corpus/tokenizer digests, and the exact selected cell.

## Measurement and output

All cells and layers pass file admission before model loading. The runtime loads bf16 through
the existing local-only HF loader, promotes to float32, pins the historical eager/TF32 flags,
and runs the same unpadded token sequence at each admitted forward width. Capture replay calls
the outer model with the row's `output_attentions` and both `logits_to_keep` positions, as the
historical capture did; fit replay uses `HFLensModel.forward`. Both read decoder-block outputs
before final norm, with one-based repository layer indexing.

Each requested width-1 residual must have exactly the stored float32 bytes, including signed
zero. No tolerance or fallback is available. Relative terms use float64 norm arithmetic on
the observed float32 vectors: `||h_fit - h_capture||_2 / ||h_capture||_2`. Zero denominators
refuse. Replicated rows must themselves be bitwise equal at the selected position.

For a mixed fit, every ordered chunk is run at its own width, including repeated widths.
**The scalar registration uses the maximum chunk term**, and every chunk's term is retained
in provenance and stated in the basis. This is an explicit conservative aggregation choice;
it is not the norm of an averaged residual or a chunk-weighted mean. Fixtures cover `[32]`,
`[16, 16, 8]`, and `[1]`; the identical schedule has zero displacement.

The output is model-keyed, with `measured_pairings` keyed by repository layer and a list of
`{pair, relative, basis}` records in positions-file order. All eleven pair fields come from
the admission, capture, and reconstructed cell. `_provenance` holds checkpoint/capture/positions
digests, admission digests, source commit and implementation file hashes, runtime versions,
actual kernel flags, upstream identity, and per-cell prompt/token/residual identities and
ordered chunk terms. A basis names its prompt, position, context, layer, and both schedules.

The table validator now accepts lists while validating each three-field record. A2 matches
T5 cell receipts as well as all eleven pair fields: two prompts with equal position/context
lengths stay separate. Generated bases carry a T5 marker, so removing provenance refuses
instead of downgrading to legacy matching. Existing legacy registrations and opaque underscore
notes retain their previous behavior. These two runtime/provenance corrections arose in review
and have regression fixtures.

## Refusal evidence

| Required refusal | Named code | CPU evidence |
|---|---|---|
| Hash mismatch | `hash_mismatch` | Checkpoint, sidecar, capture, tokenizer, positions seal, malformed manifest; no model load |
| Cell outside capture | `cell_outside_capture` | Row, position, token index/id, context, absent/duplicate/invalid layers; no model load |
| Stored residual not reproduced | `capture_residual_mismatch` | Resealed perturbation and signed-zero mismatch; no result written |
| Non-finite values | `non_finite_values` | Stored NaN, fit-forward NaN, undefined zero-norm relative term; hooks removed |
| Existing output | `existing_output` | Refuses before loading, preserving the existing bytes; exclusive create also protects publication |

Additional `path_mismatch` refusals prevent unsupported precision/width, absent or contradictory
capture attention information, wrong observed tensor shape/dtype, unequal replicas, and missing
block execution from being reported as measurements.

**Unverified on purpose:** real 4B/12B checkpoint execution, bitwise reproduction on the card,
and empirical domain of validity. The fixtures demonstrate the instrument's contract; they do
not register a measured device pairing or license an A2 interpretation.
