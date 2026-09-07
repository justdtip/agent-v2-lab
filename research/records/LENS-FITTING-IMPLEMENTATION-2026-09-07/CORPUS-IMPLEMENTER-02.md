# Task 1 corpus-builder implementation report

Status: implemented and verified; ready for controller review. This task changed only the four assigned implementation/test files and this requested local report. No checkpoint was loaded, no remote push or network download was performed, and no lock, registry, prototype fitter, runner, or live-lens file was changed.

## Public contract

- `build_agentic_corpus(sources, tokenizer, spec, manifest_path, *, max_tokens, tokenizer_files)` writes an exclusive JSON manifest and sibling `.jsonl`, returning manifest metadata.
- `build_prose_corpus(sources, tokenizer, spec, manifest_path, *, tokenizer_files, download_descriptor=None)` uses fixed nonoverlapping 1024-token windows.
- `read_corpus(manifest_path)` validates all bound inputs/assets/sequence bytes and source-step identities, then returns a list of sequence dictionaries. Returning the whole validated list ensures no earlier rows escape before later corruption is detected.
- `load_corpus_tokenizer(spec, *, local_files_only=False)` returns the tokenizer and explicit local asset paths. Exact tokenizer/config allow-patterns exclude checkpoint weights and weight-index files; `AutoTokenizer` uses local files, fast tokenization and `trust_remote_code=False`.
- `download_prose(data_dir, *, revision='main')` returns `(text_path, descriptor_path)`. Existing verified descriptors are reused offline; unresolved revisions, changed identity, corrupt text and partial outputs fail closed.
- CLI: `PYTHONPATH=src .venv/bin/python scripts/lens_corpus.py --corpus agentic --evals <paths...> --out <manifest.json>` or `--corpus prose --files <paths...>` / `--download-prose --data-dir <dir>`. Optional `--dataset-revision`, `--model`, and `--local-files-only`. Agentic CLI cap is the specified 2048-token data budget; it is not an architecture dimension.

Indices and step indices are zero-based. Held membership is `index % 5 == 4`; the index is assigned in explicit source/trajectory/step order before length filtering. Kept and dropped indices together cover every reached source step. `source` is the resolved input path. Each sequence carries domain, split, IDs, spans and n_prompt; agentic rows also retain prompt/text/offsets for audit. A prompt/completion-straddling token is labelled template and is excluded from n_prompt; the prompt is never independently tokenized. Zero-width BOS tokens receive template labels.

Manifest fields include schema_version, model_hf_id, sources in caller order, tokenizer/template identity and file hashes, sequence path/hash, counts by fit/held and all seven spans, dropped_rows, source_sequence_count, fixed split/pilot rules and domain-specific budget/window/tail metadata. `manifest_sha256` is the SHA256 of canonical JSON excluding that field. Other hashes use exact bytes. Hashes detect corruption; these are not signatures.

## Requirements and evidence

| Requirement | Implementation and exercised behavior |
|---|---|
| §3.1 runner-equivalent agentic sequences | Uses SYSTEM_PROMPT, build_prompt with DEFAULT_KEEP_LAST and generation suffix; canonical prior actions and observations; raw completion retained. Tests compare prompt equality and check hidden observation labels. |
| §3.1 terminal behavior | Includes final unparsed raw step and stops on parse failure or finish. Current raw is reparsed; contradictory recorded action/note metadata is rejected. |
| §3.1 token alignment | Single whole-text call using MLX generation's BOS policy; offsets are mandatory, typed, bounded and monotonic. Missing message content and all-zero offsets fail. Exact token label coverage, leading/existing BOS and boundary-crossing tokens tested. |
| §3.1 stable holdout | Fifth source-step held before drops; multi-source ordering, dropped first step and held membership tested. |
| §3.1/§10 leakage controls | IDs must have train/train1/train2 split, registered family, numeric ordinal and registered variant. Test/valid/validation/pilot and lookalikes fail. Summary split/keep_last and every available source-model declaration checked against ModelSpec. Missing model provenance fails. Duplicate resolved sources and trajectory identities fail. |
| §3.1 prose | Per-file tokenization, fixed windows, no overlaps, tail accounting and chat/template spans; five windows and the exact hand-derived remainder tested. |
| §3.1/§10 authorised download | Resolves Salesforce/wikitext to immutable SHA; lists names at that revision; downloads only wikitext-103-raw-v1/validation-N-of-N.parquet; reads local Parquet with PyArrow. Descriptor binds dataset/config/split/revision, actual shard names/hashes and text hash. |
| §10 immutable provenance | Exclusive creation for corpus JSONL/manifest and downloaded text/descriptor; corruption of inputs, tokenizer assets, JSONL, manifest or descriptor is rejected. Reader independently checks typed rows, split/order, source-step identities and summary counts even with a recomputed file hash. No pickle or activation artifact is written. |

## TDD record

1. Initial corpus tests: 19 failures because the required module/public implementation was absent. After the agentic/prose implementation: 19 passed.
2. Downloader/tokenizer-loading tests: three expected failures for absent public functions; after implementation these passed. The CLI cap fixture initially assumed a character-tokenized first prompt exceeded the cap; measured length was 1857, so its task text was extended explicitly to exercise dropping. This was a fixture correction, not a product change.
3. Added all-zero offset regression: failed because every token was silently labelled template. Added fail-closed validation; regression passed.
4. Added rehashed wrong-source-step regression: failed because typed/nonnegative step_index alone did not bind the source step. Reader now reconstructs source-step identities from input reports; regression passed.
5. Final source review found that installed `datasets.load_dataset` invokes `download_and_prepare` before `as_dataset(split=...)`, so selecting validation alone could fetch excluded data. Replaced permissive dataset-builder testing with a fake remote containing train/test/other-config filenames and a real local Parquet fixture. Three download tests then failed on the forbidden general builder. Implemented validation-only `hf_hub_download` calls and local Parquet reading; those tests passed. All unit tests remain network-free.

Some additional edge-case checks passed immediately against the first coherent implementation; they are regression coverage, not claimed independent red/green feature increments.

## Fresh verification

Working directory: `/Users/daniel.tipton/worktrees/lens-fitting`.

- `PYTHONPATH=src '/Users/daniel.tipton/Desktop/An app/.venv/bin/python' -m pytest tests/test_lens_corpus.py tests/test_repository_rules.py --tb=short` — exit 0, **65 passed in 3.84s** after the final downloader correction (40 corpus tests + 25 repository-rule tests).
- `'/Users/daniel.tipton/Desktop/An app/.venv/bin/ruff' check src/local_llm_lab/pipeline/lens_fitting/__init__.py src/local_llm_lab/pipeline/lens_fitting/corpus.py scripts/lens_corpus.py tests/test_lens_corpus.py` — exit 0, all checks passed.
- Ruff formatted only the four owned files.
- Offline real cached tokenizer smoke: loaded five tokenizer/config assets using local_files_only=True; no weights/network. Five synthetic training steps rendered and read successfully with the actual Qwen tokenizer. Four fit sequences contained 1977 tokens total; fifth held sequence contained 637 tokens (615 prompt tokens). All span counts summed correctly.
- Following the controller's held-lock notice, resolved the top-level import closure of corpus tests, repository-rule tests, conftest and corpus recursively before subsequent pytest runs. Local closure: agent_protocol, agent_tasks, models, pipeline.env/protocol/tasks, project, runlock, spawn and package initializers. External top-level roots: standard library, pytest and yaml. No MLX/mlx_lm/mlx_tune imports. Downloader tests import PyArrow/Hugging Face helpers only inside their test bodies and mock the remote boundary. No tiny-model tests ran.
- The full suite was deliberately left to the controller, per assignment; known unrelated broad-suite failures were not revisited.

## Review concerns and limits

- Real 600-step training reports and an actual remote prose download were not processed by this worker. Controller integration should exercise the concrete source reports and selected live dataset revision after review. The cached real-tokenizer smoke covers renderer/offset compatibility without checkpoint execution.
- The downloader intentionally fails if the pinned repository no longer exposes the expected authorised validation-Parquet layout. It never falls back to a whole-dataset builder or broader download pattern.
- Outputs use absolute source and tokenizer-asset paths and require those bytes to remain available for read verification. Move/copy workflows need an explicit future rebinding operation; this task does not silently rewrite provenance.
- Two-file publication uses exclusive opens, so a crash may leave an orphan JSONL or download text. Such outputs are never silently reused or overwritten; select a fresh output location after investigating the incomplete artifact.
- This worker did not change the base, merge primary cache changes, or integrate other task files. The parent's existing task-boundary claim covered the assigned paths.

## Commit

`31ec13ceaa6e6c6ff93164b4955569b60c6ebf3f` — `Add immutable source-bound domain lens corpora`.

The staged diff passed `git diff --cached --check`; the commit contains exactly the four assigned files. Post-commit `git status --short` was empty. This ignored local report remains available at the requested path.

## Review fix round 1

Implemented the two requested reader invariant fixes after merge `53343a3`, preserving the merged cache/rule changes and the controller's untracked build logs.

- Prose validation now constructs the expected ordered `(source, step_index, token_start)` tuples from each source's declared token budget. It validates typed nonnegative token counts, typed tails in `[0, 1024)`, exact per-source division/remainder, the global trailing-token total, the fixed chunk size, and global source-sequence count. Every row must match its expected source/chunk identity. A row rehashed as step 999 with token_start 1022976 is rejected.
- Extracted `_validate_offsets` and call it from both `_encode` and `read_corpus`. It checks the offset container, pair shape, strict integer types, bounds within the text, monotonic nonzero ranges, and presence of nonempty alignment. Negative BOS positions, reversed/out-of-text ranges, boolean positions and malformed pairs fail with an alignment error before prompt counting.
- New `pilot_exclusion` metadata adds `applies_to: agentic_task_ids`. The reader also accepts the exact prior schema-1 rule without that field, interpreting it as the historical agentic task-ID rule. Authorised WikiText validation remains permitted. Neither previously built corpus was overwritten or rebound.

Covering tests:

- `test_reader_rejects_rehashed_wrong_prose_step`
- `test_reader_reconciles_prose_sources_chunks_and_tails` — source order, token budget, source tail, global tail, sequence total, chunk size, typed tail.
- `test_reader_rejects_rehashed_invalid_offsets` — negative, reversed, out-of-text, nonmonotonic, boolean and wrong-shape offsets.
- `test_pilot_rule_scope_and_legacy_manifest_compatibility` — both agentic and prose metadata and old-rule readability.

TDD: the targeted 16-case run initially reported **15 failed, 1 passed**. The global sequence-count case was already protected. During implementation the nonmonotonic fixture was corrected from a duplicate range (legitimate for some tokenizers) to a range that actually goes backwards; the shared validator intentionally retains legitimate duplicate offsets.

Fresh final verification:

- Re-resolved the post-merge transitive top-level import closure of corpus tests, repository-rule tests, conftest and corpus before pytest. The closure remains stdlib/pytest/yaml plus the same local pure-Python modules listed above; no MLX/mlx_lm/mlx_tune imports.
- `PYTHONPATH=src '/Users/daniel.tipton/Desktop/An app/.venv/bin/python' -m pytest tests/test_lens_corpus.py tests/test_repository_rules.py --tb=short` — exit 0, **81 passed in 4.74s** (56 corpus + 25 repository-rule tests).
- `'/Users/daniel.tipton/Desktop/An app/.venv/bin/ruff' check src/local_llm_lab/pipeline/lens_fitting/corpus.py tests/test_lens_corpus.py` — exit 0, all checks passed. `git diff --check` passed.
- Read-only compatibility checks of the two existing immutable manifests passed: agentic 518 rows (418 fit, 100 held), 82 dropped; prose 255 rows (204 fit, 51 held), 164 discarded tail tokens. The manifest SHA256s were unchanged before/after reading. No model or network operation was used.

No remaining concerns for these bounded review requests. General artifact portability and crash-recovery limits from the original report remain unchanged.

Review-fix commit: `016c5f640fc89b2e5b27ef81f5663047e8f9a15f` — `Validate prose chunk identities and corpus offsets on read`. It contains only corpus.py and test_lens_corpus.py. Post-commit working-tree status was clean.
