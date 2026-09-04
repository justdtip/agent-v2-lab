# R21 implementation report: dataset write guard, render stage, B4 config repoint

Implementer: Opus slice, 2026-09-04. Task: ruling R21 (wiring map §7; issue #20). Round 3:
incorporates the Chief's ratification conditions from issue #24 (see §"Ratification round
response") on top of the round-1 review fixes. No commits made; working tree only. No model
executed; no checkpoint or weight load; no real tokenizer load either (all tests run on
fakes).

## Files changed (git numstat, this slice only)

| File | +/- |
| --- | --- |
| `src/local_llm_lab/pipeline/data.py` | +179 / -5 |
| `src/local_llm_lab/pipeline/cli.py` | +130 / -9 |
| `configs/agent_v2b_qwen35_4b.yaml` | +6 / -2 |
| `configs/agent_v2d_qwen35_4b.yaml` | +5 / -1 |
| `tests/test_cli.py` | +276 / -6 |
| `tests/test_data.py` | +309 / -1 |

No other file touched. `preflight.py`, `probes/patch.py`, `rollout.py`, `branch.py`, and
their tests were not entered; condition 1 was implementable entirely at the CLI boundary,
so no cross-lane edit was needed.

## R21(a) — write guard at the write boundary

- `PROTECTED_DATASETS` (the four irreplaceable directories): `src/local_llm_lab/pipeline/data.py:36-43`.
- `CHAT_COMPLETION_SOURCES` (the verbatim-render allowlist, ratification condition 2):
  `data.py:44-49`.
- Named errors: `DatasetWriteGuardError` (`data.py:52`), `ProtectedDatasetError` (`data.py:56`),
  `DatasetRenderError` (`data.py:60`).
- `_names_same_directory` (`data.py:64-76`): protected-set matching is by **on-disk file
  identity** (`os.path.samefile`, st_dev+st_ino) when the paths exist, with plain path
  equality as the fallback for not-yet-existing paths — closing the case-insensitive
  filesystem bypass the round-1 review proved (a mis-cased alias like `data/AGENT_V2B`
  resolves to the same directory but compares unequal as a path). `resolve()` still
  canonicalises symlinked aliases (`data.py:90`).
- `guard_dataset_write(output, *, overwrite=False, override_flag="--force-overwrite") -> bool`
  (`data.py:78-114`): protected datasets refuse unconditionally, **including every
  subpath** — the guard walks `(resolved, *resolved.parents)` against each protected root
  with the same file-identity comparison (`data.py:90-100`), so pollution of an
  irreplaceable directory is impossible, not merely refused at its root (ratification
  condition 3). Any other directory holding `manifest.json` refuses unless `overwrite`
  (`data.py:101-113`), and the returned flag lets the writer record the override. The
  refusal hint names the caller's actual flag; a stage with no override passes
  `override_flag=None` and gets an accurate message (`data.py:104-108`).
- Guard wired inside `write_dataset` itself, before any work (`data.py:309`), so programmatic
  callers hit it, with the forced overwrite recorded in the new manifest
  (`data.py:321-322`). Same guard inside `render_dataset` (`data.py:394`, recorded at
  `data.py:429-430`).
- CLI: `--force-overwrite` on `data` (`cli.py:737-741`) and `render` (`cli.py:750-754`),
  threaded through `stage_data` (`cli.py:95`, `cli.py:124`) and dispatch (`cli.py:847-852`).
- "Every stage that writes under `data/`", now **live** (ratification condition 1):
  `stage_rollout` guards its target before running (`cli.py:670-671`) and stamps it with
  `manifest.json` after `run_rollout` succeeds (`cli.py:692-703`); the `branch` dispatch
  guards `data/preferences/<split>` (`cli.py:884`) and stamps it after `run_branch_mining`
  returns (`cli.py:903-913`). The stamp helper `_write_stage_manifest` (`cli.py:166-173`)
  records stage, `generator_version`, model, split, seed, adapter, and the returned summary
  — the provenance those outputs were missing — and, because the guard fires on
  `manifest.json`, a rerun into the same split now refuses instead of silently overwriting
  mined data. A crashed run leaves no manifest, so retries of failed runs stay possible.
  Both guards pass `override_flag=None` since neither stage exposes the flag.

Tests: refusal without override and recording with it (`tests/test_data.py:510-523`); the
four refusing even with the override, before anything is written, through both writers under
a monkeypatched root (`tests/test_data.py:525-536`); the pinned protected list and the guard
rejecting the real repository paths **and their subpaths** — pure path checks, no write
attempted (`tests/test_data.py:539-551`); **subpaths of a temp protected root refusing even
when forced, including under a mis-cased parent** (`tests/test_data.py:554-575`); mis-cased
and symlinked aliases refusing even when forced, with the case-sensitive-filesystem branch
asserting a genuinely different directory stays writable (`tests/test_data.py:578-609`);
rollout stamping its manifest with the exact payload and the guard refusing the rerun
(`tests/test_cli.py:933-978`); branch stamping and refusing the rerun before mining runs
again (`tests/test_cli.py:981-1021`); the flagless refusal message accuracy
(`tests/test_cli.py:444-448`); flag propagation (`tests/test_cli.py:408-422`).

## R21(b) — the render stage

- `render_dataset(source, output, tokenizer, *, spec, overwrite=False)`
  (`data.py:377-435`): reads existing rows from `<source>/{train,valid,test}.jsonl`
  (`data.py:395-400` validates completeness first; nothing is written on a partial source),
  re-renders via the in-repo `render_rows` with the given tokenizer (`data.py:413`), and
  writes the three role files plus a manifest. The manifest records the source directory
  (`data.py:402`), the source manifest's own SHA-256 (`data.py:407`), per-split source
  SHA-256 (`data.py:412`), the source's `generator_version` carried through unchanged and
  never replaced with HEAD's (`data.py:416`; explicit `null` for a pre-versioning source),
  carried `seed`/`keep_last`/`protocol` (`data.py:417-419`), the registry `ModelSpec` and
  the effective rendering metadata (thinking, template kwargs, generation suffix,
  `data.py:420-425`). It never calls `make_tasks` (no reference exists in the function;
  proven by monkeypatch in the test).
- `stage_render` (`cli.py:145-163`): loads the registry spec and its tokenizer only
  (tokenizer load is not model execution per the issue #12 definition — the docstring cites
  it), calls `render_dataset`, and writes `provenance.json` into the render output
  directory via `write_provenance` (`cli.py:155-160`).
- CLI forms: the R21(b) explicit form `render --source <dir> --output <dir> --model <name>`
  runs without loading any run config (`cli.py:827-835`); the task's `render --config <cfg>`
  form falls back to the config's `source_rows`/`data`/`model` (`cli.py:837-846`) and errors
  actionably when neither names a source (`cli.py:840`).

Tests: byte-identical task content with changed rendering, generator version carried through
(source manifest edited to version 2; render manifest records 2, not HEAD's 4), source and
output hashes, rendering metadata, and no `make_tasks` call
(`tests/test_data.py:610-669`); output guard + forced-overwrite recording + partial/absent
source refusal before any write (`tests/test_data.py:671-695`); manifest-less source renders
with explicit unknown provenance (`tests/test_data.py:698-716`); CLI explicit form loads no
config (`tests/test_cli.py:287-325`); config fallback and missing-source error
(`tests/test_cli.py:328-356`); stage wiring — registry tokenizer only, no weight load,
render provenance into the output directory (`tests/test_cli.py:359-405`).

## R21(c) — config repoint

- `configs/agent_v2b_qwen35_4b.yaml:7-8`: `data: "data/agent_v2b-qwen35-4b"`,
  `source_rows: "data/agent_v2b"`, with the R21 rationale in the header comment (lines 1-4).
- Schema: `source_rows` is a new optional top-level key naming the dataset directory whose
  existing rows the arm re-renders; `load_config` resolves it like `data` (`cli.py:45`).
  `stage_data` for a config with `source_rows` delegates to render and makes generation
  unreachable (`cli.py:96-112`); `--extra` mixing is rejected on that path (`cli.py:99-102`).
- `configs/agent_v2d_qwen35_4b.yaml:6-7` given the same shape (`data/agent_v2d-qwen35-4b`
  from `source_rows: data/agent_v2d`) so both cross-model arms train on byte-identical task
  content; the D4 render can only run after `agent_v2d.yaml`'s data stage creates the source.
- The pairwise-recipe test pins the shape: the cross arm's `source_rows` must equal the
  base arm's `data`, its `data` must be the new `-qwen35-4b` directory, and everything else
  must remain a literal pair (`tests/test_cli.py:218-244`). Delegation test at
  `tests/test_cli.py:247-284`.

## Defect found and fixed in-scope (blocking, disclosed; ratified round 1, tightened round 2)

`render_rows` called `parse_turn` unconditionally on every assistant target. Measured against
the real source (read-only): 348 of 2,693 rows in `data/agent_v2b` are replayed chat rows
with no tool call and raise `ActionParseError` — the real B4 render, and equally the
existing rendered `write_dataset` path for any config mixing `chat_replay`, would have
crashed. Fix: rows whose metadata source is in the explicit allowlist
`CHAT_COMPLETION_SOURCES` (`data.py:44-49`; exactly `pre-expansion-policy-replay`, which
every `data/chat_replay` row carries — 240/48/60 verified read-only, and the round-1
reviewer verified all 348 mixed rows) render their content verbatim under the model's turn
terminator (`data.py:219-229`); **any other source failing to parse raises loudly**
(`data.py:220-226`) — in a controlled arm, silent is the failure mode to fear (the
docstring records this, `data.py:197-203`). The round-1 review rendered all 348 affected
real rows through the template shape and confirmed byte-identical output — the fallback is
exactly what run B trained on. Tests: `tests/test_data.py:719-751` (allowlist pin, verbatim
render, loud failure for expert/rollout/chat/None/missing-metadata sources) and the
template-equivalence test below.

## Ratification round response (issue #24, 2026-09-04)

1. **Inert rollout/branch guards made live.** Neither stage ever wrote a `manifest.json`, so
   the round-2 guard calls could never fire — a false assurance. Fixed entirely at the CLI
   boundary in owned `cli.py` (no touch of `rollout.py`/`branch.py`, which both already
   return their summary): `_write_stage_manifest` (`cli.py:166-173`) stamps the output
   after success — rollout at `cli.py:692-703`, branch at `cli.py:903-913` — giving those
   outputs provenance and arming the guard for reruns. Behaviour change, intended: a rerun
   into an existing successful rollout/branch split now refuses (fresh-split convention was
   already enforced for train/valid/test names); a crashed run leaves no manifest, so
   retries remain possible. Tests: `tests/test_cli.py:933-978`, `981-1021`.
2. **Chat-row fallback tightened to an explicit allowlist** — `CHAT_COMPLETION_SOURCES`
   (`data.py:44-49`), keyed on the verified `pre-expansion-policy-replay` tag; a rollout or
   expert row that somehow lost its tool call now fails loudly instead of being silently
   rendered as chat (`data.py:220-226`); the Chief's words are in the docstring
   (`data.py:197-203`). Test: `tests/test_data.py:719-751`.
3. **Subdirectory gap closed** — the guard walks the resolved path and all its parents
   against every protected root with the same file-identity comparison (`data.py:90-100`);
   a write targeting any subpath of a protected directory refuses even with the force flag.
   Tests: temp protected-root subpaths, including under a mis-cased parent
   (`tests/test_data.py:554-575`), and real-repository subpaths as a pure guard check
   (`tests/test_data.py:547-551`).

Accepted as implemented, unchanged: the `--force-overwrite` flag name and the flag-free
refusal message. The legacy writers and non-atomic manifest writes stay as recorded
follow-ups below.

## Decisions taken under ambiguity

1. **Flag name.** R21's text says `--overwrite`; the dispatching task specified
   `--force-overwrite` recorded in the new manifest. Implemented `--force-overwrite`;
   accepted as implemented by the Chief on issue #24.
2. **Protected refusal is unconditional**, not conditioned on `manifest.json` presence
   (`data.py:90-100`): a protected directory that lost its manifest is more damaged, not
   less protected.
3. **Issue #20's suggested test** rendering real `data/agent_v2b` under the qwen25-coder-3b
   tokeniser was not added: the task for this slice forbids tests touching `data/` and
   requires fixture rows in temp directories; the migration-equivalence property is covered
   on fixtures for both expert rows (`tests/test_data.py:252`) and chat rows
   (`tests/test_data.py:271-293`).
4. **Test placement**: issue #20 mentions `tests/test_render.py`; the task's owned paths and
   R10 (per-module test files) put the tests in `tests/test_data.py` and `tests/test_cli.py`,
   matching the modules that changed.
5. **`write_dataset` signature** gained a trailing keyword `overwrite: bool = False`
   (`data.py:296`) — additive to the §2.5 contract; all existing callers are unchanged.
6. **Render provenance location**: written into the render output data directory
   (`cli.py:155-160`), because the R21(b) explicit form has no run-output directory; the
   manifest inside `extra.dataset_manifest` carries the source provenance.
7. **Rollout/branch manifests are stamped only on success** (after the runner returns), so
   the guard never blocks the retry of a failed run; the stamp uses a direct write, not the
   atomic path, matching the metadata-size threshold of R11.

## Definition-of-done checks

- `uv run pytest` (whole tree): **635 passed, 0 failed, 0 skipped** (8.25 s), including
  `tests/test_repository_rules.py` (banned-constant AST + substring scanner) — no new hits.
  This slice's own modules in isolation: `tests/test_data.py` + `tests/test_cli.py` =
  **60 passed, 0 failed**. Note: the coordinator flagged two expected `tests/test_patch.py`
  failures from the parallel issue #22 round; in the working tree at hand-off
  `tests/test_patch.py` runs **29 passed, 0 failed** — no red observed anywhere.
- Guard refusing a manifest-holding directory: `tests/test_data.py:510-523`.
- Four irreplaceable directories refusing even with the override: `tests/test_data.py:525-551`,
  aliases by case and symlink `tests/test_data.py:578-609`, subpaths
  `tests/test_data.py:554-575`.
- Override working elsewhere and recorded: `tests/test_data.py:519-523`, `683-686`.
- Render produces byte-identical task content with changed rendering: `tests/test_data.py:653-667`.
- Generator version carried through: `tests/test_data.py:632-633` (and explicit `null` for a
  pre-versioning source, `tests/test_data.py:714`).
- Config schema parsing: `tests/test_cli.py:218-244` (shape) and `cli.py:45` (resolution).

## What only the authorised next step can verify

The real B4 render — `agent-pipeline render --source data/agent_v2b --output
data/agent_v2b-qwen35-4b --model qwen35-4b` — was deliberately not run. It is tokenizer-only
(permitted under issue #12) but is the separate authorised step on the critical path. Note
for that run: it will exercise the chat-row allowlist on exactly 348 rows (all carrying
`pre-expansion-policy-replay`; review-verified byte-identical to the legacy render); the
render manifest will record `generator_version: null` because `data/agent_v2b/manifest.json`
predates generator versioning (verified read-only: its top-level keys are seed, keep_last,
protocol, recovery_repeats, splits).

## Observed, not fixed (follow-ups for whichever lane owns these files)

- `chat_replay.py:140`: `main()` defaults `--output` to `data/chat_replay` — a
  `PROTECTED_DATASETS` member — and writes unguarded. Legacy and model-gated, but the same
  hazard class R21 closes; the follow-up is a one-line
  `guard_dataset_write(output, override_flag=None)` at its write site.
- Module-level mains bypass the CLI-boundary guard and its new manifest stamp:
  `rollout.py:193-201` (defaults to `data/rollouts/<split>`) and `branch.py:300-313`
  (defaults to `data/preferences/<split>`) — same one-line guard call each, plus the stamp
  if those entry points are kept.
- Three legacy data writers default into `data/` unguarded: `generate_agent_data.py:83`
  (`data/agent_sft`), `generate_complex_data.py:16` (`data/complex_agent_sft`),
  `build_expanded_data.py:50-53` (reads three `data/` dirs, writes
  `data/expanded_agent_sft`).
- `data/` remains untracked and unbacked (issue #20 already routes this to the Director).
