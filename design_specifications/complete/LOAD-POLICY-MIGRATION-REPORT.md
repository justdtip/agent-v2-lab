# `load_policy` registry migration — implementation report

Date: 2026-09-04. Branch: `codex/agent-v2-specs`. Work left uncommitted in the working tree.

Scope: migrate `pipeline/evaluate.py::load_policy` to the wiring-map §2.10 contract, migrate
`run_evaluation` to take a `ModelSpec`, make the evaluation JSON record the resolved spec, and
update every caller.

> **Revision 2 (2026-09-04 13:45).** Corrections round under condition K1 of
> `under_review/WAVE1-LOAD-POLICY-REVIEW-round1-2026-09-04.md`. Five claims in revision 1 were
> wrong or missing and are corrected here; every citation in the document was re-derived against
> the working tree at 13:45 and thirteen further stale line numbers were fixed. Corrections are
> marked **[C1]**–**[C5]** at the point of correction and indexed in §11. Conditions K2 (the
> `view_factory=None` test) and K3 (the `DEBT(R20)` marker) are implemented; see §6 and §5.
> The code of the migration itself is unchanged from the ratified revision.

## 1. The contract now implemented

```python
def load_policy(
    spec: ModelSpec, adapter: Path | None, *, lazy: bool = False
) -> tuple[Any, Any, ArchitectureView, ResolvedSpec]: ...
```

`src/local_llm_lab/pipeline/evaluate.py:72-92`. It is the repository's single `mlx_lm.load`
call site **for policies**: it calls `configure_local_cache()` (`evaluate.py:84`), loads by
`spec.hf_id` (`evaluate.py:87-91`), resolves the adapter path, builds the `ArchitectureView`,
and returns `spec.resolve(model, tokenizer)` (`evaluate.py:92`). It is not the repository's only
`mlx_lm.load` call site; two others remain and are listed in §7.

`run_evaluation(*, spec: ModelSpec, ...)` at `evaluate.py:377`; the resolved declaration is
written to the summary at `evaluate.py:434` (`"model": resolved.as_dict()`), which is SPEC-002
§6's outstanding shortfall and briefing §6's "every JSON records ... the resolved `ModelSpec`".

`evaluate_tasks` gained required keyword-only `spec`, `view`, `resolved` (`evaluate.py:101`,
parameters at `:105-108`) and threads all three into `run_task` (`evaluate.py:130-132`).

### **[C1]** What the behavioural payoff actually is (mechanism corrected)

Revision 1 said evaluation "no longer runs in `build_prompt` compatibility mode, so the
generation-suffix assertion is live". **That mechanism is wrong.** The assertion was already
live before this change:

- `run_task` never passes `spec=None` down to `build_prompt`. It substitutes a spec first:
  `runner.py:389` calls `_compatibility_runner_inputs`, which at `runner.py:502-511` returns
  `load_model_spec("qwen25-coder-3b")` whenever `spec is None`.
- `build_prompt` therefore always received a non-`None` spec from the evaluation path, so
  `compatibility_mode` was false (`protocol.py:262`) and the suffix assertion at
  `protocol.py:270-271` already ran.

The real gain is **which** spec the assertion runs against. Before, evaluation asserted against
a hard-coded 3B spec; now it asserts against the loaded model's own registry spec, threaded from
`load_policy` through `evaluate_tasks` (`evaluate.py:130`) into `run_task`. That is what lets it
catch a Qwen3.5 generation-suffix mismatch — under the old path a Qwen3.5 template would have
been checked against `qwen25-coder-3b`'s expected suffix, which is not a meaningful check.
The outcome claim in revision 1 ("the assertion can now catch a suffix mismatch on a non-3B
model") stands; only the stated mechanism was wrong.

The second half of the payoff is unaffected: `runner.make_turn_cache` now receives the model's
own resolved cache strategy instead of falling back to `TrimCache` (`runner.py:414-419`;
`make_turn_cache` is defined at `runner.py:200`).

## 2. Files changed (line counts from `git diff --numstat`, re-taken 2026-09-04 13:45)

| File | +/- | What |
| --- | --- | --- |
| `src/local_llm_lab/pipeline/evaluate.py` | +34/-6 | new signature, `evaluate_tasks` model context, `run_evaluation(spec=)`, resolved spec in the summary |
| `src/local_llm_lab/pipeline/preflight.py` | +18/-22 | default loader routed through `load_policy`; `view_factory`/`resolver` become overrides |
| `src/local_llm_lab/pipeline/branch.py` | +22/-6 (shared) | ~7 lines are the migration's; +3 are the K3 `DEBT(R20)` marker (`branch.py:75-77`); the rest is the in-flight spec-threading lane, preserved |
| `src/local_llm_lab/pipeline/rollout.py` | +14/-2 | registry lookup, 4-tuple unpack, model context into `collect_rollouts`, resolved spec in the summary |
| `src/local_llm_lab/probes/adapter_delta.py` | +11/-8 | both load sites, `_reuse_loaded_policy` 4-tuple, `run_evaluation(spec=)` |
| `src/local_llm_lab/pipeline/cli.py` | +6/-4 | one registry lookup per stage, shared by the screen/eval calls and `write_provenance` |
| `src/local_llm_lab/probes/assistant_axis.py` | +2/-4 | two CLI load sites; dropped the now-unused `ArchitectureView` import |
| `src/local_llm_lab/probes/patch.py` | +1/-3 | load site; removed duplicate view build and `spec.resolve` |
| `src/local_llm_lab/probes/state_probe.py` | +1/-3 | load site; dropped the now-unused `ArchitectureView` import |
| `src/local_llm_lab/pipeline/jlens.py` | +1/-2 | load site; removed duplicate view build |
| `research/cache_equivalence.py` | +1/-4 | load site; removed duplicate view build and `spec.resolve` |
| `tests/test_evaluate.py` | +161/-8 | migrated four tests; three new contract tests |
| `tests/test_preflight.py` | +115/-8 | loader fakes now return the 4-tuple; one new test in the migration, plus the K2 test and its `_RecordingControlView` fake (+71 of the 115) |
| `tests/test_adapter_delta.py` | +18/-9 | loader fakes, `_reuse_loaded_policy` arity, `spec=` assertion |
| `tests/test_integrity.py` | +14/-1 | two call sites gain the model context (see the §10 disclosure) |
| `tests/test_jlens.py` | +5/-8 | loader fakes return the view; dropped two `from_model` patches |
| `tests/test_probes_axis.py` | +6/-5 | loader fakes return the view; dropped a `from_model` patch |
| `tests/test_probes.py` | +6/-6 | loader fake returns the view; dropped a `from_model` patch |
| `tests/test_state_probe.py` | +5/-6 | loader fake returns the view; dropped a `from_model` patch |
| `tests/test_patch.py` | +2/-4 | loader fake returns the view; dropped a `from_model` patch |
| `tests/test_cache_equivalence.py` | +3/-3 | loader fake returns view and resolved spec |

`tests/test_branch.py` (+119/-2) and the bulk of `src/local_llm_lab/pipeline/branch.py` are the
other implementer's uncommitted work. `tests/test_branch.py` was not edited by this task at all,
and in `branch.py` this task only added to their spec threading (plus the K3 comment).
`tests/test_rollout.py` (untracked, theirs, 151 lines) was not modified; it was RED on arrival
and is now GREEN unchanged.

`tests/test_repository_rules.py` (+167/-2) is in the tree but belongs to the R17 scanner lane
(`01a066e6`), not to this task; it landed while these corrections were being written (file
mtime 13:41) and is the reason the suite count moved from 597 to 600 mid-session. See §9.

No file was added or removed except this report.

## 3. Callers updated (verified by grep, not by the task list)

### **[C4]** Twelve call expressions, not thirteen

`load_policy` is **called** at twelve sites. The table has thirteen rows: the last is a
monkeypatch **assignment** in `_reuse_loaded_policy`, not a call, and is kept in the table
because the substitute has to satisfy the same arity.

| # | Site | Kind | Change |
| --- | --- | --- | --- |
| 1 | `pipeline/evaluate.py:398` (`run_evaluation`) | call | `load_policy(spec, adapter)`; 4-tuple |
| 2 | `pipeline/rollout.py:128` | call | `load_model_spec` at `:127` then `load_policy(spec, adapter)`; view/resolved into `collect_rollouts` |
| 3 | `pipeline/branch.py:231` | call | 4-tuple; view/resolved into `mine_pairs` and `run_task` |
| 4 | `pipeline/jlens.py:564` | call | 4-tuple; removed `ArchitectureView.from_model` |
| 5 | `pipeline/preflight.py:262` (`_default_loader`, defined at `:256`) | call | now delegates to `load_policy(spec, adapter, lazy=lazy)` |
| 6 | `probes/assistant_axis.py:1217` (`_build`) | call | 4-tuple; removed `ArchitectureView.from_model` |
| 7 | `probes/assistant_axis.py:1280` (`_project`) | call | 4-tuple |
| 8 | `probes/adapter_delta.py:930` (`--ablate`) | call | 4-tuple; removed `from_model` and `spec.resolve` |
| 9 | `probes/adapter_delta.py:1056` (static analysis) | call | now loads a `ModelSpec` from the registry first |
| 10 | `probes/patch.py:609` | call | 4-tuple; removed `from_model` and `spec.resolve` |
| 11 | `probes/state_probe.py:2975` | call | 4-tuple; removed `ArchitectureView.from_model` |
| 12 | `research/cache_equivalence.py:49` | call | 4-tuple; removed `from_model` and `spec.resolve` |
| — | `probes/adapter_delta.py:570` (`_reuse_loaded_policy`) | **assignment** (`evaluate.load_policy = lambda ...`, restored at `:574`) | substitute returns the 4-tuple; takes `view`/`resolved` |

Callers of `run_evaluation` (signature changed `model_name: str` → `spec: ModelSpec`):

- `pipeline/evaluate.py:500` (`main`) — `spec=load_model_spec(args.model)`
- `pipeline/cli.py:483` (`stage_select`) and `cli.py:584` (`stage_eval`) — each stage now loads
  the registry spec once (`cli.py:471`, `cli.py:562`) and reuses it for both the evaluation and
  `write_provenance`, so the existing one-lookup-per-stage assertions
  (`tests/test_cli.py:292`, `:677`) still hold.
- `probes/adapter_delta.py:944` — `spec=spec` (the spec is loaded at `adapter_delta.py:929`).

Callers of `evaluate_tasks` / `collect_rollouts` (new required model context): `evaluate.py:412`,
`rollout.py:131`, `tests/test_evaluate.py:84`, `tests/test_integrity.py:413`,
`tests/test_integrity.py:557`, `tests/test_rollout.py:67` (already written to the new contract).

`ArchitectureView.from_model` duplication removed at six sites (jlens, assistant_axis `_build`,
patch, state_probe, adapter_delta `--ablate`, cache_equivalence); duplicate `spec.resolve` calls
removed at three (patch, adapter_delta, cache_equivalence).

## 4. Wiring-map rows touched

- **§2.10** — `load_policy` and `run_evaluation` now match the stated signatures. `summarize`,
  `wilson`, `mcnemar` untouched.
- **§4 gap 2** ("`load_policy` signature change → callers listed in §2.10") — closed for every
  caller that exists; see §7 for the one §2.10 entry that is not in fact a caller.
- **§5** — `outputs/<run>/evals/*.json` summary key `model` is now the resolved spec dict rather
  than the model-name string; the same is true of the rollout `summary.json` (required by
  `tests/test_rollout.py:150-151`) and of the branch `summary.json` (for consistency).
- **§2.3 / R-ratified phased `build_prompt` contract, condition (a)** — evaluation and rollout
  now thread the loaded spec, so the generation-suffix assertion runs against the loaded model's
  own spec on those paths. (Corrected wording; see **[C1]**.)
- **R20** — the `DEBT(R20)` marker required by condition K3 is at
  `src/local_llm_lab/pipeline/branch.py:75-77`.
- No document under `pending/` was edited (R3).

## 5. Decisions taken under ambiguity

1. **`run_rollout`/`run_branch_mining` keep `model_name: str`.** The map changes
   `run_evaluation`, not these. `tests/test_rollout.py:134` calls `run_rollout(model_name=...)`
   and monkeypatches `rollout.load_model_spec`, so the registry lookup was placed inside
   `run_rollout` (`rollout.py:127`). `branch.py` follows the same shape, which is what the other
   lane had already written.
2. **`preflight`'s loader seam became `load_policy`-shaped** (`(spec, adapter, *, lazy)` returning
   the 4-tuple) rather than preflight keeping a second `mlx_lm.load` call site. This is the only
   consumer of the new `lazy` keyword. `view_factory` and `resolver` were kept but demoted to
   *overrides*: `None` now means "use what the loader returned" (`preflight.py:90-94` for
   `run_preflight`, `:178-180` for `run_residual_control`). That preserves the SPEC-001 §10 fake
   seams (`_MismatchingView`, `_ControlView`) with a one-line change per test instead of a
   rewrite. The alternative — deleting those two parameters — would have forced preflight's fakes
   to merge model and view, and that was judged to be the owning lane's call.
3. **`evaluate_tasks` / `collect_rollouts` take `spec`, `view`, `resolved` as *required*
   keyword-only arguments.** Optional defaults would have silently preserved the compatibility
   path that this task exists to remove. Two call sites in `tests/test_integrity.py` were updated
   accordingly (see the §10 disclosure).
4. **`mine_pairs` takes `view`/`resolved` as *optional* keyword-only arguments**
   (`branch.py:78-79`), unlike the two above. `mine_pairs` is mid-flight in another lane whose
   tests call it with `spec=` only; optional parameters extend their contract without breaking
   it. `run_branch_mining` always passes the real values. **Ruling R20 accepts this as debt with
   an expiry**, and condition K3 is now discharged: `branch.py:75-77` carries

   ```python
   # DEBT(R20): required once the condition-4 slice threads branch.build_prompt. That slice
   # drops both defaults and updates the two optional-path callers that omit them today,
   # tests/test_branch.py:164 and :210.
   ```

   The two named call sites are
   `test_branch_pairs_keep_the_seed_step_raw_completion` (`tests/test_branch.py:160-170`, `spec=`
   at `:164`) and `test_branch_pairs_reject_seed_steps_without_raw_completion`
   (`tests/test_branch.py:206-216`, `spec=` at `:210`); both omit `view`/`resolved`.
5. **`summary["model"]` is `resolved.as_dict()`, not `asdict(spec)`.** The ratified §2.5/§5
   wording is "registry `ModelSpec` (data stage) / resolved spec (model stages)", and evaluation
   is a model stage. `tests/test_rollout.py:150` pins the same choice for rollout.
6. **`stage_select` loads the registry spec after the empty-checkpoint guard**
   (`cli.py:466-471`) so that a run with no checkpoints still fails on the checkpoint message
   before touching the registry (`tests/test_selection.py:23`).
7. **`adapter_delta` static analysis** (`--adapters` without `--ablate`) previously passed a raw
   `args.model` string to `load_policy`; it now resolves it through `load_model_spec`
   (`adapter_delta.py:1056`). Its `payload["model"]` remains `args.model`, unchanged.

## 6. Tests added (R10: per-module files)

From the migration itself:

- `tests/test_evaluate.py:306` `test_load_policy_returns_the_view_and_resolved_spec_from_one_load`
  — pins the 4-tuple order, that `hf_id` (not `name`) is loaded, adapter path resolution, `lazy`
  forwarding, `configure_local_cache` ordering, and that `spec.resolve` is called with the loaded
  pair. Uses a fake `mlx_lm` module; no checkpoint.
- `tests/test_evaluate.py:338` `test_evaluate_tasks_carries_the_loaded_model_context_into_every_run`
  — pins that spec/view/resolved reach `run_task` for every task.
- `tests/test_evaluate.py:380`
  `test_run_evaluation_writes_the_resolved_model_spec_into_the_evaluation_json` — SPEC-002 §6.
- `tests/test_preflight.py:711` `test_default_loader_is_the_one_shared_policy_loader` — pins that
  preflight has no second loading path.

Added by condition K2:

- `tests/test_preflight.py:478`
  `test_residual_control_without_a_view_factory_uses_the_loader_view` (the call is at
  `tests/test_preflight.py:493`), with its fake `_RecordingControlView` at
  `tests/test_preflight.py:126-150`. It calls `run_residual_control`
  with `view_factory` **omitted** — the configuration every non-test caller uses — and asserts
  that the view the loader returned is the instance the stage actually ran, by pinning the exact
  ordered call log recorded on that instance
  (`embed`, `masks`, `run_block:0..3`, `final_norm`, `native_manual`). It additionally
  monkeypatches `ArchitectureView.from_model` to `pytest.fail`, so a reintroduced default view
  factory fails loudly rather than silently. Fake-only: fake loader, fake tokenizer, fake view,
  `array_api=np`; no `mlx_lm.load`, no checkpoint, no inference.

  **Mutation evidence.** Re-running the same call with `view_factory=lambda model: _ControlView()`
  (the pre-migration behaviour) leaves `loaded_view.calls == []`, so the new assertion fails,
  while `fp32_manual_vs_native.max_abs_error` stays `0.0` — i.e. the numeric assertions alone
  would *not* have caught the regression, and the call-log assertion is the load-bearing one.
  Verified by a throwaway script, not committed.

No test was weakened. `tests/test_rollout.py` was treated as the specification and passes
unmodified. Existing tests changed only where a signature this task changed made them call the
old contract (loader fakes returning 2-tuples, `model_name=` keywords,
`ArchitectureView.from_model` monkeypatches that are now dead because the loader supplies the
view).

## 7. Observed, not fixed

### **[C2]** Two direct model loaders remain outside `load_policy`, not one

Revision 1 claimed `research/jspace_sweep.py` was "the last direct policy-load call site outside
`load_policy`". **That is false.** Two remain:

1. **`src/local_llm_lab/pipeline/cli.py:125-130`, `_load_training_base`** — the training path.

   ```python
   def _load_training_base(hf_id: str) -> tuple[Any, Any]:
       """Lazily load the registry's base model only while resolving training targets."""
       configure_local_cache()
       from mlx_lm import load

       return load(hf_id)
   ```

   It is **live**, not dead code: called at `cli.py:238` (resolving the effective training spec)
   and `cli.py:307` (the train stage, before `lora_config`). It takes an `hf_id` string, returns
   a 2-tuple, and never builds an `ArchitectureView`; each caller then calls
   `effective.resolve(model, tokenizer)` itself (`cli.py:239`, `cli.py:308`), which rebuilds a
   view internally. This was missed in revision 1 because the audit grepped for `load_policy`
   callers rather than for `mlx_lm.load` call sites.

   **The Chief has assigned this to the next slice**, not this one: it is a training-path load in
   SPEC-001 §7's territory and should go through `load_policy` with `adapter=None, lazy=False`
   (`under_review/WAVE1-LOAD-POLICY-REVIEW-round1-2026-09-04.md:28-30`, decision 2, and R20's
   "condition-4 completion slice"). It also sits on `cli.py`, claimed by lane `01a06858`.

2. **`research/jspace_sweep.py`** — listed in map §2.10 as a `load_policy` consumer but is not
   one. It imports `mlx_lm.load` at `research/jspace_sweep.py:62` and calls it at `:76-79` with a
   hard-coded `"mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"` (`:77`) and adapter path (`:78`),
   and builds its own view at `:82`. Migrating it means also making its model a parameter, which
   is SPEC-001 §1's lane, and it has no test to protect the change, so it was left.

### **[C5]** `run_preflight`'s `view_factory=None` branch is still untested

Condition K2 cites `cli.py:742` as the production caller of `run_residual_control`. **That
citation is wrong**: `cli.py:742` is `run_preflight(args.model)` — the preflight stage — and it
is the only `preflight` entry point in the CLI. `run_residual_control` has **no** production
caller anywhere in the repository; it is exported at `preflight.py:19`, called from
`tests/test_preflight.py:451` and `:493`, and otherwise invoked by the documented one-liner at
`under_review/GITHUB-ISSUE-15-RESIDUAL-EQUIVALENCE-DIAGNOSIS.md:153`, which also omits
`view_factory`.

The condition's *letter* has been implemented exactly as written (a test for
`run_residual_control`'s `view_factory=None` branch, §6). Its *rationale* — "this is the
production configuration at `cli.py:742`" — is however true of `run_preflight`, whose
`view_factory=None` **and** `resolver=None` branches at `preflight.py:91-94` remain unexercised:
all four `run_preflight` tests (`tests/test_preflight.py:222`, `:318`, `:351`, `:522`) pass both
overrides explicitly (`:247-248`, `:339-340`, `:372-373`, `:537-538`). That is the branch
`cli.py:742` actually reaches.
This was not fixed here because widening a ratified condition unilaterally is the Deputy's call,
not the implementer's (briefing §7.3). One further fake-only test of the same shape would close
it.

### Other observations (unchanged from revision 1)

- `jlens.py`, `assistant_axis.py`, and `state_probe.py` now receive a `ResolvedSpec` they discard
  (`_resolved`). Their output artifacts still do not record it, so briefing §6's "every JSON
  records the resolved `ModelSpec`" remains open for those three probe CLIs. That is SPEC-001 §8
  / §9 work, not this task.
- `load_policy` builds one `ArchitectureView` (`evaluate.py:92`) and `ModelSpec.resolve` builds a
  second internally (`models.py:64-66`). Removing the second would change `resolve`'s normative
  §2.1 signature, so it was left. The cost is one module walk per load, no model execution.

## 8. Banned-constant check (briefing §1.7, R17)

Grep over the added lines of `git diff -- src research tests`:

```
git diff -U0 -- src research tests | grep -E "^\+" | grep -nE \
  "(^|[^A-Za-z0-9_])(36|2048|35)([^A-Za-z0-9_]|$)|<\|im_end\|>|model\.model\.layers|\
['\"](q|k|v|o|gate|up|down)_proj['\"]|in_proj_"
```

Exit code 1, no matches for this task's lines. The repository's own AST scanner,
`tests/test_repository_rules.py:222`
(`test_banned_model_constants_are_limited_to_approved_or_legacy_modules`), passes as part of the
suite; note that this test moved from `:118` to `:208` while these corrections were being
written, because the R17 lane added `test_banned_model_constants_are_not_hidden_inside_string_literals`
(`tests/test_repository_rules.py:230`) in the same file. No hard-coded model name was introduced;
`DEFAULT_MODEL` in `evaluate.py:22` is unchanged and is still only a CLI default that flows
through `load_model_spec`.

The K2 and K3 additions introduce no constants: the K3 change is a comment, and the K2 test
reuses the existing `_spec()` fake (`tests/test_preflight.py:19-32`), whose ids are `fake-model`
and `org/fake-model`.

## 9. Suite result (R13 green gate)

```
$ uv run pytest
601 passed (602 at commit time: the R17 scanner slice committed afc0905 in the interval, +1 test) in 8.84s
exit code 0
```

Run at 2026-09-04 13:45:41 from the repository root, and confirmed unchanged by a second run at
13:50:34 (`601 passed in 9.78s`, exit code 0). `uv run ruff check src tests research` — all
checks passed, exit code 0.

The banned-constant grep restricted to this round's own added lines
(`git diff -U0 -- src/local_llm_lab/pipeline/branch.py tests/test_preflight.py`) exits 1 with no
matches.

**Counts are a moving target this session and the deltas need reading carefully.** Revision 1
recorded 596 passed. Measurements taken during this corrections round:

| When | Count | Cause |
| --- | --- | --- |
| 13:38 (first baseline this round) | 597 | +1 over revision 1 from a concurrent landing |
| 13:42 (second baseline) | 600 | R17 lane landed `tests/test_repository_rules.py` (+167/-2, mtime 13:41) |
| 13:45 (after this round's K2 test) | **601** | +1, exactly the one test added here |

Only the last row is attributable to this task's corrections. `tests/test_repository_rules.py` is
not on this task's paths and was neither read for content nor edited here. Revision 1 did not
record a baseline count before starting, so no before/after delta is claimed beyond the +1 above.

`tests/test_rollout.py` was red on arrival (its two tests were written against the new contract
that did not yet exist); both pass now with that file unmodified.

## 10. **[C3]** Cross-lane disclosure

This change writes into paths claimed by other coordination lanes. Canonical claims read from
`.codex/coordination/active/*.json` and `.codex/coordination/CURRENT.md`.

| Lane | Claimed path this change writes | Written by |
| --- | --- | --- |
| `01a066e6` Issue 14 probe policy defaults | `probes/state_probe.py`, `probes/assistant_axis.py`, `probes/patch.py`, `probes/adapter_delta.py`, `pipeline/jlens.py`, `tests/test_state_probe.py`, `tests/test_probes_axis.py`, `tests/test_patch.py`, `tests/test_adapter_delta.py`, `tests/test_jlens.py`, `tests/test_probes.py` | the migration |
| `01a06718` Rollout model-aware caller migration | `pipeline/rollout.py` | the migration |
| `01a06728` Issue 15 residual diagnosis and gate API | `pipeline/preflight.py`, `tests/test_preflight.py` | the migration, and the K2 test in this round |
| `01a06858` SPEC-001 S7 training and model wiring | `pipeline/cli.py` | the migration |
| `01a06844` Issue 10 compatibility follow-up | `pipeline/branch.py`, `pipeline/evaluate.py`, `tests/test_evaluate.py` | the migration, and the K3 marker in this round |

**The claims are orphaned.** The newest write to any coordination file is 12:44:28
(`.codex/coordination/active/01a06728-508f-7980-b9f5-e3fb0da01dd0.json` and `CURRENT.md`); the
last Codex commit is `1603159` at 12:49:53. Nothing has claimed, released, or updated a lane
since. The Chief has accepted this as disclosed and ruled that going forward cross-lane edits are
named in the dispatch rather than discovered by the reviewer
(`under_review/WAVE1-LOAD-POLICY-REVIEW-round1-2026-09-04.md:39-41`). This round's two crossed
edits — `tests/test_preflight.py` (`01a06728`) and `pipeline/branch.py` (`01a06844`) — **were**
named in advance in the dispatch.

**Specifically against the rollout lane's written plan.** `docs/superpowers/plans/2026-09-04-rollout-model-aware-load-policy.md:25`
says: do not copy the existing integrity-filter assertion from `tests/test_integrity.py`, and "if
its old direct call becomes incompatible, report that exact subtractive move as a dependency";
`:249` repeats it — "do not duplicate or edit it; send one dependency notice requesting the exact
R10 subtractive move and remain blocked until the Coordinator assigns that path."

**`tests/test_integrity.py` was edited rather than reported.** Two call sites were updated in
place:

- `tests/test_integrity.py:413-422` — `evaluate.evaluate_tasks(...)` gains `spec`/`view`/`resolved`
  at `:417-419`.
- `tests/test_integrity.py:557-570` — `rollout.collect_rollouts(...)` gains the same three at
  `:561-563`. (Condition K1 cites this as `:558-561`; the hunk starts at new-file line 558 and the
  three added lines are `:561-563`.)

Plus the import at `tests/test_integrity.py:12`. No dependency notice was sent. The edits are
additive at the call sites and no assertion was weakened, but the plan's instruction was to
report and stay blocked, and that was not done.

## 11. Correction index (condition K1)

| # | Revision 1 claim | Status | Corrected in |
| --- | --- | --- | --- |
| C1 | "evaluation no longer runs in compatibility mode, so the assertion is live" | **wrong mechanism** (assertion was always live; `runner.py:389`, `:502-511`). Outcome claim stands. | §1 |
| C2 | `research/jspace_sweep.py` is "the last direct policy-load call site outside `load_policy`" | **false** — `cli.py:125-130` also remains, live at `:238` and `:307`. Two, not one. Assigned to the next slice. | §7 |
| C3 | (no cross-lane disclosure) | **missing** — four crossed lane claims, orphaned; `tests/test_integrity.py` edited against the rollout lane's plan | §10 |
| C4 | "13 `load_policy` call sites" | **wrong count** — twelve call expressions; the thirteenth row is a monkeypatch assignment | §3 |
| C5 | preflight risk "unchanged" | **partly false** — true for `run_preflight`, false for `run_residual_control`; mitigating artifact evidence added | §12, §7 |

Stale citations found by re-deriving every file:line against the tree at 13:45 and corrected in
place: `evaluate.py:101-131` → `:101-132`; `runner.py:414-420` → `:414-419`; `branch.py:228` →
`:231`; `branch.py:75-76` → `:78-79`; `cli.py:484` → `:483`; `cli.py:585` → `:584`;
`tests/test_preflight.py:640` → `:711`; `tests/test_rollout.py:135` → `:134`;
`tests/test_repository_rules.py:118` → `:208`; `preflight.py:90-93` → `:90-94`;
`models.py:64-67` → `:64-66`; `evaluate.py:20` → `:22`; `adapter_delta.py:1053-1056` → `:1056`.
Several of these drifted because other lanes edited the same files after revision 1 was written.

## 12. **[C5]** Not verifiable without a model run (gated)

- That `mlx_lm.load(..., lazy=True)` behaves identically through `load_policy` as through
  preflight's former direct call. The keyword is passed through unchanged; only a real snapshot
  load proves the loaded module tree is the same.
- **Preflight risk is not "unchanged"; it changed for one of the two entry points.**
  Revision 1 said the risk was unchanged. Accurately:

  - **`run_preflight`: unchanged.** It always resolved a lazily loaded model. Before this change
    it called `loader(spec.hf_id, lazy=True)` and then both `view_factory(model)` and
    `resolver(spec, model, tokenizer)` — i.e. `ArchitectureView.from_model` and `spec.resolve` on
    the lazy model. It now gets the same two from `load_policy` at `preflight.py:90`, still with
    `lazy=True`. Same operations, same model, different caller.
  - **`run_residual_control`: changed.** Before, it ran only `view_factory(model)` — a view
    build, no `spec.resolve`. Now `preflight.py:178` resolves through `load_policy`, so
    `ModelSpec.resolve` (LoRA target discovery, layer-type walk, cache-strategy resolution) runs
    on the lazily loaded model on a path where it previously did not. The resolved spec is then
    discarded (`_resolved`), so this is added work on the load path, not a change to the report's
    contents.

  **Mitigating evidence that the added work succeeds on this model:** the real artifact
  `outputs/preflight/qwen35-4b.json` was produced by `run_preflight` on `qwen35-4b`
  (`mlx-community/Qwen3.5-4B-MLX-4bit`, snapshot `32f3e8ec`) and contains
  `lora.trainable_parameters: 32464896` and twelve `lora.keys` (`self_attn.{q,k,v,o}_proj`,
  `mlp.{gate,up,down}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`),
  plus a 32-entry `cache.entry_types` list and `cache.strategy_reason:
  "auto:equivalence_unverified"`. Every one of those fields is derived from
  `ModelSpec.resolve` on a lazily loaded model, so that resolution demonstrably works on this
  checkpoint. The artifact's top-level `passed: false` is unrelated: `jvp.finite` and
  `memory.within_budget` are both true and only `residual_equivalence.passed` is false, which is
  the separate issue #15 residual-equivalence investigation. On the Deputy's downgrade of this
  risk on exactly this evidence, see
  `under_review/WAVE1-LOAD-POLICY-REVIEW-round1-2026-09-04.md:9-11`.
- That the generation-suffix assertion in `build_prompt` holds for the real Qwen3.5-4B chat
  template **when checked against the Qwen3.5 spec** (see **[C1]**: the assertion already ran,
  but against a 3B spec). A mismatch will now raise during evaluation instead of passing
  silently — which is the intended safety behaviour, but the first real evaluation on a non-3B
  model is where it will be observed.
- Whether the resolved cache strategy (`none` for the unverified hybrid, per R1/R9) changes any
  evaluation output relative to the previous unconditional `TrimCache`. On the 3B model the
  registry resolves to `trim`, so 3B behaviour is unchanged by construction; this is untested
  against a checkpoint.
