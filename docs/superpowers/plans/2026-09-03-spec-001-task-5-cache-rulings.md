# SPEC-001 Task 5 Cache Rulings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan with exactly one principal implementer followed by one independent principal reviewer. Use `superpowers:test-driven-development` for every behavior change and `superpowers:systematic-debugging` for any unexpected failure. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the independently reviewed R9 hybrid-cache safety slice first, then implement and independently verify issue #1 rulings R1, R2, and R4 in the cache portion of SPEC-001 Task 5 and close R3 and the two process notes from traceable evidence produced by their existing owners.

**Architecture:** Keep declared cache policy and equivalence evidence in immutable `ModelSpec`, resolve the safe runtime strategy and an auditable reason in `ResolvedSpec`, and let one runner factory construct trim, snapshot, or no cache. `SnapshotCache` primes and deep-copies only the immutable system-plus-task prefix. The work does not change `ArchitectureView`, generator semantics, provenance ownership, or either process-note implementation.

**Tech Stack:** Python 3.13, frozen dataclasses, PyYAML, MLX/MLX-LM interfaces exercised only through fake models and caches, pytest, Codex Coordinator, GitHub CLI.

**Spec:** `design_specifications/pending/SPEC-001-model-agnostic-backbone.md` §1 and §5, overridden by `design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md` §7 rulings R1–R6 and issue #1.

## Global Constraints

- Work directly in `/Users/daniel.tipton/Desktop/An app` on `codex/agent-v2-specs`; do not create or switch a branch or worktree.
- `design_specifications/pending/` is authoritative and read-only. Do not edit it to close R3.
- Do not load a checkpoint, call `mlx_lm.load`, or run train, select, eval, rollout, branch, prefer, preflight, probe, `research/cache_equivalence.py`, or `research/jspace_sweep.py`.
- Tests use fake dense and fake hybrid models/caches only. No real pipeline or probe run is permitted.
- Do not modify or regenerate anything under `data/`, `outputs/`, or `reports/`.
- Preserve the accepted `ArchitectureView` additions (`vocab_size`, `tie_word_embeddings`, and `lora_parameter_count(keys, rank)`) and both installed DeltaNet projection layouts.
- Preserve ruling R5: `GENERATOR_VERSION = 1` names commit `97d197c`, the current accepted generator is version 2, and this task neither changes generated rows nor bumps the version.
- Preserve ruling R6 and issue #2 F1–F4. Do not change transcript header handling, generator hashes, recovery variants, the part-one SPEC-002 report, or their provenance/CLI wiring.
- Apply ruling R10 narrowly: new model/cache-resolution tests live in `tests/test_models.py`; only the cache-resolution assertion this lane touches moves out of `tests/test_probes.py`.
- `"<|im_end|>"` may occur only in `configs/models/`, `src/local_llm_lab/models.py::_default_spec`, `src/local_llm_lab/arch.py`, tests, and explicitly classified legacy modules.
- The principal implementer stages and commits only this plan's reviewed product paths with explicit pathspecs. No broad staging, pull, rebase, merge, reset, restore, stash, clean, force push, or branch switch is allowed in the shared checkout.
- Do not close issue #1 until every checkbox has committed, independently reviewed, pushed evidence.

## Dependency and Claim Gate

Implementation remains blocked until all of the following are true:

1. Native task `01a06718-057a-7bb2-a62d-71f86e904d64` has released its issue #2 claim after its independent review, so shared `tests/test_pipeline.py`, `src/local_llm_lab/provenance.py`, and `src/local_llm_lab/pipeline/cli.py` paths are no longer owned there.
2. SPEC-001 Task 3 and Task 4 have landed reviewed commits on the shared branch. Task 5 consumes Task 3's final view-based cache boundary and Task 4's `build_prompt(..., spec=..., generation=False)` rendering contract.
3. The root goal Coordinator sends an explicit unblocking message to native task `01a06728-508f-7980-b9f5-e3fb0da01dd0` with the exact revised claim.
4. This task re-lists the primary board, re-reads every claimed file, and atomically revises its own claim from blocked to active, adding the released shared paths named by the root. A path release by itself is not permission to start.

After the gate opens, record the new `HEAD`, relevant foreign dirty paths, and focused baseline test result in `.superpowers/sdd/2026-09-03-spec-001-task-5-cache-rulings/progress.md` before dispatching the implementer.

The R9 safety slice has a separate earlier gate and does not unblock Task 5: wait until native task `01a066e6-99ae-7ef3-8955-37419c5b6689` releases `tests/test_probes.py`, re-list the board, then revise this claim to add exactly `tests/test_probes.py` and new `tests/test_models.py`. The root's current assignment authorizes only that narrow R9 slice after the verified release. Keep resolver/runner implementation blocked until the later explicit root message.

## File Map

- Modify `configs/models/qwen25-coder-3b.yaml`, `configs/models/qwen35-4b.yaml`, and `configs/models/qwen35-9b.yaml`: add the explicit nullable registry evidence key without changing established strategies or model policy.
- Modify `src/local_llm_lab/models.py`: parse and validate equivalence evidence, resolve all cache branches, expose `cache_strategy_reason`, and serialize the result.
- Modify `src/local_llm_lab/pipeline/runner.py`: retain trim reuse, implement immutable-prefix snapshot reuse, select through `make_turn_cache`, and emit exactly one warning line for the unverified hybrid auto branch.
- Modify `tests/test_pipeline.py` after its later claim revision: prove registry validation, all three auto branches, reason serialization, snapshot state isolation, cache factory behavior, and the warning count with fakes.
- Inspect `src/local_llm_lab/provenance.py` and `src/local_llm_lab/pipeline/cli.py` after issue #2 releases them: verify that `ResolvedSpec.as_dict()` remains the serialization seam and that no F1–F4 behavior is duplicated. Change neither unless the root's revised claim and a failing issue #1 test demonstrate a cache-specific defect.
- Update `docs/superpowers/plans/2026-09-03-spec-001-task-5-cache-rulings.md` and `.superpowers/sdd/2026-09-03-spec-001-task-5-cache-rulings/progress.md` with execution and review evidence.

---

### Task 0: R9 safety slice — disable accidental hybrid snapshots until Task 5

**Files:**

- Modify: `configs/models/qwen35-4b.yaml`
- Modify: `configs/models/qwen35-9b.yaml`
- Modify: `src/local_llm_lab/models.py`
- Modify: `tests/test_probes.py`
- Create: `tests/test_models.py`
- Update: `.superpowers/sdd/2026-09-03-spec-001-task-5-cache-rulings/progress.md`

**Interfaces:**

- Consumes: current `ModelSpec.resolve(model, tokenizer) -> ResolvedSpec` and the reviewed hybrid ArchitectureView fake already exercised in `tests/test_probes.py`.
- Produces: explicit `cache.strategy: none` for both Qwen3.5 registry entries; a visible `TEMPORARY until R1 (Task 5)` marker on the unsafe resolver branch; a per-module model test seam carrying the displaced cache assertion.
- Preserves: every non-cache assertion in `test_model_spec_resolve_populates_every_field_from_real_architecture_view`, all R1 implementation work for the later Task 5 slice, and every issue #2/SPEC-004 change.

- [ ] **Step R9.1: Verify the release and revise only this task's claim**

  Re-list the primary board after the `RELEASED` notice. Require the SPEC-004 claim to omit `tests/test_probes.py`; silence or elapsed time is not release evidence. Re-read `tests/test_probes.py`, both Qwen3.5 configs, and `models.py`, then update this task's exact claim revision to add:

  ```text
  tests/test_probes.py
  tests/test_models.py
  ```

  Keep every existing claimed path, add no action, and set the claim active only for the R9 slice. Record that full Task 5 remains held for a later explicit root unblock.

- [ ] **Step R9.2: Write the new per-module RED test and extract only the touched assertion**

  Create `tests/test_models.py` with a lightweight monkeypatched `ArchitectureView` that supplies the current `ModelSpec.resolve` fields without loading a checkpoint. Add a normal test proving both registry files are explicit `none`:

  ```python
  @pytest.mark.parametrize("name", ["qwen35-4b", "qwen35-9b"])
  def test_hybrid_registry_disables_cache_until_r1(name: str) -> None:
      assert load_model_spec(name).cache_strategy == "none"
  ```

  Move only the cache expectation out of the broad probe test by deleting this line from `tests/test_probes.py` and leaving all of its ArchitectureView/LoRA/probe assertions unchanged:

  ```python
  assert resolved.cache_strategy == "snapshot"
  ```

  Preserve the old expectation as an explicit strict xfail in `tests/test_models.py` with a local resolver fake:

  ```python
  class _FakeHybridView:
      num_layers = 4
      hidden_size = 8
      vocab_size = 23
      tie_word_embeddings = False
      cache_trimmable = False

      def layer_kind(self, index: int) -> str:
          return "attention" if index == 3 else "linear_attention"

      def lora_targets(self, policy) -> tuple[str, ...]:
          assert policy == "auto"
          return ("linear_attn.out_proj",)

      def lora_parameter_count(self, keys, rank: int) -> int:
          assert keys == ("linear_attn.out_proj",)
          assert rank == 16
          return 256


  @pytest.mark.xfail(strict=True, reason="R1")
  def test_unverified_hybrid_snapshot_expectation(monkeypatch) -> None:
      fake_view = _FakeHybridView()
      monkeypatch.setattr(
          ArchitectureView,
          "from_model",
          classmethod(lambda cls, model: fake_view),
      )
      spec = load_model_spec("qwen35-4b")
      tokenizer = SimpleNamespace(snapshot_revision="fake-hybrid-revision")
      resolved = spec.resolve(object(), tokenizer)
      assert resolved.cache_strategy == "snapshot"
  ```

  Import `SimpleNamespace` from `types`, `pytest`, `ArchitectureView`, and `load_model_spec` directly. Do not import a test module from another test module or copy the full probe fake hierarchy.

- [ ] **Step R9.3: Run the R9 test and verify RED**

  Run:

  ```bash
  uv run pytest -q tests/test_models.py
  ```

  Expected: the two registry assertions fail because both Qwen3.5 configs still declare `auto`, and the strict xfail reports XPASS because the current unsafe `auto → snapshot` branch still satisfies the obsolete expectation. These are the intended RED failures R9 removes.

- [ ] **Step R9.4: Apply only the temporary safety behavior**

  In both Qwen3.5 registry files, replace only:

  ```yaml
  cache:
    strategy: auto
  ```

  with:

  ```yaml
  cache:
    strategy: none
  ```

  In `ModelSpec.resolve`, do not change behavior; annotate the current unsafe branch exactly:

  ```python
  if self.cache_strategy == "auto":
      # TEMPORARY until R1 (Task 5)
      cache_strategy = "trim" if view.cache_trimmable else "snapshot"
  ```

  R9 is a registry safety stopgap, not permission to add equivalence evidence, `cache_strategy_reason`, cache factories, snapshot logic, warning behavior, or runner plumbing.

- [ ] **Step R9.5: Verify GREEN with the strict xfail intact**

  Run:

  ```bash
  uv run pytest -q tests/test_models.py
  uv run pytest -q tests/test_probes.py -k 'model_spec_resolve_populates_every_field'
  uv run ruff check src/local_llm_lab/models.py tests/test_models.py tests/test_probes.py
  ```

  Expected: the registry tests pass; the unverified snapshot expectation is exactly one strict XFAIL with reason `R1`; the remaining broad ArchitectureView resolution test passes; Ruff exits 0. No model/checkpoint is loaded.

- [ ] **Step R9.6: Commit and independently review the isolated safety slice**

  Stage and commit only the five R9 product/test files with explicit pathspecs:

  ```bash
  git add -- configs/models/qwen35-4b.yaml configs/models/qwen35-9b.yaml src/local_llm_lab/models.py tests/test_probes.py tests/test_models.py
  git diff --cached --check
  git commit -m "fix: disable unverified hybrid cache snapshots" -- configs/models/qwen35-4b.yaml configs/models/qwen35-9b.yaml src/local_llm_lab/models.py tests/test_probes.py tests/test_models.py
  ```

  Dispatch exactly one independent reviewer for this commit. The reviewer checks the ratified R9/R10 text, exact diff, strict xfail state, per-module seam, no coverage loss in the probe test, and fresh fake-only commands. Findings return to the same implementer. After approval, push/release only the R9 slice boundary while retaining this task's blocked Task 5 ownership.

---

### Task 1: Implement the R1/R2/R4 cache vertical

**Files:**

- Modify: `configs/models/qwen25-coder-3b.yaml`
- Modify: `configs/models/qwen35-4b.yaml`
- Modify: `configs/models/qwen35-9b.yaml`
- Modify: `src/local_llm_lab/models.py`
- Modify: `src/local_llm_lab/pipeline/runner.py`
- Test: `tests/test_pipeline.py`
- Inspect: `src/local_llm_lab/provenance.py`
- Inspect: `src/local_llm_lab/pipeline/cli.py`
- Update: `.superpowers/sdd/2026-09-03-spec-001-task-5-cache-rulings/progress.md`

**Interfaces:**

- Consumes: `ArchitectureView.cache_trimmable`, `ArchitectureView.make_cache()`, `ArchitectureView.layer_kind(index)`, Task 4's `build_prompt(tokenizer, messages, *, spec, keep_last, generation)`, and MLX-LM cache entries exposing writable `state`.
- Produces: `ModelSpec.cache_equivalence_verified: dict[str, str] | None`, `ResolvedSpec.cache_strategy_reason: str`, `TurnCacheBase`, `TrimCache`, `SnapshotCache(model, view, prefix_tokens)`, and `make_turn_cache(model, view, resolved, *, prefix_tokens)`.
- Preserves: explicit `trim`, `snapshot`, and `none` declarations; dense legacy trim behavior; `ResolvedSpec.as_dict()` as the sole downstream serialization seam; old `Trajectory` compatibility; current generator and provenance values.

- [ ] **Step 1: Write failing registry-schema and resolution tests**

  In `tests/test_pipeline.py`, extend the existing registry tests so all three checked-in model configs expose `cache_equivalence_verified is None`. Add temporary registry mappings that reject any non-null evidence unless it is a mapping with exactly two non-empty string fields, `date` and `sha256`, and the SHA is 64 hexadecimal characters.

  Parameterize the three `auto` branches through fake dense and hybrid views:

  ```python
  @pytest.mark.parametrize(
      ("cache_trimmable", "layer_types", "evidence", "strategy", "reason"),
      [
          (True, ("attention",) * 4, None, "trim", "auto:trimmable"),
          (
              False,
              ("linear_attention", "linear_attention", "linear_attention", "attention"),
              {"date": "2026-09-03", "sha256": "a" * 64},
              "snapshot",
              "auto:equivalence_verified",
          ),
          (
              False,
              ("linear_attention", "linear_attention", "linear_attention", "attention"),
              None,
              "none",
              "auto:equivalence_unverified",
          ),
      ],
  )
  def test_auto_cache_resolution_records_safe_branch(
      monkeypatch, cache_trimmable, layer_types, evidence, strategy, reason
  ) -> None:
      spec = replace(
          load_model_spec("qwen35-4b"),
          cache_strategy="auto",
          cache_equivalence_verified=evidence,
      )
      view = install_fake_view(monkeypatch, cache_trimmable, layer_types)
      resolved = spec.resolve(view.model, object())
      assert (resolved.cache_strategy, resolved.cache_strategy_reason) == (strategy, reason)
      assert resolved.as_dict()["cache_strategy_reason"] == reason
  ```

  Add explicit-strategy cases proving reasons `explicit:trim`, `explicit:snapshot`, and `explicit:none` without weakening R1's evidence rule for `auto`.

- [ ] **Step 2: Run the new model-resolution tests and verify RED**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'cache_equivalence or auto_cache_resolution or explicit_cache_resolution'
  ```

  Expected: failures because the registry evidence field and `cache_strategy_reason` do not exist and `auto` currently resolves every non-trimmable view straight to `snapshot`.

- [ ] **Step 3: Add and validate the R4 registry evidence field**

  Add this key to the existing `cache` mapping in every checked-in model registry file, retaining `qwen25-coder-3b`'s explicit `trim` and both Qwen3.5 models' `auto`:

  ```yaml
  cache:
    strategy: auto
    equivalence_verified: null
  ```

  Add `cache_equivalence_verified: dict[str, str] | None` to `ModelSpec`. `_default_spec` supplies `None`. `_model_spec_from_mapping` accepts `null` or validates exactly `{date, sha256}` before copying the mapping into the frozen record. Missing, extra, empty, non-string, or non-SHA-256 evidence must raise a `ValueError` naming `cache.equivalence_verified`.

- [ ] **Step 4: Implement auditable cache resolution without changing ArchitectureView**

  Keep structural discovery in `ArchitectureView` and resolve only declared policy in `ModelSpec.resolve`:

  ```python
  if self.cache_strategy == "auto":
      if view.cache_trimmable:
          cache_strategy = "trim"
          cache_strategy_reason = "auto:trimmable"
      elif self.cache_equivalence_verified is not None:
          cache_strategy = "snapshot"
          cache_strategy_reason = "auto:equivalence_verified"
      else:
          cache_strategy = "none"
          cache_strategy_reason = "auto:equivalence_unverified"
  else:
      cache_strategy = self.cache_strategy
      cache_strategy_reason = f"explicit:{self.cache_strategy}"
  ```

  Add `cache_strategy_reason` to frozen `ResolvedSpec` and its `as_dict()` payload. Do not add model-family branching or alter `vocab_size`, `tie_word_embeddings`, LoRA resolution, layer types, probe layers, snapshot revision, or JVP method.

- [ ] **Step 5: Run the model-resolution tests and verify GREEN**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'model_spec or registered_models or cache_equivalence or cache_resolution'
  ```

  Expected: all selected tests pass, including the legacy raw 3B id retaining explicit trim.

- [ ] **Step 6: Write failing fake cache and factory tests**

  Rename the existing cache tests from `TurnCache` to `TrimCache` and keep every prefix/offset assertion. Add a fake `ArraysCache` whose `state` is a nested list of mutable fake arrays and a fake `KVCache` whose `state` is a tuple. The snapshot test must prove:

  - the immutable prefix is prefilled once and its saved state is a deep copy;
  - the first turn counts the prefix and suffix as encoded, with zero reuse;
  - later turns restore the saved state, count exactly `prefix_tokens` as reused, and return only the mutable suffix;
  - mutating live cache state after generation cannot mutate the saved snapshot;
  - a changed immutable prefix raises instead of silently reusing stale state.

  Add one factory test over the same dense/hybrid cases used in Step 1:

  ```python
  trim = make_turn_cache(model, dense_view, trim_resolved, prefix_tokens=4)
  snapshot = make_turn_cache(model, hybrid_view, verified_resolved, prefix_tokens=4)
  disabled = make_turn_cache(model, hybrid_view, unverified_resolved, prefix_tokens=4)
  assert isinstance(trim, TrimCache)
  assert isinstance(snapshot, SnapshotCache)
  assert disabled is None
  assert capsys.readouterr().err.splitlines() == [
      "WARNING: qwen35-4b cache auto-resolution disabled reuse: equivalence is unverified"
  ]
  ```

  Also assert explicit `none` and dense `none` return `None` without the hybrid auto warning.

- [ ] **Step 7: Run the fake cache tests and verify RED**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'turn_cache or trim_cache or snapshot_cache or make_turn_cache'
  ```

  Expected: failures because `TrimCache`, `SnapshotCache`, and `make_turn_cache` do not exist.

- [ ] **Step 8: Implement the cache protocol, trim rename, and immutable-prefix snapshot**

  Add the binding protocol and factory signatures:

  ```python
  class TurnCacheBase(Protocol):
      cache: Any
      reused_tokens: int
      encoded_tokens: int

      def prepare(self, token_ids: list[int]) -> list[int]: ...
      def commit(self, token_ids: list[int], generated: list[int]) -> None: ...


  def make_turn_cache(
      model: Any,
      view: ArchitectureView,
      resolved: ResolvedSpec,
      *,
      prefix_tokens: int,
  ) -> TurnCacheBase | None: ...
  ```

  Rename today's `TurnCache` implementation to `TrimCache` without changing its longest-common-prefix logic. `SnapshotCache.prepare` creates `view.make_cache()`, prefills `token_ids[:prefix_tokens]` through the fake-or-real model cache interface, evaluates the cache state, and stores a recursive deep copy. On later calls it restores a fresh deep copy of each saved `entry.state` and returns `token_ids[prefix_tokens:]`. Validate `0 < prefix_tokens < len(token_ids)` and exact prefix identity before reuse.

  `make_turn_cache` returns `TrimCache` for `trim`, `SnapshotCache` for `snapshot`, and `None` for `none`. For only the `auto:equivalence_unverified` hybrid branch, write the exact tested warning once to `stderr` during factory construction; one task constructs the factory once, so it cannot repeat per turn.

- [ ] **Step 9: Wire the factory at Task 5's temporary compatibility boundary**

  After Task 4 has landed, compute the immutable prefix exactly once from the initial system and task messages:

  ```python
  prefix_prompt = build_prompt(
      tokenizer,
      messages[:2],
      spec=spec,
      keep_last=keep_last,
      generation=False,
  )
  prefix_tokens = len(tokenizer.encode(prefix_prompt))
  turn_cache = (
      make_turn_cache(model, view, resolved, prefix_tokens=prefix_tokens)
      if use_cache and view is not None and resolved is not None
      else TrimCache(model) if use_cache else None
  )
  ```

  Keep the approved Task 5 compatibility seam isolated: legacy callers that have not yet been migrated by SPEC-001 Task 6 retain the 3B trim path; any caller supplying `view` and `resolved` must use the safe factory. Do not modify Task 4 parsing/thinking behavior, transcript format, cache-equivalence CLI, or downstream callers in this issue.

- [ ] **Step 10: Run focused and neighboring fake-only tests**

  Run:

  ```bash
  uv run pytest -q tests/test_pipeline.py -k 'model_spec or cache or trajectory or prompt'
  uv run pytest -q tests/test_pipeline.py
  uv run ruff check src/local_llm_lab/models.py src/local_llm_lab/pipeline/runner.py tests/test_pipeline.py
  ```

  Expected: every command exits 0. Record exact counts and any skipped artifact tests. No command may load a model or execute a real pipeline/probe.

- [ ] **Step 11: Verify R2 and cross-task non-regression evidence**

  Run a scoped constant scan and classify every hit:

  ```bash
  grep -RIn '<|im_end|>' configs/models src/local_llm_lab tests
  grep -RInE 'model\.model\.layers|\b36\b|\b2048\b|\b35\b' src/local_llm_lab/pipeline src/local_llm_lab/probes
  ```

  Allowed end-of-turn hits are registry YAML, `_default_spec`, `arch.py`, tests, and classified legacy modules only. Do not repair an out-of-scope hit silently; record it as a closure blocker for the root. Confirm `tasks.GENERATOR_VERSION` remains 2 and the version-2 reference hash test remains green. Inspect `provenance.py` and `cli.py` to confirm this task did not duplicate or weaken issue #2 F1–F4.

- [ ] **Step 12: Commit only the cache-ruling implementation**

  Inspect `git status --short`, the index, and the staged patch. Stage only the exact reviewed implementation paths and commit with explicit pathspecs:

  ```bash
  git add -- configs/models/qwen25-coder-3b.yaml configs/models/qwen35-4b.yaml configs/models/qwen35-9b.yaml src/local_llm_lab/models.py src/local_llm_lab/pipeline/runner.py tests/test_pipeline.py
  git diff --cached --check
  git commit -m "fix: enforce cache auto-resolution rulings" -- configs/models/qwen25-coder-3b.yaml configs/models/qwen35-4b.yaml configs/models/qwen35-9b.yaml src/local_llm_lab/models.py src/local_llm_lab/pipeline/runner.py tests/test_pipeline.py
  ```

  Do not include planning artifacts, foreign staged work, pending design documents, data, outputs, reports, provenance, CLI, or process-note implementation files in this commit.

---

## Independent Review and Issue Closure

- [ ] Dispatch exactly one independent principal reviewer after the implementer has committed. The reviewer reads the binding pending documents, issue #1, this plan, the implementation commit diff, and the SDD progress evidence before running fresh fake-only tests.
- [ ] The reviewer checks spec compliance first: all three R1 branches, strict R4 evidence validation, serialized reason, exactly one unverified-hybrid warning, real snapshot deep-copy/restore behavior, R2 constant locations, no ArchitectureView/generator/provenance/CLI drift, and no model-loading command.
- [ ] The reviewer checks code quality second. Any finding returns to the same implementer for a bounded fix round, followed by re-review from the same reviewer; do not spawn a second implementer.
- [ ] Verify the SPEC-004 §1 owner has a reviewed, committed, pushed implementation/report pair for the offline re-analysis slice. Treat commits `f90578b` and `75f55f2` as candidate code evidence, not automatic checkbox closure; require its final owner/reviewer report and remote commit visibility.
- [ ] Verify issue #2 F4 supplies the reviewed, committed, pushed `design_specifications/under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md` and SDD ledger evidence for accepted commits `fc38a9d` and `6dff90c`. Do not duplicate that report.
- [ ] Append a compact evidence table to `.superpowers/sdd/2026-09-03-spec-001-task-5-cache-rulings/progress.md` mapping issue #1 R1–R4 and both process notes to commits, tests/reports, reviewer verdicts, and remote URLs.
- [ ] Run final fresh verification: focused cache tests, full `tests/test_pipeline.py`, Ruff on touched Python files, the R2 scan, `git diff <review-base>..HEAD --check` scoped to this task, and `git status --short` classification. Record exact results.
- [ ] Push every reviewed commit with a normal non-force push only after confirming the shared branch has no immediate Git/index collision. Verify the remote contains the implementation and process-note evidence commits.
- [ ] Update issue #1 with links to the evidence table, reviewed commits, reports, and test results; check a box only when its evidence is traceable. Close the issue only after all six checkboxes are satisfied. If external GitHub mutation remains blocked by the host, leave the issue open and hand the exact ready-to-post payload to the root Coordinator rather than bypassing the control.

## Self-Review

- Spec coverage: R1 is covered by resolution, factory, snapshot, warning, and three fake branches; R2 by the constant scan; R3 by the no-pending-edits constraint and review; R4 by checked-in schema, strict parsing, dataclass metadata, and serialization. Both process notes are verification gates owned elsewhere.
- Placeholder scan: the implementation steps contain exact paths, APIs, expected failures, commands, reason strings, warning text, and closure evidence; no deferred code placeholder is part of the executable task.
- Type consistency: `ModelSpec.cache_equivalence_verified` is `dict[str, str] | None`; `ResolvedSpec.cache_strategy_reason` is `str`; `make_turn_cache` consumes the same `ResolvedSpec` and `ArchitectureView` produced by Tasks 1–2; runner cache objects all satisfy `TurnCacheBase`.
