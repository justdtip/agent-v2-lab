# Implementation completeness across the programme

Deputy Chief of AI Research, 2026-09-04 12:20. Five independent read-only assessors, one per
specification plus one on the shared contract. Every assessor was instructed to ignore
implementation reports, ledgers, coordination lanes and plan documents, and to score only what
it could find in source. DRAFT: the Chief ratifies.

## 0. Decision requested of the Chief: define the unblocking conditions for a training run

This is the one thing the programme cannot resolve for itself, and it now gates the next real
experiment. Below is a candidate list, offered so the Chief can accept, amend or reject specific
conditions rather than rule in the abstract. Each is checkable rather than a judgement call.

| # | Candidate condition | Status today |
| --- | --- | --- |
| 1 | Preflight passes, **or** the gate is re-scoped to the checks training actually depends on | `passed: false` on the residual-equivalence check; diagnosis in flight (issue #15). Training does **not** use `ArchitectureView`, so this failure may be blocking training for a reason that does not apply to it |
| 2 | Training rows produced by the in-repo renderer, not the library's dataset path | **Met since ~11:00.** `stage_train` now calls `mlx_lm.lora.train_model` in-process and consumes `load_rendered_splits` (`cli.py:31, 287`), under the new ruling R14 |
| 3 | A named, hash-pinned dataset exists on disk | **Not met.** `data/agent_v2d` does not exist; the recipe has never been run, and the pinned-hash oracle covers only the run C config, not Run D |
| 4 | Which arm runs first | Chief's call. The memo's controlled comparison is the same data on a new base |
| 5 | Success criteria and the selection screen fixed before the run | Screen exists and is tested (SPEC-002 §1 at 100%); the criteria in SPEC-003 §5 have no automated gate, by design |
| 6 | The single execution claim free, and an accepted cost estimate | Claim currently held by the preflight diagnosis. Estimate: ~100 min for 400 iterations at the measured 1.4x; memory verified at 3.85 GiB against a 22 GiB budget |
| 7 | Model resolution threaded through evaluation | **Not met** — see §3 below. Without it a trained Qwen3.5 adapter cannot be evaluated with its own cache strategy or rendering |

Condition 1 is the one worth deciding explicitly rather than waiting out: if training does not
depend on view equivalence, saying so unblocks the run without weakening the probe hold.

## 1. Headline numbers

| Specification | Complete | Character of the remainder |
| --- | --- | --- |
| SPEC-002 evaluation, selection, note integrity | **96%** | One gap, and it is item 7 above |
| SPEC-001 model-agnostic backbone | **78%, now higher** | §7 landed mid-assessment; the assessor's figure predates it |
| SPEC-003 run D and cross-model matrix | **50%** | Recipe complete; data never generated; matrix is runs |
| SPEC-004 probe programme revision | **50%** | Two sections done well, two not started |
| Shared contract (wiring map §2) | **87% of signatures match** | 8 of 9 integration checks genuinely tested |

**Programme estimate: roughly 75% of the code, and near zero of the experiments.** That gap is
the real headline. The engineering is in good shape; nothing has been measured with it yet.

Separating the two matters, because the raw section percentages understate code readiness. Of
SPEC-004's remainder, block ablation and P6 patching are code-complete and merely await
execution time; of SPEC-003's, the entire training matrix is configuration that exists and runs
that have not happened. Counting only work that still needs writing, the programme is closer to
**85% code-complete**.

## 2. What is genuinely finished

- **The note-integrity vertical.** All six violation kinds, ground truth from generator replay,
  and a retroactive report that an assessor verified by reading the checked-in run B and run C
  artifacts and confirming it reproduces the memo's counts exactly.
- **The architecture view and the registry.** Every member matches the contract, including the
  R3 additions, across dense, hybrid and split-hybrid fakes.
- **The Run D recipe.** Cross-validated against run B's *actual historical training data*, not
  merely against the spec's description — the templates really do match B.
- **The selection screen**, with the reversed tie-break, Wilson intervals and an exact-binomial
  McNemar, all tested against hand-computed values.
- **Cache strategy and preflight**, both implementing R1 and R9 correctly, including the
  finite-difference fallback that the first real run proved necessary.

## 3. The single highest-leverage gap

Three assessors, working independently on three different specifications, each ranked the same
thing as their top blocker: **`evaluate.load_policy` was never migrated to the registry
signature.** It still takes a model-name string and returns a two-tuple where the contract
requires a `ModelSpec` and a four-tuple carrying the architecture view and resolved spec. Every
one of its ten call sites uses the old form and then hand-rolls its own view afterwards.

Consequences, in the assessors' own findings: evaluation always runs in compatibility mode with
the generation-suffix assertion disabled; evaluation output never records a resolved `ModelSpec`,
which is SPEC-002 §6's one shortfall; and the cross-model matrix cannot evaluate a non-3B
adapter with that model's own settings, which is SPEC-003 §4.

Adjacent and best done in the same claim: `probes/policies.py::resolve_policy` is still a
hardcoded three-adapter table that never reads `ModelSpec.policies`, so asking for policy "B" on
Qwen3.5 silently returns a 3B adapter path; and `branch.py:53,124` still call `build_prompt`
without a spec, which wiring gap 1 names explicitly. A lane is in flight on the first two as of
12:16.

## 4. Claimable now, needing no model

1. **SPEC-004 §2, the P2 redesign — 0%, and the spec calls it where the remaining probe budget
   goes.** Observation stubbing, the difficulty-split plan, the `compare` subcommand, dual-position
   capture, the disjointness test. None exist.
2. **SPEC-004 §4's four pre-conditions**, all pure code: the closure note, a matched-design flaw
   where the default persona still receives more prompts than the roles, an unbalanced exemplar
   set, and a judge that still defaults to on for same-policy scoring. These were pre-registered
   as prerequisites before P1 could ever be rerun, and none were applied.
3. **Run D data generation.** Needs no model under the Director's definition, and nothing else in
   SPEC-003 can proceed without it.
4. **The layer-fraction work**: `--layers` parses integers only and defaults to a literal 3B layer
   list ending at 35, which is out of range on the 32-layer new base, while
   `spec.probes.layer_fractions` sits in every registry file unread.

## 5. Two quality notes worth the Chief's attention

**The banned-constant scanner has a blind spot.** It is an AST scan for exact constant values, so
it does not see `"35"` inside the longer string `"6,12,18,24,30,35"`. The project's own guard
therefore passes over a live instance of exactly the pattern it exists to catch. A substring check
alongside the AST scan closes it.

**Reports over-claim relative to code.** Three assessors independently flagged deliverables that
reports described as done and that do not exist in source. Ruling R13 fixed the suite-gate half of
this; the report-accuracy half is unaddressed. Worth requiring that each report's claims name the
file and line that satisfies them.

## 6. Confidence and caveats

The repository was under live concurrent edit throughout. Two assessors watched files change
under them, and SPEC-001's headline figure is already stale in the direction of being too low,
because its largest gap closed mid-assessment. Section percentages are in fixed 25-point steps by
instruction, so they are comparable across specs but not precise within one. Two tests are failing
at this moment, both inside one lane's dirty working files, mid-work rather than regressions.
