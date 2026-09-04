# Programme progress and specification fidelity, DRAFT (2026-09-03, HEAD 60dfd51)

**DRAFT — not ratified. Deputy Chief of AI Research. Nothing here is approved.**

Method: evidence only. Git history, the files present in `src/` and `tests/`, the review files
and implementation reports in `under_review/`, Codex's ledger and coordination lanes read but
not trusted as evidence of code. No model was run. `uv run pytest`: 242 passed at `60dfd51`.

Coverage: complete for SPEC-001 §1 and §2, the SPEC-002 correction commit, the integration-check
list, and the briefing's anchor table. Two passes were lost to repeated API 529 failures and are
still outstanding: the full cross-cutting rule audit and the complete signature-drift table
against wiring map §2. Both were relaunched; §5 below states what is not yet covered.

## 1. Progress by section, from evidence

| Section | Deliverable | Status | Evidence |
| --- | --- | --- | --- |
| SPEC-001 §1 | model registry, `ModelSpec`, `configs/models/` | landed | `models.py:26-138`; three config files; 7 tests |
| SPEC-001 §2 | architecture view **and** the `capture.py` / `jlens.py` rewrites onto it | **partial** | view landed `arch.py:45-272`, 11 tests; consumers untouched since `bda5ff5` |
| SPEC-001 §3 | one prompt renderer, `generation_suffix`, `render_completion`, `strip_thinking` | not started | all three absent from `src` |
| SPEC-001 §4 | thinking as a mode | not started | no thinking fields anywhere |
| SPEC-001 §5 | cache strategies, `SnapshotCache` | not started | `SnapshotCache` absent |
| SPEC-001 §6 | LoRA target discovery wired into training | not started | `arch.lora_targets` exists as a view method only |
| SPEC-001 §7 | training entry on the resolved spec | not started | no evidence |
| SPEC-001 §8 | probes on the view, per-model policies table | not started | no `--model` registry plumbing in the three probe CLIs |
| SPEC-001 §9 | provenance | partial | `provenance.py` gained generator-version recording in `60dfd51`; the §9 preservation recommendation is not implemented |
| SPEC-001 §10 | preflight stage | not started | absent |
| SPEC-002 §3, §5 | verdict hardening, known bugs | landed, approved | `fc38a9d`, `6dff90c`, review round 1 |
| SPEC-002 F1-F4 | review corrections | landed | `60dfd51`, report `under_review/SPEC-002-IMPLEMENTATION-REPORT-part1.md` |
| SPEC-002 §1 | difficulty parameter, family-balanced screen, `wilson`, `mcnemar` | not started | `task_from_id`, `wilson`, `mcnemar` all absent |
| SPEC-002 §2 | `pipeline/integrity.py`, six checks, retroactive B-vs-C report | not started | module does not exist |
| SPEC-002 §4 | runner and evaluator fields | not started | no evidence |
| SPEC-003 | run D data recipe and training matrix | not started | no evidence |
| SPEC-004 §1 | offline re-analysis | landed, reviewed, **sent back** | `f90578b`, `75f55f2`; corrections C1-C6 owed |
| SPEC-004 §2-5 | P2 redesign, ablation, P1, P6 | not started | gated on SPEC-001 |

## 2. Fidelity findings

### 2.1 (blocking for the state of play) SPEC-001 §2 is recorded as complete and is about half done

The view itself is good: it matches the wiring map member for member, carries the three R3
additive fields, dispatches per-kind masks and caches on a hybrid stack (`arch.py:105-138`),
handles tied and untied embeddings through `_TEXT_MODULE_PATHS` (`arch.py:18, 319-325`) with no
hard-coded layout, counts LoRA parameters correctly (`arch.py:224-234`), and honours the
residual indexing convention (`arch.py:236-272`). Eleven tests exercise it on dense, hybrid and
split-hybrid fakes.

But §2 also requires `probes/capture.py` and `pipeline/jlens.py` to run on the view, and it
states that `InjectionHook` must wrap `run_block` rather than `block.__call__` and must read the
absolute position from whichever cache kind is present, tracking the offset in the view for
`ArraysCache`. Neither file has changed since the baseline commit. `grep -rl ArchitectureView
src` returns only `arch.py` and `models.py`, so nothing consumes the view.

Consequence on a real hybrid checkpoint: those paths reach `model.model.layers`
(`capture.py:81, 86, 92`; `jlens.py:151-165`), which does not exist on the Qwen3.5 `Model`, and
where they do run they build one attention mask for every block (`jlens.py:135-146`, imported by
`capture.py:23`), which is the failure the briefing calls out as silently invalidating every
downstream number.

### 2.2 Ruling R1 is violated in live configuration, though the fix is scheduled

`ModelSpec.resolve` turns `auto` into `snapshot` whenever the cache is not trimmable
(`models.py:67-69`), with no equivalence gate and no recorded reason. R1 says that resolution is
not acceptable. Both hybrid configs ship `strategy: auto`
(`configs/models/qwen35-4b.yaml:21`, `qwen35-9b.yaml:21`) and a hybrid cache is never trimmable,
so both new models resolve to an unverified snapshot today. The 3B config sets `trim`
explicitly, so nothing currently running is affected.

Mitigation on the record: this is scheduled as SPEC-001 Task 5
(`docs/superpowers/plans/2026-09-03-spec-001-task-5-cache-rulings.md`), whose lane is `blocked`
behind the issue #2 lane, and the plan sequences it behind Tasks 3 and 4. Nothing in the code
marks the branch as temporary, and a passing test bakes the behaviour in
(`tests/test_probes.py:552`).

### 2.3 R4's field and config key are absent

`ModelSpec` has no `cache_equivalence_verified` and none of the three config files carry
`cache.equivalence_verified` (`configs/models/*.yaml:20-21`). Same deferral as 2.2.

### 2.4 R2's literal survives outside the permitted places

`END_OF_TURN = "<|im_end|>"` at `pipeline/protocol.py:38`, used at `protocol.py:189, 202` and
aliased in `branch.py:17, 30`. `git log -S` dates it to the baseline commit `bda5ff5`, so it is
pre-existing legacy rather than newly introduced, and SPEC-001 §3 is what removes it. Until then
the end-of-turn token is pipeline-wide rather than per model, which is exactly what §1 forbids
downstream.

### 2.5 Three integration checks could be tests today and are not

Wiring map §6 marks ten checks, most with T. Implementable now and missing:

- Check 1, the import-without-a-model test: no `importlib` test exists in `tests/`. It cannot
  fully pass yet either, since it imports `pipeline.integrity`, which SPEC-002 §2 has not built.
- Check 2, the banned-constant grep as a test: absent. This is the check that would have
  surfaced 2.4 mechanically at every hand-off.
- Check 4, the `Trajectory` round-trip: no test constructs `Trajectory(**d)`. The briefing warns
  that every new `Trajectory` field needs a default or old evaluations stop loading, and
  SPEC-002 §4 is about to add fields.

Check 6 was superseded by ruling R5 and is satisfied in the new form: the pinned-hash test at
`tests/test_pipeline.py:1846` compares a fresh generation from the reference config against real
hex constants keyed by generator version 2. It is not tautological. One nit: the `expected`
table holds only version 2, so a bump to 3 raises `KeyError` rather than failing with a message
telling the implementer to pin new hashes.

Commit `6325d3a`, which landed during this review, narrowed that test. It changed the call
from the reference config's `chat_replay` directory to `chat_dir=None` and re-pinned all three
hashes (`tests/test_pipeline.py:1854-1862`). The test is now hermetic and independent of
`data/chat_replay`, which is a protected, irreplaceable directory that may be absent, so the
motive is sound. The cost is that R5 asks for hashes pinned "for the reference config", and
`configs/agent_v2c.yaml:22-23` sets `chat_replay: data/chat_replay`, so what is pinned is now a
chat-free variant rather than the shipped dataset composition. A generator change affecting only
how chat rows merge (`data.py:122-129`) would not be caught. Pinning both, with the
chat-inclusive case skipped when the directory is missing on the existing `state-base.npz`
pattern, would close that without losing hermeticity.

Checks 3, 5 and 8 depend on unstarted work. Check 7 is satisfied only for the view contracts;
capture equivalence, snapshot-cache equivalence and block-mask coverage have nothing to test yet.

### 2.6 Half the briefing's repository-facts anchors no longer point where they claim

The briefing tells every implementer to use those anchors instead of opening files. Verified
still correct: `capture.py:11-13`, `data.py:20-71`, `capture.py:145-285`,
`adapter_delta.py:614-623`, `assistant_axis.py:8-10`, `assistant_axis.py:958`, `tasks.py:53-81`,
and `protocol.py:214-230`.

Drifted, with the current location:

| Briefing fact | Claimed | Actual |
| --- | --- | --- |
| `transcripts.jsonl` append mode | `transcript.py:118` | `transcript.py:153` |
| `checkpoint_dirs` re-copies config | `cli.py:169-170` | `cli.py:174` |
| selection tie-break | `cli.py:213-216` | `cli.py:247-257` |
| saved-artifact regression test | `tests/test_probes.py:1231-1246` | `tests/test_probes.py:1910` |
| `--strip` rewrites assistant only | `state_probe.py:280-286` | `state_probe.py:304` |
| rows come from expert replay | `state_probe.py:1778-1785` | `state_probe.py:47, 161` |
| mixed difficulty-0 rows | `state_probe.py:90` | drifted into `REANALYSIS_TARGETS` |
| test split all-clean | `tasks.py:90-91` | `tasks.py:103` |
| `chat_replay` rows mixed in | `data.py:119-121` | `data.py:122-129` |

The four `state_probe.py` anchors moved because the 874-line reanalysis insertion pushed them.

### 2.7 Two name collisions will bite when SPEC-001 §3 lands

`render_completion` already exists at `branch.py:33` with the signature
`(thought: str, action: Any)`, which is not the map's §3 contract. `_preflight_section` exists at
`state_probe.py:2514` and is a report section, unrelated to the §10 preflight stage. Neither is
an error today; both are traps for an implementer grepping for a name.

### 2.8 Review coverage is uneven

`fc38a9d` and `6dff90c` were reviewed. `f90578b` and `75f55f2` were reviewed and sent back. The
five commits that landed SPEC-001 §1 and §2 have no review file in `under_review/`; issue #1
reviewed the plan, not the code. `60dfd51` has an implementation report but no review. Sections
2.1 to 2.3 above are, in effect, that first review of the SPEC-001 code.

### 2.9 Structural: two test files serialise the programme

`tests/test_probes.py` and `tests/test_pipeline.py` carry every lane's tests, so file-level
claims serialise work that the dependency graph would allow to run in parallel. SPEC-004's
corrections, the remainder of SPEC-001 §2, and Task 3 all queue on the first file; SPEC-002 §1
and §2 and Task 4 queue on the second.

## 3. What the evidence says can start next

Using the map's §1 dependency graph against what has actually landed:

1. SPEC-002 §1 then §2. Needs only SPEC-001 §1 config plumbing, which is landed. §2 is the
   highest-value unstarted work in the programme and needs no GPU.
2. SPEC-004 §1 corrections C1 to C6. No dependencies.
3. SPEC-001 §2's remainder, the `capture.py` and `jlens.py` rewrites. Needs only the view.
4. SPEC-001 §3 then §4, then §5 to §7.

Items 1 and 2 are independent of each other and of item 3 in dependency terms. They are serial
today only because of 2.9.

## 4. Cross-cutting rule audit: clean on all eight checks

Audited across `bda5ff5..6325d3a`.

| Rule | Result |
| --- | --- |
| 1.7 banned constants | clean for this branch. The two projection lists at `adapter_delta.py:56-62` and `cli.py:77-85` blame to the baseline commit and were not touched. The `model.model.layers` hits are live attribute traversal, not string literals. Remaining numeric hits are task counts, a success rate and an ANSI colour code |
| 1.2 protected directories | clean. Only three files under `outputs/` postdate the first implementer commit: a Finder metadata file and the two permitted reanalysis outputs. `data/agent_v2c` untouched; `reports/` untouched |
| 1.3 research documents | clean. No commit touches `research/`. Two files have later mtimes but empty diffs, a filesystem touch only |
| 1.1 no model runs | clean, and better than required. No test loads a checkpoint; the reanalysis tests wrap the loader in a guard that raises if called (`tests/test_probes.py:1905, 2460`). The one artifact-dependent test skips when the file is missing |
| 1.8 determinism | clean. Every `random.Random` is seeded from a config-derived string and every `default_rng` takes an explicit seed. No bare RNG construction |
| 1.5 float32 | clean. The four `float16` mentions in probe paths are comments explaining why it is avoided. No cast exists |
| 1.9 atomic writes | the large writer (reanalysis JSON and Markdown) uses temp file, fsync and `os.replace`. Two new small direct writers noted for judgement: `provenance.py:26` and `transcript.py:44-46`, both metadata-sized. Compliant in spirit; recorded, not raised as violations |
| §6 dependencies | clean. `pyproject.toml` and the lock file are unchanged on this branch |

## 5. Signature conformance against wiring map §2

The full table was walked, subsection by subsection. The result is better than it first looks:
almost every "not implemented" row belongs to a section that has not started, which is expected
progress, not drift. Sorting them apart:

**Landed and conforming.** `pipeline/env.py` §2.8 matches on all three rows (`Verdict`'s new
fields at `env.py:32-45`, the simulator's two new attributes at `env.py:154-155`, the new reason
string at `env.py:268`). `parse_turn` and `turn_is_complete` match. `strip_state_fields` matches.
`provenance.write_provenance` matches its signature exactly at `provenance.py:12`. The
`agent-v2-probe-state` and `agent-v2-probe-axis` CLIs carry all five common flags. The
`reanalyse` subcommand is present.

**Not started, therefore expected absences.** Everything in §2.9 (`pipeline/integrity.py`, seven
signatures, SPEC-002 §2); §2.6 (`tuner_data.py`, SPEC-001 §7); `SnapshotCache`,
`TurnCacheBase`, `make_turn_cache` (SPEC-001 §5); `wilson`, `mcnemar`, `task_from_id`,
`SplitSpec`, `Task.difficulty` (SPEC-002 §1); `generation_suffix`, `strip_thinking`,
`Turn.thinking` (SPEC-001 §3-4); `preflight`, `patch.py`, `stub_observations`, the `compare`
subcommand, `source_tree_hashes`. None of these is a fidelity failure today.

**Two real gaps in landed code:**

- **Provenance is wired into one stage of five.** The map (§2.11) requires every stage to call
  `write_provenance`. Only `stage_data` does, at `cli.py:58`. `stage_train` (`cli.py:133`),
  `stage_select` (`203`), `stage_eval` (`271`) and `stage_rollout` (`319`) do not. The function
  conforms; the wiring is one-fifth done. This is SPEC-001 §9, which §1 above records as partial;
  this is the specific shortfall.
- **The J-lens CLI has no GPU guard.** `probes/guard.py` is imported by `adapter_delta`,
  `state_probe` and `assistant_axis`, and not by `pipeline/jlens.py`, so `agent-v2-jlens` has no
  `--allow-busy-gpu`. Briefing §4.9 says keep the guard and keep the flag. Pre-existing rather
  than newly introduced, and SPEC-001 §8's common-flag requirement would close it.

**Confirmations of §2.1 above, from the signature side.** Every `capture.py` and `jlens.py`
signature still takes `model` rather than `view` (`capture.py:53, 99, 184, 263`;
`jlens.py:151, 169, 196`), `_causal_mask` is still defined and used (`jlens.py:135, 163, 184`),
and `_distribution` and `_encode` were never promoted to public names, so `jspace_sweep.py:62`
and `adapter_delta.py:442` still reach for the private ones. This is the same unstarted work
seen from a different angle.

**Name collision confirmed.** `render_completion` exists at `branch.py:33` with the signature
`(thought, action)`, where the map places it in `protocol.py` with a `spec` keyword. See §2.7.

## 6. Not covered by this review

- Any review of `60dfd51` and `6325d3a` beyond confirming F1 to F4 exist, the pinned-hash test
  is genuine, and the hermetic narrowing in §2.5.

## 7. Work in flight at the time of writing

`src/local_llm_lab/probes/state_probe.py` and `tests/test_probes.py` are dirty in the working
tree: the SPEC-004 remediation lane is executing C1 to C6 now. Anything in §2 about the
reanalysis output describes `6325d3a` and will be superseded when that lands.
