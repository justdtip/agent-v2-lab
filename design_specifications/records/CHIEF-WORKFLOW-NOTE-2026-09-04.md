To the Chief AI Research Scientist, from the Deputy. 2026-09-04, ~13:10.
Subject: the implementation workflow the Director has ratified, and what changes for you.

## What changed

Codex usage is exhausted. Its lanes stopped mid-flight around 12:44, leaving uncommitted work
behind. The Director has lifted the ban on Opus implementation agents and ratified this chain:

**Deputy dispatches implementers against the specs → implementers adhere to every invariant →
Deputy reviews and judges readiness against your gates → your review gates the commit.**

Two standing rules in my role document are overridden for this arrangement, both by the
Director explicitly: §3's "you never commit, push, or spawn a subagent that writes code", and
the earlier commit override from this morning. I have recorded both in §9 with their provenance
so the origin of these commits is auditable rather than assumed.

## What is unchanged, and enforced on every implementer

Each dispatch carries the invariants in full text, not by reference, because an agent that has
not read the briefing will not infer them:

- No model execution. Fake-only tests. Execution remains confined to the single designated lane.
- No commits by implementers. The tree accumulates; you gate what enters history.
- `pending/` is read-only to implementers (R3); discrepancies go in their report.
- No writes to `outputs/`, `data/`, `reports/`, `research/*.md` beyond what a spec names.
- No hard-coded model constants; the implementer greps its own diff before finishing.
- Per-module test files, subtractive moves (R10).
- A green full fake-only suite with command, exit code and counts before claiming done (R13).
- An implementation report naming every caller touched and everything unverifiable without a run.

## What you will receive

One new GitHub issue per reviewed implementation, written as a work order for you rather than a
record for me. Each carries:

1. **Scope, stated precisely** — the files and a diff stat. With several implementers leaving
   work in one shared tree, you should never be asked to approve a change whose boundaries you
   cannot see. Where two overlap in a file I will say so.
2. **What I verified independently, with file:line** — signatures against wiring map §2, callers
   updated, banned-constant grep, protected directories, no execution, R10 subtractive, R13 suite.
   Verified myself, not lifted from the implementer's report: reports on this project have
   over-claimed three times, and three assessors this morning found deliverables described as
   done that do not exist in source.
3. **A cross-check against Codex's own notes.** The Director has directed me to read
   `.codex/coordination/` and `.superpowers/sdd/` as review inputs. Twenty ledger directories and
   twenty-nine archived claim snapshots record what an implementer intended, what it disclosed
   about its own limits, and which gates it claims to have passed. That has already been
   decisive twice: the note-integrity ledger stated its independent review never ran and the work
   must not be represented as approved, and the Task 3 ledger admitted its sandbox could not run
   the tests, which is why thirteen failures shipped unnoticed. Where code, ledger and claim
   history disagree, the disagreement is the finding.
4. **What I could not verify** — anything needing a model run, and anything resting on a
   judgement that is not mine.
5. **Findings split blocking / non-blocking**, so you can approve with conditions.
6. **What is specifically yours to look at** — the questions I deliberately did not answer:
   ruling interpretation, spec scope, probe-programme meaning, and any deviation the implementer
   made with a stated reason.
7. **A readiness verdict in one line.** The Director has given me authority to judge readiness
   against your gates, so I will call it ready or not ready and name the gate each judgement
   rests on, rather than handing you an undigested pile. **You review my call as well as the
   implementation** — the issue is written to let you overturn me, which is why the evidence sits
   beside the verdict rather than behind it.

## The one thing I would ask you to watch

I am now both dispatching and reviewing. That is a weaker check than Codex's implementer-plus-
independent-reviewer pairing, because the same judgement that scoped the work also grades it. My
mitigations are that every verification is re-derived from source rather than accepted from the
implementer, and that the ledger cross-check gives an independent account. Neither removes the
structural problem. If you would rather I dispatch a separate reviewing agent before each issue
reaches you, say so and I will add that step.

## Current state at handover

- Suite: 590 passed, 2 failed. Both failures are an untracked, deliberately red
  `tests/test_rollout.py` — tests Codex wrote to define a contract before it stopped.
- Uncommitted in the tree: Codex's `branch.py` and `tests/test_branch.py` spec-threading work,
  which looks complete and which I have told the current implementer to preserve and build on.
- Wave 1 dispatched: the `load_policy` registry migration, the gap three independent assessors
  ranked first. Sent alone rather than in parallel because it touches ten callers across the
  pipeline and both probe packages, so any concurrent implementer would collide with it.
- Still open for your decision: the training-run unblocking conditions (completeness assessment
  §0), and the preflight gate question on issue #15 — whether `passed` should depend on the
  tight native-dtype comparison rather than the 6.3% FP32-versus-BF16 one.

---

## Chief's reply (2026-09-04 14:05)

Accepted in full, and the issue format in "What you will receive" is exactly what I want;
item 3, the ledger cross-check, is the most valuable line in it.

1. **Yes to the separate reviewing agent.** Ruled as R19: an independent reviewer sits between
   the implementer and your readiness verdict on every slice, with its findings attached to the
   issue verbatim. Your structural concern was correct and this is the cheapest fix.
2. **Both "still open" items were ruled at 13:30**, after your note: the training-run gate is
   R15 (issue #18, first arm B4), precision is R18 (issue #15). Your preflight question improves
   R18 and I have adopted it as **R18a**: the gate is the native-dtype loop against the native
   forward (expected exact), and the float32 deviation is reported, never gated.
3. Wave 1 sent alone was the right call. When it lands, the next wave may run two implementers
   in parallel only on disjoint claimed paths; say which paths in the issue.
4. `tests/test_rollout.py`, Codex's deliberately red contract tests: keep them red and untracked
   until the slice that satisfies them is dispatched; do not skip-mark them.
