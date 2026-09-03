# SPEC-003: Run D data recipe and the cross-model training matrix

> Read first: `01-IMPLEMENTER-BRIEFING.md` (standing rules, traps, hand-off) and `02-INTERFACE-AND-WIRING-MAP.md` (exact shared signatures, file ownership, implementation order, integration checks). Signatures in the wiring map override any looser wording here.

Status: pending. Author: Claude. Date: 2026-09-03.
Depends on: SPEC-001 (model registry, rendering, LoRA discovery), SPEC-002 (screen, verdict,
integrity). GPU: §1-3 are code and data only; §4-5 are training and evaluation runs and are gated.

## 0. Hypotheses this run tests

H1 (regimen): run B's note templates plus a ban on completion assertions and length mixing lift
`aggregate_report` and `batch_update` off zero on the 3B base and restore `cross_reference`.
H2 (parameters): Qwen3.5-4B trained on run B's unchanged data beats the 3B base on the same
180 tasks, with the gain concentrated in the long-horizon families.
H3 (interaction): the D recipe helps the 4B base less than the 3B base (a stronger base needs
less scaffolding), or not; either answer is informative.

## 1. Note templates (`pipeline/tasks.py`)

1. Revert the three run-C templates to run B's: append-only `values so far:` for
   `aggregate_report`, the full `loads so far:` table for `conditional_update`, and
   `Inspected/Applied k of N … pending: …` for `batch_update`. The evidence: B's format survives
   extrapolation from N=4 to N=6, C's does not, and C's format caused negative transfer into
   untouched families.
2. Keep from C only what the failure analysis did not implicate: naming the queue head in
   `batch_update` apply notes (`Next: worker-0.ini mode=safe -> mode=fast`), because the
   `test_batch_update_apply_phase_names_a_queue_head_then_transitions_to_verify` invariant
   protects it, but express the phase transition solely as `pending: none`.
3. `aggregate_report` subtotal split: state the split point as a number derived from the total
   in every note (`split after 3 of 6`), no `(full)` marker, values listed in full on both sides.
4. Design rule, enforced by a generator invariant test: **no note asserts completion.** The only
   permitted completion expression is `pending: none`, and it may appear only when the
   ground-truth remaining set is empty. The test scans every note of every task at every
   difficulty for the SPEC-002 completion patterns.
5. Progress is always `k of N` with `N` read from the listing or manifest, never from the prompt.
   A test asserts `N` equals the true total in every note.
6. `cross_reference`: every hop note restates the current key and hop number
   (`Hop 2: record gave Next-Key REF-…-1; no Resolution yet.`); a test asserts consecutive notes
   differ and each carries the key used by the next action.
7. Generalise the aggregate-only "reading drops no value" invariant to all five long families:
   every required-carry value (SPEC-002 §2.1) must appear in the note immediately preceding the
   step that needs it.

## 2. Recovery variants

- The failing step in `wrong_path` and `stale_path` states the guessed path in its note, so the
  context no longer contains a note/call mismatch and the recovery cannot copy the correct name
  from the previous note.
- Multipliers: keep run B's (`transient 1, wrong_path 2, unknown_tool 2, stale_path 6,
  failed_edit 6`) so the template change is the only data variable between B and D. Record the
  multipliers in the manifest (already done) and in `provenance.json`.

## 3. Length mixing and splits

Difficulty is an explicit parameter (SPEC-002 §1). Splits for run D:

| Split | Tasks | Difficulty | Perturbed | Use |
| --- | --- | --- | --- | --- |
| `train` | 240 | 0 | yes | as before |
| `train1` | 120 | 1 | yes | length mixing; `train` and `train1` are concatenated |
| `valid` | 24 | 1 | no | screen, family-balanced (SPEC-002) |
| `valid2` | 24 | 2 | no | screen at extrapolation length |
| `test` | 180 | 2 | no | unchanged task ids, so B and C comparisons stay paired |
| `test3` | 60 | 3 | no | new: one step longer than anything screened; reported, never selected on |

Difficulty 3 is defined by the generator as `level = 3` in every `+level` expression (7 invoices,
5 workers, 6 hops); tests assert horizons at each level are strictly increasing.

Chat-replay rows unchanged (240 train rows). Row counts and hashes go into the manifest; the
data stage is deterministic and must regenerate byte-for-byte under a test.

## 4. Training matrix (gated)

| Arm | Base (registry) | Data | LoRA keys | Thinking | Purpose |
| --- | --- | --- | --- | --- | --- |
| B (exists) | `qwen25-coder-3b` | B | attention+mlp | unsupported | reference |
| D3 | `qwen25-coder-3b` | D | attention+mlp | unsupported | H1 |
| B4 | `qwen35-4b` | B | all-linear (auto) | off | H2, paired with B |
| D4 | `qwen35-4b` | D | all-linear | off | H3, paired with D3 |
| D4-think | D4's adapter | none | none | inference (512-token budget) | evaluation-only arm: does intra-turn reasoning fix list handling without retraining |
| D9 (stretch) | `qwen35-9b` | D | all-linear | off | only if `preflight` memory estimate fits at batch 1 |

Recipe held fixed across arms: rank 16, scale 32, dropout 0, AdamW 3e-5, effective batch 4,
400 iterations, max sequence 2688, gradient checkpointing, checkpoints every 100, screen at
every checkpoint with the SPEC-002 selector. For the 4B base the `all-linear` policy adds the
DeltaNet projections; the parameter count is recorded and a secondary arm `B4-attn` with
`attention+mlp` keys is run only if B4 underperforms B, to separate target choice from base.

Cost on the M4 Pro at the 3B model's measured rates: about 70 minutes training plus 55 minutes
for the 180-task evaluation per arm; the 4B model is expected at roughly 1.4× that. The full
matrix is one working day of GPU time.

## 5. Pre-registered success criteria

| Arm | Criterion |
| --- | --- |
| D3 vs B | total ≥ 150/180 (McNemar p < 0.05 on paired ids); `aggregate_report` ≥ 8/15; `batch_update` ≥ 8/15; `cross_reference` ≥ 13/15; no family drops by more than 2 |
| B4 vs B | total higher with p < 0.05, or a family-level gain ≥ 5 in at least one long family with no loss elsewhere |
| D4 vs D3 | reported with intervals; no threshold |
| `test3` | reported per arm as the extrapolation curve (levels 2 and 3), never used for selection |
| Integrity | fraction of failures explained by a note-integrity violation per arm; D arms must show fewer `premature_completion` and `verbatim_copy` events than B by construction of the data |

## 6. Later experiment (not in this run): trained thinking

If D4-think helps, a `trained` mode row synthesises a short reasoning block from generator ground
truth (the remaining set, the current accumulator, the next action's justification) and trains
the model to emit it inside `<think>`. Rendered via SPEC-001 §4. Deferred until D exists.

## 7. Acceptance

- Generator tests in §1 pass at difficulties 0-3.
- `agent-pipeline data --config configs/agent_v2d.yaml` writes the six splits with a manifest
  and `provenance.json`.
- New config files: `configs/agent_v2d.yaml` (3B) and `configs/agent_v2d_qwen35_4b.yaml`,
  `configs/agent_v2b_qwen35_4b.yaml` (B data on the 4B base), differing only in `model:` and
  `output:`.
