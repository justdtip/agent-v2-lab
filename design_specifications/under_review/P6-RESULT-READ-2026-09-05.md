# P6 result — the Deputy's read of the third attempt (for the Chief)

**Artifact:** `outputs/probes/patch-C-2026-09-04/patch.json` (SHA-256
`34542df54136b66b4a2bff4d0396a55275626556ac2042d98fdc759a40146aa4`) and `patch.md`
(`b740e17dd62e5f9dce1d7915bb8d17185e1d5d20eef99696b9dee605ad793fc6`); schema `p6-patch-r25`;
policy C on the 3B; five scoring-version-stable cases, none excluded; layers 6/12/18/24/30/36
(registry default); seven cells; alignment table identical to the pre-run measurement. Ran
under the pre-R26 code, so there is no `run.log` for this attempt.

**What a flip is (read from the code):** `_score_patch` (`patch.py:1163-1190`) generates the
decision-step turn with the injection active, substitutes its thought into the failing steps,
and `_is_flip` (`:741-746`) calls `check_trajectory`; the case flips when no `value_drop`
violation remains at the decision step, i.e. when the generated note now carries every
required fact (`integrity.py:238-254`). A parse failure is not a flip.

## 1. The numbers

Treatment (counterfactual residuals from run B's context into run C's):

| cell | L6 | L12 | L18 | L24 | L30 | L36 |
| --- | --- | --- | --- | --- | --- | --- |
| previous_notes | 1.0 | 1.0 | 0.8 | 0.2 | 0.2 | 0.0 |
| dropped_value_slot | 0.8 | 0.8 | 0.2 | 0.0 | 0.0 | 0.0 |
| shared_value_tokens, system, task, last_two_observations, final_token | 0.0 throughout |

Controls: `unrelated_task` (another case's aligned rows, same cell) is **as high or higher** on
the note-region cells — previous_notes 0.8/1.0/0.8/1.0/1.0/0.0, shared_value_tokens
1.0/1.0/1.0/0.8/0.8/0.0, dropped_value_slot 0.6/0.8/0.0/0.0/0.0/0.0 — and 0.8 on
last_two_observations at L6–30. `random_positions` (the same rows at random non-treatment
positions) is low on the note cells (previous_notes 0/0.6/0/0/0.2/0; the two value cells 0
throughout) and moderate elsewhere (system 0.4 at L6/18/24; observations 0.8/0.6/0.6/0.2/0.2).
n = 5, so a rate of 0.8 carries a Wilson interval of about [0.38, 0.96].

## 2. What the design can and cannot say

**It localises the decision.** Nothing injected at the system prompt, the task, or the final
token moves the outcome; injections at the note positions in the early layers (6–12) move it
almost every time; by layer 24 the effect is gone. The drop is decided early and at the note
positions.

**It does not isolate the dropped value's representation — corrected after re-measuring with
the checker's own rules.** A required fact that is visible in the retained observations is
never a drop (`integrity.py:230-236`, `_fact_is_visible`), so by construction none of the five
dropped values sits in the two retained observations. Where the value's text does sit in the
failing prompt, measured with `_fact_is_referenced`: in an EARLIER note in four of five cases
(0031 → 85, 0127 → 89, 0139 → 100 at note 5 of 7; 0175 → 54 at note 4 of 6) — the value was
carried and then dropped, so the model can re-copy it from its own earlier note; in 0163 the
value 32 appears in no note, no retained observation, and neither the task prompt nor the
system prompt (all measured with `_fact_is_referenced`): its only text source is a hidden,
stubbed observation, so a flip on 0163 would be genuine retrieval from the injected rows. A flip
therefore does not require the injected content in at least four cases, and the controls show
exactly that: foreign note rows (`unrelated_task`) flip as often as run B's, and foreign rows
over the observation positions flip 0.8 at L6. Because the artifact is aggregate-only, which
cases flipped under which condition is not recorded, so the one case where retrieval would be
genuine (0163) cannot be separated. The `dropped_value_slot` cell is the closest to the
write-side question and it says: writing *any* value-like row at the separator after the last
shared value reopens the list (treatment 0.8, unrelated 0.6–0.8), while the same row at a
random position does nothing (0.0) — **position-specific, content-nonspecific**.

**Reading:** run C's failure at these steps is a premature closure of the note's value list at
the note-writing positions, which any coherent perturbation of that region undoes; the
evidence does not show that the value's identity is carried by the injected residual. That
bears on the memo's regimen-versus-parameters question, but it does not settle it in the
direction the write-side hypothesis anticipated, and five cases cannot separate 0.8 from 0.6.

## 3. Recommendations (the Chief's decision)

1. **Strict flip, recorded per generation.** Score a second outcome: the generated note
   parses canonically AND carries the specific dropped value AND the case's dropped value is
   not visible in the retained observations. Record the generated note text (or its parsed
   values) per generation in the artifact, which today is aggregate-only by design, so any
   future re-scoring is offline. This is a small code slice on `patch.py` and a rerun
   (~1.5 h on the lane).
2. **A content control.** Inject run B's note rows with the dropped value's own positions
   replaced by an unrelated value's rows. If the flip rate is unchanged, content is irrelevant
   and the finding is "position gate"; if it collapses, the value's representation matters.
3. **More cases.** The bound secondary condition (`aggregate_report` failures with the
   generator-v4 note, labelled designed-correct) adds cases and a second family; run it after
   B4 training on the same lane.
4. **Do not draw the memo's verdict from this run alone.** Record it as: decision localised to
   early note positions; specificity not established.

## 4. Bookkeeping

Probes checklist B3 ticked as "artifact on disk"; the interpretation above is not a tick.
The lane is free: B4 training (condition 8 met) waits only on the Director's condition 6.
