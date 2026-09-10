# Steering pilot — experiment 1 in its minimal form: does the residual at the action position carry the target?

**Chief, 2026-09-10 UTC, on the Director's "Pilot: go for it". Status: designed and smoke-tested; the
run is scheduled for the card's phase-0 window beside the 4B W-3b pass; nothing read.** The form is
Codex's proposal §6 (the private atlas proposal of this date, evaluated by the Chief) reduced to what
the existing corpus and the existing scripts can do this rental: whole-residual donor replacement at
the action position, complete calls generated and parsed, on matched decisions the corpus already
contains. Written before the run.

## Design

The base instruction-tuned Gemma 3 4B (the checkpoint the float32 lenses and captures are on), float32,
eager attention, determinism pinned, greedy decoding. A recipient is an aggregate_report decision
(clean variant, step 2, 3 or 4) whose expert reads `metric-(step−1).txt` in the episode's own
directory. Its sequence is the rendered prompt plus the expert's note and the JSON prefix through
`P_act`, the token before the tool name; the expert's note is teacher-forced. At each layer L of the
sweep {4, 8, 12, 16, 20, 24, 28, 32}, the residual at block L's output at `P_act` is replaced by a
donor's residual at the donor's own `P_act` (a forward hook on the prefill; decode steps untouched),
and up to 48 tokens are generated greedily and parsed as the tool call.

Arms per recipient and layer, with the prediction under "the residual at `P_act` carries the target":

| arm | donor | prediction if the target is carried there | what else it tests |
|---|---|---|---|
| baseline | none | — | what the base model itself emits at this prefix (it is not the expert) |
| same_state | the recipient's own residual | token-for-token identical to the baseline | the patch mechanism is a no-op when it should be |
| opposite_target | another episode, same step, same metric index, different directory | the path's directory switches to the donor's; the tool stays | directory identity |
| pending_file | the same episode's next step (the next metric) | the file switches, the directory stays; the tool stays | "which file is pending", the Director's example |
| operation | the same episode's later `calculate` step | the tool switches to calculate | operation, the other half of the dissociation |
| unrelated_family | a `read_file` decision from search, read or pointer_chain | the rival: if the whole plan moves, the call becomes the foreign donor's | discriminates "target" from "the entire next call" only weakly; a whole-residual swap moves everything at that position |

Recipients: 24, balanced over steps 2–4 by a seeded round-robin over the 104 eligible decisions
(111 candidates, 104 with a valid next step; no same-target variant donor exists in this corpus, each
episode having one variant). Donors chosen by seed among the eligible. Per row the generated text, the
parsed tool and path, JSON validity, the path classified against the recipient's expert path, the
donor's expert path and the recipient's own baseline path, the directory-switched and
file-switched-same-directory flags, and the first generated token's logit for each of the six tools.

## Claim ceiling, stated before the run

A whole-residual swap is a localisation test. A donor-consistent switch at layer L is a
counterfactual effect on the generated call at that layer and position; it does not identify a
target component (everything at that position moved), it is not task success (nothing is executed),
and it says nothing about the tuned agent (the base model is patched). A switch under the unrelated
donor as well as the targeted ones reads as "the next call is carried whole", not as a target
representation. The baseline decides what "switch" is measured from: where the base model's own call
already differs from the expert's, the comparison is against the model's baseline, and the expert's
path is reported beside it. The same-state arm failing to reproduce the baseline anywhere voids that
layer's readings. Unit: the recipient decision (one per episode by construction); 24 is a pilot, not
an estimate with a bound.

## Scripts

`scripts/steer_pilot.py` (948c91086e45), `scripts/chief_pilot_gated.sh` (5192a4b80905; starts on the W-3b
pass's start line, seat chief-pilot, expandable segments, from `/workspace/chief`), `scripts/pilot_test.sh`
(the one-recipient CPU smoke test). Output `captures/pilot-4b/`:
`run.json` (identity, pairs, seed), `pilot.jsonl` per recipient, `manifest.json` at the end.

## Smoke test (CPU, one recipient, layers 8 and 20, 20 new tokens; a mechanism check, not a reading)

Every patch applied exactly once per generation; the same-state arm reproduced the baseline token for
token at both layers; the parser and classifier ran. The 20-token budget truncated every path before its
file name, so no path was classified (the run uses 48 tokens, enough for the longest call). One
observation, on one row: at layer 20 the operation donor produced a well-formed `calculate` call
(`{"expression": "10 + 10"}`) where every other arm kept `read_file`; at layer 8 nothing switched. The
base model's own baseline began the expert's directory (`lab/train1/0227`).

## Results

Not yet. The run is armed to start beside the 4B W-3b pass.
