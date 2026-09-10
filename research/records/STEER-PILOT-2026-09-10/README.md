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

`scripts/steer_pilot.py` (b3b676a0467c), `scripts/chief_pilot_gated.sh` (5192a4b80905; starts on the W-3b
pass's start line, seat chief-pilot, expandable segments, from `/workspace/chief`; the first attempt's driver,
see below), `scripts/chief_pilot_gated2.sh` (5d8ec3d3b0dd; the re-run's driver, gated on a GO file and on the
card's free memory), `scripts/pilot_test.sh` (the one-recipient CPU smoke test). Output `captures/pilot-4b/`:
`run.json` (identity, pairs, seed), `pilot.jsonl` per recipient, `manifest.json` at the end.

## Smoke test (CPU, one recipient, layers 8 and 20, 20 new tokens; a mechanism check, not a reading)

Every patch applied exactly once per generation; the same-state arm reproduced the baseline token for
token at both layers; the parser and classifier ran. The 20-token budget truncated every path before its
file name, so no path was classified (the run uses 48 tokens, enough for the longest call). One
observation, on one row: at layer 20 the operation donor produced a well-formed `calculate` call
(`{"expression": "10 + 10"}`) where every other arm kept `read_file`; at layer 8 nothing switched. The
base model's own baseline began the expert's directory (`lab/train1/0227`).

## Codex's review of the pilot before its run (59e6b11, local to Codex's worktree), applied

Five findings, all reproduced by Codex with model-free fixtures and all correct. **F1**, a crash: an
operation donor (a `calculate` step) has no path, and where the model kept a complete `read_file` call
the classifier split `None`; the 20-token smoke test truncated every path and never reached the branch,
and the run would have died at its first recipient. **F2**: the same-state invalidation the design
promised was recorded, not applied. **F3**: a "switch" was donor agreement, not a change from the
model's own baseline. **F4**: truncated calls carrying a donor path counted. **F5**: the selector did
not enforce one recipient per episode, and the reporter admitted missing, duplicate or unfinished stores.

Applied in the producer: path outcomes are inapplicable (None) for a pathless donor, never a crash;
every outcome is a paired state from the model's own baseline call to the arm's call on jointly
complete calls (unchanged / switched_to_donor / changed_to_other, for the path and for the tool),
donor agreement kept as a descriptive label; the same-state arm is a hard stop; the selector admits one
decision per episode; the baseline's prefill residuals at the patch site are compared with the direct
forward's capture (a hard stop beyond 1e-4, the figure recorded); the manifest names the requested
recipients. In the reporter: admission (manifest, exact requested set, unique episodes, every layer and
prescribed arm present, patch applied once, prefill check) before any count; a (row, layer) whose
same-state failed is void and excluded; switches require jointly complete calls and a change from the
baseline; the full cohort stays the denominator with the invalid pairs counted; an empty or incomplete
store is refused. `scripts/pilot_classify_fixtures.py` runs Codex's cases (and three more) on the
script's own source: eight pass. The smoke test re-ran with the full 48-token budget and two recipients
so that the crash branch is exercised (result below).

**Second smoke test (CPU, two recipients from two episodes, layers 8 and 20, 48 tokens).** Both
baselines reproduce the expert's exact call (tool and path); the generate prefill's residuals at the
patch site equal the direct forward's to the bit (difference 0.0 at both layers); same-state identical
everywhere; every patch applied once; the reporter admits with no void cell. On these two rows the
operation donor at layer 20 produced a well-formed `calculate` call in both (`10 + 10`, `24 + 60`), at
layer 8 nothing moved, and no path arm switched at either layer. Two rows: a mechanism check, not a
reading. The reporter's refusal fixtures (`scripts/pilot_reporter_fixtures.sh`) on corrupted copies of
that output: a duplicated row refused; a failed same-state at one (row, layer) admitted with that cell
void and the operation switch count at that layer falling from two to one; a missing manifest refused;
an empty store refused; a patch applied zero times refused. Digests as armed on the card:
`steer_pilot.py` b3b676a0467c, `steer_pilot_analyze.py` 7f646b820e82.

## The first attempt died at model load (06:28Z), before any recipient

The driver started the pilot 60 s after the 4B W-3b pass's start line, as designed — "beside W-3b on an
otherwise empty card". The card was not otherwise empty. The 12B fit's last chunk (c3) was still running:
the re-run 4B capture finished at 06:27Z, twelve minutes *before* c3 ended at 06:39Z, the reverse of the
order the driver assumed. At 06:28:45Z the card held c3 (62.25 GiB of process memory) and W-3b (23.01 GiB),
and the pilot's `model.to(float32)` failed on a 2.50 GiB allocation with 1.82 GiB free. The process had
written its pair selection (111 candidates, 104 pairs available, 24 chosen from 24 episodes, 8 per step,
all with an operation donor; corpus sha256 790cefff…) and nothing else. Cause: a driver gated on another
job's start rather than on the card's free memory — the Chief's planning error, not the pilot's. The
attempt is kept on the card at `captures/pilot-4b.crashed-0628Z/` and `steer-pilot-4b.crashed-0628Z.log`.

The re-run driver `scripts/chief_pilot_gated2.sh` (5d8ec3d3b0dd) waits for `/workspace/chief/GO-PILOT`,
a file the Chief touches on the D-CRO's message ("timed jobs done"; the card's phases are entered on
messages, never on a clock), then refuses to launch until nvidia-smi shows at least 30 GiB free,
checking every 30 s; seat chief-pilot, lease 240 min, the same command line as the first attempt (24
recipients, layers 4 to 32 by 4, 48 new tokens). The pilot is untimed and may share the card with
another untimed job. Both smoke tests ran on the CPU, so the re-run is the pilot's first pass on CUDA;
its `peak_gib` field is `max_memory_allocated`, and process memory is read from nvidia-smi during the run
for the shared-card rule.

## Results — the base 4B, 24 recipients, 07:25–07:58Z

**Admission.** Manifest present, 24 rows written for 24 requested, the recipients exactly the requested
set, 24 unique episodes, every layer of the sweep and every arm present in every row, every patch
applied exactly once, the prefill's residuals at the patch site identical to the direct forward's in all
24 rows (difference 0.0), the same-state arm token-for-token identical to the baseline at every (row,
layer): no void cell; the reporter admits in strict mode (`results/pilot-4b-analysis.json`,
`results/pilot-4b-manifest.json`; producer b3b676a0467c, reporter 7f646b820e82). All 24 baselines
reproduce the expert's exact call, tool and path. 33.9 minutes on the shared card, 15.9 GiB peak
allocated, about 17 GiB of process memory.

**Outcomes.** Paired states from the model's own baseline call to the arm's call, all 24 pairs jointly
complete at every layer and arm. Δ margin: the first generated token's logit for the named tool minus
the best other tool, arm minus baseline, in logits, median over the 24 with the range in brackets.

| layer | operation: tool → donor's `calculate` | operation: Δ margin of the expert tool | opposite_target, pending_file, unrelated_family: any path or tool change |
|---|---|---|---|
| 4 | 0 of 24 | +0.04 [−0.0, +0.1] | none (0 of 24 each) |
| 8 | 0 of 24 | +0.08 [−0.4, +0.5] | none |
| 12 | 0 of 24 | +0.55 [−0.0, +1.4] | none |
| 16 | 0 of 24 | −4.70 [−11.7, −0.9] | none |
| 20 | 24 of 24 | −35.9 [−39.8, −30.6] | none |
| 24 | 24 of 24 | −44.6 [−49.5, −38.5] | none |
| 28 | 24 of 24 | −44.4 [−49.4, −37.5] | none |
| 32 | 24 of 24 | −42.1 [−46.0, −35.3] | none |

Under the three path donors the Δ margin of the expert tool stays within a few logits at every layer
(the widest median +2.2 for pending_file at layer 32, ranges within −10.6 to +7.5); no path switched to
the donor's, no directory switched, no file switched within the directory, and the tool never changed,
at any of the eight layers, in any of the 24 recipients. Under the operation donor the switched calls
are well-formed `calculate` calls with freshly generated expressions (`"10 + 10"`, `"74 + 65 + 83"`);
the donor record carries no expression. Codex's post-hoc audit (WORKSPACE-DATA-READOUT-2026-09-10), verified here
on the corpus joined by task id and step: the same expression appears at all four switching layers; 12 of 24 equal
the donor's expert expression exactly, and every one of the 24 is built from the metric values listed in the
recipient's own note — one value at step 2 (plus 0, or repeated), the two at step 3, the three at step 4 — the
donor matches being the recipients whose donor later summed those same values. The vector moved the operation; the
arguments were assembled from the recipient's context. Codex's proposed discriminating measurement, forcing only
the token `calculate` into the unmodified recipient, would say whether the rest is ordinary continuation.

**Against the predictions.** The operation prediction held completely from layer 20: the tool switches
to the donor's in every recipient, the expert tool's first-token margin falling by thirty to fifty
logits, with a sub-threshold drop at layer 16 and nothing at 4–12. The two target predictions failed
completely: neither the opposite directory nor the pending file moved the generated path at any layer.
The rival failed too: the unrelated donor moved nothing, so "the next call is carried whole" is not
what happened either. Reading, under the claim ceiling: at the action position a whole-residual swap at
a single layer, from layer 20 on, carries the operation and does not carry the target file. One
caveat bounds it. The operation is emitted at the very next token, read from the patched position; the
path is emitted several tokens later, at decode steps the patch does not touch, from a context that
still names the recipient's own file. So the dissociation may reflect *when* each piece is read — the
next token from `P_act`, the path from the context afterwards — rather than what the residual at `P_act`
holds. Distinguishing the two needs a patch that persists through the decode steps or a readout of the
path from the residual itself; both are outside this pilot. Base model, not the tuned agent; 24
recipients, one family, one donor per arm; a localisation test, not target identification.
