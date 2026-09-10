# Workspace experiments, set 1 — an observational screen at the agent's decision positions

**Chief, 2026-09-10 UTC. Status: captures in progress; nothing read.** Order:
`design_specifications/pending/WORKSPACE-EXPERIMENTS-2026-09-10.md`, with the rulings on Codex's
design review (`WSA-WORKSPACE-AND-PREREG-2026-09-10`, 3494a2c), closure review (232fe6a) and
run-capabilities review (712f78c) applied before any reading. The mechanism claims — ignition,
functional availability, causal broadcast, report versus computation — are reserved for the steering
experiments; this record describes. Vocabulary from 712f78c: (a) content a researcher can decode,
(b) content the model can use, report or manipulate, (c) a model of its own access; this screen is (a).

## What is measured, by name

| experiment | quantity | instrument | control / null |
|---|---|---|---|
| W-1 | first confidence crossing of the expert action; readout competitor | six-tool conditional mass through the fitted lens, full-vocabulary p beside, log-odds beside | the logit lens; the tool prior |
| W-2 | cross-family linear decodability of the expert action at `P_act` | PCA-r on the training fold + ridge readout; folds by episode; leave-families-out | within-family permutation null; transfer null (per-family label bijection) |
| W-3 | direct attention to tagged spans from `P_note` and `P_act` | eager attention, per layer and head, the 300-sample | the local-mask gate; span sizes and distances preserved |
| W-3b | the action under masked carriers | key masks by kind and the current note; the leaky-mask self-check; hard-stop gates | the unmasked forward; same-kind arms; the random-token arm (count only) |
| W-4 | early versus late linear decodability of the expert action (`P_note` against `P_act`) | W-2's readout at both positions; the note trajectory split at the fence into prose and syntax | the same nulls; the tool prior |
| W-5 | lens validity by position band and source layer | last-layer lens readout against the model's own | — |

## Defects found before any reading

**The current-note arm of W-3b masked nothing (Codex 712f78c, P1).** In the script as first written,
every completion token was labelled `note`, syntax and the supplied tool-name token included, and the
query slice of the mask began one past the last note token — one past the end of the sequence. The
arm blocked no edge; `all_carriers_and_current_note` equalled `all_carriers`. Confirmed three ways:
the slice read from the source; Codex's model-free reproduction (`check.py` in
`WSA-RUN-CAPABILITIES-2026-09-10`); and the one-row smoke output, where the current-note arm's
readouts equal the unmasked arm's bit for bit and the combined arm's equal the carriers arm's.

**The one-row smoke test of W-3b was itself mismatched.** It took positions from a capture made from
the test split (`out/wtest`, 23:22Z) while taking prompts from the concatenated corpus, and that
capture predates the doubled-BOS fix (23:51Z), so its positions are one too high. The old script
truncated the completion silently. Neither defect touched the production capture, which runs the
fixed capture script on the concatenated corpus on both sides.

## The repair, and its test

`scripts/workspace_w3b.py` (sha256 `b2d674d3…`; the empty-cut version is kept on the card as
`workspace_w3b.py.v1-empty-note-cut`, sha256 `f70eb66a…`). The note is the completion text before
the ```json fence, its trailing newline included, cut as the previous-note spans are cut; the keys
are the completion tokens whose character offset starts before the fence; the queries are every
token from the fence onward through `P_act` (`q0` = the first token starting at or after the fence),
so no relay through the syntax between the note and the action stays open; the supplied tool-name
token is labelled `tool_name` and is never read; a completion with no prose before the fence records
the arm under `skipped_arms`, never as a no-effect result. Checks, each a hard stop: every masked arm
blocks at least one edge at `P_act`; the attention on every masked edge, over all queries, layers and
heads, is exactly zero (`attention_on_masked_edges_all_queries`); local layers place zero mass beyond
the window at `P_act`; the leaky negative control (one carrier key left open) receives attention and
the rest none; the current-note arm leaves `P_note`'s readouts bit-identical to the unmasked
forward's; the in-context tool-name token equals the standalone first token the six-tool readout
uses, and the six are distinct; the capture and the pass read the same corpus, by digest. Rows are
written as they complete. Owed: a fixture in which a deliberately misplaced cut fails for its stated
reason (Codex's request; standing request 5 may specify it).

Test, CPU, 4B float32, a fresh three-row capture from the test split made by the current capture
script (`out/wtest2`, corpus digest `6518f957…`), its one sampled decision
(`test-search-0121-clean`, step 1, `read_file`; S = 568, `P_note` 531, `P_act` 566, 28 prose tokens,
7 syntax tokens, `q0` 560, no straddling token):

| arm | edges blocked at `P_act` | edges blocked in all | attention on masked edges | margin at `P_act` (logits) |
|---|---|---|---|---|
| unmasked | 0 | 0 | — | 19.94 |
| previous_note | 29 | 1,073 | 0.0 | 20.29 |
| previous_call | 29 | 1,073 | 0.0 | 22.39 |
| latest_result | 20 | 740 | 0.0 | 18.04 |
| all_carriers | 78 | 2,886 | 0.0 | 22.98 |
| current_note | 28 | 224 | 0.0 | 18.19 |
| all_carriers_and_current_note | 106 | 3,110 | 0.0 | 18.12 |
| random_equal_count | 78 | 2,886 | 0.0 | 26.43 |

Leaky control: the open key received 0.0345 (summed over layers), the masked keys 0.0. The
current-note arm's `P_note` readouts are bit-identical to the unmasked forward's. The two note arms
now differ from their controls. One row, a test of the instrument; not a reading.

The earlier mismatch was found by the repaired script's own tool-token check, which refused the stale
capture (`row 0: in-context tool token 236779 is not the standalone first token 1399 of read_file`).

## Deviations from the order in the capture as run (recorded, not repaired mid-run)

1. On the 300-sample the capture saves six-tool readouts at every note token, not the full
   residuals the order asked for. The 12B capture pass, not yet started, saves them; a supplementary
   sample-only pass adds them for the 4B after its W-3b pass. Until then W-4's secondary is the
   readout its text describes.
2. The note trajectory runs over the whole completion prefix. The analysis
   (`scripts/workspace_analyze.py`, with `--corpus --snapshot`) splits it at the fence by the repair's
   offsets rule: a prose position sits on the prompt's last token or a prose token, a syntax position
   on the fence or the JSON prefix, the last at `P_act`; the crossing is reported within each. W-3's
   kinds are relabelled the same way (`note`, `note_syntax`, `tool_name`). On the test row the
   taken tool became the resolved top only at `P_act`, every prose position unresolved by the floor.
3. The behavioural measure at `P_act` is the tool's first token; the complete call is scored only in
   the steering pilot.
4. Attention arrays are stored as float16; the in-forward gate is float32 and is what the
   zero-attention claim rests on.
5. Per-cell token digests are not in the capture manifest; the binding is the corpus digest plus
   positions re-derived deterministically at analysis, and every row's token after `P_act` is checked
   against the tool's first token when the index closes (count to be reported here).

## Provenance

Sources as run are frozen in `scripts/` (digests below). The on-disk files predate the processes
that loaded them and were not modified after: `workspace_capture.py` last modified 23:51:38Z, the 4B
capture (pid 46534) started 00:22:51Z; `fit_lens_f32.py` last modified 14:30Z on the 9th, chunk c3
(pid 46410) started 00:24:21Z.

| file | sha256 (first 12) | role |
|---|---|---|
| `workspace_capture.py` | 958ccdcf0bae | the capture as running for the 4B (and queued for the 12B) |
| `workspace_w3b.py` | b2d674d3469e | W-3b, repaired |
| `workspace_w3b_analyze.py` | 85948cba4b7f | W-3b aggregate, schema 2 |
| `workspace_analyze.py` | da680b6e5773 | W-1/W-3/W-4/W-5 with the prose–syntax split |
| `workspace_w2.py` | 033ae371ef3e | W-2 and W-4 primary |
| `fit_lens_f32.py` | 67fdf5559a1a | the float32 exact lens fit (4B full; 12B chunks) |
| `merge_chunks.py` | 5c39d626e129 | chunk merge, weighted by requested rows (valid when no row skipped: c1 70/70/0, c2 70/70/0; c3 checked at its end) |
| `chief_4b_passes.sh`, `chief_12b_passes.sh`, `overnight3.sh`, `chief_analyze.sh` | 0ef283871e02, c97b46a212bb, f3d6edd6541b, 49d0dc5589ba | drivers |

Corpus (the three splits concatenated, 7,629 distinct decisions, 1,128 episodes, twelve families):
sha256 `790cefffc29b…`. 4B lens archive `out/lens4b-f32/exact-maps.npz`: sha256 `56c7b49e1c71…`
(201 prompts, float32, width 16 fit; anchor 16). 12B lens: merged after c3. Checkpoints by the
loader's complete hash manifest (`load_report_sha256` in each capture manifest). Arithmetic: float32,
TF32 off, highest matmul precision, eager attention, width 1 at capture; determinism pinned. Sample
rule: per family, sorted (task_id, step), prompts ≤ 1,500 tokens, every k-th to 25; 300 decisions.

## Results

Not yet. Each experiment reports in three states with its unit count, its unresolved count and its
tail, and nothing in this record is a claim about a workspace property.
