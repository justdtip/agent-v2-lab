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

## The first 4B capture died at 03:22Z, 128 rows from its end; the re-run, with the fatal allocation removed

At row 7,501–7,628 of 7,629 the capture raised `torch.OutOfMemoryError`: it tried to allocate
4.20 GiB with 3.41 GiB free, the 12B fit's chunk c3 holding 62.25 GiB of process memory (its
`max_memory_allocated` was 59.72 GiB; the process figure is the one to plan with) and the capture
29.30 GiB (22.12 allocated, 6.52 reserved and unallocated). The allocation was the full-vocabulary
logits of a 4,321-token row — 4,321 × 262,144 float32 — of which the pass reads two positions. The
W-3b pass then failed at once for want of the capture's manifest. The dead capture is kept as
`captures/4b-died-0322Z` (7,501 rows of residuals, reconstructable by `reconstruct_index.py`); its
per-row readouts were in the end-only index and are lost, so the clean recovery is a full re-run with
the index written per row, beside the fit, whose remaining hours would otherwise leave a third of the
card idle.

The fix, in both scripts: the model's logits are computed only at the two positions read
(`logits_to_keep=[P_note, P_act]`), which removes the spike entirely; the first row of every run also
computes the full logits and records the maximum difference at those positions, a hard stop beyond
1e-2 (a guard against a gross error such as wrong positions; the reduction-order difference itself is expected at 1e-5 to 1e-4 and was 7.9e-5 on the GPU's first row). The kept logits are **not bit-identical** to the full-sequence ones — 3.8e-5 on the CPU
regression, the lm_head matmul's reduction order depending on its shape — and the residuals, the
sample and the lens readouts are unchanged. Every reading of the model's own logits in this
programme (the capture's six-tool fields, the W-3b arms' six-tool fields and margins) now takes the
same path, so comparisons within and across passes share it; the figure is stated here so that no
later pass is compared with a full-logits one without knowing it. Scripts as re-run:
`workspace_capture_v2.py` v2.1 (6f87954ed5dd; the 4B re-run loaded 422435f2f0b7, identical but for the guard's threshold), `workspace_w3b.py` v3.1 (23c09ec1fe36), driver
`chief_4b_passes_v2.sh` (387bed5cf3a1), from `/workspace/chief` with expandable segments; CPU regression in
`recover_4b_test.sh` (residuals and sample identical to v2; receipts and oracle pass; the analyzer
admits).

**Scheduling after the re-run.** The re-run ends about fifteen minutes after the 12B fit's last chunk,
and the D-CRO's repeat gate and ladder must then run alone (both measure time), so the W-3b launch was
taken off the running driver (the driver stopped, the capture untouched) and put behind a GO file of
its own (`scripts/chief_w3b_4b_gated.sh`, e0ac7c3b0ba4); the 12B passes were already behind one. The four
phases, each entered on a message and none on a clock: (1) "card clear" from the Chief → the D-CRO's
timed jobs alone; (2) "timed jobs done" → the D-CRO's 4B pass beside the Chief's 12B capture;
(3) "4B pass ended" → the Chief's 4B W-3b beside the 12B capture; (4) "W-3b ended" → the D-CRO's 12B
pass beside the 12B capture, on the rule that the two measured process-memory peaks leave 9 GiB.

**A cosmetic bug in the re-run's progress line, found from its own output.** v2's outer loop variable
is `k`, and the sample-row block reuses `k` for two inner loops, so after a sample row the progress line
reports that row's sequence length as the count (row 909, a sample row of 501 tokens, printed as
"n_done 501" between 901 and 1,001). The memmap writes and the index row are written before that block
with the correct index, and the index is keyed by the global row number; verified on the live run (index
contiguous, memmaps non-zero through the last row written). The inner variables are renamed in the copy
the 12B run reads (d4651269f23c); the 4B run keeps its silly progress lines and its correct data.

## Codex's audit of the repair (63d0549, standing request 5): placement certified, admission and the retained metric corrected

Codex found three things in the repaired script and its analysis, none a finding that the production
cut was wrong: a misplaced but non-empty cut passed every check, because the checks asked only whether
the supplied mask was non-empty and received zero attention; the analysis aggregated rows without a
validity state, so a failed gate, a duplicate id or a stale manifest would have entered the estimates;
and "still top" counted a tool that became the winner under the mask as having remained it, with an
empty denominator written as zero. All three are applied before any W-3b reading.

`scripts/workspace_w3b.py` v3 (9bf3f0e66581), the script the 4B pass runs. Placement is certified three ways,
each a hard stop: a self-test at start reproduces Codex's hand-labelled keys, `q0` and straddle on
four synthetic tokenisations at four prompt offsets and rejects the six wrong masks (delayed start,
wrong key, missing key, extra key, future-only key, empty) with an expected-edge oracle; a text oracle
per row, independent of the builder, requires the note keys to decode to exactly the prose before the
fence and the first query token to begin the fence (exact on all 7,629 rows of the corpus); and a
receipt per arm compares the mask's applied edges with the expected set (no missing, no unexpected)
and reads the effective mask from the returned attentions — zero on every masked edge over all
queries, no mass above the diagonal, zero beyond the window for every query in the local layers, and
the queries before the cut still attending to the masked keys. The hook must run on every layer and
find the native mask. The declaration now says the true endpoint: queries through S−1, the supplied
tool-name token, an unread query; `P_act` = S−2. `run.json` names the requested sample at start.
One-row CPU test: self-test 16 cases and 6 rejections; every arm's receipt passes; effective checks
all zero or positive as required (`scripts/w3b_v3_test.sh`).

`scripts/workspace_w3b_analyze.py` (30ffe6d01889): admission before any estimate — the rows on disk must be
exactly the requested sample, unique, the manifest's row count consistent, every arm's gate complete,
finite and passing, the current-note arm's `P_note` identity true, and a v3 row's receipts and oracle
passing; an ineligible row is counted with its reasons and never averaged; strict mode refuses (exit
1), `--diagnostic` renders labelled. The retained metric is a paired transition on rows where the
baseline and the arm are both resolved and untied — retained, lost, gained, neither — with unresolved
and tie counts kept separately and a null share on an empty denominator; the old metric survives as
`masked_expert_top_agreement`; the comparison with the random control is paired on jointly eligible
rows. Tested (`scripts/analyzer_test.sh`): the clean output admitted; a copy with a nonzero masked
attention, a duplicate row and a stale manifest refused with exit 1 and rendered only in diagnostic
mode; a gained winner classified as gained with a null retained share.

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

## The capture's end-only index, and the insurance written before the pass could fail

The capture as run for the 4B (`workspace_capture.py`, 958ccdcf…) writes `index.jsonl` and
`manifest.json` only when it finishes; the residual memmaps and `sample.jsonl` are written as it goes.
Sharing the card with the 12B fit's third chunk, the allocator warned nine times in an hour (free
memory down to 0.7 GB), recovering each time; 15 of the remaining rows were longer than any yet
processed (4,321 tokens against 3,950). Rather than restart under the same pressure, the exposure was
closed while the pass ran, without touching it:

- `scripts/reconstruct_index.py` (279b16d1b79f) rebuilds the deterministic index
  (positions, tool, token counts, sample membership from `sample.jsonl`) for every non-zero memmap row;
  the model's own six-tool fields need a forward and stay null; the lens and logit-lens readouts are a
  pure function of the saved residual, the maps and the unembed, recomputed without a forward. Verified
  read-only against the live pass at 5,040 rows: no holes, every tool token aligned, every sample row
  on disk.
- `scripts/workspace_capture_v2.py` (165d7b59eaf4), for the 12B pass and any
  re-capture: the index written per row; `--rows-from/--rows-to` (global corpus indices; memmap row =
  index − rows_from); `--sample-ids-file` so a tail keeps the original sample rows. Regression on the CPU
  (`capture_v2_test.sh`, `capture_v2_test2.sh`): index and memmaps bit-identical to v1 on the three-row
  test capture; the ranged run and the sample-ids run exact.
- `scripts/merge_tail_capture.py` (4d292cf78120) copies a tail re-capture into a
  dead pass's memmaps, refusing rows that are already non-zero. A simulated death (rows 1–2 zeroed, the
  index and manifest removed, reconstructed, the ranged tail merged) reproduced the intact capture
  exactly, memmaps and deterministic index fields.

The 12B passes run from `chief_12b_passes_v2.sh` (f70cae68e67f), gated on a GO
file so they cannot start into the D-CRO's timed slot, launched with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (memory management only, recorded here) and from
`/workspace/chief`, not the shared checkout.

## Provenance

Sources as run are frozen in `scripts/` (digests below). The on-disk files predate the processes
that loaded them and were not modified after: `workspace_capture.py` last modified 23:51:38Z, the 4B
capture (pid 46534) started 00:22:51Z; `fit_lens_f32.py` last modified 14:30Z on the 9th, chunk c3
(pid 46410) started 00:24:21Z.

| file | sha256 (first 12) | role |
|---|---|---|
| `workspace_capture.py` | 958ccdcf0bae | the capture as running for the 4B (and queued for the 12B) |
| `workspace_w3b.py` | 23c09ec1fe36 | W-3b v3: placement certified (the repaired v2, b2d674d3469e, is kept on the card as `workspace_w3b.py.v2-repaired`) |
| `workspace_w3b_analyze.py` | 30ffe6d01889 | W-3b aggregate with admission and paired transitions |
| `workspace_analyze.py` | da680b6e5773 | W-1/W-3/W-4/W-5 with the prose–syntax split |
| `workspace_w2.py` | 033ae371ef3e | W-2 and W-4 primary |
| `fit_lens_f32.py` | 67fdf5559a1a | the float32 exact lens fit (4B full; 12B chunks) |
| `merge_chunks.py` | 5c39d626e129 | chunk merge, weighted by requested rows (valid when no row skipped: c1 70/70/0, c2 70/70/0; c3 checked at its end) |
| `chief_4b_passes.sh`, `chief_12b_passes.sh`, `overnight3.sh`, `chief_analyze.sh` | 0ef283871e02, c97b46a212bb, f3d6edd6541b, 49d0dc5589ba | drivers |
| `workspace_capture_v2.py`, `reconstruct_index.py`, `merge_tail_capture.py`, `chief_12b_passes_v2.sh` | 6f87954ed5dd, 279b16d1b79f, 4d292cf78120, f70cae68e67f | the 12B capture and the insurance (section above) |

Corpus (the three splits concatenated, 7,629 distinct decisions, 1,128 episodes, twelve families):
sha256 `790cefffc29b…`. 4B lens archive `out/lens4b-f32/exact-maps.npz`: sha256 `56c7b49e1c71…`
(201 prompts, float32, width 16 fit; anchor 16). 12B lens: merged after c3. Checkpoints by the
loader's complete hash manifest (`load_report_sha256` in each capture manifest). Arithmetic: float32,
TF32 off, highest matmul precision, eager attention, width 1 at capture; determinism pinned. Sample
rule: per family, sorted (task_id, step), prompts ≤ 1,500 tokens, every k-th to 25; 300 decisions.

## Results

Not yet. Each experiment reports in three states with its unit count, its unresolved count and its
tail, and nothing in this record is a claim about a workspace property.
