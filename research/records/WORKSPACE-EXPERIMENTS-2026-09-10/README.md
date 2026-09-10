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
- `scripts/workspace_capture_v2.py` (d4651269f23c as the 12B pass reads it; its lineage is in the provenance table), for the 12B pass and any
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

## The passes as they ended (06:27Z–06:45Z), the verification, and the schedule as executed

The re-run 4B capture ended at 06:27Z: 7,629 rows in 173.7 minutes (1.37 s per row beside the 12B fit),
manifest present. `verify_capture_index.py` (3caf3d683c0e) at 07:02Z reconstructed the index from the
corpus and the memmaps and compared it with the index as written: 7,629 rows both ways, the sample's
300 rows, 0 differences on the deterministic fields, no P_act readout below the mass floor — the index
is consistent with the corpus. The 12B fit's chunk c3 ended at 06:39Z with 61 rows requested and 61
written (no row skipped, the progress log's row events counted), and the merge returned rc=0 at 06:40Z:
201 prompts, 47 layers, digest in Provenance; the weighting by requested rows is therefore valid.

The 4B W-3b pass (v3.1, 23c09ec1fe36) ran alone from 06:27Z to 06:45Z: 300 rows in 18.6 minutes,
23.01 GiB of process memory. Every row carries receipts — 1,886 arm receipts, each with no missing and
no unexpected edge, attention exactly zero on every masked edge over all queries, no mass above the
diagonal, no local mass beyond the window, the queries before the cut still attending — and the
placement oracle exact on all 300; no arm skipped; the analyzer (30ffe6d01889) admits 300 of 300 in
strict mode, 300 unique episodes. Every arm is resolved at P_act in every row (the format leaves the
six tool names as the only next tokens).

Schedule as executed. Phase 0 — the W-3b alone as soon as the capture's manifest existed (the change of
04:05Z; its GO file touched at 03:45Z) — ran 06:27–06:45Z; "card clear" went to the D-CRO at 07:01Z and
their timed jobs held the card from 07:02Z. The steering pilot, whose driver started on the W-3b's
start line, died at model load: c3 was still on the card, its end having fallen twelve minutes *after*
the capture's rather than before (the pilot record). The CPU analyses of the capture started at 07:02Z
with the BLAS threads capped at four of sixteen during the D-CRO's timed slot, disclosed to them. The
remaining phases are entered on the D-CRO's "timed jobs done": the pilot (`GO-PILOT`), the 4B W-3b
repeat with the corrected control (`GO-W3B-4B-V32`, next section) and the 12B capture and W-3b
(`GO-12B`), each driver refusing to start until the card shows the memory it needs.

## The random control of the 4B pass is degenerate (found 07:12Z in a row-level check), and its correction

v3.1 drew the equal-count control's keys from the positions before P_note whose kind is not `task`.
Every position before the note is the task statement, a carrier span or a format token, so that pool is
the carrier spans plus the format tokens — in the 187 rows with carriers, a median of 208 carrier tokens
beside 69 format tokens — and a draw of the all-carriers count from it is mostly the all-carriers mask
itself: three quarters of the drawn keys are carrier tokens (median 0.75, minimum 0.57), `<bos>` is
among them in 152 of 187 rows, and the arm's transition counts (137 retained, 42 lost, 1 gained, 7
neither) shadow the all-carriers arm's (142, 37, 0, 8), its lost rows overlapping all_carriers' in 27
of 42. The `paired_vs_random` figures in the v3.1 output are comparisons between two versions of the
same mask and are not read. Masking the sink is not what drives the arm's losses (lost share 0.23 with
`<bos>` masked, 0.20 without, on 152 and 35 rows). The same-kind comparisons that the basis line names as
the role- and contiguity-matched ones — older versus previous note, older versus previous call — do not
depend on the control and stand. The current-note arm never had a count-matched control.

v3.2 corrects the pool and adds the missing control: `random_equal_count` draws the all-carriers count
from the positions 1 … P_note−1 that lie in no carrier span (the task statement and the format tokens;
never position 0), masked from P_note as the carrier arms are; `random_equal_count_note` draws the
current note's prose-token count from the same pool, masked from the fence onward (q0 … S−1) as the
current-note arm is, and must leave P_note's readouts bit-identical like it; each row records the pool's
size and kind composition and each draw's kind composition; a draw that touched a carrier token or
position 0 is an assertion failure. The analyzer pairs each carrier arm with the first control and the
current-note arm with the second, names the control arm it used, and labels a v3.1 output's control
degenerate in its `control_status` so that no later reader takes its `paired_vs_random` for a control
comparison. The 12B pass runs v3.2 (its driver reads `workspace_w3b.py`, promoted before `GO-12B`); the 4B
pass is repeated with v3.2 (19 minutes, untimed, behind `GO-W3B-4B-V32`) so that both models carry the
same control; the v3.1 pass stays as recorded here, its receipts and every non-control reading intact.
Digests and the CPU regression of v3.2 are recorded in Provenance when they exist.

## Provenance

Sources as run are frozen in `scripts/` (digests below). The on-disk files predate the processes
that loaded them and were not modified after: `workspace_capture.py` last modified 23:51:38Z, the 4B
capture (pid 46534) started 00:22:51Z; `fit_lens_f32.py` last modified 14:30Z on the 9th, chunk c3
(pid 46410) started 00:24:21Z.

| file | sha256 (first 12) | role |
|---|---|---|
| `workspace_capture.py` | 958ccdcf0bae | v1, the first 4B capture (died at 03:22Z); kept on the card as `workspace_capture.py.v1-as-run-4b` |
| `workspace_w3b.py` | 23c09ec1fe36 | W-3b v3: placement certified (the repaired v2, b2d674d3469e, is kept on the card as `workspace_w3b.py.v2-repaired`) |
| `workspace_w3b_analyze.py` | 30ffe6d01889 | W-3b aggregate with admission and paired transitions |
| `workspace_analyze.py` | da680b6e5773 | W-1/W-3/W-4/W-5 with the prose–syntax split |
| `workspace_w2.py` | 033ae371ef3e | W-2 and W-4 primary |
| `fit_lens_f32.py` | 67fdf5559a1a | the float32 exact lens fit (4B full; 12B chunks) |
| `merge_chunks.py` | 5c39d626e129 | chunk merge, weighted by requested rows (valid: no row skipped — c1 70/70/0, c2 70/70/0, c3 61/61/0) |
| `chief_4b_passes.sh`, `chief_12b_passes.sh`, `overnight3.sh`, `chief_analyze.sh` | 0ef283871e02, c97b46a212bb, f3d6edd6541b, 49d0dc5589ba | drivers |
| `workspace_capture_v2.py` | d4651269f23c | the capture with the index per row, as the 12B pass reads it. Lineage, each version in this record's history: 165d7b59eaf4 (v2, tested; commit 3e82317) → 422435f2f0b7 (v2.1, the copy the 4B re-run loaded at 03:33Z; commit 4315b44) → 6f87954ed5dd (the guard's stop widened to 1e-2; e6845f6) → d4651269f23c (the inner loop variables renamed; 1757284) |
| `reconstruct_index.py`, `merge_tail_capture.py`, `verify_capture_index.py`, `chief_12b_passes_v2.sh` | 279b16d1b79f, 4d292cf78120, 3caf3d683c0e, f70cae68e67f | the insurance and the index verification (section above); the gated 12B driver |
| `chief_w3b_4b_gated.sh`, `chief_pilot_gated.sh` | e0ac7c3b0ba4, 5192a4b80905 (in the pilot record) | the phase-0 W-3b driver; the pilot's first driver, which died at load (the pilot record) |

Corpus (the three splits concatenated, 7,629 distinct decisions, 1,128 episodes, twelve families):
sha256 `790cefffc29b…`. 4B lens archive `out/lens4b-f32/exact-maps.npz`: sha256 `56c7b49e1c71…`
(201 prompts, float32, width 16 fit; anchor 16). 12B lens `out/lens12b-f32/exact-maps.npz`: sha256 `e7942f1d6a73…` (201 prompts, 47 layers; c1 and c2 at width 16, c3 at width 8, merged 06:40Z weighted by rows, none skipped). Checkpoints by the
loader's complete hash manifest (`load_report_sha256` in each capture manifest). Arithmetic: float32,
TF32 off, highest matmul precision, eager attention, width 1 at capture; determinism pinned. Sample
rule: per family, sorted (task_id, step), prompts ≤ 1,500 tokens, every k-th to 25; 300 decisions.

## Results — 4B, first readings (07:20Z; W-2 pending)

Each experiment reports in three states with its unit count, its unresolved count and its tail, and
nothing in this record is a claim about a workspace property. Unit: the decision (7,629; the 300-sample
where stated). Base Gemma 3 4B, float32, width 1, the expert's note teacher-forced.

**The anchor (W-4's model readout).** At P_act the model's own six-tool argmax is the expert's action in
7,398 of 7,629 decisions (0.970; six-tool mass median 1.0, none unresolved): ≥ 0.998 in ten families,
0.943 in cross_reference (862), 0.829 in batch_update (1,053). At P_note the six-tool mass is below the
floor in 7,587 of 7,629 (median 5.6e-11; the next token there is the note's first word), so there is no
six-tool readout at P_note by construction: W-1's P_note cells and W-4's secondary P_note readouts are
void (never rank 1 in 0.968 and 0.774 of rows through the lens and the unembedding; the 42 resolved rows
agree with the expert). Early-versus-late decodability rests on W-2.

**W-1, first confidence crossing at P_act.** Through the fitted float32 lens the expert action first
ranks first at median depth 0.727 — layer 24 of 33 (p10 layer 24, p90 layer 26) — never in 162 rows
(0.021), and once first it stays (crossing width median 0, p90 0). A late readout competitor (another
tool first in the late layers before the crossing) in 596 rows (0.078), at median layer 29, the prior
favourite `read_file` in only 0.15 of them (replace_text 242, calculate 139, list_files 123, read_file
92), concentrated in batch_update (0.34 of its 1,053; replace_text 211). By family the median crossing
is layer 24 in eight families, 25 in update, 26 in pointer_chain, synthesis, batch_update and read; the
never share is 0.142 in batch_update and ≤ 0.007 elsewhere. Through the unembedding (the plain logit
lens) the crossing is earlier — median depth 0.606, layer 20 (p10 18, p90 24) — wider (width median 2
layers, p90 4), never in 97 (0.013), with a late competitor in 2,939 rows (0.385) at median layer 23,
the prior favourite `read_file` in 0.83 of those (the calculate family: all 223, read_file 200). W-5
explains the lens's crossing: through the fitted maps the six-tool readout at P_act is below the floor
in every row through layer 23 in every prompt-length band, resolves at layer 24 (1,897 of 3,184, 1,493
of 3,476 and 526 of 969 rows resolved in the ≤ 1,024, ≤ 2,048 and > 2,048 bands; agreement with the
model's own argmax 0.99, 0.99, 0.98) and is fully resolved from layer 26 (agreement 0.92–1.00 through
33); the unembedding resolves from layer 18 (agreement 0.48, 0.96, 0.92 by band) and agrees fully from
layer 24; no band dependence at layers ≥ 24. The fitted lens is silent and then right: its "first
crossing" is its first resolved layer, a property of the readout at this floor, not of the model.

**W-3, direct attention to the tagged spans (300-sample, eager kernel).** Gate: local-layer mass beyond
the window is 0 in all 17,400 cells. In the global layers (6, 12, 18, 24, 30) the mass beyond the
window has median 0, p90 0.36 and maximum 0.79 over 3,000 cells. Mean direct attention mass from P_act:
in the local layers, format 0.31, the note's syntax 0.24 (0.64 at layer 3), task 0.15, the note's prose
0.12 (at most 0.29, layer 17), previous_call 0.04, hidden_result 0.04, latest_result 0.03, previous_note
0.02, the older spans ≤ 0.02; in the global layers, task 0.40, format 0.33, previous_call 0.07,
older_call 0.07, the note's syntax 0.04, its prose 0.03, the results ≤ 0.02. From P_note: format 0.59
and 0.48 (local, global), task 0.23 and 0.33, previous_note 0.05 and 0.08, older_note 0.02 and 0.06,
latest_result 0.045 and 0.02, hidden_result 0.04 and 0.01. Routing weights, not causal shares.

**W-4's secondary, the note trajectory (300-sample, lens at the last layer).** The expert tool is first
at the note's start in 0 of 300 rows and at its end in 288 (0.96); the first position where it is first
sits at the note's end (median 1.0 of the note's length, p10 0.625), it moves at most once (changes per
note median 0, maximum 1), and the readout is unresolved on a median 37 of the note's ~38 tokens. Split
at the fence: first at the prose's end in 0 of 300; the crossing lies inside the seven syntax tokens in
255 rows, inside the prose in 33 (at 0.63 of the prose's length, median), nowhere in 12. The six-tool
readout lights only where a tool name is the next token; this cell says nothing about report versus
computation, as its basis line states.

**W-3b, the carriers and the current note masked in the attention (300 rows; 187 with carriers, 131
with older carriers, 71 with a hidden result; v3.1, receipts on every arm).** The paired transition of
the expert tool's argmax from the unmasked forward to the arm, on jointly resolved untied rows; Δ margin
is the taken tool's logit lead over the best other tool, arm minus unmasked, in logits.

| arm | rows | retained | lost | gained | neither | retained share | Δ margin median (p10, p90) |
|---|---|---|---|---|---|---|---|
| previous_note | 187 | 177 | 2 | 0 | 8 | 0.989 | −0.15 (−3.4, +0.9) |
| previous_call | 187 | 172 | 7 | 1 | 7 | 0.961 | −0.40 (−5.5, +3.0) |
| latest_result | 187 | 179 | 0 | 0 | 8 | 1.000 | −0.31 (−2.1, +0.9) |
| older_note | 131 | 129 | 1 | 0 | 1 | 0.992 | −0.07 (−1.3, +0.8) |
| older_call | 131 | 120 | 10 | 0 | 1 | 0.923 | +0.23 (−4.9, +3.2) |
| older_result | 131 | 130 | 0 | 0 | 1 | 1.000 | +0.01 (−0.6, +0.5) |
| hidden_result | 71 | 70 | 0 | 0 | 1 | 1.000 | +0.55 (−0.1, +1.6) |
| all_carriers | 187 | 142 | 37 | 0 | 8 | 0.793 | −2.5 (−28.1, +4.1) |
| current_note (queries from the fence) | 300 | 225 | 63 | 2 | 10 | 0.781 | −4.0 (−23.8, +0.2) |
| all_carriers_and_current_note | 187 | 124 | 55 | 2 | 6 | 0.693 | −7.3 (−31.2, +2.6) |
| random_equal_count (degenerate, section above) | 187 | 137 | 42 | 1 | 7 | 0.765 | −4.2 (−23.9, +6.3) |

A lost winner is lost outright: Δp of the expert tool ≤ −0.5 in all 63 lost rows of the current-note
arm and all 37 of all_carriers. No single carrier span is necessary for the six-tool winner at the
expert-forced prefix (retained ≥ 0.92; the result spans 1.00). All carriers together lose it in 37 of
179 — pointer_chain 11 of 16, read 8 of 12, cross_reference 5 of 20, batch_update 4 of 19,
aggregate_report 4 of 16 (the lost tool read_file 24, finish 8, search_files 5) — where the single-span
arms lost at most 7: what the winner needs in those families (the next file to read) is carried
redundantly by the previous note, the previous call and the latest result, and only their joint removal
takes it away. The current note's prose, removed from every query after the fence (no relay exists),
leaves the winner in 225 of 288; the 63 losses fall on step 0 in 36 (of 113 step-0 rows, 0.32; later
steps 27 of 187, 0.14) — the first decision of an episode, where the note is the only place the plan has
been written — and the masked winner is list_files in 34 of 63 (from read_file 28, from search_files 6):
a generic look-around replaces the specific read. By family: conditional_update 11 of 25, update 10,
search 8, synthesis 7, batch_update 7, pointer_chain 5, list 5, cross_reference 4, read 3,
ledger_reconcile 2, aggregate_report 1, calculate 0. Prose lengths of lost and retained rows are alike
(median 33 and 31 tokens). Carriers and note together: 55 of 179 lost. Survival shows non-necessity of
the masked edges under this intervention and does not date the decision (the order's ruling); the
count-matched comparisons wait for the v3.2 repeat.

**W-2 (the primary of W-4: early versus late linear decodability of the expert action at P_note and
P_act, rank ladder with the permutation and transfer nulls).** Running on the host CPU at this writing;
its readings follow in this section.
