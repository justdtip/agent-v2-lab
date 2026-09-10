# Review ready: what the running tools can establish, and an empty current-note intervention

Source inspection for the Director's question about access consciousness and the jobs in their 01:18Z snapshot. No model, tensor archive, unfinished scientific result, process state or GPU was accessed. Six small source/driver files were read from the authorised `rtx6000` host and frozen here, alongside the Git order and upstream adapter. Hashes and read time are in `sources.json`. These are the files present on disk at inspection, not independent proof of the bytes already loaded by the running processes. No running or queued source was changed.

## Immediate review finding: current-note masking is a no-op

**P1 — `workspace_w3b.py`, lines 107 and 115–119.** The script labels every completion-prefix position from `len(p_ids)` through `S-1` as `note`, including the JSON prefix and the first tool-name token supplied at the end. It then masks note keys only for query positions starting at `note_keys[-1] + 1`, which is `S`. The query dimension has length `S`, so this slice is empty. `P_act` is `S-2`, before that slice.

Consequences: `current_note` blocks no edges anywhere; `all_carriers_and_current_note` equals `all_carriers`. Equality of their readouts with the corresponding controls is built into the intervention and cannot support any report-versus-computation conclusion. A zero masked-attention sum is vacuously satisfied for the empty mask. This defect does not by itself invalidate the distinct earlier-carrier masks or the unmasked residual capture.

`check.py` executes the producer's actual mask-building AST with a set-backed array, using standard-library code only. Sixteen synthetic prompt/prefix lengths reproduce both equalities. A deliberately different note boundary leaves a suffix and produces blocked edges at the action position, demonstrating that the check distinguishes a nonempty intervention. These are source/slice checks, not device acceptance or independent scientific observations.

Required repair before interpreting these two arms: distinguish the prose note from the JSON syntax using token offsets; select the intended query cut explicitly; demonstrate nonempty forbidden edges at `P_act` where a note exists; compare intended versus applied masks and run the leaky negative control. The existing guard only logs numbers and a `passes` value; it does not refuse a failed cut. No repair or new run is performed by this review.

## What is running and what follows

- `fit_lens_f32.py`: exact upstream autograd Jacobians at each non-final block output, targeting the pre-final-normalisation residual. It averages across selected positions and corpus rows. The Chief's third 12B chunk adds the remaining 61 prompts to the earlier two chunks; the intended merged corpus is 201 fit prompts, capped at 128 tokens. It changes the instrument, not the model's trained weights.
- The parameters load as bf16 and are then cast to float32 before computation. This is float32 arithmetic on those stored values, not a native-bf16 behavioural study and not recovery of precision discarded before loading.
- `workspace_capture.py`: one width-one forward per distinct expert decision, over the prompt plus a teacher-forced completion prefix. Full float32 residuals are saved at the last prompt position and just before the first tool-name token. The source layers are those with fitted maps: ordinarily 33 in 4B and 47 in 12B. The final block is read as the model output reference, rather than another fitted source map.
- The capture saves model six-tool probabilities and top-five decoded labels at the two positions; per-layer lens and logit-lens six-tool distributions and their original probability mass. On the balanced short-prompt sample, it saves per-head attention to every key from the two query positions, and six-tool lens trajectories through the supplied prefix.
- The drivers queue `workspace_w3b.py` after capture, on that sample, and queue the analogous 12B passes after the merge. This is a source-level schedule statement; no claim is made here that those later jobs have started or passed.

## Claim ceiling of the outputs

| Question | What the outputs support after verification and analysis | What they do not identify |
|---|---|---|
| What content is readable at decisions? | Layerwise tool readouts; arbitrary later probes of the two saved residual vectors, with appropriate dictionaries/readout weights | A complete transcript of thought; all content the model itself can access |
| When does an expert action become readable? | Early/late held-out decodability, confidence crossings, reversals and competing tool readings | The time an autonomous decision was made, or an ignition mechanism |
| Is action information shared across families? | Episode-held-out cross-family decoding versus frozen nulls; comparison of raw and lens-transformed representations | Functional global availability merely from researcher decoding; a unique abstract action representation rather than common syntax |
| What receives attention? | Direct attention distributions over correctly tagged spans, layers, heads and distances | Causal contribution percentages or absence of multihop routes |
| Does earlier transcript access matter? | For working masks, intervention-induced changes in the model's own tool logits/mass at a fixed expert prefix, relative to suitable controls | Persistent memory, removal of every route carrying the fact, or autonomous task success |
| Does the note matter? | Early/late associations from capture | A causal answer from the currently empty note-mask arms |
| Does size change the picture? | Comparison of these two checkpoints under declared shared tasks, arithmetic and analysis | A general scaling law or a consciousness ranking |

The two source positions have dense residuals, so many future analyses can reuse them without recapture. The sample **does not save the full residual at every note token**, despite the order requesting it: it saves only six-tool readouts there. The current loop also includes JSON syntax in what is called the note trajectory. Attention arrays are cast to float16 on disk, although the forward is float32; tiny saved zeros cannot alone certify an exact structural mask. W-3b's in-forward attention diagnostics use float32 but only log the gate outcomes.

The capture only reduces lens readouts to the six first-token tool scores; it does not save vocabulary-wide lens top words. Such words could be computed later from the saved decision residuals and the necessary model readout parameters. The script computes standalone first-token IDs, but does not assert they are distinct or equal to the actual in-context token at the labelled tool boundary. That alignment must be checked before labelling a six-token score an action readout. A tool name does not include arguments or establish that an executable call would succeed.

## Instrument qualifications that remain material

1. The fit caps prose at 128 tokens; the agent reading extends much further and into a different domain. The revised order correctly treats per-source-layer, position-banded readout agreement as a diagnostic, not a proof that the average map is the correct derivative in every new context. Retained residuals remain useful independently of lens quality.
2. Float32 alone does not establish the same arithmetic path: the fit uses `dim_batch` as its forward width; capture uses one. The supplied c3 line declares width eight and earlier fit metadata declared other widths. The merge retains chunk widths but does not demonstrate their equivalence to capture or to one another.
3. `merge_chunks.py` weights by requested `manifest.n_rows`, not successful `nu.pair_weighting.n_prompts`; these agree if no rows were skipped. Check successful counts, disjoint rows, compatible identities and selectors before certifying the merged population. This review has not read completed chunk results.
4. The remote capture does not implement all provenance requirements in the amended order: it emits whole-corpus/map hashes but does not store per-cell consumed token IDs or rendered-prompt hashes, bind source hashes in its own manifest, or save full note residuals. The separate state-programme capture's recently repaired guards do not automatically protect this Chief script.

## Access consciousness: the useful next distinctions

The Director's functional question can be pursued by distinguishing (a) content an external researcher can decode, (b) content the model can report or use flexibly itself, and (c) a self-model of what it knows, attends to or can control. These are different targets; a word such as “aware” appearing in a lens is not a test of the latter two.

The current capture gives a broad observational screen of (a), and properly implemented carrier interventions add restricted evidence about information use. To establish the agentic version of (b), intervene on a candidate content representation while holding the task fixed, then test coherent changes in executable tool calls and arguments, transfer to other task families, and report of the altered content when queried. Preserve fluency and unrelated capabilities under structurally matched controls. To test (c), compare a model's report of an experimentally changed internal state with that hidden change, ruling out answers derivable from the visible transcript.

For example, an internally represented “which file still needs checking” should steer the selected file across different tasks, rather than merely increase the token probability of “check.” That would be evidence about access to a usable content representation. If the model can also report the induced change without being told it occurred, that addresses access to its own state. These are proposed next measurements, not findings from the running jobs.

Anthropic's [workspace paper](https://transformer-circuits.pub/2026/workspace/index.html) makes functional reportability, directed modulation, internal reasoning, flexible generalisation and selectivity the targets, supported by interventions as well as readouts. Adopting access consciousness as the philosophical interpretation still leaves those operational tests to be replicated here. The current jobs produce instruments and candidate locations for them; completion alone does not complete that replication.

## Verification and unexecuted work

Executed: eight source hashes checked; 16 producer-AST mask cases; a nonempty-boundary negative control; standard-library-only execution. Unexecuted: real-model mask validation, capture integrity, token alignment, result analysis, fit-width equivalence, merged-population verification, behavioural steering and consciousness-related hypotheses. The existing jobs were neither changed nor interrupted.
