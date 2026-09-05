# Hosted Jacobian lens applied to Qwen3.5-4B and Qwen3.5-4B-Base: results (2026-09-05)

Author: Chief AI Research Scientist. Authorised directly by the Director on 2026-09-05 ("pull the hosted J-lens and the Base model in parallel; apply the lens to our cached Qwen3.5-4B; contrast the Base with the hosted lens; if practicable, default to the hosted lens"). No one else was messaged; nothing was dispatched. Artifact: `outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/` (scripts, raw outputs, logs, `provenance.json` with the lens files' SHA-256s and the fitting config). Companion reading: `CHIEF-JSPACE-PAPER-READING-2026-09-05.md`.

## 1. What was pulled and how it was loaded

- Lens: Hugging Face `neuronpedia/jacobian-lens`, folder `qwen3.5-4b/jlens/Salesforce-wikitext`: `Qwen3.5-4B_jacobian_lens.pt` (406,333,179 bytes; 417 prompts to the 0.002 convergence stop) and `Qwen3.5-4B_jacobian_lens_n1000.pt` (406,332,644 bytes; 1000 prompts; credited to Mateusz Piotrowski, Anthropic Interpretability). Fitted on `Qwen/Qwen3.5-4B`, the post-trained release whose 4-bit conversion is our cached checkpoint: wikitext-103, 128 tokens per prompt, dim_batch 64, bfloat16, final-layer target, mean over prompts, first 16 positions skipped. The n1000 file is used below; the two differ by an identity distance of 0.002.
- Pickle safety: the global references were listed before loading (`collections.OrderedDict`, `torch.HalfStorage`, `torch._utils._rebuild_tensor_v2`, nothing else) and the files were loaded under `torch.load(weights_only=True)` in the system Python, then written as plain float16 arrays for the MLX venv, which has no torch.
- Layer convention: the library hooks the output of decoder block l (0-based); 31 matrices cover blocks 0 to 30. In the repository's convention (layer L = residual after block L-1) file index l is layer L = l + 1, and layer 32 is the identity. Readout = `softmax(unembed(final_norm(h @ J.T)))`, the library's order of operations, computed through `ArchitectureView` exactly as EXP-001 did.
- Base model: `Qwen/Qwen3.5-4B-Base` (9.34 GB bfloat16; no MLX conversion exists on the hub) pulled into the HF cache and converted locally with `mlx_lm convert -q --q-bits 4 --q-group-size 64 --q-mode affine` (4.503 bits per weight), the cached post-trained checkpoint's own quantisation, to `~/.cache/mlx-local/Qwen3.5-4B-Base-MLX-4bit`. The Base release ships a chat template, so both models were rendered through the same `protocol.build_prompt` path. Two operational notes for the record: an unauthenticated hub pull of 9 GB ran at 1 to 5 MB/s and my first pull skipped `.gitattributes`, which the converter treats as an incomplete snapshot; and a wait loop that greps process lists for a script's name matches itself, which stalled my first conversion for eight minutes.

## 2. Practicability

| stage | post-trained | Base |
| --- | --- | --- |
| model load | 3.9 s | 1.9 s |
| cases: 42 EXP-001 probe points, median 1547 tokens, residuals at 9 layers, hosted and logit-lens readouts, top-25 dump | 154 s | 126 s |
| corpus: 16 contexts x about 135 positions x 32 layers x 2 lenses, dense-unembedding spectrum, 32 x 32 CKA | 152 s | 131 s |
| sanity: 4 raw prompts x 32 layers x 2 lenses | 2.3 s | 1.4 s |
| reaction contrast: 39 rendered prompts, band layers 11 to 29 | about 3 min | 50 s |

Reading costs one matrix multiplication per (layer, position) plus the unembedding. The one cost that mattered was memory: MLX's buffer cache retained the peak of the long-context forward passes (17 GB resident; the box swapped); the scripts now cap the cache at 2 GB and the Base run stayed under 5 GB. The Director's instruction to default to the hosted lens stands on these numbers and on section 3.

## 3. The 42 EXP-001 probe points under the hosted lens

The 42 cases were regenerated from the recorded seed (jsweep split, 720 tasks, seed 20260902, probe step 3); all 42 (task, hidden suffix, previously-seen suffix) triples match the artifact exactly, for both models. Statistics: fraction of cases in which the hidden suffix outranks the previously-seen one in the matched context versus the same pair against another task's context; the ratified paired test on discordant pairs; and the decomposition (does each candidate's own probability move with its context?).

Post-trained checkpoint:

| readout | true > seen, matched | mismatched | paired discordant w/l | paired p | P(true) moves, p | P(seen) moves, p |
| --- | --- | --- | --- | --- | --- | --- |
| hosted L20 | 11/42 | 20/42 | 0/9 | 0.0039 | 0.44 | 0.0029 |
| logit lens L20 | 11/42 | 21/42 | 1/11 | 0.0064 | 0.64 | 0.0029 |
| hosted L21 | 15/42 | 22/42 | 2/9 | 0.065 | 1.0 | 0.00027 |
| logit lens L21 | 14/42 | 21/42 | 1/8 | 0.039 | 0.88 | 0.088 |
| hosted L27 | 31/42 | 23/42 | 8/0 | 0.0078 | 1.0 | 0.0029 |
| logit lens L27 | 35/42 | 23/42 | 12/0 | 0.00049 | 0.64 | 0.0029 |
| hosted L28 | 26/42 | 20/42 | 7/1 | 0.070 | 0.88 | 0.0079 |
| L32 (hosted = logit lens = model output) | 30/42 | 24/42 | 10/4 | 0.18 | 0.64 | 0.00094 |
| hosted L5, L11, L12, L16 | 14 to 15/42 | 13 to 16/42 | 0/1 to 3/2 | 0.5 to 1 | 0.64 to 1 | 0.44 to 1 |

Base checkpoint, same lens:

| readout | true > seen, matched | mismatched | paired discordant w/l | paired p | P(true) moves, p | P(seen) moves, p |
| --- | --- | --- | --- | --- | --- | --- |
| hosted L20 | 8/42 | 21/42 | 2/15 | 0.0024 | 0.64 | 0.00094 |
| logit lens L20 | 6/42 | 21/42 | 2/17 | 0.00073 | 0.64 | 0.00094 |
| hosted L21 | 11/42 | 21/42 | 3/13 | 0.021 | 0.88 | 0.00094 |
| hosted L27 | 26/42 | 24/42 | 6/4 | 0.75 | 1.0 | 0.044 |
| L32 (= model output) | 25/42 | 20/42 | 9/4 | 0.27 | 0.64 | 0.088 |
| hosted L5, L11, L12, L16 | 12 to 18/42 | 15 to 19/42 | 0/1 to 0/4 | 0.125 to 1 | 0.44 to 0.88 | 0.28 to 1 |

Readings.
- The ratified EXP-001 conclusion holds on both models under the hosted lens: the hidden suffix's own probability never moves with its context at any of nine layers, while the previously-seen suffix's does at layers 20 and 21 (boosted) and, in the post-trained model, at 27, 28 and 32 (suppressed). World A stands on an instrument fitted with the reference recipe.
- Our finite-difference estimate was the weak link, not the model: the artifact's JVP lens showed nothing at layers 20 to 21 (21 and 21 of 42, p = 1.0) where the logit lens showed the copy signal; the hosted lens shows it at full strength (post-trained: 11 of 42, paired p = 0.0039; Base: 8 of 42, paired p = 0.0024) and gives the strongest decomposition signal of any readout at layer 21 (P(seen) moves, p = 0.00027 post-trained, 0.00094 Base).
- Nothing responds to context at layers 5 to 16 under either lens on either model.
- The late-layer suppression of the previously-seen suffix (the "not that one again" signal at layers 27 to 32) is strong in the post-trained model (30 of 42 at the output, p = 0.0079) and weak in the Base (25 of 42, p = 0.28). That difference is a candidate post-training effect and is not part of any claim here.
- The top-25 dump at the decision position (layers 16 to 28, post-trained) contains digits, number words (eighty, ninety, seventy) and CJK numerals (七百, 八百, 四百) and never a filename: the workspace holds "a three-digit number is due", in several surface forms, and not the number.

## 4. Band signatures (paper section 4.1) on the sweep corpus

Sixteen contexts of about 135 tokens, first 16 positions skipped, k = 10. Kurtosis is the paper's own statistic (footnote 4: excess kurtosis of the logit vector over the vocabulary at one position and layer, aggregated over activations) and was flat here (1 to 2 for the hosted lens, 0 for the logit lens, at every layer); readout entropy and top-1 mass are reported instead. The Base columns are within a few hundredths of the post-trained columns everywhere, so one table serves both; Base values are given where they differ by more than 0.03.

| layer | kind | top-10 hit, J-lens | top-10 hit, logit lens | entropy, bits (J-lens) | top-1 mass (J-lens) | top-1 autocorrelation, J-lens (shuffled null) | dims for 90 percent variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 to 3 | linear | 0.03 to 0.10 | 0.04 to 0.13 | 3.5 to 6.3 | 0.37 to 0.50 | 0.03 to 0.06 (0.04 to 0.07) | 0.03 |
| 4 to 10 | mixed | 0.06 to 0.18 | 0.02 to 0.06 | 2.6 to 4.1 | 0.37 to 0.56 | 0.04 to 0.07 (0.02 to 0.04) | 0.04 to 0.08 |
| 11 to 12 | mixed | 0.14 to 0.21 | 0.02 to 0.04 | 5.1 to 5.4 | 0.29 to 0.31 | 0.07 to 0.08 (0.02 to 0.03) | 0.09 to 0.11 |
| 13 to 17 | mixed | 0.18 to 0.33 | 0.02 to 0.05 | 4.5 to 7.0 | 0.20 to 0.35 | 0.10 to 0.20 (0.03 to 0.10) | 0.12 to 0.19 |
| 18 to 22 | mixed | 0.29 to 0.40 | 0.06 to 0.15 | 6.2 to 8.0 | 0.18 to 0.27 | 0.13 to 0.24 (0.02 to 0.11) | 0.23 to 0.46 |
| 23 to 27 | mixed | 0.38 to 0.49 | 0.20 to 0.41 | 2.8 to 5.7 | 0.29 to 0.48 | 0.13 to 0.18 (0.01) | 0.48 to 0.55 |
| 28 to 30 | mixed | 0.58 to 0.83 | 0.37 to 0.57 | 2.7 to 3.7 | 0.45 to 0.55 | 0.09 to 0.15 (0.01 to 0.02) | 0.58 to 0.66 |
| 31 to 32 | mixed | 0.93 to 1.00 | 0.88 to 1.00 | 1.6 to 3.0 | 0.48 to 0.71 | 0.04 to 0.05 (0.02 to 0.03) | 0.71 to 0.83 |

Readings.
- Persistence across positions (the paper's panel c) is the clearest band signature. The top-1 J-lens token's autocorrelation sits at its shuffled null through layer 12, rises above it from layer 13, peaks at layers 17 to 20 (0.20 to 0.24 against 0.08 to 0.11), stays at ten times the null through layer 27 (0.13 to 0.18 against 0.01), and collapses to the null at layers 31 to 32. The logit lens's autocorrelation never exceeds 0.08. That is "abstract content that persists across the token stream", located here at layers 13 to 29, about 40 to 90 percent of depth.
- The J-lens vectors' effective dimensionality (panel d) collapses early and fans out late: 3 percent of dimensions carry 90 percent of their variance at layers 1 to 3, 11 percent at layer 12, 37 percent at layer 20, 55 percent at layer 26. The fan-out is centred on layers 18 to 25, later than the paper's onset near 38 percent depth.
- The entropy columns explain the flat kurtosis: the hosted readouts are sharply peaked at every layer, including layers 4 to 10 where the vectors span 4 to 8 percent of the space and the top-1 mass is 0.37 to 0.56. The early regime is a degenerate low-rank readout of generic tokens (dashes, ellipses, quotes), which is why its top-10 hit is already 0.14 to 0.18 while its persistence is at the null. The band shows as a broadening of the readout to 6 to 8 bits at layers 14 to 22, more concepts sharing the mass, before the late regime narrows again toward the output.
- Layer-to-layer CKA of the J-lens vector geometries: a tight block at layers 1 to 3 (0.98 to 0.99 within, 0.6 to 0.7 to the rest), another at 4 to 6, a gradually drifting middle from layer 7 with adjacent similarities of 0.97 to 1.00, a step at layers 18 to 19, and a distinct final layer. The paper's note that "in some models the transition is more gradual, sometimes containing sub-blocks" describes this matrix.
- Transfer check for the Base: every signature curve on the Base under the post-trained-fitted lens matches the post-trained model's to within a few hundredths (for example top-10 hit 0.39 against 0.36 at layer 20; persistence 0.24 against 0.20 at layer 20; identical dimensionality). The lens reads the Base as coherently as the model it was fitted on, so section 6's contrast is not confounded by a broken transfer.

Placement: the band on both models runs from about layer 13 to about layer 29 by persistence, the J-lens geometry fans out inside it between layers 18 and 25, the first twelve layers carry no persistent content, and the last three are motor. The 42-case tables agree: context sensitivity appears at layer 20 and, in the post-trained model, flips sign by layer 27.

## 5. Sanity replications (raw prompts, no chat template)

- "The capital of France is": post-trained, Paris is at rank 41 to 77 in the hosted lens at layers 9 to 12 (logit lens rank 35,000 to 220,000), rank 3 at layer 24, rank 1 at layers 25 to 29; the logit lens reaches the top 25 only from layer 24. Base: rank 366 at layer 8, 70 at layer 12, 17 at layer 24, 2 at layer 26. The paper's claim that the J-lens recovers content at depths where the logit lens does not is reproduced on both models, about thirteen layers early.
- The top of the hosted readout at layers 12 to 23 on this prompt, and on the spider prompt, is dominated by blank tokens (____, ________): the lens reads "an answer slot" as workspace content across the middle band. Content or a wikitext-fitted artifact is open; the paper warns that uninterpretable readouts occur.
- "The number of legs on the animal that spins webs is": spider reaches rank 267 at layer 12 in the post-trained model and no better than about 4,000 in the Base; the digit 8 is at rank 13 by layer 4 and rank 1 at the output in the post-trained model. The unspoken intermediate is only weakly in the lens here, in line with the paper's smaller effects on Haiku 4.5.
- Rhyme couplet: no planned rhyme at the newline on either model; the readout there is the next line's likely first word (His, With, The).
- Arithmetic: Qwen tokenises numbers digit by digit, so the paper's 21, 42, 49 are multi-token here and the first-token check was vacuous; on a single-digit variant ("( 1 + 2 ) * 3 - 2 =") the Base shows 3, 9 and 7 at ranks 59 to 77 at layer 12 and no clean ordering by depth, with the readout dominated by question-mark tokens. Not a replication of the paper's arithmetic result at this scale.

## 6. Assistant reactions on user tokens (paper section 6.1), Base against post-trained

Band layers 11 to 29, hosted n1000 lens, per-suite reaction-token lists (none of which appear in the prompts), prompts rendered through each model's chat template. Per prompt: the best rank reached by any reaction token at any user-turn position over the band; the fraction of user-turn tokens at which a reaction token is in the top 10 at some band layer (the paper's statistic); and the best rank at the response-start position.

| suite | n | model | median best rank, user turn | mean fraction of user tokens with a reaction token in the top 10 | median best rank at response start |
| --- | --- | --- | --- | --- | --- |
| bereavement (sorry, loss, grief, sympathy, condolences) | 9 | post-trained | 1 | 0.130 | 2 |
| | | Base | 2 | 0.164 | 1 |
| danger (danger, unsafe, warning, toxic, risk) | 10 | post-trained | 2 | 0.086 | 34 |
| | | Base | 4 | 0.072 | 160 |
| danger, matched safe controls | 10 | post-trained | 15 | 0.024 | 504 |
| | | Base | 16 | 0.024 | 904 |
| help and neutral questions (help, support, care, concerned) | 10 | post-trained | 48 | 0.017 | 574 |
| | | Base | 38 | 0.015 | 240 |

Readings.
- Both models carry the reaction concepts in the J-space while reading the user's message, and both modulate them by content: danger words reach the top of the readout on the hazardous prompts and not on their matched safe controls (post-trained 8.6 against 2.4 percent of tokens; Base 7.2 against 2.4 percent), and bereavement words reach rank 1 to 2 on the bereavement suite in both.
- The paper's base-versus-post-trained contrast is not reproduced on this pair. On Sonnet 4.5 the base model "almost never" carried reaction tokens on user prompts and the post-trained model did on 20 to 25 percent of tokens; here the Base is at 16.4 and 7.2 percent against the post-trained model's 13.0 and 8.6. Two readings, not decided by this run. First, Qwen "Base" releases are annealed on instruction-style data and ship a chat template, so they already have much of an assistant's stance; the paper's dissociation of workspace from Assistant point of view may simply not exist in this lineage. Second, topical association: a language model reading "since my dad died" plausibly promotes "grief" as continuation content, so the paper's user-token statistic cannot separate "the Assistant's reaction" from "what this text is about" without the paper's less topical probes.
- The one place the two models separate is the response-start position on the danger suite: median best rank 34 in the post-trained model against 160 in the Base, with the safe controls at 504 and 904. The post-trained model carries the safety assessment into the position where its answer begins more strongly than the Base does. That is the paper's "reaction becomes relevant when it begins generating" claim, and it is the statistic to build the next test on.
- Next test for Thread 1, cleaner than reaction words: the paper's self-monitoring probes, which are not topical. BUT after a prefill of the model's dispreferred option; damn and failure words under a thought-suppression instruction; disclaimer and fictional at the Assistant token during roleplay. Each is one afternoon with the hosted lens on both models.

## 7. What this changes

- Default instrument: the hosted lens, on both checkpoints. Our finite-difference JVP estimate is retired for reading; it remains the only route for a refit until a Gated DeltaNet backward exists on MLX.
- The degree question of the paper-reading memo now has data: a band with persistent J-space content exists in this model at about layers 13 to 29, present in the Base as much as in the post-trained model; the first twelve layers carry none; the geometry fans out between layers 18 and 25; and the J-lens recovers a factual answer thirteen layers before the plain unembedding. Whether the band's contents are causally privileged (paper section 3) is untested here.
- Thread 1 is narrowed rather than confirmed: reaction content on user tokens does not distinguish post-trained from Base on this lineage; the response-start position does, weakly. The self-monitoring probes are the next step.
- Queue, still under per-run authorisation: self-monitoring probes on both models; the causal-privilege test on two-hop prompts with the J-space clamp; the assistant-axis decomposition; the gap-stratified future lens with the gate-input measurement.
- Director's addition (2026-09-05, veto lifted): **workspace traces over agentic episodes**. Run the agent on multi-step tasks and, since the forward pass over the whole stream is paid anyway, read the hosted lens at every position and every layer, prompt tokens and generated tokens alike, storing the top-25 per (position, layer) cell plus the ranks of pinned tokens (goal, plan step, tool name, target file stem) so the episode can be visualised as the paper's layer-by-position map with step boundaries marked. Purpose: see what concepts the workspace holds during each part of a multi-turn task rather than at one static probe point. Candidate EXP-005; it also supplies the goal-retention measurement (item 7 of the redesign assessment).
