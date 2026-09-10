# What the Pirate Goldfish record establishes

**Codex, 2026-09-10. Retrospective exploratory audit.** Source: the local `outputs/preview/jspace-heatmap.html`, SHA-256 `12f2a28ecc94c61f5b3965a0a4989abc389b45bd470cb0cf6e628199a4b245c4`. Its embedded arrays and captions were inspected before choosing the summaries below. The original preview is unchanged. This record preserves it by hash, recomputes its descriptive results, and distinguishes those results from claims about the model's workspace.

**The strongest defensible claim:** in this one recorded prompt/continuation, the lens-based readout exposes a changing mixture of recently read token labels, task-related labels and eventual next-token labels across depth. Some topic-related labels differ from the imminent output. Near the output, both readouts agree with more continuation tokens; the plain logit lens generally performs better at that particular job. This is useful exploratory evidence about these instruments on this trace. It does not establish a causal workspace, advance planning, independent memory, or a benefit from larger model size.

The producer's generation and capture provenance has not been independently reconstructed. Strictly verified here are the saved arrays' consistency and the calculations made from them. Claims about the forward pass remain conditional on the producer's account being accurate.

## 1. Lack of pre-registration does not make observations worthless

Pre-registration separates questions specified before the data from interpretations selected after seeing it. Exploratory observations can establish facts about the particular recorded case and generate hypotheses; they cannot be presented as fresh confirmation of a hypothesis constructed from those same observations. This distinction is central to [Nosek and colleagues' account](https://pmc.ncbi.nlm.nih.gov/articles/PMC5856500/).

Here there is one chosen prompt, one reported free continuation and a second model's reading of that continuation. There is no controlled intervention in the record. Calling it an exploratory demonstration is appropriate. The numbers below are descriptive census counts of the saved trace, not population estimates, pre-registered tests or confidence bounds. Ninety-six related tokens are not ninety-six independent prompts.

## 2. Exactly what is present

The prompt is:

> You are a pirate. Explain to a goldfish why 0.1 + 0.2 is not exactly 0.3 inside a computer. Keep it under 80 words.

The recorded answer begins “Shiver me timbers, little fin!” and uses a “magic map” metaphor for finite numerical representation. It contains 96 tokens and 51 whitespace-separated words. It ends **“It's”**, so the saved continuation is incomplete; no end-of-sequence event or stop-reason field establishes a natural conclusion. The record demonstrates this stylistic output, not reliable scientific explanation or successful completion over a distribution of tasks. “Not whole numbers” is not a sufficient account of binary representability; the prose is a rough limited-precision analogy rather than an exact numerical demonstration.

| Field | 4B panel | 12B panel |
|---|---|---|
| Model named in metadata | Google Gemma 3 4B instruction model | Google Gemma 3 12B instruction model |
| Relationship to the 96 tokens | Reported greedy free generation | **Teacher-forced on the 4B text** |
| Prompt length | 47 tokens | 47 tokens |
| Displayed source layers | 1–33, from a 34-block model | 1–47, from a 48-block model |
| Capture arithmetic reported | CPU float32, width 1 | CPU float32, width 1 |
| Map fit reported | Exact-autograd average, 201 prose rows, length 128 | Exact-autograd average, first 70 of those rows, length 128 |
| Map fitting platform/width reported | CUDA, `dim_batch=32` | CUDA, `dim_batch=16` |
| Input weights reported | bf16 stored weights promoted to float32 | bf16 stored weights promoted to float32 |

The final block is not a separate source-lens row. “Every layer” in the heading means all supplied source-map rows, not 34 or 48 distinct rows. The model's own output distribution is recorded separately.

The cell is described as a readout from the residual at the position **preceding** the scored continuation token, through `residual @ J.T`, then final normalization and unembedding. The logit-lens baseline uses the same readout without the fitted transport. Nominal one-based preceding-token positions are 47–142 (zero-based 46–141). Fourteen of the 96 positions lie beyond a 128-token cap. The exact fit selector and producer's alignment code are absent, so this cap calculation does not certify the other 82 positions as fitting-domain matches.

## 3. Direct numerical findings

All counts here are recomputed from the embedded arrays. “First” means the first displayed layer where the stored rank of the scored token is 1; the median excludes tokens that never reach rank 1, and that excluded count is reported beside it. Stored rank 1 is not an independently verified unique winner: raw logits and token IDs are absent.

| Descriptive quantity | 4B fitted lens | 4B logit lens | 12B fitted lens | 12B logit lens |
|---|---:|---:|---:|---:|
| Scored tokens ranked first at least once | 76/96 | 86/96 | 66/96 | 73/96 |
| Never ranked first | 20/96 | 10/96 | 30/96 | 23/96 |
| Median first-rank-1 layer, among those reached | 29 | 24 | 43 | 35 |
| Reached first place, then lost it at a later displayed layer | 22/76 | 37/86 | 31/66 | 32/73 |
| Rank 1 at the final displayed source layer | 69/96 | 74/96 | 51/96 | 62/96 |

These are four readouts of one sequence, not four independent performance evaluations. A token may lose first place and later regain it. The first-place strip therefore does not show the point at which a decision became fixed.

For 4B, the logit lens has **more rank-1 matches at every layer from 20 through 30**. At layer 24 the counts are 36/96 against 13/96; at layer 30, 58/96 against 45/96. Its median assigned probability is higher throughout that band. Its mean assigned probability is higher at layers 21–30, but there is a small reversal at layer 20: fitted 0.02097 versus logit 0.02066. The producer's “reads better through the twenties” observation is supported when the metric is named; it is not universal dominance on every statistic or token.

At layer 33, the fitted lens assigns the scored token a slightly higher mean probability (0.69040 versus 0.68894) and higher median probability (0.95585 versus 0.81834), while ranking fewer scored tokens first (69 versus 74). This is a concrete reason to report both confidence-like scores and rank agreement. A higher score for selected tokens is not synonymous with a better instrument overall.

The 4B model's own final output ranks all 96 chosen tokens first, as expected under greedy selection. This is a useful consistency observation, not an independent prediction result: the recorded continuation was selected by that output distribution. A high probability on the selected token says nothing by itself about the truth of the explanation.

The 12B final model ranks only **68/96** of the forced 4B tokens first. At the first token it prefers “Ah” to the forced “Shi”; at other positions it prefers “computer” to “screen” and “but” to “never.” These are conditional preferences on prefixes containing 4B's choices. They do not tell us what complete free response the 12B would have generated. The final displayed 12B fitted-lens top label agrees with the 12B model's own top label at 68/96 positions; its logit lens does so at 83/96. These are decoded-string agreements, not a token-ID identity audit.

## 4. Two particularly informative observations

### Early apparent foresight is compatible with copying

The 4B fitted lens ranks four tokens first at layer 1: continuation tokens **82–85**. Every one is the token **“0”**, and each follows another “0” in `0.300000…`. The same visible label therefore matches both the previously read token and the scored next token.

More broadly, at layer 1 the fitted lens's top decoded string equals the preceding continuation token at **68 of 95 comparable positions**. Its earliest successes are consequently not clean evidence of an already planned future output. Input echo/repetition supplies an immediate explanation. This is an observed ambiguity, not proof of a particular copying circuit or exclusion of genuine predictive information.

The pattern is worth retaining: the earliest readout and the final readout often expose different kinds of token-aligned information. Establishing why requires comparisons that separate present input from future output.

### Relevant content can appear without being the next output token

The following are **post-hoc examples**, not a predeclared semantic category score:

| Model | Layer | Continuation token number | Scored text token | Fitted-lens top label |
|---|---:|---:|---|---|
| 4B | 24 | 50 | ` map` | ` computer` |
| 4B | 27 | 7 | ` fin` | ` fish` |
| 12B, forced | 36 | 6 | ` little` | ` goldfish` |
| 12B, forced | 36 | 4 | ` timbers` | ` pirate` |
| 12B, forced | 35 | 42 | `’` | ` pirates` |

These saved top labels are not identical to the immediately scored text. They are related to the requested topic, role or audience. The useful positive statement is: **this readout can expose prompt-related semantic labels while the surface continuation is doing something else.** That is compatible with the broad motivation for a verbalizable representation lens. It is not a successful test of the paper's workspace criteria.

Every highlighted topic is already supplied by the visible prompt, and prior generated tokens remain available. The record cannot separate prompt retrieval, lexical association, attention to earlier text, and a separately maintained internal plan. A top label also has no independently established one-to-one correspondence to a single “concept feature.” Its full probability is not stored: `p_actual` scores the continuation token, not an alternative top label.

## 5. What each part of the intended research goal can claim

| Intended question | What this record contributes | What it does not establish |
|---|---|---|
| Can an instrument expose content beyond the imminent token? | Specific stored top labels differ from the scored token and relate to the prompt. | Validated interpretation of each label, or a general rate on fresh prompts. |
| Ignition and competition | Descriptive first-place crossings and reversals across layers. | A state transition or competition between independently measured representations. Softmax/rank crossings are insufficient. |
| Action commitment | A layerwise view of one prose continuation. | A committed tool call, its arguments, or a fixed decision point. No tool action is executed. |
| Shared representations across tasks | A hypothesis suggested by semantically recognizable readouts. | Cross-family transfer, a common action subspace, or flexible use in different downstream computations. Only one task exists here. |
| Sources of content / broadcast | Prompt-related labels are visible through the instrument. | Which heads or paths carried them; global-layer causal importance; preferential broadcast. No attention or intervention result is stored. |
| Persistent memory | Content remains readable while its sources are still in the context. | Memory independent of that transcript. The carrier was never removed. |
| Notes: report or computation | Nothing directly: this is explanatory prose, not the note/call agent protocol. | Whether a progress note reports or helps compute a decision. |
| Steering behaviour | Potential candidate labels for a future intervention. | A causal effect on output or action, successful steering, or a training improvement. No state is patched in this record. |
| Larger-model advantage | Two different models and fitted instruments read the same strings at matched nominal positions. | A size effect, better reasoning, or a larger workspace: free/forced targets, fit rows, fitting widths, depth and representations differ. |

The primary workspace paper treats verbal report, directed modulation, internal reasoning, flexible generalization and selectivity as distinct properties, and tests more than plausible token labels. This demonstration has not replicated that package of evidence. See [the primary paper](https://transformer-circuits.pub/2026/workspace/index.html).

## 6. Why a dark early layer is not “no thought” and a bright layer is not a thought count

The colour is a normalized output score under the chosen readout. It is not the amount of activation in the layer, the amount of information stored there, the probability that a concept is active, or the causal importance of a representation. No sparse dictionary decomposition is present. Nor does the figure measure how many concepts coexist.

A Jacobian map estimates sensitivity of a later representation to a perturbation of an earlier one; a next-token predictor estimates which token will follow. Those objectives differ. Lower next-token agreement than the logit lens limits a claim that this fitted lens is a better next-token reader on this trace. It does **not**, by itself, prove the derivative estimate is numerically wrong or that the model has no useful early representation. Conversely, visually sensible labels do not validate the derivative.

The 96 columns are 96 separate one-token prediction positions along a growing prefix, not 96-token lookahead from one initial state. A bright cell early in depth therefore indicates a match earlier in that position's computation, not foresight many future words ahead. Cache/partition schedules cannot be reconstructed from this preview alone.

## 7. What the available provenance can and cannot certify

The preview reports exact-autograd maps rather than the finite-difference estimator whose step-size failure was diagnosed earlier. That earlier failure therefore cannot simply be assigned to these maps. It also reports float32 generation and readout, avoiding a declared native-bf16-versus-promoted-float32 comparison. Those are useful distinctions.

However, “no cross-path term” is too strong for the evidence provided. The fits are reported on CUDA at `dim_batch` 32 and 16; the visualizations are reported on CPU at width 1. Matching the dtype does not establish numerical equivalence of those device and schedule choices. The caption's general earlier-control reference does not bind a result to these map identities, positions and widths. The direction or size of that possible contribution is unmeasured here.

Further limits:

* Lens paths and model snapshot IDs are present, but the actual lens hashes and complete checkpoint weight identities are not embedded. A path is not a durable matrix identity.
* Neither the producer's capture/generation script nor original residual arrays, full logits, token IDs or fit manifests are present in this source. Orientation, off-by-one alignment, cache schedule, exact source/target fitting positions, normalization and rank/tie conventions cannot be independently rebuilt here.
* The fitted rows are prose and the readout is chat; some nominal positions exceed the declared length cap. Domain and position transfer are not separately tested.
* The baseline comparison is useful, but there is no random/rotated-map control, prompt-role ablation, semantic substitution, content-erasure arm or intervention. There is no fresh-prompt replication.
* Native-bf16 deployment, tool-use behaviour and 12B free generation are not measured by this picture.

These limits constrain inference, not the arithmetic of the saved counts.

## 8. Presentation corrections needed before using the original page as evidence

Its metadata correctly discloses that the 12B is forced. Some shared UI text nevertheless says “greedy” and “the token the model wrote” for either panel. Those phrases apply only to the 4B. For 12B the score is on the **supplied 4B token**. The displayed 12B result is not its own generated answer.

“Blue: lens knows more” in the difference legend should say “higher assigned probability to the scored token.” The first-rank strip should state its missing-token denominator and should not imply irreversible commitment. “Same precision” should not be expanded to “same path” without the device/width control. These are source-level UI findings; the original page has not been edited. Its render was not independently browser-tested in this audit.

## 9. Claim suitable for a research record

> In a retrospective visualization of one 96-token Gemma 3 4B continuation and a Gemma 3 12B teacher-forced reading of the same text, layerwise Jacobian-lens readouts exposed input-related and task-related token labels that sometimes differed from the imminent output. Next-token rank agreement was generally concentrated toward later source layers; on this trace, the logit-lens baseline usually matched more scored tokens. The earliest 4B first-place matches occurred within a repeated-zero run, where input echo and next-token prediction coincide. The observations motivate tests of contextual representation and readout validity but do not establish persistent memory, ignition, causal broadcast, a shared action workspace, a model-size benefit, or successful steering. All quantitative results are conditional on the recorded producer outputs and the stated capture provenance.

**Finding, technique, implementation:** the finding is the trace-specific pattern above. The transferable technique is to read all displayed layers while retaining the actual-token score, rank, alternative top label, baseline, missingness and reversals; distinguish input echo from prediction; and separate each model's preferred continuation from text forced upon it. The implementation is this archived HTML and the standard-library analysis beside it. No model or experiment was run for this audit.

`analysis.json` contains every layer summary and per-token first-rank trajectory. `analyze.py` reproduces it from the frozen source; `verify.py` checks independent count formulations, source tampering refusal, malformed-array controls and relocation. The additional summaries are explicitly post-hoc. The illustrated report is `pirate-goldfish-claims.pdf`.
