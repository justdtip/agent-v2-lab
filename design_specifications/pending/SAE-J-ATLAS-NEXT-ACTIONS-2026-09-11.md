# SAE–J atlas: the next concrete actions

**Chief, 2026-09-11, on the Research Director's instruction to record the sequence before proceeding.** This is the working list, in order, with what each costs and what it would settle. It supersedes nothing; the atlas work order of 2026-09-11 remains the design, and this is its execution queue. **E2's 10,000-resample interval is not on this list and is not a dependency of anything on it.**

## A. The three experiments

**A1 — analyse the atlas. DONE**, recorded at `research/records/SAE-J-ATLAS-ANALYSIS-2026-09-11` (4ad0b07). With the model's own prediction beside the expert's and the lens's: the model departs from the expert in two of four action cells; at layer 24's action position the lens argmax equals the model's in four of four, including both departures, though only one clears W-5's 0.001 mass floor; the features supply half to all of the tool score gap there, one feature carries about 40% of it, and feature 10084 is active in exactly the three `read_file` cells across three task families while 3461 is its `replace_text` complement. Four cells. Diagnostic, correlational, unlabelled.

**A2 — the behavioural pilot on the whole action population. SUBSTITUTION DONE, CALLS RUNNING.** All 300 action cells at layers 18 and 24, reconstruction against same-state, norm-matched random and angle-matched controls. Substitution result: flips are confined to the lowest margin quintile — at layer 18, 16 of 60 cells below 11.9 logits flip against 11.7% for random and 7.5% for angle-matched, and 1 flip in the 240 cells above; at layer 24 the reconstruction flips **nothing in 300** while random errors flip 1.8%. The complete-call half follows.

**A3 — the feature intervention. NEXT.** For each cell take its own top separating feature, subtract that feature's contribution from the residual and leave the dictionary's unexplained error exactly where it is, then generate the complete call. Controls: same-state write-back; a random direction of the norm removed; and a sham feature, a different active feature with a similar contribution to a tool the model did not choose. Pilot of 32 cells at layer 24's action position, about ten minutes of card. This is the first causal claim the instrument could earn; everything above it is correlational.

## B. The cheap capture over a whole action

The capture holds **two** residuals per decision, at the note position and the action position, and nothing between or after. So we can say which features were active when the tool was chosen and what the model then wrote, and not which features were active while it wrote the path argument.

**B1 — the teacher-forced version, cheap.** The corpus completion is fixed text, so one forward per decision records the residual at every token of the call the expert wrote: about 5 GB for 300 decisions across all 33 layers, hours not days. Its limit is that the action is the expert's.

**B2 — the model's own action, the honest version.** Requires recording during generation, which the capture tool does not do. Build after B1 shows whether a feature trace across an action is legible at all.

## C. Gaps identified in the last stretch, with costs

1. **The bundle records agreement as null below the mass floor.** `lens_vs_model` returns null whenever either side's raw six-tool mass is under 0.001, which is correct by W-5's resolution rule but means the file never records the four-of-four argmax agreement I had to compute myself. Add a field for argmax equality separate from resolution, labelled as below-floor evidence. Small.
2. **The atlas covers 2 of 33 layers.** Pure extraction: the exporter takes a layer list and every layer's dictionary is on the card. About ten minutes of CPU for the existing four episodes across all layers, and it turns the layer 18 against layer 24 contrast into a depth curve, which is the scientific question behind "does it support the full network".
3. **There is no dictionary browser.** Only features that fired in the selected cells exist in the bundle, so you cannot look a feature up and then ask where it fires. The per-feature top-token tables already exist in the A1 output for every one of the 16,384 features at each layer run; joining them is the highest-value single piece of work, because it turns feature 10084 from an index into something readable.
4. **One bundle does not scale.** 16 cells is 4.4 MB served as a single static file; the full population at all layers is hundreds of megabytes to gigabytes the browser must load before showing anything. Sharding by layer, or a query endpoint, before widening past a few hundred cells.
5. **No labels, and a labelled dictionary is sitting unused.** The published Neuronpedia labels index `resid_post/layer_17_width_16k_l0_medium` — a different site and sparsity from the `resid_post_all` `l0_small` dictionaries used so far, and feature indices do not transfer between training runs. That exact folder was fetched to the card on 2026-09-11 with its examples. **Run the atlas at layer 18 against the labelled dictionary**: every active feature then carries a natural-language label written by someone who never saw this corpus, which is both a reading aid and a free check on us — if the features separating `read_file` from `list_files` are labelled about files and paths, the bridge is telling the truth. One export, minutes.
6. **Behavioural tests intervene at one layer at a time.** The steering pilot showed whole-residual swaps carry the tool decision across layers 20–32; whether a single *feature* has that property at more than one layer is unknown. Run A3 at the two or three layers the depth curve of C2 picks out rather than at all of them.
7. **Deployment.** The reviewed bundle with the model's predictions is published at the card's authenticated atlas route, digests verified against the export.

## D. Order of execution

A3, then C5 (the labelled dictionary, because it makes everything after it readable), then C2 (all layers), then C1 and C3 together, then B1, then C6 at the layers the curve names, then C4 when the population grows. B2 and any widening past a few hundred cells belong to the next rental.
