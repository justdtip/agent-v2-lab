# Moving one feature: a dose–response test of the tool decision

**Chief, 2026-09-11.** The first test in this programme that can make a causal statement about an SAE feature. Script [feature_intervention.py](feature_intervention.py); outputs [feature-dose-l24.json](feature-dose-l24.json), [feature-dose-l18.json](feature-dose-l18.json), and the inert first pilot [feature-l24.json](feature-l24.json).

## 1. The construction

The dictionary writes an activation as `h = b + Σ_i z_i d_i + e`. Removing feature `k` means setting `h' = h − z_k d_k`, which changes the reconstruction and **leaves the unexplained error `e` exactly as it was**: the new vector's distance from the dictionary's span is the same vector `e` it always was. The model is then run from `h'` and the **whole tool call is generated greedily**, forty-eight tokens, so the outcome is a parsed call rather than a next token.

`k` is the active feature whose contribution most separates the tool the **model itself** chose from the model's own runner-up, recomputed per cell from the capture rather than read from a bundle.

Five arms at every rung of a multiplier ladder, all at the same magnitude:

| arm | what it does | what it controls for |
|---|---|---|
| knockout | `h − m·z_k d_k` | the claim |
| **amplify** | `h + m·z_k d_k` | **the same direction and magnitude with the opposite sign** |
| sham | `h − m·z_j d_j` for an active feature `j` the decomposition says is not carrying this decision | a feature of similar size that should not matter |
| random | `h + r`, `‖r‖` matched | any perturbation of that size |
| angle-matched | `‖·‖` and signed projection on `h` matched | a perturbation of that size *and geometry* |

Cells are the 32 lowest first-token margins among the 300 action positions (1.2 to 9.8 logits), because the whole-population run showed that decisions with a margin do not move for anything. The same-state write-back changes nothing in any run.

## 2. Layer 24, the site the domain statement licenses

| multiple | moved, share of ‖h‖ | knockout | amplify | sham | random | angle-matched | knockout vs pooled controls | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x1 | 2.3% | 1 of 32 (3%) | 0 of 32 | 0 of 32 | 0 of 64 | 0 of 64 | 31250000.0× | 0.1667 |
| x2 | 4.6% | 5 of 32 (16%) | 0 of 32 | 1 of 32 | 1 of 64 | 0 of 64 | 12.5× | 0.0016 |
| x4 | 9.2% | 9 of 32 (28%) | 0 of 32 | 1 of 32 | 2 of 64 | 1 of 64 | 11.2× | <0.0001 |
| x8 | 18.4% | 20 of 32 (62%) | 0 of 32 | 9 of 32 | 10 of 64 | 8 of 64 | 3.7× | <0.0001 |
| x16 | 36.8% | 30 of 32 (94%) | 0 of 32 | 22 of 32 | 43 of 64 | 41 of 64 | 1.4× | 0.0007 |

**There is a window, and in it the feature is specific.** At twice and four times its own contribution the knockout changes the model's tool in 16% and 28% of cells against 1% and 2% for the pooled matched controls — eleven to twelve times the rate, p = 0.0016 and below 0.0001. Below that window nothing moves at all; above it everything moves, and by sixteen times the ratio has fallen to 1.4 because a 36% displacement breaks the computation whatever its direction.

**The sign control is the strongest evidence here.** At sixteen times, removing the feature changes the tool in 30 of 32 cells and *adding the same vector at the same magnitude changes it in none*. Same direction, same size, opposite sign, opposite outcome, at every rung of the ladder. No account in terms of "a perturbation of that magnitude disrupts the model" survives that, because the amplified arm is exactly such a perturbation and it disrupts nothing.

## 3. Layer 18, where the lens does not track the model

| multiple | moved, share of ‖h‖ | knockout | amplify | sham | random | angle-matched | knockout vs pooled controls | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x1 | 1.1% | 0 of 32 (0%) | 4 of 32 | 0 of 32 | 1 of 64 | 1 of 64 | 0.0× | 1.0000 |
| x2 | 2.2% | 2 of 32 (6%) | 5 of 32 | 1 of 32 | 2 of 64 | 1 of 64 | 2.5× | 0.2621 |
| x4 | 4.4% | 7 of 32 (22%) | 6 of 32 | 6 of 32 | 7 of 64 | 6 of 64 | 1.8× | 0.1130 |
| x8 | 8.8% | 11 of 32 (34%) | 8 of 32 | 3 of 32 | 18 of 64 | 15 of 64 | 1.5× | 0.1165 |
| x16 | 17.5% | 28 of 32 (88%) | 13 of 32 | 16 of 32 | 41 of 64 | 38 of 64 | 1.5× | 0.0014 |

The separation is weak and only reaches significance at the largest displacement, where the ratio is 1.5 and the amplified arm also changes 13 of 32 — the signature of a perturbation that is simply too big rather than of a feature being removed. That is the same site-dependence as everything else in this programme: the decomposition means something where the lens is licensed and does not where it is not.

## 4. What it does and does not establish

**It does not establish that the feature as it actually fires decides the tool.** At its own magnitude, one times, the knockout changes 1 cell of 32 and the controls change none; that is not a result. You have to remove two to four times what is there. So the honest statement is about direction, not sufficiency: *the axis the decomposition names is the axis along which the decision is sensitive, and it is sensitive in one direction only.* Whether the feature is necessary at its own size is untested and, on this evidence, probably false.

**It is 32 cells, one layer, one model, chosen for small margins**, which is where the whole-population run said any effect would have to live. It is not a population estimate.

**The first pilot was inert and is kept.** It moved 2.4% of the residual norm at layer 24 on cells not chosen for margin — a fifth of a reconstruction error already shown harmless over 300 cells — and produced zero changes in all six arms. That null measured the design, not the model; it is recorded because the corrected run is only interpretable beside it.

**No labels.** The features are indices in a dictionary whose published labels index a different training run, and nothing here may be read as a published label.
