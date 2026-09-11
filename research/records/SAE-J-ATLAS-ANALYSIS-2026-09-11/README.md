# The atlas, read: expert, model and lens at four decisions

**Chief, 2026-09-11.** First analysis of the SAE/J atlas after the model's own prediction was added beside the expert's token and the lens's readout (`SAE-J-ATLAS-MODEL-PREDICTIONS-2026-09-11`, bundle `63a81a62790f7db6…`). Script [analyse_atlas.py](analyse_atlas.py), output [analysis.json](analysis.json), source bundle kept as [atlas-source.json](atlas-source.json).

**Sixteen cells: four episodes, two positions, two layers.** This is a pilot. Nothing here is a population estimate, and the A2 error budget refused a feature ranking on all sixteen, so every contribution below is a **diagnostic** quantity, as the bridge order requires. What is not in doubt is the arithmetic: the decomposition identity holds to 1e-12 per cell, so "this feature supplies 42% of the gap" is exact. Reading that feature as *the read_file feature* is an interpretation of the arithmetic and is not established by it.

## 1. Why the model's own prediction had to be added

Until this export the atlas carried the expert corpus's next token and not the model's. The two are different: the corpus is teacher-forced expert demonstration, so the model is fed the expert's text whatever it would have chosen. With both present the question separates cleanly, and the answer is that **the model disagrees with the expert in two of the four action cells** — it picks `replace_text` where the expert wrote `read_file`, and `read_file` where the expert wrote `replace_text`. Any comparison of the lens against the expert would have scored those as lens failures. They are not.

## 2. Agreement, by site

| site | cells | lens argmax = model | lens above the 0.001 mass floor | lens ranks the model's pick first | feature share of the gap (median) | error share | raw reconstruction error |
|---|---:|---:|---:|---:|---:|---:|---:|
| L18/P_note | 4 | 0 of 4 | 0 | 0 of 4 | +0.73 | -0.09 | 0.066 |
| L18/P_act | 4 | 1 of 4 | 0 | 1 of 4 | +0.46 | +0.38 | 0.067 |
| L24/P_note | 4 | 1 of 4 | 0 | 2 of 4 | +1.22 | -0.16 | 0.123 |
| L24/P_act | 4 | 4 of 4 | 1 | 4 of 4 | +0.87 | +0.12 | 0.134 |

**At layer 24 at the action position — the one site the domain statement licenses — the lens's six-tool argmax equals the model's in four of four cells, including both cells where the model departs from the expert.** That is the strongest evidence so far that the lens reads the model rather than the corpus, and it could not have been seen before this export. The honest qualification is next to it: only one of the four clears W-5's 0.001 mass floor, so by the recorded resolution rule three are unresolved, and argmax agreement below the floor is suggestive rather than established.

Away from that site it degrades exactly as the domain statement says it should. At layer 18 at the action position the lens ranks the model's own pick first in one cell of four. At either note position it never does, and the model's six-tool mass is 10⁻⁹ or smaller because the model is not choosing a tool there at all: the expert token at those cells is `Plan`, `Ins`, `Applied`, `Replacement` — prose, mid-note. The six-tool renormalisation is meaningless at the note position, and the model's six-tool argmax there is `read_file` in all four cells, which is the prior showing through rather than a decision.

## 3. What carries the tool decision at the licensed site

For each cell, the lens score gap between the tool the **model** chose and its own runner-up, decomposed into the feature sum and the dictionary's error:

| episode | expert | model | lens | gap | features / gap | error / gap | top feature | its share | top1 / top3 of the feature separation |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| train-pointer_chain-0150-clean | read_file | read_file | read_file | 12,285 | +1.00 | +0.07 | #10084 | 42% | 42% / 74% |
| train-batch_update-0466-unknown_ | read_file | replace_text | replace_text | 14,901 | +0.99 | +0.07 | #3461 | 40% | 40% / 76% |
| train-batch_update-0010-clean | replace_text | read_file | read_file | 11,769 | +0.74 | +0.17 | #10084 | 39% | 52% / 85% |
| train-update-0172-failed_edit | read_file | read_file | read_file | 10,957 | +0.50 | +0.41 | #10084 | 38% | 76% / 119% |

The features supply between half and all of the gap and the error supplies little. One feature carries 38 to 42 per cent of the whole gap on its own, and three features carry three quarters of the feature separation.

## 4. The same feature, across three task families

| cell | model picks | activation and per-tool contribution |
|---|---|---|
| train-pointer_chain-0150-clean | read_file | 922 | read +5,725 fini +623 repl +414 calc +106 sear -337 list +604 |
| train-batch_update-0466-unknown_ | replace_text | not active |
| train-batch_update-0010-clean | read_file | 792 | read +4,917 fini +535 repl +356 calc +91 sear -289 list +519 |
| train-update-0172-failed_edit | read_file | 717 | read +4,456 fini +485 repl +322 calc +83 sear -262 list +470 |

Feature 10084 is active in exactly the three cells where the model chooses `read_file`, absent from the one where it chooses `replace_text`, pushes toward `read_file` by an order of magnitude more than toward anything else, and pushes **against** `search_files`. The three episodes are from three different task families (`pointer_chain`, `batch_update`, `update`), so it is not a family artefact.

| cell | model picks | activation and per-tool contribution |
|---|---|---|
| train-pointer_chain-0150-clean | read_file | not active |
| train-batch_update-0466-unknown_ | replace_text | 830 | read -15 fini -298 repl +5,892 calc +22 sear +88 list +229 |
| train-batch_update-0010-clean | read_file | not active |
| train-update-0172-failed_edit | read_file | not active |

Feature 3461 is its complement: active only in the `replace_text` cell, pushing toward `replace_text` and slightly against `read_file` and `finish`.

## 5. What this does and does not license

It licenses the next experiment and nothing more. Four cells, one layer, one position, one model. The A2 budget refused ranking on every one, the lens mass clears the floor on one, and the whole account is correlational: feature 10084 being active and pushing toward `read_file` does not show that changing it changes what the model does. That is a behavioural question and it is the one worth asking next — hold the unexplained residual fixed, move the feature, and read the completed tool call.

There are also **no labels**. The published Neuronpedia labels index `resid_post/layer_17_width_16k_l0_medium`, a different site and sparsity from the `resid_post_all` `l0_small` dictionaries here; feature indices do not transfer between training runs. Nothing in this record may be presented as a published label, and the description "the read_file feature" is ours, not anyone's.
