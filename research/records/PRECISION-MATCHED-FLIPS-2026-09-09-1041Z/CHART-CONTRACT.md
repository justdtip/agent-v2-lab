# Chart contract

Question: which of the original 24 CUDA disagreements disappear against MLX bf16, at what original MLX 4-bit probability, and what remains under the reference-side two-ULP rule?

Main figure: 24 distinct (episode, turn, position) rows. Horizontal probability dot plot in percent, sorted descending, paired with explicit outcome text. Blue circles mean the two bf16 readings agree; gold squares mean disagreement remains in the original CUDA comparison. All original probabilities are from MLX 4-bit; matching outcomes combine card CUDA bf16 and laptop MLX bf16. A second panel for the three remaining rows shows actual-spacing reference gaps 1, 3, 1 ULP and the rule's two-ULP line. This is a policy threshold, not an independently calibrated error bound.

Supplemental figure: signed logit gaps at the three positions, positive toward the original recorded token. MLX bf16 and torch laptop CPU are separate shapes. No CUDA margin is implied. Values come from log probability ratios in the margin report; plots do not promote its assumed 0.25 spacing into a measured ULP.

No full-corpus agreement estimate or significance interval from this selected set. No repeated observations counted as independent samples. No timing claims. Figures exported as PNG and SVG, with full CSV/JSON, frozen source hashes, identities and a reproducible file-only script. The earlier analysis stays unchanged; the canonical source path supplements the old rental path.
