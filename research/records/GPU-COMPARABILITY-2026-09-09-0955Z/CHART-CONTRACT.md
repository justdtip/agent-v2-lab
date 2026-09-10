# Chart contract, before plotting

Question: does moving the torch comparison from the card's GPU to its CPU remove
the disagreement with the same saved MLX reference?

Surface: standalone scientific figure, Matplotlib PNG and SVG, generated locally
without importing model libraries. Two aligned horizontal dot-plot panels, one row
per episode: all-position disagreement percentage, and confidence-gated mismatch
count. Both x axes start at zero. CPU and GPU remain separate series. The two
available device conditions are a comparison, not a time trend.

Grain: 15 unique episodes per condition, matched on exact label and compared count;
5,245 recorded positions in each condition. Counts over correlated positions are
not independent sample sizes. No confidence intervals or inferential test.

Takeaway to verify: CPU does not eliminate the recorded confidence-gated mismatch
count; equal aggregate counts conceal changes in two episode counts. This is
instrument comparability evidence, not a model-behavior finding or proof of cause.

Palette: two roots, blue GPU and gold CPU, plus charcoal and gray. Distinguish the
series by circle versus square markers as well as color. Direct denominators and
source precision in the subtitle; the source's confidence definition on the count
axis. Name missing port probabilities and position identities in the caption.

Expected output: approximately 1,700 by 1,100 PNG pixels, SVG for export, a richer
CSV of the plotted rows, value-basis JSON, and a script that reconstructs all
outputs from the immutable source copies. Inspect the PNG before handoff.
