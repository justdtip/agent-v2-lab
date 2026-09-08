# Depth-profile figure contract

Question: how do the two stored linear maps compare across all nonfinal layers, and how much
selection-set residual error does the regression leave at the same depths?
Takeaway: matrix alignment generally increases toward the final layer while regression selection
error decreases; this does not identify the cause of instrument differences or token accuracy.

Three line-and-point panels, one point per layer, 33 observed layers; no interpolation claim,
no smoothing, no uncertainty bands. This is an ordered decoder-depth axis, not time. Final-layer
identity is an open diamond shown separately from the learned curves. Report all layers, with
exact values available in comparison.csv and lens-metadata.json. The repeated line family is
appropriate because all three questions concern ordered depth, with three distinct units.

Sources: comparison.json and the unchanged output sidecar copied to lens-metadata.json. The
selection panel uses 100 * held SSE / sum(target residual squared), 402 windows, 51,456 positions;
it is not independent generalization. The regression used 1,608 fit windows, 205,824 positions.
Both fits used 128 tokens, below the 1,024 sliding window; carry this caveat in the figure.

Delivery: standalone scientific figure exported by Matplotlib as PNG and SVG, linked from the
research record. Three vertically stacked panels in a 9 by 10.2 inch figure, final QA on the PNG.
Palette: single blue root (#2563a6), white background, charcoal text, neutral grid and identity
reference. Filled points and solid lines for measured maps; open diamond for identity. No
architecture-family colors or comparisons. Descriptive titles and explicit denominators.
