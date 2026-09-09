# Verification

The local analysis completed successfully with no model library imported. All four source hashes match the remote reports. The JSON and JSONL summaries agree, fifteen episode identities are unique, denominators match, and episode totals and confidence reductions reconcile.

A second run into a fresh directory reproduced all five generated outputs byte for byte: PNG, SVG, CSV, analysis JSON and README. A deliberately altered report was rejected by the source-hash gate. These are file-only checks; no model tests were run.

The PNG was opened and inspected at full figure scale. Episode labels, axes, legend, source precision and the confidence limitation are legible without clipping. Color is paired with marker shape. Every point uses its own episode denominator; overall agreement uses the pooled token count within each condition, not a mean of episode percentages.

The downloaded WS-D manifest and both implementation source files also match remotely read SHA-256 hashes. SNAPSHOT.json records their identity, native-capture evidence and the one-row departure. The ongoing progress file is a snapshot, never a declaration that the run completed.

Plotting dependencies were installed into an isolated temporary directory. The shared research Python environment was unchanged. No timing or model-quality inference is made.
