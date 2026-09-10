# Workspace experiments, set 1 — an observational screen at the agent's decision positions

**Chief, 2026-09-11 UTC. Status: captures in progress; nothing read.** Order:
`design_specifications/pending/WORKSPACE-EXPERIMENTS-2026-09-10.md`, with the rulings on Codex's
design review (`WSA-WORKSPACE-AND-PREREG-2026-09-10`, 3494a2c) applied before any reading. The
mechanism claims — ignition, functional availability, causal broadcast, report versus computation —
are reserved for the steering experiments; this record describes.

## What is measured, by name

| experiment | quantity | instrument | control / null |
|---|---|---|---|
| W-1 | first confidence crossing of the expert action; readout competitor | six-tool conditional mass through the fitted lens, full-vocabulary p beside, log-odds beside | the logit lens; the tool prior |
| W-2 | cross-family linear decodability of the expert action at `P_act` | PCA-r on the training fold + ridge readout; folds by episode; leave-families-out | within-family permutation null; transfer null |
| W-3 | direct attention to tagged spans from `P_note` and `P_act` | eager attention, per layer and head, the 300-sample | the local-mask gate; span sizes and distances preserved |
| W-3b | the action under masked carriers | key masks by kind and the current note; the leaky-mask self-check | the unmasked forward; same-kind arms; the random-token arm (count only) |
| W-4 | early versus late linear decodability of the expert action (`P_note` against `P_act`) | W-2's readout at both positions | the same nulls; the tool prior |
| W-5 | lens validity by position band | last-layer lens readout against the model's own | — |

## Provenance

To be filled from the capture manifests: checkpoint identities (the loader's complete hash
manifest), lens archives and their digests and fitting declarations, the corpus digests, the sample
rule and keys, the scripts' hashes as run, the arithmetic settings, the widths, the positions.

## Results

Not yet. Each experiment reports in three states with its unit count, its unresolved count and its
tail, and nothing in this record is a claim about a workspace property.
