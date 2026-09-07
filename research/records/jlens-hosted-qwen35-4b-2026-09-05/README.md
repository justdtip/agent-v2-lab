# Hosted J-lens and read-weight measurements, 2026-09-05

The record of the measurements cited in
`design_specifications/pending/WP3-TRANSPORT-WEIGHTS-DESIGN-2026-09-05.md` and
`design_specifications/under_review/HOSTED-JLENS-QWEN35-4B-RESULTS-2026-09-05.md`.

## Why it is here

These files were produced in `outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/`, which is
gitignored. The design sections cite the JSON by name while the code that made it existed on one
machine only, in a directory git is told to forget. A later run that disagrees with these numbers
could not have been compared against the method that produced them. R37 — land the code and the
artifacts together — is the rule this closes.

`outputs/` keeps its copy. That copy is untracked, gitignored, and is not to be a launcher again.

## The scripts do not run

Every script that loads a model refuses at the top, before its imports, unless invoked with
`--i-am-a-record`:

    refusing to run: this file is a record of the 2026-09-05 measurements, not a launcher;
    re-run through the package entry points of issues 79 and 81 once the issue 83 lock has landed

They each carried an ad hoc process check and no lock. The one operating constraint on this machine
is a single model load at a time, and it is enforced by the loader wrapper and the model-run lock of
issue #83, not by a note. Runnable versions arrive as package entry points under **#79** (hosted
lens as the reading instrument) and **#81** (transport weights).

`build_atlas.py` is deliberately unguarded: it reads `dashboard_data.json` and writes
`atlas.html`, and loads no model.

## What is not here

The two hosted-lens weight files (`.npz`, ~406 MB each) are excluded. `provenance.json` carries
their SHA-256 digests and their Hugging Face location, so they are recoverable and verifiable
without being carried in the tree.

## Reading the measurements

| file | what it holds |
| --- | --- |
| `overlap_spread.json` | the current far-source overlap: 7 read positions spanning 25–100% of the context, 230 head-context entries, with summary and provenance |
| `overlap_test.json` | the superseded overlap on 4 back-half read positions, kept because §10h names it as superseded and a reader needs to see which way the correction ran |
| `overlap_by_cosine.json` | the per-head cross of overlap against query cosine, marked superseded for that question |
| `query_cosine.json` | query and key cosines for all 1,536 (context, block, value head) entries, with the long-gate flag |
| `alpha_calibration.json` | the reconstruction-gate calibration behind the 1e-5 gate and the 1e-6 change line |
| `atlas.html`, `dashboard_data.json` | the workspace atlas and its data |

Cosine fields are properties of the **key** head: value head `h` reads key head `h // 2`, so
consecutive value heads carry identical values. Any count or percentile over them states the
distinct key-head n.

## Where the lens lives now (2026-09-07)

The float16 `.npz` was regenerated on 2026-09-07 at the Director's instruction ("save the lens persistently"),
after the scratch copy the WP12 scripts read (`../jlens_hosted/`) was lost with the 06:03 restart. It now
lives inside the tree, outside version control (`models/` is ignored):

| path | what |
| --- | --- |
| `models/jlens/Qwen3.5-4B_jacobian_lens_n1000.npz` | 31 float16 arrays `J0`..`J30`, 2560×2560; sha256 `381c089dcffead8147ee91f944496f468cce2c7d593e0a1b17230745055aea12` |
| `models/jlens/Qwen3.5-4B_jacobian_lens_n1000.json` | sidecar: source digest, pickle globals, layer convention, per-layer identity distance |
| `models/jlens/source/` | the original `.pt` (sha256 `1f9a8f8f…8534e`, matching `provenance.json`), `config.yaml`, `CREDIT.md`, the convergence csv |

Regenerate with `scripts/convert_jlens.py` (a runnable converter; it loads no model, so it may run beside
training or evaluation). The venv has no torch: run it under Homebrew's python3 (torch 2.9.1). The key
convention is unchanged: `J{L-1}` is repo layer `L`; layer 32 is the identity and is not stored.
