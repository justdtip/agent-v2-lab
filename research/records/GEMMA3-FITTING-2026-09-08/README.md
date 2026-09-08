# Gemma lens validation preparation

The deliverable is the [findings and reproducible technique](FINDINGS-AND-TECHNIQUE.md).
The [next-run protocol](NEXT-RUN-PROTOCOL.md) separates correctness checks from comparisons
between different fitting objectives and lists what must be frozen before runtime work.

Completed here: explicit prose corpus lengths (128 and 2,048 tested, 1,024 default preserved),
70 passing CPU-only corpus checks, read-back compatibility of the surviving frozen corpora,
Gemma tokenizer/count and initial-prompt whitespace audit, and exact-arithmetic examples showing
why derivative/value fits can disagree and why a recurrent derivative needs a stated history.
No Gemma model was loaded, no lens was fitted, no cloud compute was purchased, and the later
Scope composition was not started. This completes a preparation slice, not the research programme.

## Reproduce

From this worktree, using the existing project Python environment:

```sh
python research/records/GEMMA3-FITTING-2026-09-08/reproduce_methods.py
python research/records/GEMMA3-FITTING-2026-09-08/verify_cpu.py
python research/records/GEMMA3-FITTING-2026-09-08/audit_inputs.py --primary '/path/to/primary-checkout'
```

The first needs only Python 3. The audit additionally needs the existing tokenizer backend and
bound local assets, but imports no model runtime. Tokenizer previews are not frozen Gemma corpora.
The CPU wrapper prevents actual MLX imports, including during test collection. The mathematical
examples are exact counterexamples, not findings about a loaded checkpoint.

The archived input-audit records the HEADs observed at measurement; SHA256SUMS.json identifies
the actual modified source/evidence files. The main checkout advanced independently while this
preparation ran. A formatting-only audit-script revision was rerun and produced the same payload
apart from the independently moving primary HEAD. The three production/test files are the complete
code scope of this patch. Lint, formatting and git whitespace checks passed.

## Next dependency

The registry has landed in the primary tree, and the identity patch is delivered. Fitting still
requires the released architecture port, its above-window native conformance/negative-control
record, and the Chief's official BF16 conversion identity. Import those into the fitting worktree
before running it. The pilot path can progress independently under the work order.

Before long-prose capture, section 14 also needs its Gemma-specific sample/prefix registration;
the old literal 51 windows and 824-token prefix do not establish the new long-context regime.

The code-only CORPUS-WINDOWS.patch uses zero context so the archived diff has no trailing-space
context lines. Check/apply it with git apply --unidiff-zero; the check passed against the primary
tree. Source and test changes remain on codex/lens-fitting for the Chief to integrate.
