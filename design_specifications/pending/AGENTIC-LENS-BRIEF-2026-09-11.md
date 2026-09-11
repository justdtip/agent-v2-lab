# Brief: an agentic, long-context J-lens for Gemma 3

**Chief, 2026-09-11, for the Research Director to hand to Codex.** This is a brief, not an order: it states what exists, what is in use, why the gap matters, what will be hard, and the decisions that are the Director's. It authorises no run and changes no sealed artefact.

## 1. The gap, stated precisely

Every Gemma 3 lens in use — `out/lens4b-f32`, `out/lens12b-f32-remerged` and its three chunks — is fitted on `prose-gemma3-4b-cuda-bf16.json`: **domain `prose`, 201 prompts, `max_seq_len` 128, positions 16–126**. That lens is behind the SAE–J bridge's A1 and A2, behind the atlas, and behind every reading in `SAE-J-DEVICE-RUNS-2026-09-10` and `SAE-J-ATLAS-ANALYSIS-2026-09-11`. The cells it is read on sit at **positions 415–4,300 in contexts up to 4,300 tokens** of agent transcript, which is why the positions file carries a domain-of-validity statement declaring every cell outside the fit domain.

## 2. What already exists, and what it was built for

`src/local_llm_lab/pipeline/lens_fitting/corpus.py` has `build_agentic_corpus`, reached by `scripts/lens_corpus.py --corpus agentic --evals …`. It renders **every step of every training trajectory** into a row carrying `task_id`, `step_index`, `split`, content spans, the prompt/answer boundary and the token ids, drops rows over a declared `max_tokens`, and writes an immutable manifest with `domain: "agentic"`, the tokenizer identity and per-source digests. It refuses non-training splits and a `keep_last` that disagrees with the runner's window policy.

It was exercised on **Qwen3.5-4B under MLX** on 2026-09-07 (`research/records/LENS-FIT-agentic-regression-2026-09-07`), with a frozen manifest of 518 sequences up to 2,044 tokens. That record is a resource calibration; **no fitted agentic lens artefact exists anywhere in the tree**, and `models/lenses/` does not exist. Treat the Qwen work as a precedent for the corpus path, not as a lens.

## 3. The input problem, which is the first thing to settle

`build_agentic_corpus` consumes **evaluation files** (`{summary, trajectories:[{task_id, steps:[…]}]}`) and re-derives each step's prompt by replaying the message window, re-parsing the action and checking it against the recorded one. The only such files in the tree are Qwen's, under `outputs/agent-v2e-qwen35-4b/evals/`.

The workspace corpus the captures actually use, `all-splits.jsonl` on the card (8,907 rows), is a **different shape**: each row is `{messages, metadata, prompt, completion}` with `metadata.task_id`, `family`, `variant`, `step`, `source`, `recovery`, `difficulty`, `perturb`, and the message window already rendered. It is not an evaluation file and `build_agentic_corpus` will not read it.

So the first decision is whether to (a) write an adapter from the rendered corpus into the corpus builder's row contract, keeping `task_id` and `step` as the builder does, or (b) produce Gemma trajectory evaluation files and use the existing path. **(a) is strongly preferred**: it reuses the exact rows the workspace captures, the W-3b masking passes and the behavioural tests were computed on, so a lens fitted on them is in-domain for those readings by construction rather than by argument. (b) introduces a second rendering of the same tasks and a second chance for them to disagree.

## 4. Why this is worth doing, from measurements rather than principle

Two of today's results look like **domain** failures rather than failures of the lens idea:

- **The readout resolves but cannot say so.** At layer 24's action position the prose lens's six-tool argmax equals the model's in four of four atlas cells, including both where the model departs from the expert — but its raw six-tool mass clears W-5's 0.001 floor in only one, so three are formally unresolved. An in-domain lens is the obvious candidate for putting mass where the answer already is.
- **The lens does not propagate a perturbation.** Passing the lens's residual change through the model's own readout derivative leaves a relative error of 1.3–1.6 at every strength, and before any readout the cosine between the lens-propagated change and the model's exact response is −0.04 at layer 18 and 0.19–0.46 at layer 24. Reading a state and carrying a perturbation are different uses; the second is what steering needs and the prose lens does not supply it.

Neither is a dictionary problem and neither is fixed by more SAE work. Note also the **negative** result that must be preserved: the prose lens *does* extrapolate for the tool readout, at 0.99 agreement with the model's own argmax from layer 24 (W-5). Any agentic lens must be shown to keep that, not only to improve the two above.

## 5. What will be hard

- **Cost.** The fit takes finite-difference Jacobians per position. The prose fit was 201 prompts × 128 tokens and took a night of the card. A 2,000-token corpus is roughly sixteen times the work per prompt, so the run must be scoped by prompt count, by a sampled position set, or both — and the sampling is a declared property of the lens, not an implementation detail. `fit_upstream_jacobian` already takes a `position_selector`; it was `None` for the prose fit.
- **Batch invariance.** This card's float32 forward is not batch-invariant: a fit at dimension batch 32 and a capture at width 1 differ, which is why every reading carries a measured pairing. An agentic lens inherits that and needs its own pairing measurement (`scripts/measure_pairings.py`) before any cell is read through it.
- **Admission.** A new lens is a **new artefact with its own admission**, not a replacement. `out/lens4b-f32/admitted-maps.npz` (`1c8d2bd7…`) is referenced by the registered pairings, the atlas bundle and every recorded result; it must not be overwritten or its digest reused. Admit the new one beside it through `scripts/admit_device_lens.py` and give it its own domain-of-validity statement.
- **Comparability.** The whole point is a paired comparison, so the agentic lens must be readable on the **same** 600 cells at the **same** layers, and the report is prose-lens against agentic-lens on identical cells: six-tool argmax agreement with the model, raw mass against the floor, and the perturbation-propagation cosine of §4.

## 6. Decisions that are the Director's

1. Adapter from `all-splits.jsonl` (preferred) or Gemma evaluation files.
2. The scoping of the fit: how many prompts, what context length, and whether positions are sampled — this fixes the cost and is a declared property of the resulting lens.
3. Whether the 4B alone first, or both models.
4. Whether this precedes or follows the SAE work already queued in `SAE-J-ATLAS-NEXT-ACTIONS-2026-09-11.md`. The Chief's view: it precedes the dictionary items, because a dictionary trained elsewhere and a lens fitted elsewhere are two separate domain gaps and this is the one we can close ourselves.

## 7. What must not move

The existing lens archives and their admissions; the registered pairings; the sealed captures; and the recorded results that cite them. Nothing in this brief is a reason to re-run, re-admit or restate any of them.
