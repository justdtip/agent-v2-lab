# ArchitectureView: see through adapter wrappers (wave-1 regression fix)

Author: the Deputy, under the Director's direct authorisation for a single-point fix
(2026-09-04 ~16:50). Not committed; the Chief reviews. Every claim cites file:line (R16).

## What broke

The Director's live P6 run (issue #26 command) passed case selection against the real files
and then crashed at model load:

    load_policy (evaluate.py:92) -> spec.resolve (models.py:68) -> lora_targets (arch.py:236)
    ValueError: LoRA policy 'attention+mlp' matched no linear modules

Root cause, confirmed from source rather than inferred:

- `mlx_lm.load(..., adapter_path=...)` applies `load_adapters`, which replaces each target
  projection with `LoRALinear` (`.venv/.../mlx_lm/tuner/lora.py:11-33`): a plain `nn.Module`
  that stores the original projection at `.linear`.
- The view's `_linear_modules` admitted only `isinstance(module, (nn.Linear,
  nn.QuantizedLinear))` (pre-fix `arch.py:298-306`). The wrapper at `self_attn.q_proj`
  failed that test; its child passed it but at path `self_attn.q_proj.linear`, whose suffix
  `linear` matches no LoRA suffix. Net: zero matches, hence the raise.
- Since wave 1 (`a2f003c`) every `load_policy` resolves, so **every adapter-loaded path was
  broken**: P6 patching, block ablation, evaluation of any adapter, rollout, branch mining.
  The preflight never hit it (base model only) and no fake carried LoRA wrappers, which is why
  three reviews passed it. Verified by a probe of `named_modules()` on a wrapped block, which
  yields `self_attn.q_proj` as `LoRALinear` and `self_attn.q_proj.linear` as `Linear`.

## The fix (`src/local_llm_lab/arch.py:295-329`, +31/−5)

`_linear_modules` now collects a block's entries, identifies wrappers (a module that is not a
linear type but whose `.linear` attribute is one), and reports each wrapper under **its own
path with the base module**, skipping the base's own `<path>.linear` entry so nothing is
counted twice. Consequences, both deliberate:

- Paths are identical with or without an adapter (`self_attn.q_proj` either way), which
  adapter comparison and provenance rely on.
- `_linear_dimensions` (`arch.py:362-374`) receives the base, the module that owns `weight`
  and `bits`; the wrapper owns neither and would fail there.
- No module is matched on the mere presence of a `.linear` attribute: the type check is on
  the attribute's value. Unwrapped models behave exactly as before.

Audit for a second copy of the filter: `grep isinstance(...Linear` across `src/local_llm_lab`
excluding the briefing's legacy modules returns nothing outside `arch.py`. Single point of fix.

## Tests (`tests/test_arch.py:140-236`, +121)

Proven **red before the fix** (the first two raised the exact production error; the third
failed on my own expectation, corrected — see below), green after.

- `_wrap_dense_projections` (`:140`) wraps every dense LoRA-target projection on the existing
  arch fakes with a real `mlx_lm.tuner.lora.LoRALinear.from_base`, the same shape an adapter
  load produces. No checkpoint.
- `test_lora_targets_see_through_adapter_wrappers_on_dense_and_hybrid_fakes` (`:166`): for
  the dense fake under `attention+mlp` and the hybrid fake under `auto`, targets and
  `lora_parameter_count` on the wrapped model equal the unwrapped result; asserts the fake
  really carries wrappers and that no reported path ends in `.linear`.
- `test_model_spec_resolve_succeeds_on_an_adapter_wrapped_policy` (`:197`): the exact
  crashing chain, `load_model_spec("qwen25-coder-3b").resolve(...)` on a wrapped fake, now
  succeeds with `lora_keys` and `trainable_parameters` equal to the unwrapped resolve.
- `test_lora_wrapper_over_a_quantized_base_reports_dequantized_input_width` (`:216`): the
  real 3B is 4-bit, so the base is a `QuantizedLinear`; `_linear_dimensions(base) == (8, 64)`,
  the wrapper shape mlx-lm produces is pinned, and `_linear_dimensions(wrapper)` must raise —
  proving dimension readers need the base, which is what the fix hands them.

Correction during red-first: my first expectation for the third test omitted the intermediate
`self_attn` module from the traversal list; fixed to match reality, not the code under test.

## Verification

- `uv run pytest` run bare with exit status checked: **exit 0, 642 passed** (previous tree:
  639). No skips, xfails or xpasses.
- Scope: `git status` shows only `arch.py` and `tests/test_arch.py` beyond the unrelated
  uncommitted #26 slice. No writes to `outputs/`, `data/`, `reports/`, `pending/`.
- Banned-constant grep on added lines: clean; the repository-rules scanner ran green in the
  suite.
- No model was loaded.

## Process disclosure

This slice did not go through an R19 independent reviewer: the Director authorised a direct
single-point fix and said the Chief would review. The dispatched implementer was stopped before
it wrote anything, so there is one author. Reachability, the lesson from the R21 inert-guard
finding, is covered here by the resolve-chain test, which drives the production call path on a
wrapped model rather than the mechanism in isolation.

## What this unblocks on commit

P6 (probes list B3, the Director's command unchanged), block ablation (B2), evaluation of any
adapter, rollout and branch mining — everything the training arm's own evaluation will need.
