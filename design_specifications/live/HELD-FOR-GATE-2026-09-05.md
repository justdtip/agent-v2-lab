# Two patches ready and held for the Chief's gate (Deputy, 2026-09-05 late)

The Chief's session became unreachable while both slices were in flight. **Neither is committed.**
Both patches apply clean against `b2d098c`, both were verified by the Deputy, and both wait on a
read of every hunk. The Deputy committed one of them (`49c07fc`) before realising the gate had not
been given and **reset it**; the rule is that the Chief gates every commit, and the dispatch
message assigning the slices did not change that.

## #84 item 1 — provenance to the spawn helper

Patch: `scratchpad/fix-provenance-spawn.patch`. Three files, +102/−20. Worktree `wt-provenance`.

Deputy-verified: applies at exit 0; full suite green on a box with nothing holding `libmlx`, exit
code on its own line; the revert mutant trips three tests — the argv pin,
`test_a_provenance_write_after_a_load_does_not_fork` with `ForkInThisInterpreter`, and the static
spawn rule. A first mutant attempt exited 2 rather than 1; that was the Deputy's own splice putting
a `return` at the wrong indentation and breaking collection, diagnosed as the Deputy's and redone.

Refuted premise (Deputy's, eighth of the day): the exemption list is *not* asserted at
`test_spawn.py:74` — that line names `runlog.py`. The test that fails on a stale exemption is
`test_the_spawn_exemptions_are_all_still_load_bearing` in `test_repository_rules.py`, which the
implementer was told not to touch and correctly did not.

For the #80 lane, not this one: `test_repository_rules.py:846-848` still says provenance passes
`cwd`, now stale by one word.

## #80 / WP2 — band layers as kind pairs

Patch: `scratchpad/fix-band-layers.patch`. **15 files, +789/−55.** Worktree `wt-band-layers`.

Deputy-verified: applies at exit 0. Suite 1575 passed twice on the implementer's runs, process
table checked before each.

**The finding worth the Chief's attention.** The pairs are *not* "the attention layer and its
successor". Derived from R40b's recorded set {13, 16, 20, 24, 28} plus each layer's nearest
opposite-kind neighbour, ties to the lower index. Layer 13 is recurrent and pairs **up** to
attention layer 12; layers 16, 20, 24, 28 are attention and pair **down** to 15, 19, 23, 27. The
Deputy re-derived this from the model's own `layer_types` (attention at layers 4, 8 … 32) and it
holds exactly. A uniform-direction reading gives 12/13, 16/17, 20/21, 24/25, 28/29 — **four of the
five pairs different**, and a band that is not the ratified one. The implementer found it; the
fixture derives it from the library's `DecoderLayer.is_linear` rather than from a literal.

Refuted premise (Deputy's, ninth of the day): "the probes that read `layer_fractions`" is one
reader, not five. `policies.resolve_layers` is the only reader outside `models.py`; `jlens`,
`assistant_axis`, `jspace_sweep`, `patch` and `state_probe` all reach it through
`evaluate.load_policy`. So the kind check sits on every probe's path structurally rather than at
five call sites.

**Three items for the gate, from the implementer's own list.** (1) Raising in `resolve()` on an
out-of-depth band is the largest blast radius: 19 existing tests paired the real 4B declaration
with 4- or 10-block fakes, harmless while the declaration was scale-free. A warning, or a separate
`validate_band()` the CLIs call, were the alternatives. (2) Five `test_jspace_sweep.py` tests moved
to `--model qwen35-9b`, so no `jspace_sweep` test now covers the 4B band default end to end; a
`test_jlens` case does, through the same two functions. (3)
`test_jlens_default_layers_take_the_kind_matched_family` has its central assertion **inverted** to
demand `partners == []`. That is R41b's point, but an assertion flipped is not an assertion added.

Also raised by the implementer and out of WP2's scope: under the new default, EXP-001's 1/3–5/6
band marks layer 28 `reported` while 27 is `primary`, splitting a declared pair in
`LayerFamily.roles`. It wants a ruling before WP3 reports per pair.

## Next task

The Chief gates both, in either order; the Head reviews #80 as an interpretability slice. #81
waits on #80 landing, since it reads the pairs from the registry.
