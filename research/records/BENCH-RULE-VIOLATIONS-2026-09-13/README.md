# Five model constants the repository rule refuses, and who put them there

`tests/test_repository_rules.py::test_banned_model_constants_are_limited_to_approved_or_legacy_modules`
has been failing for at least thirty commits. Recorded rather than silenced, because sanctioning a
rule at 2am to fit code written at 2am is how rules stop meaning anything.

| where | constant | whose | what it actually is |
|---|---|---|---|
| `scripts/inject_repl.py:210` | `<end_of_turn>` | bench, 2026-09-11 | a CANDIDATE stop-token name, probed against the tokenizer and discarded if the vocabulary lacks it. The guard is real: `convert_tokens_to_ids` answers UNK for a missing name, so the id has to name the token back. Arguably not an assumption at all. |
| `scripts/steer_server.py` ×3 | `2048` | tonight | the output token cap, an argparse default. The rule's own sanctioned list already carries four scripts whose 2048 is "a token budget ... not a model assumption" — this is the same thing, and it coincides with the 3B width by accident. |
| `src/local_llm_lab/lora_torch.py:25` | the projection-name list | tonight | `DEFAULT_TARGETS`, the default of an overridable argument. This one IS a model-family assumption: `q_proj`/`k_proj`/… is a Llama-lineage naming convention. It is mitigated by `apply_lora` refusing a target that is not an `nn.Linear` rather than skipping it, so a model that names things differently fails loudly instead of training an unadapted network. |

**The call for Daniel.** The first two look like the precedent the sanctioned list already sets. The
third is a genuine assumption and the honest fix is to derive the target list from the model's own
module tree rather than name it — worth doing, not worth doing tonight, and not worth pretending
the rule is wrong about it.
