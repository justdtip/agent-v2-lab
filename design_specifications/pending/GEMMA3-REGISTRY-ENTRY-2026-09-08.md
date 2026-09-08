# Phase 0: the Gemma 3 registry entry, read from the checkpoint rather than from the audit

**For the Chief's gate. Not landed, per "land nothing".** Every value below was read from the
files on disk at
`.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767/`
or from the installed `mlx_lm`, and the two were cross-checked against the **weight shapes**, not
against each other. Unruled fields are marked and left at values that cannot mislead.

---

## 1. Ground truth, and two corrections to the pivot document

`config.json`'s `text_config` is **far sparser than the audit assumed**. It declares only
`num_hidden_layers: 34`, `hidden_size: 2560`, `intermediate_size: 10240`, `sliding_window: 1024`
and `model_type: gemma3_text`. Everything else comes from MLX defaults, so the defaults are
load-bearing and had to be checked against the weights rather than trusted.

Checked, by reading the safetensors headers:

| | from the weights | source of the value MLX uses |
|---|---|---|
| decoder layers | indices 0–33, 34 blocks | config |
| hidden size | `embed_tokens.weight` `[262208, 2560]` | config |
| **vocabulary** | **262,208** | `gemma3.ModelArgs.vocab_size` default, which overwrites `text_config` |
| query heads | `q_proj [2048, 2560]` ÷ head_dim 256 = **8** | `gemma3.py:22-24` default |
| key-value heads | `k_proj [1024, 2560]` ÷ 256 = **4** | `gemma3.py:25-27` default |
| intermediate | `gate_proj [10240, 2560]` | config |
| dtype | BF16 | file |
| total size | 8.60 GB, 444 text tensors + 439 vision | index |

**Correction 1: the vocabulary is 262,208, not 262,144.** The audit's figure is
`gemma3_text.ModelArgs`'s default, which the VLM wrapper overwrites at `gemma3.py:21` before the
text args are built. The embedding matrix on disk is 262,208 rows. The difference is trivial in
size and not trivial in kind: any token-denominated constant should be derived from the view, not
from either document.

**Correction 2: `lm_head` is absent from the checkpoint, so the readout is tied.**
`gemma3_text.Model.__init__` sets `tie_word_embeddings = False` and builds an `lm_head`, but
`sanitize` (`gemma3_text.py:237-241`) sets it True and pops the layer when the weights carry no
`lm_head.weight`. Gemma 3 4B carries none. This matters because `arch.py`'s `unembed` and
`native_readout` branch on that flag, and it is set during load rather than in the constructor.

**A third fact, unremarked in the audit and load-bearing for the live-lens pilot.**
`gemma3_text.Model.make_cache` (`gemma3_text.py:247-257`) returns `KVCache()` for the six global
blocks and **`RotatingKVCache(max_size=1024)` for the other 28**. So 28 of 34 caches are rotating
on every Gemma run, not only long ones, and `arch.py:671`'s `type(c) is not KVCache` refuses
capture on this model outright. Fails closed, correctly, and it is the gate the live-lens pilot
has to clear.

---

## 2. The Phase 0 urgency is overstated, and I would rather say so than quietly agree

The Chief's reason for putting the registry entry first is that preflight indexes the training
batch size and sequence length bare and dies after the weights load, burning a slot. The bare
indexes are real — `preflight.py:959`, `:960`, `:1038`, `:1068` — but they sit inside
`_activation_bytes`, `_resolve_row_tokens` and `_training_footprint`, all reached only under
`require_preflight(consumer="training")`.

**No caller anywhere in `src/` passes that consumer.** The pivot document says as much in its own
§4 ("the footprint rejection runs only under the training consumer, and the training path does not
pass one"), and Gemma training is deferred regardless. So the key error cannot fire on the first
Gemma run and does not gate it.

The entry should still exist first, for a better reason: **without it every Gemma run takes
`_default_spec`**, which hardcodes ChatML's `<|im_end|>` as the turn ending and declares
`cache_strategy: none` and `thinking: unsupported` with no record of why. The entry is what makes
a run's provenance mean something. I have declared the training fields anyway — they cost two
lines and remove the question.

---

## 3. The entry

```yaml
# configs/models/gemma3-4b.yaml
name: gemma3-4b
hf_id: google/gemma-3-4b-it
family: gemma3
chat:
  # Gemma 3 has no thinking mode and no template switch for one.
  thinking: "unsupported"
  template_kwargs: {}
  # tokenizer_config.json's eos_token, and the second of the two ids config.json declares
  # (`eos_token_id: [1, 106]`, `<eos>` and `<end_of_turn>`). Appended as text by
  # protocol.py:140, branch.py:165 and data.py:295; not a generation stop, which MLX takes
  # from the config's id set.
  end_of_turn: "<end_of_turn>"
  # Declared empty deliberately. The field is read by nothing outside models.py (issue 98),
  # and Gemma's terminators already reach the generation loop through the config; declaring a
  # token here would be a false entry in every provenance record this model writes.
  extra_stop_tokens: []
lora:
  keys: auto
  rank: 16
  scale: 32.0
  dropout: 0.0
train:
  # UNRULED: training on Gemma is deferred (pivot §6). These exist so no consumer meets a bare
  # key error, and they are the Qwen recipe's values, which are evidence about a different
  # model. Nothing may cite them as a measured envelope for this one.
  max_seq_length: 2688
  batch_size: 1
  grad_accumulation_steps: 1
  learning_rate: 3.0e-5
  grad_checkpoint: true
cache:
  # Explicit rather than `auto`, per the pivot document. `none` rather than `trim` for the
  # first runs: 28 of 34 caches are RotatingKVCache and no equivalence has been measured on
  # this model. A reuse strategy is an equivalence claim, and this model has no acceptance
  # record behind one. UNRULED, and the first thing to revisit once a smoke test exists.
  strategy: none
  equivalence_verified: null
probes:
  # The same six fractions as every other entry. They are a depth convention, not a model
  # constant, and on 34 layers they land on 6, 11, 17, 23, 28, 34.
  layer_fractions: [0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
  capture_dtype: native
  # NOT DECLARED: `live_lens_pairs` and `partner_tie_breaks`. The band is a ruling and no
  # ruling exists for this model. The pivot's §2.3 derives the kind-matched family
  # 6, 11, 12, 17, 18, 23, 24, 28, 30, 34 from period 6, and that derivation belongs in a
  # record with the Director's word behind it before it becomes a declaration here.
memory:
  # A cap and a declaration of intent; preflight resolves min(this, the device's recommended
  # working set) and records both (R32b). Same as the 4B Qwen entry because it is the same
  # box, not because it is the same model.
  budget_gib: 22
policies: {}
```

---

## 4. What this entry cannot express yet, and what breaks first if it lands

**`sliding_window_pattern` has no home.** The band machinery reads `full_attention_interval`
(`jlens.py:190`), and Gemma's periodicity is not in its config at all — MLX supplies 6 by default.
Neither the registry nor the model file states it, so on this entry `hybrid_period` returns
`(None, "dense")` and the layer family degrades with a false reason. That is the finding in
`HYBRID-ASSUMPTIONS-READING-2026-09-08.md` §1, and it is not fixed by any value this entry can
carry. It needs the field-name generalisation in `kind_matched_layer_family`.

**`hf_id` points at the official BF16 repository**, because that is what is on disk and its
provenance is nameable. The pivot document's ruling 4 asks the Director for 4-bit for evaluation,
to stay comparable with the Qwen baseline, and BF16 for lens work. **UNRULED**: if he wants a
4-bit evaluation, the entry becomes two entries or the `hf_id` changes, and a converted checkpoint
needs its own provenance line. I have not converted anything.

**The tool role is unruled and this entry cannot encode either answer.** `chat_template.json`
raises on `(message['role'] == 'user') != (loop.index0 % 2 == 0)`, folds a system message into the
first user turn with a trailing blank line, and has branches for user, assistant and system only.
Every trajectory here is system, user, assistant, tool, assistant, tool, so an observation at an
even index raises. Re-roling observations as user turns changes the training distribution; a
custom template leaves the model's own distribution. The Chief's recommendation to the Director is
the former. **Nothing here is built for either.**

## 5. Landing checklist, for whoever lands it

- `configs/models/` is excluded from the model-assumption rule (`test_repository_rules.py:75`,
  `_SANCTIONED_CONFIG_DIRS`), and `test_jspace_sweep.py`'s config glob covers `configs/` top level
  only, so neither rule moves.
- No test asserts the exact registry list; `test_models.py:152` monkeypatches the directory. So
  the entry adds no coverage and breaks none. **A parametrised case in `test_models.py` should be
  added with it**, or the entry is the first registry file nothing checks.
- `test_models.py` and `test_repository_rules.py` are the suite that covers this, and both are
  MLX-free, so they run without the box.
