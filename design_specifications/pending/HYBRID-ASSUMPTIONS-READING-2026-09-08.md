# Where the tree assumes a hybrid: a code-grounded reading, no changes

**The Chief's request, phrased as a reading rather than an order to build.** Every claim carries a
file and line and was read in the installed source, not inferred. Nothing here is changed.

The short answer: the eval path is clean, the probe path degrades **silently and with a false
explanation**, and the capture path fails closed except in one place where it does not.

---

## 1. The finding that matters most: the guard against a degraded family is blind on Gemma

`jspace_sweep.py:690-700` refuses to spend a sweep when a hybrid produces no kind-matched partner:

```python
decoder_kinds = {probe_layer_kind(view, layer) for layer in range(1, view.num_layers + 1)}
if family.derived and not family.partners and {ATTENTION_KIND, RECURRENT_KIND} <= decoder_kinds:
    parser.error("this decoder has both attention and linear-attention blocks, but no "
                 "kind-matched partner was derived ... Refusing to spend the sweep on a "
                 "degraded layer family")
```

That guard exists because a real sweep once ran the six registry fractions instead of the
pre-registered nine-layer family and logged `hybrid_period=-` beside a correct list of block
kinds. On Gemma it cannot fire, and the chain is worth following in full because every link is a
correct decision made about a different model:

1. `arch.py:91` `layer_kind` reads `block.is_linear`. Gemma's window flag is one level down, at
   `self_attn.is_sliding` (`gemma3_text.py:55`), so **every Gemma block reports `attention`**.
2. `jlens.py:_structural_period` inverts `is_linear` over the blocks. With no recurrent kind
   present it returns `(None, ...)`: a dense backbone.
3. `jlens.py:_configured_period` walks the config for `full_attention_interval`
   (`jlens.py:190`). Gemma's field is `sliding_window_pattern`, so this is `None` too.
4. `kind_matched_layer_family` therefore takes `_degenerate_reason`'s first branch and returns
   the bare registry fractions with the reason **"no hybrid period in the model configuration
   (`full_attention_interval` absent): a dense backbone has one block kind, so no kind-matched
   partner exists"**.
5. The guard asks whether both kinds are present. `decoder_kinds == {"attention"}`, so it passes.

**So the sweep runs, on a degraded family, carrying a sentence into the artifact that is false.**
Gemma is not a dense backbone with one block kind; it has two attention spans, and the pivot
document's §2.3 works out that the contrast is *better* than Qwen's — same module class, same
parameter shapes, differing in exactly the mask window and the RoPE base.

This is the guard's own failure mode reappearing one level up. It was built to catch a family
silently degraded by a bad period read, and it is defeated by a period that is honestly absent
under a field name that no longer applies.

**The fix is not in the guard.** The guard is right to ask whether the contrast the sweep
pre-registers is in the run. What is wrong is that three different things all key on
`full_attention_interval` and `is_linear`, and on Gemma all three answer "dense" for three
different correct-in-isolation reasons.

---

## 2. Every place the hybrid is assumed, by path

### The evaluation path: clean, with one blocker that is not about the hybrid

Confirmed model-agnostic: `spec.family` is read **nowhere** in `src/` (only one test assertion);
`LORA_POLICIES` carries a dense suffix set beside the linear-attention one (`arch.py:26-40`);
`cache_strategy: auto` resolves from `view.cache_trimmable` rather than a name
(`models.py:110-119`).

The blocker the Chief found is real and is not a hybrid assumption: `protocol.py:292` hard-codes
`"<|im_start|>assistant\n"`. That is a chat-template assumption, and the pivot document's ruling
to move it into `ChatSpec` **with the assertion kept** is right for the reason it gives.

### The probe and lens path: three keys on the same two names

| site | reads | on Gemma |
|---|---|---|
| `arch.py:91` `layer_kind` | `block.is_linear` | every block `attention` |
| `jlens.py:190` `HYBRID_PERIOD_FIELD` | `full_attention_interval` | absent |
| `jlens.py:_structural_period` | inverts `layer_kind` | dense |
| `jspace_sweep.py:690` guard | both kinds present | never fires |
| `jlens.py:_degenerate_reason` | — | emits a false sentence |

Issue 86's `kind_matched_layer_family`, delivered this morning, needs only the field name and the
predicate's other kind: the two formulas are the same shape, `is_linear = (i+1) % p != 0` against
`is_sliding = (i+1) % p != 0`. The pivot document's derived family for 34 layers at period 6 is
**6, 11, 12, 17, 18, 23, 24, 28, 30, 34**, four adjacent sliding-global pairs.

### The capture path: fails closed twice, open once

- `arch.py:671` — `if any(type(c) is not KVCache for c in entries): raise`. An exact type check,
  so `RotatingKVCache` is refused. **Closed.**
- `arch.py:737` — `if not isinstance(att, Qwen3NextAttention): raise "head capture currently
  supports Qwen3NextAttention only"`. Reached only when head capture is requested, and it names
  what it refuses. **Closed**, and the import at `arch.py:684` does not fail on a Gemma-only run
  because the class exists in the installed library.
- `arch.py:786` — `if not hasattr(keys, "shape"): raise "quantized or rotating KV caches are not
  supported for capture"`. **This does not catch a rotating cache.** `RotatingKVCache.state[0]`
  is an array and has a shape; the guard catches the *quantised* case, whose state is a tuple,
  and its message claims a coverage it does not have. The pivot document is right that widening
  the type gates would be wrong here, because the surrounding code reads column *j* as absolute
  source position *j*.

### Training: hybrid-specific and deferred, with one live hazard

`training/gated_delta_chunked.py` and `gated_delta_chunkwise.py` install recurrences and have no
meaning on a dense backbone. The pivot document defers Gemma training, and the machinery is
opt-in per registry entry, so an omitted launch block never reaches it. The hazard the document
names is the inverse and I agree it is the real one: an author who declares a recurrence mode to
silence an error gets a passing run whose log claims a schedule that never executed.

---

## 3. The single-token assumption, swept as a class rather than an instance

The Chief asked for every place a project string is assumed to be one token in the model's
vocabulary. **There is exactly one in `src/`:**

`runner.py:467-468`

```python
with contextlib.suppress(Exception):  # tokenizer wrappers vary
    stop_ids.add(tokenizer.convert_tokens_to_ids("</tool_call>"))
```

Confirmed, and **the tests are why it survived.** Both tokenizer fixtures that implement this
method raise:

- `tests/test_runner.py:114` — `def convert_tokens_to_ids(self, token): raise KeyError(token)`
- `tests/test_runner.py:534` — the same

So every test of this path exercises the branch where the suppress *does* catch something, and no
test exercises what a real tokenizer does, which is to return an id. The suppress reads as
protective because the only tokenizers it has ever been shown raise. A fixture that returned an
unknown-token id would have failed the day it was written.

**Everything else that looks like a token is a text operation and is model-agnostic.**
`protocol.py:177-178` compiles `<tool_call>` as a regex and holds `</tool_call>` as a string;
`protocol.py:231-236` and `runner.py:419-430` use `str.find` on `<think>` and `</think>`. Those
run on decoded text, and the protocol instructs the model to emit those strings, so they are
correct on any family. Every other tokenizer interaction goes through `encode`, or reads
`bos_token` / `eos_token_ids` from the tokenizer itself (`profiles.py:537` handles both the set
and the scalar). None assumes a project string is one token.

`models.py:310`'s `<|im_end|>` in the unregistered-model fallback is a third ChatML assumption,
alongside `protocol.py:292`. It is not a single-token assumption, since `end_of_turn` is appended
as text (`protocol.py:140`, `branch.py:165`, `data.py:295`).

---

## 4. What I would want ruled, having read it

**Does `layer_kind` stay a two-value module property, with span as a separate reader?** The
architecture-view port design argues yes and the pivot document requires it for manifests and
the capture refusal. But §1 above shows that once it does, three probe-path sites conclude
"dense" from a property that is true (Gemma's blocks *are* all attention modules) and draw a
conclusion that is false (that no contrast exists). Whatever answers the period question for the
probe path has to read the span, not the kind, and the false reason string is the thing that would
mislead a reader of an artifact.

**Is a false reason string a bug of the same rank as a wrong number?** I think it is worse, and I
would rather `_degenerate_reason` refuse to characterise a backbone it cannot classify than
assert dense. That is a change and I have not made it.
