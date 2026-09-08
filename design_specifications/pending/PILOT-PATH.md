# The pilot path: the cache guard, narrowly, and the generation prefix declared

Patch: `design_specifications/pending/PILOT-PATH.patch`, against `65bf87e`. Nine files, 147
insertions. **Not landed.** Work orders, the Deputy's item 2, first two bullets. Issue 99 is
delivered separately (`LENS-IDENTITY.patch`); the pilot script's constants are the fourth bullet
and are not in here.

**876 passed, 1,095 skipped, exit 0**, against the worktree's own source. `ruff check` clean on
every changed Python file.

**One of the two is not yet exercised and I am not reporting it as green.** The cache guard's test
lives in `test_live_lens_native.py`, which loads the model library, so it is skipped while the
1,200-row evaluation holds the box. The generation prefix's tests are in `test_models.py` and ran.

---

## 1. The cache guard, narrowed to what it can justify

Two different risks sat behind one exact-type check, and only one of them is real.

**Residual capture is safe under rotation.** It emits against `self._offset`, the monotone count
of tokens the model has seen, and `RotatingKVCache` maintains that identically:
`cache.py` does `self.offset += keys.shape[2]`, read rather than assumed. Nothing in that path
indexes a cache column.

**Head capture is not.** It reads `cache.state[0]` and treats column *j* as absolute source
position *j*. Under rotation a column is not its position and at most `max_size` of them exist, so
the same arithmetic stays well-formed and produces plausible per-head numbers against the wrong
positions — which is the failure worth a loud refusal rather than a widened gate.

So: any unquantised cache is accepted, and a rotating one is refused **only when head capture is
requested**, with the column arithmetic named in the message and the two ways out given.

This is the difference between running and not running. Gemma's `make_cache` returns a
`RotatingKVCache` for every block whose index plus one is not divisible by six — **29 of its 34**,
on every run rather than only long ones — so the old check refused the model outright.

The test uses `max_size=2` against a four-token sequence, so the cache genuinely rotates within the
test rather than merely being of the rotating type, and asserts three things: the captured logits
match an uncaptured run on the same rotating caches, the residuals still arrive, and head capture
raises with the column arithmetic in the message.

## 2. The generation prefix, declared per model, with the assertion kept

`protocol.generation_suffix` held ChatML's assistant marker as a literal and `build_prompt` asserts
that the rendered prompt ends with it. So **every generation-side render raises on Gemma**, whose
template opens a turn with `<start_of_turn>model\n` — read from the checkpoint's own
`chat_template.json`, not from a document.

`ChatSpec` gains `generation_prefix`, declared in all four registry entries, and
`generation_suffix` reads it. **The assertion stays**, for the reason the work orders give: the
same value is stamped into training manifests and into the lens corpus's tokenizer identity, which
the prose stage re-derives and hash-compares, so deleting the guard would turn a loud failure into
a corpus identity that is falsified and self-consistent.

**Two lines of defence, after the Chief's ruling.** The first draft gave the dataclass a ChatML
default so the eight positional constructors kept working, and the Chief refused it: *a ChatML
default on the one field whose purpose is to stop a ChatML value being assumed is the same defect
one level down.* Accepted, and the argument is stronger than the one I made — the parser covers
registry files, and `_default_spec` builds a `ChatSpec` **directly**, which is precisely where
`<|im_end|>` had been hiding all along.

So the field is **required and keyword-only**. Keyword-only for the reason the ruling gives:
removing a default from the middle of a signature would let every positional constructor silently
mis-assign, and the symptom would be a turn ending that is some other field's value. Named, they
fail to construct, which is a list of sites rather than a bug. Eight call sites now name it, and
the test asserts both defences — `TypeError` from a positional construction, and the parser's
refusal of a registry document with the key removed.

`thinking` and `end_of_turn` were checked for the same shape and have none: neither carries a
default, so neither can be assumed from one model. They stay positional.

**What I left alone and why.** The empty thinking block, `<think>\n\n</think>\n\n`, is still a
literal and still Qwen's. It is reached only under `thinking == "off"`, which no other registered
model declares, so generalising it now would be inventing a schema from one example. Gemma's
suffix is the bare marker, asserted in the test.

## 3. What this does not cover

The pilot script's pinned constants (`LENS`, `LENS_SHA`, `REGISTRY`) are still module-level, so
`--model` moves nothing. That is the fourth bullet and it is next.
