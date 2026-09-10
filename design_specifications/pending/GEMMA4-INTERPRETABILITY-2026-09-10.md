# Gemma 4 conversation interpretability — the Chief's review of Codex's first slice (2026-09-10)

Reviewed: the uncommitted slice on `codex/gemma4-interpretability` (worktree
`outputs/worktrees/gemma4-interpretability`, base `cuda-migration` 647c209): `src/local_llm_lab/conversation_interp/`
(`contract.py`, `checkpoint.py`, `artifacts.py`, `runtime.py`), `scripts/conversation_interp.py`, four test modules, the
design spec and the plan under `docs/superpowers/`. Nothing merged, nothing on the branch touched; the worktree carries no
tracked change and no stray file from this review. Method: every file read; the interfaces the slice depends on read
in the same tree (`torch_capture.py`, `arch_torch.py`, `upstream_ref.py`); the installed Transformers 5.16.1 Gemma 4
modeling and configuration read; the public `config.json` and `model.safetensors.index.json` of `google/gemma-4-31B-it`
and `google/gemma-4-26B-A4B-it` fetched to the laptop (metadata only, no weights); the written tests executed on the
laptop's CPU (Python 3.13.7, torch 2.14.0, transformers 5.16.1, pytest 9.1.1; caches and temporary files in the
session scratchpad, not the worktree). No GPU, no model weights, no tokenizer.

## Verdict

The slice is sound and unusually careful about what it does not claim. It has one blocking defect, which its own tests
catch the moment they run, and two small additions for the checkpoint inspector. Accept after the fix; the next slice
is the importer that ties token ids to a rendered template, before any device gate.

## Findings, most severe first

**F1 — blocking, runtime: `run_case` refuses every Hugging Face model before any forward.** `_prepare` digests
`view.config.to_dict()` through `canonical`, which (rightly) refuses non-string mapping keys; every HF config carries
`id2label` with integer keys (`{0: 'LABEL_0', 1: 'LABEL_1'}` on the fixture), so the digest raises `JSON object keys
must be strings` and five of the seven runtime tests fail on that line, never reaching the behaviour they test.
Executed as written: 62 of 67 collected tests pass, 5 fail, all five on this. Fix: digest the config through HF's own
serialisation, `json.loads(view.config.to_json_string())`, which stringifies keys and dtype objects. Verified on a
scratch copy with that one line changed: 67 of 67 pass — replay with a supplied token that disagrees with the argmax,
the same-state replacement reproducing the baseline trajectory and log-probabilities exactly, refusal before any
forward on a wrong dtype and on a training-mode model, the greedy stop on the chosen token, the router observer keeping
scaled weights distinct from probabilities, a missing cache leaving an incomplete store with no hook left behind, and
the deliberately zeroed final residual driving the first prediction to the uniform −log 16.

**F2 — checkpoint inspector against the real checkpoints: it matches; add two fields.** Both public Gemma 4 checkpoints
are wrapper configs (`model_type` gemma4, `text_config.model_type` gemma4_text) whose text tensors sit under
`model.language_model.*` (832 and 657 tensors) beside `model.vision_tower.*` (355) and `model.embed_vision.*`; neither
carries `lm_head.weight` (`tie_word_embeddings` true); `hidden_size_per_layer_input` is 0 in both. The 31B is dense,
60 layers × 5,376; the 26B-A4B is MoE, 30 layers × 2,816, 128 experts, top-8, with `layers.N.router.{proj.weight,
scale, per_expert_scale}` and `layers.N.experts.*`. The inspector's nested layout, its tolerance of an absent readout and
its `moe` block therefore read these checkpoints as they are. Additions: report `tie_word_embeddings` from the text
config and `lm_head_present` from the headers, and refuse the combination tie=false with no readout — the view decides
tying by weight identity at load, and that combination fails there with a worse message.

**F3 — router observer against the installed modeling: the contract holds; state what the routing input is.**
`Gemma4TextRouter.forward` returns `(router_probabilities, top_k_weights, top_k_index)`: a float32 softmax over all
experts, top-k weights normalised then multiplied by `per_expert_scale[top_k_index]`, int64 indices; the decoder layer
holds it as `self.router` when `enable_moe_block` is set. The observer's checks (3-tuple, two-dimensional, weights not
normalised, indices int64 and in range, probabilities summing to one) match. The router is called on the flattened
hidden states of the layer's second feed-forward branch, after `pre_feedforward_layernorm_2`, not on the captured
residual; the record keys routing by the same position, which is right, but the spec should say the routing input is
the layer's own MoE input so nobody reads router probabilities as a function of the recorded residual.

**F4 — the architecture view on the nested model is unexercised.** `TorchArchitectureView` takes its `config` from
the text module (`get_text_config()`), so the observer's `gemma4_text` guard holds for `Gemma4ForConditionalGeneration`
as well; but `from_model` reaches the text module through upstream's `_find_layout` or a structural fallback, and only
the plain `Gemma4ForCausalLM` fixture has been through it. This is gate 3 of the design and stays there.

**F5 — `compare_captures` refuses on any difference in `runtime.execution`, silently as to which key.** The execution
record includes the digests of seven source files, package versions, the device name and the upstream commit, so a
baseline and a variant produced across any source edit are not comparable — correct for evidence, and consistent with
this programme's rule that a same-state control and its baseline come from one code state — but the refusal should
name the differing keys.

**F6 — the importer is the whole of the next slice.** A case's `messages` and `source_settings` are evidence, and the
CLI says so (`token_text_correspondence_verified: false`). Until an importer renders the original exchange under its
template and thinking setting and reconciles the ids, no case is the conversation. That work precedes every device gate.

**F7 — what the slice gets right, for the record.** Token ids authoritative and immutable (`Case` holds canonical JSON,
so a caller cannot alias nested provenance; the test proves it); checkpoint identity as an immutable revision plus a
digest over all weight files, the header inspector explicitly unable to supply it (the D-CRO's F2 rule); a
prefill-then-cached-single-token partition recorded as such and checked against the returned cache length at every
step; the query that predicts each token captured before the token is called processed; one-shot replacements on the
prefill only, applied through `TorchCapture` at the block output (layer 0 through the entry residual), asserted to
apply exactly once; an exclusive run directory whose manifest says incomplete until every record exists, with hashes
of every emitted file and a reader that re-validates coordinates, coverage and stop reasons rather than trusting the
hashes; a deliberately destructive control among the tests. Conventions carried over from the workspace records:
admission before reading, refusal on an incomplete store, the same-state control as a hard requirement.

## What this review did not do

It did not run anything on the card, load a Gemma 4 checkpoint, or fetch weights; it did not merge or commit on the
branch; it did not execute the design's device gates. The runtime is verified on a two-layer, sixteen-wide random
Gemma 4 text model on a CPU, which certifies the adapter's bookkeeping, not Gemma 4 compatibility at scale.

## Requested of Codex

1. F1: digest the config through `to_json_string()`; run the four test modules and record the collected count and the
   environment in the plan's acceptance status (67 collected on the reviewer's laptop).
2. F2: the two inspector fields and the refusal.
3. F3 and F5: the two sentences in the spec, and the differing keys in the comparison's refusal.
4. Then the importer (F6), with its own tests: rendering under the original template and thinking setting, ids
   reconciled against the supplied trajectory, the correspondence flag set only by that path.

## Second review, Chief, 2026-09-10 (~11:05Z) — the corrections and the importer, uncommitted on `codex/gemma4-interpretability`

Read: `importer.py` (286 lines) and its 26 tests, and the four corrections in place. Executed on the laptop's CPU
with caches in the session scratchpad: 107 collected, 107 passed, none skipped (the worktree untouched).

**The four corrections are in.** F1: the config digest goes through `to_json_string(use_diff=False)`, so every
HF config serialises; the five runtime tests that failed on the integer `id2label` keys pass. F2: the inspector
reports `tie_word_embeddings` and `lm_head_present` and refuses an untied checkpoint without a readout. F3: the spec
now says routing is observed on the layer's own MoE input after `pre_feedforward_layernorm_2`, not on the recorded
residual. F5: a comparison refusal names the differing runtime keys.

**The importer does what the first review asked for, and refuses the right things.** It renders the case's
messages through the installed HF chat-template renderer with the literal template text (a template name is
refused), tokenises the rendered text and checks it against the tokenizer's own direct tokenisation, then
requires the supplied `prompt_ids` to equal the rendered ids exactly, reporting the first differing index. For
replay it renders the full turn, requires the prompt rendering to be a prefix, and requires the supplied
`replay_ids` to equal the suffix — so a missing terminator is a refusal, never an appended token. The thinking
setting must be a known boolean bound to a named template argument that the template actually references;
reserved rendering controls cannot be smuggled through `template_kwargs`; time-dependent and random templates
are refused; the tokenizer's backend is fingerprinted before and after and must not change. The receipt
carries the rendering, the tokenizer pin and fingerprint, the rendered texts and their digests, the package
versions and the importer's own source hash, and is digested; `verify_import` re-renders and compares rather
than trusting the stored boolean, which is the property that matters (`test_saved_boolean_does_not_establish_
correspondence`). Two limits are stated correctly in the code: `tokenizer_revision_verified` is false — the pin
is the caller's claim until the files are hashed against a snapshot — and a receipt shows correspondence with a
template, not that the source service used that template.

**Requested before the real exchange is imported.** (1) The tokenizer pin should be checked the way the checkpoint
is: hash `tokenizer.json`, `tokenizer_config.json` and the template file of the named snapshot into the receipt
and compare on verification, so `tokenizer_revision_verified` can become true by evidence rather than stay false
by design. (2) One real-tokenizer regression on the CPU with the actual Gemma 4 tokenizer files once Daniel has
downloaded a checkpoint — the synthetic fast tokenizer proves the reconciliation rules, not Gemma 4's template.
Neither blocks committing the slice. Verdict: accept; commit on the branch; the next slice is the import of the
actual exchange, then gates 1–5 as designed. Nothing merged, nothing run on the card.

*2026-09-10 UTC, ~11:35Z:* committed on the branch: `542a1cd` (the accepted foundation and importer) and `7ff43d2` (the tokenizer snapshot files hashed into the receipt, the verified flag requiring matching local snapshot, template and tokenizer files, and a real-tokenizer regression that is skipped until a Gemma 4 checkpoint exists — 130 passed, 1 skipped). Both requests of the second review met. Nothing merged; the branch waits on the direction going live and on a checkpoint Daniel downloads.
