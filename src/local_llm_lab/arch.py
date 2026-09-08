"""Structural access to decoder-only language-model backbones.

The view is the single compatibility boundary for hand-running decoder blocks.  In particular,
it always asks the model's mask helpers for both attention kinds; passing ``None`` as a mask
shortcut would make multi-token probe activations disagree with inference.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any, Literal

__all__ = ["ArchitectureView", "LORA_POLICIES", "NativeCapture"]

_ACTIVE_CAPTURES: set[int] = set()

LORA_POLICIES = frozenset({"auto", "attention+mlp", "all-linear"})

_TEXT_MODULE_PATHS = ((), ("model",), ("language_model", "model"))
_TEXT_MODULE_MEMBERS = ("embed_tokens", "layers", "norm")
_ATTENTION_KIND = "attention"
_LINEAR_ATTENTION_KIND = "linear_attention"
_DENSE_LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_LINEAR_ATTENTION_LORA_SUFFIXES = (
    "in_proj_qkvz",
    "in_proj_ba",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


class ArchitectureView:
    """A model-type-independent view of one decoder's embeddings, blocks, and readout."""

    def __init__(self, model: Any, text_module: Any) -> None:
        self.model = model
        self.text_module = text_module
        self.blocks = list(text_module.layers)
        if not self.blocks:
            raise ValueError("text module must contain at least one decoder block")
        self.num_layers = len(self.blocks)
        self.vocab_size, self.hidden_size = _embedding_dimensions(text_module.embed_tokens)
        self._unembed_module = _find_unembedding(model, text_module)
        self.tie_word_embeddings = self._unembed_module is None
        if self.tie_word_embeddings:
            if not callable(getattr(text_module.embed_tokens, "as_linear", None)):
                raise ValueError("tied embed_tokens must expose as_linear")
        else:
            output_size, input_size = _linear_dimensions(self._unembed_module)
            if (output_size, input_size) != (self.vocab_size, self.hidden_size):
                raise ValueError(
                    "lm_head dimensions must be (vocab_size, hidden_size); "
                    f"got {(output_size, input_size)}"
                )
        self._attention_mask = _find_mask_helper(model, text_module, "create_attention_mask")
        self._ssm_mask = _find_mask_helper(model, text_module, "create_ssm_mask")

    @classmethod
    def from_model(cls, model: Any) -> ArchitectureView:
        """Discover the decoder structurally, without consulting a model-type string."""
        for path in _TEXT_MODULE_PATHS:
            candidate = model
            for member in path:
                candidate = getattr(candidate, member, None)
                if candidate is None:
                    break
            if candidate is not None and all(
                hasattr(candidate, member) for member in _TEXT_MODULE_MEMBERS
            ):
                return cls(model, candidate)
        members = ", ".join(_TEXT_MODULE_MEMBERS)
        raise ValueError(
            "could not find a text module owning "
            f"{members} at model, model.model, or model.language_model.model"
        )

    def layer_kind(self, index: int) -> Literal["attention", "linear_attention"]:
        """Return the block's structurally declared attention kind."""
        block = self.blocks[self._validate_block_index(index)]
        if bool(getattr(block, "is_linear", False)):
            return _LINEAR_ATTENTION_KIND
        return _ATTENTION_KIND

    def attention_span(self, index: int) -> str | None:
        """Which mask the model hands this block: ``"global"``, ``"sliding"``, or ``None``.

        **Beside :meth:`layer_kind`, deliberately not inside it.** ``layer_kind`` answers what
        module the block is, and that answer is written into ``ResolvedSpec.layer_types`` in every
        recorded manifest and gates attention capture, which refuses any block whose kind is not
        ``attention``. Widening it to carry the window would reinterpret every existing record and
        refuse five Gemma blocks in six.

        This answers what the block can *see*, which is a property of the mask and not of the
        module. On Gemma the two are orthogonal: every block is ``attention``, and five in six are
        ``sliding``. ``None`` where the family hands every block the same mask, which is every
        model this repository ran before Gemma.

        Read from the attention module's own ``is_sliding`` where it exists
        (``gemma3_text.py:55``), so this reports what the model dispatches on rather than a rule
        rederived from a period.
        """
        block_index = self._validate_block_index(index)
        attention = getattr(self.blocks[block_index], "self_attn", None)
        sliding = getattr(attention, "is_sliding", None)
        if not isinstance(sliding, bool):
            return None
        return "sliding" if sliding else "global"

    def embed(self, ids: Any) -> Any:
        """Return the layer-zero residual as a batched float32 activation.

        **The model's own layer-zero residual, observed rather than reconstructed.** This used to
        be ``embed_tokens(ids)``, which is the residual only on a family whose entry transform is
        the identity. Gemma 3 multiplies the embedding by ``sqrt(hidden_size)``, about 50.6 at
        2560, rounded through bfloat16 first, so the reconstruction was out by a factor of fifty
        on every layer of every reading. A family that adds or normalises instead would have been
        wrong differently, and nothing here would have said so.

        One observing forward: whatever the model does before its first block has been done by the
        time the first block is called, so this inherits it without naming it.
        """
        import mlx.core as mx

        token_ids = mx.array(ids).astype(mx.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
        if token_ids.ndim != 2:
            raise ValueError(f"token ids must have one or two dimensions; got {token_ids.ndim}")
        entry, _ = self._observe_forward(token_ids)
        return entry.astype(mx.float32)

    def native_readout(self, h: Any) -> Any:
        """Apply the installed norm and unembedding without altering native precision."""
        normalized = self.text_module.norm(h)
        if self._unembed_module is None:
            return self.text_module.embed_tokens.as_linear(normalized)
        return self._unembed_module(normalized)

    def masks(
        self,
        h: Any,
        cache: list[Any] | None,
        *,
        hidden_spans: Sequence[tuple[int, int]] | None = None,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the attention and SSM masks through the model's own helpers.

        ``hidden_spans`` hides absolute key positions from the **attention** blocks only, as
        half-open ``(start, end)`` ranges over ``[0, offset + N)``.  The recurrent blocks keep
        the mask the model's own SSM helper returns, because the asymmetry between the two
        paths is the measurement EXP-002 makes; masking both would measure nothing.  Hiding is
        never deletion: the columns stay in place, so every later position keeps its index and
        the rotary embeddings are untouched.

        ``None`` (the default) leaves this method exactly as inference runs it.  An empty
        sequence forces the explicit array and hides nothing, which is the only way to compare
        the array route against the sentinel route on an otherwise identical forward.

        ``record``, when given, is filled in **by the branch that runs** with the route's name,
        the resolved column ranges the mask indexed and the shape it came out at.  Two mask
        forms exist by design and either can be correct, so an artifact that only said which
        spans were *requested* would leave a reader unable to tell which code path produced the
        number in front of them -- and the two come apart precisely when something is wrong.
        """
        import mlx.core as mx

        # Ids of the right length, not ``h``. The mask constructors read the sequence length and
        # the cache and never the token values, so this observes exactly the masks the model
        # would build for a sequence of this shape -- in the model's own embedding dtype, which
        # is what inference uses and better than the caller's. Passing ``h`` itself would be
        # wrong twice over: callers hand mid-network residuals to this method
        # (``lens_fitting.jacobian.pre_norm_tail`` does), and Gemma would scale whatever it was
        # given by about fifty and return it looking untouched.
        stand_in = mx.zeros((1, int(h.shape[1])), dtype=mx.int32)
        _, observed = self._observe_forward(stand_in, cache=cache)
        # The model builds its masks in its own embedding dtype, which is right -- they are its
        # masks. But a caller running the loop in float32 applies them to a float32 residual, and
        # this method used to hand back masks built from the caller's own `h` and therefore in its
        # dtype. An additive mask carried at a narrower precision than the stream it is added to
        # is a silent precision change, so the content stays the model's and the dtype follows the
        # residual. Sentinels and `None` pass through untouched.
        by_block = {
            index: (
                observed[index].astype(h.dtype)
                if isinstance(observed[index], mx.array)
                else observed[index]
            )
            for index in range(self.num_layers)
        }
        if hidden_spans is None:
            if record is not None:
                attention_cache = self._first_cache(
                    self._cache_entries(cache), _ATTENTION_KIND
                )
                record["attention_mask_route"] = "model helper, unmodified"
                record["hidden_spans"] = ()
                record["query_tokens"] = int(h.shape[1])
                record["cache_offset"] = _cache_offset(attention_cache)
                record["attention_mask_shape"] = _mask_shape(
                    by_block[self._first_attention_block()]
                )
            return by_block

        # Span masking hides key columns from attention blocks only, because EXP-002's whole
        # measurement is the asymmetry between an attention path and a recurrent one. On a family
        # where every block is attention there is no asymmetry to measure, and applying it to all
        # of them would answer a different question in the same artifact.
        kinds = {self.layer_kind(index) for index in range(self.num_layers)}
        if _LINEAR_ATTENTION_KIND not in kinds:
            raise ValueError(
                "hidden_spans measures the asymmetry between an attention path and a recurrent "
                "one, and this decoder has no recurrent block; the span-masked experiment is "
                "deferred on a dense backbone rather than silently applied to every block"
            )
        entries = self._cache_entries(cache)
        attention_cache = self._first_cache(entries, _ATTENTION_KIND)
        masked = self._span_masked_attention(h, attention_cache, hidden_spans, record)
        return {
            index: masked if self.layer_kind(index) == _ATTENTION_KIND else by_block[index]
            for index in range(self.num_layers)
        }

    def _first_attention_block(self) -> int:
        """The lowest block the view calls attention; for the record's shape field only."""
        for index in range(self.num_layers):
            if self.layer_kind(index) == _ATTENTION_KIND:
                return index
        return 0

    def _span_masked_attention(
        self,
        h: Any,
        attention_cache: Any | None,
        hidden_spans: Sequence[tuple[int, int]],
        record: dict[str, Any] | None = None,
    ) -> Any:
        """The model's own attention mask as a boolean array, with hidden columns set False.

        Two returns have to be handled because they are the same library function on different
        inputs, both measured against the real caches rather than read off ``base.py``:

        * an array comes back for ``N > 1``, shaped ``(N, offset + N)`` with ``True`` meaning
          *attend*.  The span is **ANDed into it**, never substituted for it, or the cache's
          own causality and offset handling would be discarded along with it.
        * ``None`` comes back at ``N == 1``, with ``return_array=True`` as well and whether or
          not a cache is present, so no flag forces an array out of the library at a single
          scored step.  There is nothing to AND into, so the whole ``(1, offset + 1)`` row is
          built here -- through the library's own causal-mask constructor, so the row agrees
          with what the cached route would have produced.
        """
        import mlx.core as mx
        from mlx_lm.models.base import create_causal_mask

        queries = int(h.shape[1])
        offset = _cache_offset(attention_cache)
        total = offset + queries
        mask = self._attention_mask(h, attention_cache, return_array=True)
        route = "library array, hidden columns ANDed"
        if mask is None:
            mask = create_causal_mask(queries, offset)
            route = "row constructed at a single step"
        elif not isinstance(mask, mx.array):
            # A helper that answers the sentinel despite ``return_array`` would take a
            # different attention path than the array the span has to be ANDed into.
            raise ValueError(
                f"attention mask helper returned {mask!r} for return_array=True; "
                "an explicit boolean array is required to hide a span"
            )
        columns = mx.arange(total)
        hidden = mx.zeros((total,), dtype=mx.bool_)
        resolved: list[tuple[int, int]] = []
        for span in hidden_spans:
            start, end = _validate_hidden_span(span, total)
            resolved.append((start, end))
            hidden = hidden | ((columns >= start) & (columns < end))
        # The column axis is last on both the 2-D and the padded 4-D mask shapes, so one row
        # of hidden columns broadcasts across every query and head without reshaping.
        masked = mask & ~hidden
        if record is not None:
            # Written after the AND, from the values the mask was actually built from, so the
            # artifact carries the derivation's output rather than its input.
            record["attention_mask_route"] = route
            record["hidden_spans"] = tuple(resolved)
            record["query_tokens"] = queries
            record["cache_offset"] = offset
            record["attention_mask_shape"] = _mask_shape(masked)
        return masked

    def run_block(
        self,
        index: int,
        h: Any,
        masks: dict[str, Any],
        cache_i: Any | None,
    ) -> Any:
        """Run one block with the mask the model itself hands that block.

        Keyed by **block index**, not by kind. Kind was enough while every family built one mask
        per kind; Gemma builds a global mask and a windowed one and dispatches on the index, so
        two blocks of the same kind get different masks and a kind-keyed mapping cannot say which.
        """
        import mlx.core as mx

        block_index = self._validate_block_index(index)
        if block_index not in masks:
            raise ValueError(f"missing mask for block {block_index}")
        return self.blocks[block_index](
            h, mask=masks[block_index], cache=cache_i
        ).astype(mx.float32)

    def final_norm(self, h: Any) -> Any:
        """Apply the decoder's final norm and return a float32 activation."""
        import mlx.core as mx

        return self.text_module.norm(h).astype(mx.float32)

    def diagnostic_native_final_residual(self, ids: Any) -> Any:
        """Replay the native no-cache residual loop without diagnostic dtype promotion."""
        import mlx.core as mx

        token_ids = mx.array(ids).astype(mx.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
        if token_ids.ndim != 2:
            raise ValueError(f"token ids must have one or two dimensions; got {token_ids.ndim}")
        hidden = self.text_module.embed_tokens(token_ids)
        masks = self.masks(hidden, None)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, mask=masks[index], cache=None)
        return self.text_module.norm(hidden)

    def unembed(self, h: Any) -> Any:
        """Project hidden states to vocabulary logits through the tied or untied readout."""
        import mlx.core as mx

        if self.tie_word_embeddings:
            logits = self.text_module.embed_tokens.as_linear(h)
        else:
            logits = self._unembed_module(h)
        return logits.astype(mx.float32)

    def make_cache(self) -> list[Any]:
        """Create the model-native per-block cache and validate its cardinality."""
        factory = getattr(self.model, "make_cache", None)
        if callable(factory):
            cache = list(factory())
        else:
            from mlx_lm.models.cache import make_prompt_cache

            cache = list(make_prompt_cache(self.model))
        if len(cache) != self.num_layers:
            raise ValueError(
                f"cache factory returned {len(cache)} entries for {self.num_layers} blocks"
            )
        return cache

    @property
    def cache_trimmable(self) -> bool:
        """Whether every native per-block cache supports prefix trimming."""
        cache = self.make_cache()
        for entry in cache:
            check = getattr(entry, "is_trimmable", None)
            if not callable(check):
                raise ValueError(f"cache {type(entry).__name__} does not expose is_trimmable()")
            if not bool(check()):
                return False
        return True

    def lora_targets(self, policy: str | tuple[str, ...]) -> tuple[str, ...]:
        """Resolve a policy to explicit, existing linear-module paths."""
        modules = self._linear_modules()
        available = {path for _, path, _ in modules}
        if isinstance(policy, tuple):
            if not policy:
                raise ValueError("explicit LoRA targets must not be empty")
            if len(set(policy)) != len(policy):
                raise ValueError("explicit LoRA targets must not contain duplicates")
            unknown = tuple(path for path in policy if path not in available)
            if unknown:
                raise ValueError(f"unknown LoRA target(s): {', '.join(unknown)}")
            return policy
        if policy not in LORA_POLICIES:
            raise ValueError(
                f"unknown LoRA policy {policy!r}; expected one of {sorted(LORA_POLICIES)}"
            )
        if policy == "auto":
            hybrid = any(
                self.layer_kind(index) == _LINEAR_ATTENTION_KIND
                for index in range(self.num_layers)
            )
            policy = "all-linear" if hybrid else "attention+mlp"
        suffixes = list(_DENSE_LORA_SUFFIXES)
        if policy == "all-linear":
            suffixes.extend(_LINEAR_ATTENTION_LORA_SUFFIXES)
        selected: list[str] = []
        for suffix in suffixes:
            for block_index, path, _ in modules:
                if path.rsplit(".", 1)[-1] != suffix:
                    continue
                if suffix in _LINEAR_ATTENTION_LORA_SUFFIXES and self.layer_kind(
                    block_index
                ) != _LINEAR_ATTENTION_KIND:
                    continue
                if path not in selected:
                    selected.append(path)
        if not selected:
            raise ValueError(f"LoRA policy {policy!r} matched no linear modules")
        return tuple(selected)

    def lora_parameter_count(self, keys: tuple[str, ...], rank: int) -> int:
        """Count implied LoRA A/B parameters without changing or dequantizing weights."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"LoRA rank must be a positive integer; got {rank!r}")
        resolved = set(self.lora_targets(tuple(keys)))
        total = 0
        for _, path, module in self._linear_modules():
            if path in resolved:
                output_size, input_size = _linear_dimensions(module)
                total += rank * (input_size + output_size)
        return total

    def residuals(self, ids: Any, layers: Sequence[int]) -> dict[int, Any]:
        """Capture requested residual indices in one pass, stopping at the deepest index."""
        wanted = sorted({int(layer) for layer in layers})
        if not wanted or wanted[0] < 0 or wanted[-1] > self.num_layers:
            raise ValueError(
                f"layers must be a non-empty sequence within [0, {self.num_layers}]; got {wanted}"
            )
        h = self.embed(ids)
        masks = self.masks(h, None)
        captured: dict[int, Any] = {}
        if 0 in wanted:
            captured[0] = h
        for index in range(wanted[-1]):
            h = self.run_block(index, h, masks, None)
            if index + 1 in wanted:
                captured[index + 1] = h
        return captured

    def _observe_forward(self, ids: Any, cache: Any = None):
        """Ask the model what it does before and between its blocks, instead of describing it.

        Runs the text module's own ``__call__`` with every block replaced by a proxy that records
        the tensor and the mask it was handed and returns the tensor unchanged. Nothing is
        computed: no attention, no MLP. What comes back is ``(first_input, {block index: mask})``.

        **Why observe rather than reimplement.** Everything before the first block has run by the
        time the first proxy is called -- the embedding lookup, Gemma 3's ``sqrt(hidden_size)``
        entry scale rounded through bfloat16 (``gemma3_text.py:190``), any future family's entry
        transform -- and every mask the model would build has been built by the model's own code
        from its own cache selection. Gemma builds two, a global one from ``cache[pattern - 1]``
        and a windowed one from ``cache[0]``, and dispatches on the block index; Qwen builds one
        per block kind and dispatches on ``layer.is_linear`` (``qwen3_5.py:268-273``). Neither is
        described here.

        **The proxy forwards attribute access to the real block**, because Qwen dispatches on the
        block's own ``is_linear`` while Gemma dispatches on the loop index. A proxy that answered
        neither would change the very masks it exists to observe.

        **Driven by token ids and never by ``input_embeddings``.** Two reasons, and the second is
        the one that changed this design. Gemma applies its entry scale *after* the
        ``input_embeddings`` branch, unconditionally, so a residual handed in that way comes back
        multiplied by about fifty and looking untouched — a footgun for any caller holding a
        mid-network residual, which ``pre_norm_tail`` does. And requiring that keyword narrows
        what this can read: several stand-in decoders accept ids and nothing else, and a view that
        cannot observe them is a view that cannot be tested without a checkpoint. Masks need only
        a sequence length and a cache, so ids of the right length are enough and are what every
        decoder takes.
        """

        module = self.text_module
        blocks = list(module.layers)
        seen: dict[int, Any] = {}
        entry: list[Any] = []

        class _Watch:
            """A block-shaped proxy that records and computes nothing."""

            def __init__(self, inner, index):
                self._inner = inner
                self._index = index

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __call__(self, x, mask=None, cache=None, *args, **kwargs):
                if not entry:
                    entry.append(x)
                seen[self._index] = mask
                return x

        original = list(module.layers)
        try:
            module.layers = [_Watch(block, index) for index, block in enumerate(blocks)]
            module(ids, cache)
        finally:
            module.layers = original
        if not entry or len(seen) != len(blocks):
            raise ValueError(
                f"the decoder ran {len(seen)} of {len(blocks)} blocks under observation; its "
                "forward does not iterate its own layer list, so this view cannot read it"
            )
        return entry[0], seen

    def native_residuals(self, ids: Any, layers: Sequence[int]) -> dict[int, Any]:
        """The same residuals as :meth:`residuals`, from the model's **own** forward.

        Identical contract -- ``{layer: (1, positions, hidden_size)}``, no cache, the final
        layer taken after the last block and before the final norm -- and a different producer.
        :meth:`residuals` re-runs the decoder through this class's own loop, which is where the
        model-specific assumptions live: the entry transform, the mask construction, the mask
        selection per block. This one taps the model's forward through :class:`NativeCapture`,
        so all three are the model's.

        **Why both exist.** On a family whose loop this class describes correctly the two agree
        exactly, and the check below proves it rather than asserting it. On Gemma 3 they do not,
        because the hand-run loop omits a ``sqrt(hidden_size)`` entry scale and builds one mask
        where the model builds two; the native path inherits all of it and is correct today.
        That disagreement is not a defect in this method -- it is the measurement the
        architecture-view port has to close, and it is the port's acceptance evidence.

        A fit that needs residuals and not derivatives can therefore run on Gemma now. Anything
        that perturbs a residual and re-runs a tail still needs the loop, and still needs the
        port.
        """
        import mlx.core as mx

        wanted = sorted({int(layer) for layer in layers})
        if not wanted or wanted[0] < 0 or wanted[-1] > self.num_layers:
            raise ValueError(
                f"layers must be a non-empty sequence within [0, {self.num_layers}]; got {wanted}"
            )
        token_ids = mx.array(ids).astype(mx.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[None, :]
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("native residuals take an unpadded batch of one")

        captured: dict[int, Any] = {}

        class _Collect:
            """The sink protocol's three methods; only ``residual`` carries anything here."""

            def residual(self, layer, offset, h):
                captured[int(layer)] = h

            def output(self, offset, ids, logits):
                return None

            def head(self, *args, **kwargs):  # pragma: no cover - head capture is off
                return None

        with NativeCapture(self, _Collect(), layers=tuple(wanted)) as wrapped:
            wrapped(token_ids)
        missing = set(wanted) - set(captured)
        if missing:  # pragma: no cover - defensive; the tap emits every requested layer
            raise ValueError(f"native capture emitted no residual for layers {sorted(missing)}")
        return captured

    def residual_source_agreement(
        self, ids: Any, layers: Sequence[int]
    ) -> dict[int, float]:
        """Per-layer maximum absolute difference between the two residual producers.

        Zero on every layer means this class's loop reproduces the model's forward exactly, and
        substituting one for the other is a change of producer and not of result. Non-zero says
        which layers the loop gets wrong and by how much, which is what the port's acceptance
        reads.
        """
        import mlx.core as mx

        loop = self.residuals(ids, layers)
        native = self.native_residuals(ids, layers)
        return {
            layer: float(
                mx.max(mx.abs(loop[layer].astype(mx.float32) - native[layer].astype(mx.float32)))
                .item()
            )
            for layer in sorted(loop)
        }

    def cached_logits(
        self,
        ids: Any,
        cache: list[Any] | None,
        *,
        hidden_spans: Sequence[tuple[int, int]] | None = None,
        position: int = -1,
        record: dict[str, Any] | None = None,
    ) -> Any:
        """Run every block over ``cache`` and return the logits at one position.

        ``residuals`` runs with no cache and ``tail`` takes none, so neither can score a
        decision under a cache that persists across turns.  This one does: it advances the
        entries it is given, so successive calls continue one sequence rather than restarting
        it, and ``cache=None`` runs the plain uncached forward.

        The per-kind masks are built here rather than accepted from the caller because their
        shape is a function of the very ``h`` and cache offset this forward uses; a dict built
        anywhere else can only be right by accident.  ``hidden_spans`` is therefore how a
        caller hides text from the attention blocks -- see ``masks``, which also documents
        ``record``: the mask route and resolved columns this forward really used, for an
        artifact that has to say what happened rather than what was asked for.

        Only the scored row is unembedded.  Projecting every position to a vocabulary this size
        would dominate the cost of a probe point that reads exactly one distribution.  The
        residual stream is bit-identical to the model's own forward either way; unembedding one
        row rather than the whole sequence is a differently shaped matmul, which costs up to one
        float32 epsilon on a logit.  Two calls through this method cancel it, so an identity
        gate comparing two such calls is unaffected.
        """
        entries = self._cache_entries(cache)
        h = self.embed(ids)
        index = _validate_scored_position(position, int(h.shape[1]))
        masks = self.masks(h, entries, hidden_spans=hidden_spans, record=record)
        for block_index in range(self.num_layers):
            h = self.run_block(block_index, h, masks, entries[block_index])
        return self.unembed(self.final_norm(h)[:, index, :])

    def tail(self, layer: int) -> Callable[[Any], Any]:
        """Return ``blocks[layer:] + final_norm`` with model-native per-kind masks."""
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer <= self.num_layers
        ):
            raise ValueError(f"layer must lie in [0, {self.num_layers}]; got {layer!r}")

        def apply_tail(h: Any) -> Any:
            import mlx.core as mx

            hidden = h.astype(mx.float32)
            masks = self.masks(hidden, None)
            for index in range(layer, self.num_layers):
                hidden = self.run_block(index, hidden, masks, None)
            return self.final_norm(hidden)

        return apply_tail

    def _cache_entries(self, cache: list[Any] | None) -> list[Any | None]:
        """One cache entry per block, or one ``None`` per block when running uncached."""
        if cache is None:
            return [None] * self.num_layers
        entries = list(cache)
        if len(entries) != self.num_layers:
            raise ValueError(
                f"cache must contain one entry per block; got {len(entries)} "
                f"for {self.num_layers}"
            )
        return entries

    def _first_cache(self, cache: list[Any | None], kind: str) -> Any | None:
        for index, entry in enumerate(cache):
            if self.layer_kind(index) == kind:
                return entry
        return None

    def _linear_modules(self) -> list[tuple[int, str, Any]]:
        """Every linear projection per block, seen through adapter wrappers.

        Loading a policy with an adapter (``mlx_lm.load(..., adapter_path=...)``) replaces
        each projection with a ``LoRALinear``: a plain ``nn.Module`` that keeps the original at
        ``.linear``. The wrapper is not a linear type, and its child sits at ``<path>.linear``,
        whose suffix matches no LoRA target. So a wrapper is reported under *its own* path with
        the *base* module, and the base is not reported a second time. Paths therefore stay
        identical with or without an adapter, which adapter comparison and provenance rely on,
        and dimension readers receive the module that owns ``weight`` and ``bits``.
        """
        import mlx.nn as nn

        linear_types = (nn.Linear, nn.QuantizedLinear)
        found: list[tuple[int, str, Any]] = []
        for index, block in enumerate(self.blocks):
            if not callable(getattr(block, "named_modules", None)):
                continue
            entries = [(path, module) for path, module in block.named_modules() if path]
            wrapped: dict[str, Any] = {}
            for path, module in entries:
                if isinstance(module, linear_types):
                    continue
                base = getattr(module, "linear", None)
                if isinstance(base, linear_types):
                    wrapped[path] = base
            for path, module in entries:
                if path in wrapped:
                    found.append((index, path, wrapped[path]))
                    continue
                if any(path.startswith(prefix + ".") for prefix in wrapped):
                    continue
                if isinstance(module, linear_types):
                    found.append((index, path, module))
        return found

    def _validate_block_index(self, index: int) -> int:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.num_layers
        ):
            raise ValueError(f"block index must lie in [0, {self.num_layers - 1}]; got {index!r}")
        return index


def _mask_shape(mask: Any) -> tuple[int, ...] | None:
    """The mask's shape for the record, or ``None`` where the route produced no array.

    Both the ``'causal'`` sentinel and the library's ``None`` at a single step are legitimate
    returns of the unmodified helper, and neither has a shape; reporting ``None`` for them says
    so, where reaching for ``.shape`` would turn a recorded fact into an exception.
    """
    shape = getattr(mask, "shape", None)
    return tuple(int(value) for value in shape) if shape is not None else None


def _cache_offset(entry: Any | None) -> int:
    """Absolute key position already held by an attention cache entry.

    Every attention cache mlx-lm builds carries ``offset``; a missing one is raised on rather
    than defaulted to zero, because a silent zero would put the span's columns at the wrong
    absolute positions and hide the wrong text while the forward still ran cleanly.
    """
    if entry is None:
        return 0
    offset = getattr(entry, "offset", None)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError(
            f"attention cache {type(entry).__name__} must expose a non-negative integer "
            f"offset; got {offset!r}"
        )
    return offset


def _validate_scored_position(position: Any, length: int) -> int:
    """Resolve the scored row, allowing Python's negative indexing from the end."""
    if isinstance(position, bool) or not isinstance(position, int):
        raise ValueError(f"scored position must be an integer; got {position!r}")
    if not -length <= position < length:
        raise ValueError(
            f"scored position must lie in [{-length}, {length - 1}]; got {position}"
        )
    return position


def _validate_hidden_span(span: Any, total: int) -> tuple[int, int]:
    """Check one half-open ``(start, end)`` span against the key range it must lie in.

    An empty or inverted span is rejected rather than ignored: it would hide nothing while the
    arm still reported itself as masked, which is the failure mode that reads as a null.
    """
    try:
        pair = tuple(span)
    except TypeError as error:
        raise ValueError(f"hidden span must be a (start, end) pair; got {span!r}") from error
    if len(pair) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in pair
    ):
        raise ValueError(f"hidden span must be a pair of integers; got {span!r}")
    start, end = pair
    if not 0 <= start < end <= total:
        raise ValueError(
            f"hidden span must satisfy 0 <= start < end <= {total}; got {span!r}"
        )
    return start, end


def _embedding_dimensions(embedding: Any) -> tuple[int, int]:
    quantized_vocab = getattr(embedding, "num_embeddings", None)
    quantized_hidden = getattr(embedding, "dims", None)
    if isinstance(quantized_vocab, int) and isinstance(quantized_hidden, int):
        if quantized_vocab <= 0 or quantized_hidden <= 0:
            raise ValueError("embed_tokens dimensions must be positive")
        return quantized_vocab, quantized_hidden
    weight = getattr(embedding, "weight", None)
    shape = getattr(weight, "shape", ())
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError(f"embed_tokens.weight must have positive rank-2 shape; got {shape}")
    return int(shape[0]), int(shape[1])


def _find_unembedding(model: Any, text_module: Any) -> Any | None:
    owners = (model, getattr(model, "language_model", None), text_module)
    for owner in owners:
        head = getattr(owner, "lm_head", None) if owner is not None else None
        if head is not None:
            return head
    return None


def _find_mask_helper(model: Any, text_module: Any, name: str) -> Callable[..., Any]:
    owners = (text_module, model, getattr(model, "language_model", None))
    for owner in owners:
        helper = getattr(owner, name, None) if owner is not None else None
        if callable(helper):
            return helper
    for owner in owners:
        module = sys.modules.get(type(owner).__module__) if owner is not None else None
        helper = getattr(module, name, None) if module is not None else None
        if callable(helper):
            return helper
    from mlx_lm.models import base

    helper = getattr(base, name, None)
    if not callable(helper):
        raise ValueError(f"model does not provide required mask helper {name}")
    return helper


def _linear_dimensions(module: Any) -> tuple[int, int]:
    import mlx.nn as nn

    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", ())
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError(f"linear weight must have positive rank-2 shape; got {shape}")
    output_size, input_size = int(shape[0]), int(shape[1])
    if isinstance(module, nn.QuantizedLinear):
        bits = getattr(module, "bits", None)
        if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0 or 32 % bits:
            raise ValueError(f"quantized linear has unsupported bits value {bits!r}")
        input_size = input_size * 32 // bits
    return output_size, input_size


class NativeCapture(AbstractContextManager):
    """Observe installed native forwards without replacing their arithmetic or masks.

    Attention rows are diagnostic recomputations from native queries and cached keys.
    Head contributions use the actual post-gate projection input. Neither diagnostic feeds
    generation. Wrappers are scoped and restored on every exit.
    """

    def __init__(self, view, sink, *, layers, attention_blocks=(), injection=None,
                 head_vectors=True):
        self.view = view
        self.model = view.model
        self.sink = sink
        self.layers = set(layers)
        self.attention_blocks = set(attention_blocks)
        self.injection = injection  # (post-block residual layer, absolute source, delta)
        self.head_vectors = head_vectors
        self._restore = []
        self._active = False
        self._injected = False
        self._offset = 0
        self._attention = {}
        if not self.layers or any(
            type(layer) is not int or not 0 <= layer <= view.num_layers for layer in self.layers
        ):
            raise ValueError("capture layers outside model")
        if any(
            not 0 <= b < view.num_layers or view.layer_kind(b) != "attention"
            for b in self.attention_blocks
        ):
            raise ValueError("attention capture requires an attention block")
        if injection is not None:
            layer, source, delta = injection
            if (
                type(layer) is not int
                or not 1 <= layer <= view.num_layers
                or type(source) is not int
                or source < 0
                or len(delta) != view.hidden_size
                or any(not math.isfinite(float(value)) for value in delta)
            ):
                raise ValueError("invalid injection layer, source or direction")

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, ids, *args, **kwargs):
        if not self._active:
            raise RuntimeError("native capture must be entered before forwarding")
        if len(ids.shape) != 2 or ids.shape[0] != 1:
            raise ValueError("capture supports an unpadded batch of one")
        cache = kwargs.get("cache", args[0] if args else None)
        entries = [
            cache[i]
            for i in range(self.view.num_layers)
            if cache is not None and self.view.layer_kind(i) == "attention"
        ]
        self._offset = int(entries[0].offset) if entries else 0
        if any(int(c.offset) != self._offset for c in entries):
            raise ValueError("attention caches disagree on the current position")
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        # Two different things were behind one type check, and they are not the same risk.
        #
        # **Residual capture is safe under rotation.** It emits against ``self._offset``, the
        # monotone count of tokens the model has seen, which a rotating cache maintains exactly
        # as an ordinary one does; nothing in that path indexes a cache column.
        #
        # **Head capture is not.** It reads ``cache.state[0]`` and treats column *j* as absolute
        # source position *j* (see the attention reconstruction below). Under rotation a column
        # is not its position and at most ``max_size`` columns exist at all, so the same code
        # would produce plausible per-head numbers against the wrong positions.
        #
        # Gemma 3 makes this the difference between running and not: ``make_cache`` returns a
        # ``RotatingKVCache`` for every block whose index plus one is not divisible by six, which
        # is 29 of its 34, on every run rather than only long ones. So the gate narrows to what
        # it can actually justify rather than widening to a type.
        if any(type(c) not in (KVCache, RotatingKVCache) for c in entries):
            raise ValueError("capture requires unquantized KV caches")
        if self.attention_blocks and any(type(c) is RotatingKVCache for c in entries):
            raise ValueError(
                "head capture cannot read a rotating KV cache: it maps weight column j to "
                "absolute source position j, and under rotation a column is not its position "
                "and only the last max_size of them exist. Residual capture is unaffected and "
                "runs on a rotating cache; turn head capture off, or restrict it to the "
                "global-attention blocks, whose caches do not rotate."
            )
        if self._offset == 0:
            self._injected = False
        if self.injection is not None and self._offset > self.injection[1] and not self._injected:
            raise ValueError("injection source is already cached; rebuild from a fresh prefill")
        logits = self.model(ids, *args, **kwargs)
        self.sink.output(self._offset, ids[0].tolist(), logits)
        return logits

    def __enter__(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen3_next import Qwen3NextAttention

        if self._active or id(self.model) in _ACTIVE_CAPTURES:
            raise RuntimeError("native capture cannot be nested")
        self._active = True
        _ACTIVE_CAPTURES.add(id(self.model))
        owner = self

        class Tap(nn.Module):
            def __init__(self, inner, before=None, after=None):
                super().__init__()
                self.inner = inner
                self.before = before
                self.after = after

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if name == "inner":
                        raise
                    return getattr(super().__getattr__("inner"), name)

            def __call__(self, *args, **kwargs):
                if self.before is not None:
                    self.before(args, kwargs)
                out = self.inner(*args, **kwargs)
                return self.after(args, kwargs, out) if self.after is not None else out

        class ModelTap(Tap):
            # Keep the original model visible to MLX parameter traversal / wired_limit.
            def __call__(self, *args, **kwargs):
                return owner(*args, **kwargs)

        def replace(obj, key, value):
            previous = obj[key] if isinstance(obj, list) else getattr(obj, key)
            self._restore.append((obj, key, previous))
            if isinstance(obj, list):
                obj[key] = value
            else:
                setattr(obj, key, value)

        def remember(block, name):
            def after(args, kwargs, out):
                owner._attention[block][name] = (args[0], out)
                return out

            return after

        try:
            for index, block in enumerate(self.view.blocks):
                if index in self.attention_blocks:
                    att = block.self_attn
                    if not isinstance(att, Qwen3NextAttention):
                        raise ValueError("head capture currently supports Qwen3NextAttention only")
                    self._attention[index] = {"module": att, "projection": att.o_proj}
                    for name in ("q_norm", "k_norm", "o_proj"):
                        replace(att, name, Tap(getattr(att, name), after=remember(index, name)))

                def before(args, kwargs, i=index):
                    if i == 0 and 0 in owner.layers:
                        owner.sink.residual(0, owner._offset, args[0])
                    if i in owner.attention_blocks:
                        info = owner._attention[i]
                        info["mask"] = kwargs.get("mask", args[1] if len(args) > 1 else None)
                        info["cache"] = kwargs.get("cache", args[2] if len(args) > 2 else None)
                        if i not in owner.layers:
                            owner.sink.residual(i, owner._offset, args[0])

                def after(args, kwargs, out, i=index):
                    if owner.injection is not None and i + 1 == owner.injection[0]:
                        _, source, delta = owner.injection
                        local = source - owner._offset
                        if 0 <= local < out.shape[1]:
                            direction = mx.array(delta).astype(out.dtype)
                            # Zero is a strict no-op, preserving native rounding and sign bits.
                            if bool(mx.any(direction != 0).item()):
                                out = out.at[0, local].add(direction)
                            owner._injected = True
                    if i + 1 in owner.layers:
                        owner.sink.residual(i + 1, owner._offset, out)
                    if i in owner.attention_blocks:
                        owner._emit_head(i, mx)
                    return out

                replace(self.view.text_module.layers, index, Tap(block, before, after))
            return ModelTap(self.model)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _emit_head(self, block, mx):
        info = self._attention[block]
        att = info["module"]
        q = info["q_norm"][1].transpose(0, 2, 1, 3)
        q = att.rope(q, offset=self._offset)[:, :, -1:, :]
        cache = info["cache"]
        if cache is None:
            keys = att.rope(info["k_norm"][1].transpose(0, 2, 1, 3))
        else:
            keys = cache.state[0]
            if not hasattr(keys, "shape"):
                raise ValueError("quantized or rotating KV caches are not supported for capture")
        repeats = att.num_attention_heads // att.num_key_value_heads
        scores = (
            q.astype(mx.float32)
            @ mx.repeat(keys, repeats, axis=1).astype(mx.float32).swapaxes(-1, -2)
        ) * att.scale
        mask = info["mask"]
        if mask is not None and not isinstance(mask, str):
            last = mask[..., -1:, :]
            scores = (
                mx.where(last, scores, -float("inf")) if last.dtype == mx.bool_ else scores + last
            )
        elif isinstance(mask, str) and mask != "causal":
            raise ValueError(f"unsupported attention mask: {mask}")
        weights = mx.softmax(scores, axis=-1)[0, :, 0, :]
        gated, native_total = info["o_proj"]
        last = gated[:, -1:, :]
        contributions = []
        if self.head_vectors:
            for h in range(att.num_attention_heads):
                isolated = mx.zeros_like(last)
                begin, end = h * att.head_dim, (h + 1) * att.head_dim
                isolated = isolated.at[..., begin:end].add(last[..., begin:end])
                contributions.append(info["projection"](isolated)[0, 0])
        target = self._offset + gated.shape[1] - 1
        written = mx.stack(contributions) if self.head_vectors else None
        self.sink.attention(block, target, weights, written, native_total[0, -1])
        for key in ("q_norm", "k_norm", "o_proj", "cache", "mask"):
            info.pop(key, None)

    def __exit__(self, *exc):
        for obj, key, old in reversed(self._restore):
            if isinstance(obj, list):
                obj[key] = old
            else:
                setattr(obj, key, old)
        self._restore.clear()
        self._attention.clear()
        if self._active:
            _ACTIVE_CAPTURES.discard(id(self.model))
        self._active = False
        return False
