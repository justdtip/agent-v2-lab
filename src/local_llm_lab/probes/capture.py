"""Shared activation-capture infrastructure for the probe suite.

Everything here obeys the two numerical rules stated in ``research/representation_probes.md``
§3: every captured activation is float32 (float16 overflowed in the J-lens work), and every
capture uses the model's own causal mask. The mask helper is imported directly from
:mod:`local_llm_lab.pipeline.jlens` rather than reimplemented, so "the same mask the model
builds itself" is true by construction and cannot drift: calling transformer blocks directly
bypasses ``Qwen2Model.__call__``, which is where ``create_attention_mask`` is normally called,
and passing ``None`` would give a multi-token prompt bidirectional self-attention.

Layer indexing follows :func:`jlens.residual_at`: layer ``0`` is the embedding output, layer
``L`` is the residual stream after block ``L - 1``, and layer ``len(model.model.layers)`` is
the residual stream after the final block and before the final norm.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Sequence
from typing import Any

from local_llm_lab.pipeline.jlens import _as_token_ids, _causal_mask, _encode

__all__ = [
    "InjectionHook",
    "capture_residuals",
    "lora_block_mask",
    "response_mean_activations",
    "strip_state_fields",
]


# --------------------------------------------------------------------------- capture


def _take(h: Any, positions: str | Sequence[int]) -> Any:
    """Slice a ``(1, T, d)`` float32 residual stream down to the requested positions."""
    import mlx.core as mx

    length = h.shape[1]
    if positions == "last":
        return h[0, -1]
    if positions == "all":
        return h[0]
    indices = [int(index) if index >= 0 else length + int(index) for index in positions]
    for index in indices:
        if not 0 <= index < length:
            raise ValueError(f"position {index} out of range for a length-{length} sequence")
    return mx.take(h[0], mx.array(indices), axis=0)


def capture_residuals(
    model: Any,
    token_ids: Any,
    layers: Sequence[int],
    positions: str | Sequence[int] = "last",
) -> dict[int, Any]:
    """Post-block residual streams at several layers from **one** forward pass.

    ``layers`` are indices in ``[0, len(model.model.layers)]`` as described in the module
    docstring; the blocks are iterated once and a snapshot is taken whenever the index just
    completed was requested, so capturing six layers costs one forward pass rather than six
    (which is what calling :func:`jlens.residual_at` per layer would do).

    ``positions`` is ``"last"`` (one ``(d,)`` vector per layer), ``"all"`` (``(T, d)``), or an
    explicit sequence of token indices (``(len(positions), d)``, negatives counted from the
    end). Returns ``{layer: mx.array}`` in float32.
    """
    import mlx.core as mx

    if isinstance(positions, str):
        if positions not in ("last", "all"):
            raise ValueError(f"positions must be 'last', 'all', or a sequence: {positions!r}")
    else:
        positions = list(positions)

    wanted = sorted({int(layer) for layer in layers})
    if not wanted:
        raise ValueError("capture_residuals: at least one layer is required")
    depth = len(model.model.layers)
    if wanted[0] < 0 or wanted[-1] > depth:
        raise ValueError(f"layers must lie in [0, {depth}]; got {wanted}")

    ids = _as_token_ids(model, token_ids)
    h = model.model.embed_tokens(ids).astype(mx.float32)
    mask = _causal_mask(h)
    captured: dict[int, Any] = {}
    if wanted[0] == 0:
        captured[0] = _take(h, positions)
    deepest = wanted[-1]
    for index, block in enumerate(model.model.layers[:deepest]):
        h = block(h, mask, None).astype(mx.float32)
        if index + 1 in wanted:
            captured[index + 1] = _take(h, positions)
    return captured


def response_mean_activations(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
    layers: Sequence[int],
    *,
    stats: dict[str, int] | None = None,
) -> dict[int, Any]:
    """Mean residual stream over the *response* tokens only, per layer.

    This is the quantity the assistant-axis paper measures ("mean post-MLP residual stream
    activations at all response tokens"). The response span is found by tokenising the prompt
    alone and taking everything after it, which is only valid if the prompt's ids really are a
    prefix of the joint tokenisation; a merge across the prompt/response boundary can break
    that. The prefix is therefore asserted, and when it fails the joint sequence is rebuilt as
    ``encode(prompt) + encode(response)`` and the failure is counted in ``stats`` (keys
    ``"sequences"`` and ``"prefix_mismatch"``) so a caller can tell a clean run from one whose
    spans were repaired.
    """
    import mlx.core as mx

    if not response_text:
        raise ValueError("response_mean_activations: response_text is empty")
    prompt_ids = _encode(tokenizer, prompt_text)
    joint_ids = _encode(tokenizer, prompt_text + response_text)
    start = len(prompt_ids)
    if joint_ids[:start] != prompt_ids or len(joint_ids) <= start:
        joint_ids = [*prompt_ids, *_encode(tokenizer, response_text)]
        if stats is not None:
            stats["prefix_mismatch"] = stats.get("prefix_mismatch", 0) + 1
    if stats is not None:
        stats["sequences"] = stats.get("sequences", 0) + 1
    if len(joint_ids) <= start:
        raise ValueError("response_mean_activations: response tokenised to nothing")

    captured = capture_residuals(model, joint_ids, layers, positions="all")
    return {
        layer: mx.mean(activation[start:].astype(mx.float32), axis=0)
        for layer, activation in captured.items()
    }


# --------------------------------------------------------------------------- injection


class _InjectingBlock:
    """Wraps one transformer block, keeping its ``(x, mask, cache)`` signature."""

    def __init__(self, block: Any, hook: InjectionHook) -> None:
        self.block = block
        self.hook = hook

    def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
        return self.hook._apply(self.block(x, mask, cache), cache)

    def __getattr__(self, name: str) -> Any:  # delegate e.g. .self_attn for other tooling
        return getattr(self.__dict__["block"], name)


class InjectionHook:
    """Context manager that adds ``alpha * vector`` to a block's output.

    Wraps ``model.model.layers[layer]`` in place with a callable of the same
    ``(x, mask, cache)`` signature, so it works for a plain forward pass, for
    ``mlx_lm.stream_generate``, and with :class:`pipeline.runner.TurnCache` (the KV cache then
    stores post-injection keys, which is the intended behaviour: the steered state is what the
    rest of the trajectory attends to). The original block is restored on exit, including when
    the body raises.

    ``positions`` selects where the vector is added, in **absolute** token positions of the
    whole sequence:

    * ``"all"`` -- every position of every forward call;
    * ``("from", index)`` -- positions ``>= index``;
    * ``("at", index)`` or a bare ``int`` -- exactly that position.

    During cached generation a call only sees a slice of the sequence, so the absolute position
    of the first row is taken from the cache's ``offset`` (the value before the block updates
    it); without a cache the slice starts at zero.

    ``calls`` counts the forward calls intercepted and ``injected`` the subset in which at
    least one position was actually modified.
    """

    def __init__(
        self,
        model: Any,
        layer: int,
        vector: Any,
        alpha: float = 1.0,
        positions: str | int | tuple[str, int] = "all",
    ) -> None:
        import mlx.core as mx

        depth = len(model.model.layers)
        if not 0 <= int(layer) < depth:
            raise ValueError(f"layer must lie in [0, {depth - 1}]; got {layer}")
        self.model = model
        self.layer = int(layer)
        self.vector = mx.array(vector).astype(mx.float32).reshape(-1)
        self.alpha = float(alpha)
        self.mode, self.index = self._parse_positions(positions)
        self.calls = 0
        self.injected = 0
        self._original: Any = None

    @staticmethod
    def _parse_positions(positions: str | int | tuple[str, int]) -> tuple[str, int]:
        if positions == "all":
            return "all", 0
        if isinstance(positions, int):
            return "at", int(positions)
        if isinstance(positions, tuple) and len(positions) == 2 and positions[0] in ("from", "at"):
            return str(positions[0]), int(positions[1])
        raise ValueError(
            f"positions must be 'all', ('from', index), ('at', index), or an int; got {positions!r}"
        )

    def _apply(self, out: Any, cache: Any) -> Any:
        import mlx.core as mx

        self.calls += 1
        length = out.shape[1]
        offset = int(getattr(cache, "offset", 0)) if cache is not None else 0
        weights = mx.zeros((1, length, 1), dtype=mx.float32)
        if self.mode == "all":
            weights = weights + 1.0
        elif self.mode == "from":
            local = max(0, self.index - offset)
            if local >= length:
                return out
            weights[0, local:, 0] = 1.0
        else:
            local = self.index - offset
            if not 0 <= local < length:
                return out
            weights[0, local, 0] = 1.0
        self.injected += 1
        delta = weights * (self.alpha * self.vector).astype(mx.float32)
        return (out.astype(mx.float32) + delta).astype(out.dtype)

    def __enter__(self) -> InjectionHook:
        self._original = self.model.model.layers[self.layer]
        self.model.model.layers[self.layer] = _InjectingBlock(self._original, self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._original is not None:
            self.model.model.layers[self.layer] = self._original
            self._original = None


# --------------------------------------------------------------------------- adapter masking


def _lora_modules(root: Any) -> list[tuple[str, Any]]:
    """Every submodule of ``root`` (itself included) carrying a ``lora_b`` tensor."""
    # ``named_modules`` walks the whole subtree; the fallback covers non-mlx stand-ins.
    pairs = list(root.named_modules()) if hasattr(root, "named_modules") else [("", root)]
    return [(name, module) for name, module in pairs if hasattr(module, "lora_b")]


@contextlib.contextmanager
def lora_block_mask(model: Any, keep_layers: set[int]) -> Iterator[int]:
    """Temporarily zero the LoRA adapter everywhere except ``keep_layers``.

    ``mlx_lm.tuner.lora.LoRALinear.__call__`` (lines 95-98) computes
    ``y = linear(x) + scale * ((dropout(x) @ lora_a) @ lora_b)``, so ``lora_b = 0`` removes the
    adapter's contribution exactly while leaving the base weights untouched -- no dequantise,
    no re-fuse, and reversible. Yields the number of modules masked and restores every original
    tensor on exit, including when the body raises.
    """
    import mlx.core as mx

    saved: list[tuple[Any, Any]] = []
    try:
        for index, block in enumerate(model.model.layers):
            if index in keep_layers:
                continue
            for _name, module in _lora_modules(block):
                saved.append((module, module.lora_b))
                module.lora_b = mx.zeros_like(module.lora_b)
        yield len(saved)
    finally:
        for module, original in saved:
            module.lora_b = original


# --------------------------------------------------------------------------- note stripping


# Field labels whose *contents* are task state the model could simply read back. The
# "... complete" spellings are the same fields at the step where the generator declares the
# bucket full ("first half complete: 12 + 34"), so they carry the same bucket contents and are
# stripped as the same field; "held skipped" is the ledger's spelling of "held (skip)" at the
# summing step. Longest labels are tried first so "first half complete" wins over "first half".
_STRIP_LABELS = (
    "pending",
    "first half complete",
    "second half complete",
    "first half",
    "second half",
    "approved",
    "held (skip)",
    "held skipped",
    "highest so far",
)
_LABEL_RE = re.compile(
    "(?P<label>" + "|".join(re.escape(label) for label in _STRIP_LABELS) + r")[ \t]*:[ \t]*"
)
# One list entry: a run with no whitespace, comma or semicolon, never ending in "." (so a
# sentence-final period is not swallowed while "invoice-2-537.txt" is kept whole), optionally
# followed by the generator's "(full)" / "(final)" annotation.
_ITEM_RE = re.compile(r"[^\s,;]*[^\s,;.](?: \((?:full|final)\))?")
_STRIPPED = "[stripped]"


# The two ways the generator joins a bucket's contents: ``_join`` (", ") for lists it is still
# filling, and the calculator expression (" + ") at the step where it declares the bucket full.
_SEPARATORS = (", ", " + ")


def _value_end(text: str, start: int) -> int:
    """End of the field value beginning at ``start``, or ``start`` if there is none.

    The value is a run of items joined by :data:`_SEPARATORS`. An item that is not followed by
    a separator counts only when it is immediately followed by a real terminator (end of
    string, ``;``, ``,``, or ``.`` before whitespace/end). That is what stops the scan from
    running on into prose: in ``highest so far: service-1=87 (final), above threshold, so
    throttle x.ini.`` the item ``above`` is followed by a bare space, so the value ends at
    ``(final)``.
    """
    end = start
    position = start
    while True:
        match = _ITEM_RE.match(text, position)
        if match is None or match.end() == position:
            return end
        after = match.end()
        separator = next(
            (sep for sep in _SEPARATORS if text.startswith(sep, after)),
            None,
        )
        if separator is not None:
            end = after
            position = after + len(separator)
            continue
        following = text[after : after + 1]
        if following in ("", ";", ","):
            terminated = True
        elif following == ".":
            terminated = text[after + 1 : after + 2] in ("", " ", "\t", "\n")
        else:
            terminated = False
        return after if terminated else end


# batch_update carries its state as a queue rather than a labelled list: ``Next: worker-2.ini
# mode=audit -> mode=fast. Remaining after this: worker-3.ini ...; ...`` (and, during
# verification, ``Next: worker-2.ini, expect mode=fast``). Items contain spaces, so the
# label/item scanner above would stop after the first token; strip the whole clause instead.
_QUEUE_RE = re.compile(
    r"(?P<label>Next|Remaining after this)[ \t]*:[ \t]*(?P<value>[^\n]*?)(?=(?:\. |\.$|\n|$))"
)


def _strip_queue_fields(text: str) -> str:
    return _QUEUE_RE.sub(lambda m: f"{m.group('label')}: {_STRIPPED}", text)


def strip_state_fields(text: str) -> str:
    """Remove the task-state contents from a progress note, leaving everything else intact.

    Strips the value of ``pending:``, ``first half:`` / ``second half:`` (including their
    ``... complete:`` spellings), ``approved:``, ``held (skip):`` / ``held skipped:`` and
    ``highest so far:``, replacing each with ``<field>: [stripped]``. This is the stripped
    condition of the decisive contrast in §5 of ``research/representation_probes.md``: if a
    probe's accuracy collapses here, the state lived in the note and the model was reading it.

    Fields whose value is empty or unparsable are left exactly as they are.
    """
    text = _strip_queue_fields(text)
    out: list[str] = []
    position = 0
    for match in _LABEL_RE.finditer(text):
        if match.start() < position:
            continue
        end = _value_end(text, match.end())
        if end == match.end():
            continue
        out.append(text[position : match.start()])
        out.append(f"{match.group('label')}: {_STRIPPED}")
        position = end
    out.append(text[position:])
    return "".join(out)
