"""Streaming capture with explicit coordinates and reproducible, hash-chained records.

Only the last query of each native forward is observed for attention (one row per decode
step). Prompt source readings cover the full prefix. Forward chunks are retained so replay
uses the identical prefix and partition, including the generator's un-emitted lookahead.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from local_llm_lab.arch import NativeCapture
from local_llm_lab.forward import ForwardLedger, encode_prompt
from local_llm_lab.pipeline.live_lens.records import FutureRanks, transport_overlap


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class RecordWriter:
    """Exclusive new file; hash every event and close with a required footer."""

    def __init__(self, path: Path, manifest: dict):
        self.stream = path.open("x", encoding="utf-8")
        self.previous = "0" * 64
        self.count = 0
        self({"kind": "manifest", "schema_version": 1, "provenance": manifest})

    def __call__(self, event):
        payload = {"event": event, "previous": self.previous, "sequence": self.count}
        digest = hashlib.sha256(_encoded(payload)).hexdigest()
        self.stream.write(_encoded(payload | {"sha256": digest}).decode() + "\n")
        self.stream.flush()
        self.previous = digest
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self({"kind": "end_record", "status": "complete" if error_type is None else "aborted"})
        finally:
            self.stream.close()


def read_record(path: Path):
    previous = "0" * 64
    events = []
    for sequence, line in enumerate(path.read_text().splitlines()):
        payload = json.loads(line)
        digest = payload.pop("sha256")
        if (
            payload["sequence"] != sequence
            or payload["previous"] != previous
            or hashlib.sha256(_encoded(payload)).hexdigest() != digest
        ):
            raise ValueError("capture record hash chain mismatch")
        events.append(payload["event"])
        previous = digest
    if (
        not events
        or events[0]["kind"] != "manifest"
        or events[-1]["kind"] != "end_record"
        or events[-1]["status"] != "complete"
    ):
        raise ValueError("capture record is incomplete")
    return events


class LensReadout:
    """J{L-1} maps to the final pre-norm residual; the final layer is native identity.

    Intermediate matrix products and their readout use float32, recorded explicitly by the
    caller. Identity retains native dtype through norm and unembedding before returning fp32.
    """

    def __init__(self, view, lens):
        self.view = view
        self.lens = lens
        self._maps = {}

    def logits(self, residual, layer):
        import mlx.core as mx

        if layer == self.view.num_layers:
            mapped = residual
        else:
            if layer not in self._maps:
                self._maps[layer] = mx.array(self.lens.maps[layer])
            mapped = residual.astype(mx.float32) @ self._maps[layer].T
        return self.view.native_readout(mapped).astype(mx.float32)

    def __call__(self, residual, layer):
        import mlx.core as mx

        return np.array(mx.softmax(self.logits(residual, layer), axis=-1))


def top_tokens(probabilities, k):
    # Token-id tie breaking is stable; foreknowledge uses competition ranks separately.
    threshold = np.partition(probabilities, len(probabilities) - k)[-k]
    candidates = np.flatnonzero(probabilities >= threshold)
    return candidates[np.argsort(-probabilities[candidates], kind="stable")[:k]].tolist()


def _logit_hash(logits):
    import mlx.core as mx

    digest = hashlib.sha256()
    # Avoid materializing a prefill's full-vocabulary matrix as Python float objects.
    for row in logits.reshape(-1, logits.shape[-1]):
        digest.update(np.array(row.astype(mx.float32)).astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


class SpanLabeller:
    """Per-position span labels, decided from the model's own output as it arrives.

    The pre-registration's segmentation (2026-09-08 amendment), emitted at write time so the facet
    is a property of the data rather than of a later analysis. Four labels:

    ``note``
        generated tokens from the start of an agentic turn up to and excluding the opening fence.
    ``call_skeleton``
        tokens inside the fence whose value is fixed by the calling convention: the fence itself,
        the key names, the punctuation, the tool name.
    ``call_argument``
        tokens inside the fence carrying a task-dependent value — the path, query, expression,
        replacement text or answer.
    ``chat_prose``
        every generated token of a chat episode.

    **A token that straddles a boundary counts as an argument**, and that is a decision rather than
    a detail. Gemma emits `' "/'` as one token: a space, the opening quote and the first character
    of a path. Assigning by the token's first character would file the whole of it as skeleton, and
    that particular token is the one the whole `update-0028` finding turns on. A token any part of
    which lies inside an argument value carries task-dependent content, so it is argument.

    The scanner is deliberately small and deliberately not a JSON parser. It tracks whether the
    fence has opened, whether it is inside a string, and which key the current string is the value
    of. Malformed output — which this model produces — must not raise here: an unparseable turn
    still emitted tokens, and dropping their labels would silently shrink one facet.
    """

    ARGUMENT_KEYS = frozenset(
        {"path", "directory", "query", "expression", "old", "new", "answer"}
    )

    def __init__(self, kind: str):
        self.kind = kind
        self._fence = False
        self._in_string = False
        self._escaped = False
        self._buffer = ""
        self._pending_key = None
        self._last_key = None
        self._after_colon = False

    def _label(self, character: str) -> str:
        """The label of one character, decided from the state *before* it is consumed."""
        if self.kind != "agentic":
            return "chat_prose"
        if not self._fence:
            # The fence itself is skeleton; the note is what precedes it. Backticks appear in this
            # protocol only as the fence, so the first one ends the note.
            return "call_skeleton" if character == "`" else "note"
        argument = self._last_key in self.ARGUMENT_KEYS
        if self._in_string:
            # The terminating quote closes the value and carries none of it.
            closing = character == '"' and not self._escaped
            return "call_skeleton" if closing or not argument else "call_argument"
        if character == '"' and self._after_colon and self._pending_key in self.ARGUMENT_KEYS:
            # The opening quote of an argument value. Gemma emits it fused to the value's first
            # character as a single token, so filing it as skeleton would file that token as
            # skeleton — and that token is the one the update-0028 finding turns on.
            return "call_argument"
        return "call_skeleton"

    def _advance(self, character: str) -> None:
        if not self._fence:
            if character == "`":
                self._fence = True
            return
        if self._in_string:
            if self._escaped:
                self._escaped = False
            elif character == "\\":
                self._escaped = True
            elif character == '"':
                self._in_string = False
                if not self._after_colon:
                    self._pending_key = self._buffer
                self._buffer = ""
            else:
                self._buffer += character
            return
        if character == '"':
            self._in_string = True
            self._buffer = ""
            if self._after_colon:
                self._last_key = self._pending_key
        elif character == ":":
            self._after_colon = True
        elif character in ",{":
            self._after_colon = False
            self._pending_key = None

    def feed(self, piece: str) -> str:
        """Label the token whose decoded text is ``piece``, and advance past it.

        A token any part of which carries argument content is an argument token; otherwise a token
        that begins in the note is a note token; otherwise skeleton. Malformed output never raises
        here — an unparseable turn still emitted tokens, and dropping their labels would silently
        shrink one facet of the comparison.
        """
        if not piece:
            return "chat_prose" if self.kind != "agentic" else "call_skeleton"
        seen = []
        for character in piece:
            seen.append(self._label(character))
            self._advance(character)
        if "call_argument" in seen:
            return "call_argument"
        return seen[0]


class CaptureSession:
    def __init__(
        self,
        view,
        readout,
        emit,
        *,
        layers,
        attention_blocks=(),
        top_k=10,
        audit_modulus=23,
        audit_seed=0,
        retain_logits=False,
        final_readout_gate=None,
    ):
        if not 0 < top_k <= view.vocab_size or audit_modulus < 1:
            raise ValueError("invalid top-k or audit sampling modulus")
        self.view, self.readout, self.emit = view, readout, emit
        #: How far the readout may disagree with the model's own logits at the final layer, in
        #: absolute logit units, before the run is refused. The map there is the identity, so a
        #: disagreement is the residual tap or the unembedding and not a fitted quantity.
        #:
        #: The default is deliberately loose and is a **tripwire, not a tolerance**: it catches a
        #: wrong tap, a transposed unembedding or a missed norm, all of which move logits by
        #: whole units, and it does not pretend to a precision nobody has measured on this model.
        #: Every forward records its own figure as `final_readout_max_abs_error`, so the gate can
        #: be tightened onto observed behaviour rather than guessed at now — which is the same
        #: mistake as an absolute difference published without the quantity's own scale.
        self.final_readout_gate = 1.0 if final_readout_gate is None else float(final_readout_gate)
        self.layers = tuple(sorted(set(layers)))
        self.attention_blocks = tuple(sorted(set(attention_blocks)))
        self.top_k, self.audit_modulus, self.audit_seed = top_k, audit_modulus, audit_seed
        self.retain_logits = retain_logits
        self.ranks = FutureRanks()
        self.context = {}
        self.turn = -1
        self._active = False

    def set_context(self, **metadata):
        if self._active:
            raise RuntimeError("cannot replace context during a turn")
        self.context = metadata

    def _write(self, kind, **values):
        self.emit({"kind": kind, "turn": self.turn, **values})

    @contextmanager
    def generation(self, model, tokenizer, prompt, *, turn_cache):
        if self._active or model is not self.view.model:
            raise ValueError("capture must use its own model and a single active turn")
        if turn_cache is not None:
            raise ValueError("capture requires the resolved no-reuse cache strategy")
        prompt_ids = encode_prompt(tokenizer, prompt)
        self.ledger = ForwardLedger(prompt_ids)
        # One labeller per turn: the spans are a property of a single generation and the scanner
        # carries state across it. `kind` comes from the context the caller set; anything that is
        # not explicitly a chat episode is an agentic one, because the agentic path does not
        # declare a kind and adding a required field there would break every existing caller.
        self._tokenizer = tokenizer
        self._spans = SpanLabeller(self.context.get("kind", "agentic"))
        self.turn += 1
        self.ranks.reset()
        self.prompt_ids = prompt_ids
        self.generated = self.ledger.generated
        self._residuals = {}
        self._sources = {block: {} for block in self.attention_blocks}
        self._active = True
        self._write(
            "begin_turn",
            prompt=prompt,
            prompt_ids=prompt_ids,
            context=self.context,
            layers=self.layers,
            attention_blocks=self.attention_blocks,
            top_k=self.top_k,
            audit_modulus=self.audit_modulus,
            audit_seed=self.audit_seed,
            rank_horizons=self.ranks.horizons,
            distribution_capacity=self.ranks.capacity,
            readout_precision=(
                "fp32 intermediate maps; final layer reports the model's own softmax and is "
                "checked against the readout's identity branch every forward"
            ),
            final_readout_gate=self.final_readout_gate,
        )
        completed = False
        try:
            with NativeCapture(
                self.view, self, layers=self.layers, attention_blocks=self.attention_blocks
            ) as captured:
                yield captured
            completed = True
        finally:
            self._write(
                "end_turn",
                emitted_count=len(self.generated),
                forwarded_count=self.ledger.offset,
                status="complete" if completed else "aborted",
                unresolved_futures="censored at turn boundary",
            )
            self._residuals.clear()
            self._sources.clear()
            self.ranks.reset()
            self._active = False

    def residual(self, layer, offset, h):
        if layer in self.layers:
            self._residuals[layer] = h
        if layer in self._sources:
            for local in range(h.shape[1]):
                position = offset + local
                top = top_tokens(self.readout(h[0, local], layer), self.top_k)
                self._sources[layer][position] = top
                self._write("source", layer=layer, position=position, top=top)

    def attention(self, block, target, weights, written, total):
        import mlx.core as mx

        # Check decomposition descriptively; acceptance bounds are fixed by the caller.
        error = float(mx.max(mx.abs(written.sum(0) - total)).item())
        positions = list(range(weights.shape[-1]))
        sources = [self._sources[block][position] for position in positions]
        for head in range(weights.shape[0]):
            top = top_tokens(self.readout(written[head], block + 1), self.top_k)
            row = np.array(weights[head]).astype(float)
            score = transport_overlap(
                top, sources, row, source_positions=positions, target_position=target
            )
            score.pop("source_positions")
            payload = dict(
                block=block,
                head=head,
                position=target,
                written_top=top,
                head_sum_max_abs_error=error,
                **score,
            )
            key = [self.audit_seed, self.turn, block, head, target]
            if int(hashlib.sha256(_encoded(key)).hexdigest(), 16) % self.audit_modulus == 0:
                payload["audit"] = {
                    "attention": row.tolist(),
                    "positions": positions,
                    "source_top": sources,
                }
            self._write("head", **payload)

    def output(self, offset, ids, logits):
        import mlx.core as mx

        self.ledger.validate(offset, ids)
        if set(self._residuals) != set(self.layers):
            raise ValueError("native path did not capture every requested layer")
        # The next forward can precede the yield. Score only tokens actually yielded, below.
        # The final layer's distribution stays the model's own softmax, because that is what a
        # reader of a layer-34 row should be able to assume it is. What changes below is that the
        # readout path is *also* run there and compared against it.
        #
        # It used to be only substituted. That made the layer-34 rank-1 rate a statement about
        # decoding and position bookkeeping and nothing else -- no lens map, no residual tap, no
        # unembedding -- while the record and a published visualisation both described it as the
        # instrument proving itself. The CRO found it by reading this function. Run for real, the
        # identity branch of `LensReadout.logits` exercises the residual tap and
        # `view.native_readout`, so agreement with the native logits is a genuine end-to-end
        # check of everything the readout does except the fitted maps themselves.
        final = self.view.num_layers
        # `None` and `0.0` are different statements and the field carries both: no readout means
        # the check did not run, and a reader must never take a missing figure for a passing one.
        # A session constructed without a readout is the native-only path, which can ask for the
        # final layer alone and has nothing to compare against.
        checkable = self.readout is not None and final in self._residuals
        error = 0.0 if checkable else None
        for local in range(len(ids)):
            position = offset + local
            probabilities = {}
            for layer, h in self._residuals.items():
                if layer == final:
                    native = logits[0, local].astype(mx.float32)
                    if checkable:
                        through_readout = self.readout.logits(h[0, local], layer)
                        error = max(
                            error, float(mx.max(mx.abs(through_readout - native)).item())
                        )
                    probabilities[layer] = np.array(mx.softmax(native))
                else:
                    probabilities[layer] = self.readout(h[0, local], layer)
            self.ranks.capture(position, probabilities)
            self._write(
                "reading",
                position=position,
                top={str(layer): top_tokens(p, self.top_k) for layer, p in probabilities.items()},
            )
        if checkable and error > self.final_readout_gate:
            raise ValueError(
                f"the readout disagrees with the model's own logits at the final layer by "
                f"{error:.3g}, above the gate of {self.final_readout_gate:.3g}. At layer "
                f"{final} the map is the identity, so this is the residual tap or the "
                "unembedding, and every reading below it is suspect"
            )
        record = {
            "logits_sha256": _logit_hash(logits),
            "logits_shape": list(logits.shape),
            "argmax": np.array(mx.argmax(logits, axis=-1)).tolist(),
            "final_readout_max_abs_error": error,
        }
        if self.retain_logits:
            record["logits"] = np.array(logits.astype(mx.float32)).tolist()
        self._write("forward", offset=offset, input_ids=ids, **record)
        self.ledger.record(offset, ids)
        self._residuals.clear()

    def _decode(self, token_id):
        """One token's text, with a failure that costs a label rather than the run.

        Tokenizers differ in whether `decode` is available on the wrapper the pilot passes, and a
        span label is worth less than the episode it would abort.
        """
        try:
            return self._tokenizer.decode([int(token_id)])
        except Exception:  # noqa: BLE001 - any decoder failure is the same failure here
            return ""

    def emitted(self, token_id):
        if not self._active:
            raise RuntimeError("token emitted outside capture context")
        position = len(self.prompt_ids) + len(self.generated)
        self.ledger.emitted(int(token_id))
        # The span label is written here, with the token, rather than derived later: the
        # pre-registration requires the facet to be a property of the data, and a segmentation
        # computed after a record is read is a boundary chosen after seeing the answer.
        self._write(
            "emitted",
            position=position,
            token_id=int(token_id),
            span=self._spans.feed(self._decode(token_id)),
        )
        for row in self.ranks.observe(position, int(token_id)):
            self._write("rank", **row)


def replay(view, rows, *, atol, rtol):
    """Replay complete turns' native forward partitions with a fresh cache per turn.

    Use the same checkpoint for equivalence. Different adapters may consume the same fixed
    histories, but their output differences do not imply capture error or require lens refits.
    """
    import mlx.core as mx

    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("replay bounds must be finite and nonnegative")
    cache = None
    max_error = 0.0
    forwards = 0
    tokens_equal = True
    for row in rows:
        if row["kind"] == "begin_turn":
            cache = view.make_cache()
        elif row["kind"] == "forward":
            if cache is None:
                raise ValueError("forward without turn prefix")
            if (atol or rtol) and "logits" not in row:
                raise ValueError("nonzero replay tolerances require retained logits")
            logits = view.model(mx.array([row["input_ids"]]), cache=cache)
            if list(logits.shape) != row["logits_shape"] or not bool(mx.all(mx.isfinite(logits))):
                raise ValueError("replay output shape or finiteness mismatch")
            if "logits" in row:
                actual = np.array(logits.astype(mx.float32))
                expected = np.asarray(row["logits"], dtype=np.float32)
                error = float(np.max(np.abs(actual - expected)))
                max_error = max(max_error, error)
                if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                    raise ValueError(f"native replay exceeds declared tolerance: {error}")
            elif _logit_hash(logits) != row["logits_sha256"]:
                raise ValueError(
                    "native replay logit hash differs (retain logits for tolerance checks)"
                )
            tokens_equal &= np.array_equal(np.array(mx.argmax(logits, -1)), row["argmax"])
            forwards += 1
        elif row["kind"] == "end_turn":
            if row["status"] != "complete":
                raise ValueError("cannot accept an aborted turn")
            cache = None
    if not forwards or cache is not None:
        raise ValueError("replay requires complete captured turns")
    return {
        "forwards": forwards,
        "max_abs_error": max_error,
        "argmax_equal": bool(tokens_equal),
        "atol": atol,
        "rtol": rtol,
    }
