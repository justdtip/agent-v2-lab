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
    ):
        if not 0 < top_k <= view.vocab_size or audit_modulus < 1:
            raise ValueError("invalid top-k or audit sampling modulus")
        self.view, self.readout, self.emit = view, readout, emit
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
            readout_precision="fp32 intermediate maps; native final identity",
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
        for local in range(len(ids)):
            position = offset + local
            probabilities = {
                layer: (
                    np.array(mx.softmax(logits[0, local].astype(mx.float32)))
                    if layer == self.view.num_layers
                    else self.readout(h[0, local], layer)
                )
                for layer, h in self._residuals.items()
            }
            self.ranks.capture(position, probabilities)
            self._write(
                "reading",
                position=position,
                top={str(layer): top_tokens(p, self.top_k) for layer, p in probabilities.items()},
            )
        record = {
            "logits_sha256": _logit_hash(logits),
            "logits_shape": list(logits.shape),
            "argmax": np.array(mx.argmax(logits, axis=-1)).tolist(),
        }
        if self.retain_logits:
            record["logits"] = np.array(logits.astype(mx.float32)).tolist()
        self._write("forward", offset=offset, input_ids=ids, **record)
        self.ledger.record(offset, ids)
        self._residuals.clear()

    def emitted(self, token_id):
        if not self._active:
            raise RuntimeError("token emitted outside capture context")
        position = len(self.prompt_ids) + len(self.generated)
        self.ledger.emitted(int(token_id))
        self._write("emitted", position=position, token_id=int(token_id))
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
