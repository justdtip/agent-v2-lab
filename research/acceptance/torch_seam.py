"""Small WS-A acceptance primitives for the shared remote gate runner.

The caller loads a model, declares its device/dtype/attention implementation and supplies
frozen token IDs. These functions never download weights, choose a device or infer a tolerance.
Development execution on a small model is not full-checkpoint golden acceptance.
"""

from __future__ import annotations

import torch
from jlens.fitting import _check_layer_indices

from local_llm_lab.arch_torch import TorchCapture


def structural_report(view):
    sources, target = _check_layer_indices(None, None, view.num_layers)
    return {
        "num_layers": view.num_layers,
        "hidden_size": view.hidden_size,
        "vocab_size": view.vocab_size,
        "attention_spans": [view.attention_span(i) for i in range(view.num_layers)],
        "upstream_source_blocks": sources,
        "upstream_target_block": target,
        "source_residuals": [i + 1 for i in sources],
        "target_residual": target + 1,
        "entry_residual": 0,
    }


def decode_readout_probe(view, prompt_ids, *, steps, readout=True, control=None):
    """Capture native greedy decisions and the actual row-wise final-residual readout.

    The normal comparison mirrors LiveLensSession.output: norm/unembed one residual row
    against the corresponding native logit row. The controls alter only the comparator inputs,
    so they cannot change the generated history. A missing comparator reports None, not zero.
    The returned distribution is evidence from which a separately declared gate is calibrated;
    this helper never selects or relaxes the gate itself.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if control not in (None, "skew_after_prefill", "corrupt_residual"):
        raise ValueError("unknown readout control")
    if not readout and control is not None:
        raise ValueError("a readout control requires the comparator")
    errors, emitted, identity_failures = [], [], []

    class Probe:
        final = None
        readout_top = None

        def residual(self, layer, offset, h):
            self.final = h

        def output(self, offset, ids, logits):
            if self.final is None:
                raise ValueError("final residual was not captured")
            error = None
            if readout:
                error = 0.0
                for local in range(len(ids)):
                    h = self.final[0, local]
                    if control == "corrupt_residual":
                        h = h.clone()
                        h[0] += 1.0
                    predicted = view.native_readout(h).float()
                    native = logits[0, local].float()
                    if control == "skew_after_prefill" and offset > 0:
                        native = native + 1.0
                    error = max(error, float((predicted - native).abs().max()))
                self.readout_top = int(view.native_readout(self.final[0, -1]).float().argmax())
            errors.append(error)
            self.final = None

    probe = Probe()
    cache = view.make_cache()
    ids = view._ids(prompt_ids)
    with torch.no_grad(), TorchCapture(view, probe, layers=(view.num_layers,)) as wrapped:
        for step in range(steps):
            result = wrapped(ids, cache=cache)
            logits = result if torch.is_tensor(result) else result.logits
            token = int(torch.softmax(logits[0, -1].float(), dim=-1).argmax())
            emitted.append(token)
            if readout and probe.readout_top != token:
                identity_failures.append(step)
            ids = view._ids([token])
    return {
        "emitted_ids": emitted,
        "identity_failures": identity_failures,
        "readout_errors": errors,
        "control": control,
    }
