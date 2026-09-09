"""Small WS-A acceptance primitives for the shared remote gate runner.

The caller loads a model, declares its device/dtype/attention implementation and supplies
frozen token IDs. These functions never download weights, choose a device or infer a tolerance.
Development execution on a small model is not full-checkpoint golden acceptance.
"""

from __future__ import annotations

import copy
import math
from contextlib import suppress

import torch

from local_llm_lab.arch_torch import TorchCapture
from local_llm_lab.upstream_ref import load_upstream

_check_layer_indices = load_upstream().fitting._check_layer_indices


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


def _residual_error(actual, reference, *, require_same_dtype=False):
    """One max-norm comparison; every arm uses the same frozen native denominator."""
    if actual.shape != reference.shape or actual.device != reference.device:
        raise ValueError("residual comparison shapes or devices differ")
    if require_same_dtype and actual.dtype != reference.dtype:
        raise ValueError("native-dtype residual loop changed the native dtype")
    actual_dtype, native_dtype = str(actual.dtype), str(reference.dtype)
    actual, reference = actual.float(), reference.float()
    difference = float((actual - reference).abs().max())
    scale = float(reference.abs().max())
    if not math.isfinite(difference) or not math.isfinite(scale):
        raise ValueError("non-finite residual comparison")
    if scale == 0 and difference != 0:
        raise ValueError("nonzero residual error has a zero native reference norm")
    return {
        "actual_dtype": actual_dtype,
        "native_dtype": native_dtype,
        "max_abs": difference,
        "native_max_abs": scale,
        "max_norm_relative": difference / scale if scale else 0.0,
    }


def _residual_summary(rows):
    return {
        "by_layer": rows,
        "max_abs": max(row["max_abs"] for row in rows.values()),
        "max_norm_relative": max(row["max_norm_relative"] for row in rows.values()),
    }


def residual_precision_probe(view, ids, *, checkpoint=None):
    """Separate exact native-dtype seam agreement from a descriptive promoted-fp32 floor.

    One text-only native forward supplies independent frozen residual references through the
    existing TorchCapture hooks and the actual kwargs through the view's observer. No vocabulary
    logits are allocated. Loop outputs are reduced a layer at a time; only native references
    and the observed bundles persist across arms. The hook-site control reads each preceding
    native site, and the entry control bypasses the embedding subclass's forward with a raw
    weight lookup. Neither control invents an activation perturbation.

    ``checkpoint(phase, phase_report)`` receives scalar-only running/measured/failed state before
    and after each arm. It may stop execution, for example after persisting a memory reading.
    ``status=measured`` is execution evidence, not acceptance: the caller applies the declared
    zero seam gate and control margins at its required lengths. No tolerance is selected here.
    """
    if not torch.is_tensor(ids):
        rows = ids if isinstance(ids, (list, tuple)) else ()
        if rows and isinstance(rows[0], (list, tuple)):
            rows = [token for row in rows for token in row]
        if not rows or any(type(token) is not int for token in rows):
            raise ValueError("probe requires nonempty integer token IDs")
    elif ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("probe requires integer token IDs")
    ids = view._ids(ids)
    if ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("probe requires one nonempty unpadded sequence")
    if bool(torch.any((ids < 0) | (ids >= view.vocab_size))):
        raise ValueError("probe token IDs lie outside the vocabulary")
    if view.model.training:
        raise ValueError("residual precision probe requires evaluation mode")
    if not isinstance(view._embed_tokens, torch.nn.Embedding):
        raise NotImplementedError("entry omission requires an nn.Embedding raw weight lookup")
    spans = [view.attention_span(i) for i in range(view.num_layers)]
    if "global" not in spans or "sliding" not in spans:
        raise NotImplementedError("mask dispatch control requires both global and sliding blocks")
    global_index = spans.index("global")
    layers = range(view.num_layers + 1)
    controls = ("mask_dispatch", "hook_site_off_by_one", "entry_transform_omission")
    report = {
        "status": "running",
        "tokens": int(ids.shape[1]),
        "relative_definition": "max(abs(actual-native)) / max(abs(native)); zero/zero is zero",
        "native_capture": {"status": "unexecuted"},
        "native_dtype_loop": {"status": "unexecuted"},
        "promoted_fp32_loop": {"status": "unexecuted"},
        "controls": {name: {"status": "unexecuted"} for name in controls},
    }
    references, bundles = {}, {}
    entry = None

    def phase(name, operation):
        destination = report["controls"] if name in controls else report
        destination[name] = {"status": "running"}
        try:
            if checkpoint is not None:
                checkpoint(name, copy.deepcopy(destination[name]))
            destination[name] = {"status": "measured", **operation()}
            if checkpoint is not None:
                checkpoint(name, copy.deepcopy(destination[name]))
        except Exception as error:
            report["status"] = "failed"
            detail = f"{type(error).__name__}: {error}"
            if destination[name]["status"] == "measured":
                # A post-measurement resource/persistence refusal does not unexecute the arm.
                # Preserve its status and numbers; record the failed checkpoint separately.
                destination[name]["checkpoint_error"] = detail
            else:
                destination[name] = {"status": "failed", "error": detail}
            if checkpoint is not None:
                # Preserve the original failure, including a resource refusal.
                with suppress(Exception):
                    checkpoint(name, copy.deepcopy(destination[name]))
            raise

    class NativeReference:
        def residual(self, layer, offset, h):
            if offset != 0 or layer in references:
                raise ValueError("native probe must visit each residual exactly once")
            references[layer] = h.detach().clone()

        def output(self, offset, ids, logits):
            raise AssertionError("the residual probe must not allocate vocabulary logits")

    def capture_native():
        nonlocal entry, bundles
        # _observe_forward calls the text module directly. TorchCapture's block hooks still
        # observe genuine native sites; its full-model output hook is deliberately not visited.
        with TorchCapture(view, NativeReference(), layers=layers):
            observed_entry, observed_bundles = view._observe_forward(ids)
        if set(references) != set(layers):
            raise ValueError("native capture did not emit every requested residual")
        entry = observed_entry.detach().clone()
        # All tensors are no-grad leaves here; deepcopy preserves sharing between layer-type
        # bundles while insulating subsequent controls from the original forward's objects.
        bundles = copy.deepcopy(observed_bundles)
        report["native_dtype"] = str(references[0].dtype)
        return {
            "dtype": report["native_dtype"],
            "layers": list(references),
            "retained_residual_bytes": sum(
                h.numel() * h.element_size() for h in references.values()
            ),
            "basis": "TorchCapture block hooks during one text-only native observer forward",
        }

    def loop(*, promoted=False, mask_control=False, omit_entry=False):
        h = entry.clone()
        if omit_entry:
            # Gemma's scale lives inside its embedding subclass. Calling that module would
            # retain the transform; functional embedding reads only the stored weight rows.
            h = torch.nn.functional.embedding(ids, view._embed_tokens.weight)
        if promoted:
            h = h.float()
        masks = bundles
        if mask_control:
            masks = {
                index: {**values, "attention_mask": bundles[global_index]["attention_mask"]}
                for index, values in bundles.items()
            }
        rows = {0: _residual_error(h, references[0], require_same_dtype=not promoted)}
        for index in range(view.num_layers):
            h = (
                view.run_block(index, h, masks, None)
                if promoted else view._block(index, h, masks, None)
            )
            rows[index + 1] = _residual_error(
                h, references[index + 1], require_same_dtype=not promoted
            )
        return _residual_summary(rows)

    def control(values):
        floor = report["promoted_fp32_loop"]
        values["margins_vs_floor"] = {
            field: values[field] - floor[field] for field in ("max_abs", "max_norm_relative")
        }
        for layer, row in values["by_layer"].items():
            row["margins_vs_floor"] = {
                field: row[field] - floor["by_layer"][layer][field]
                for field in ("max_abs", "max_norm_relative")
            }
        return values

    def hook_control():
        rows = {}
        for layer in layers:
            observed_layer = max(0, layer - 1)
            rows[layer] = {
                **_residual_error(references[observed_layer], references[layer]),
                "intended_layer": layer,
                "observed_layer": observed_layer,
            }
        return {
            **_residual_summary(rows),
            "basis": "block input reported at the residual index reserved for that block's output",
        }

    try:
        with torch.no_grad():
            phase("native_capture", capture_native)
            phase("native_dtype_loop", lambda: {
                **(values := loop()), "exact": values["max_abs"] == 0.0,
                "basis": "_block at the native entry dtype, without run_block's fp32 promotion",
            })
            phase("promoted_fp32_loop", lambda: {
                **loop(promoted=True),
                "basis": "existing residuals arithmetic: fp32 entry and run_block; descriptive",
            })
            phase("mask_dispatch", lambda: control({
                **loop(mask_control=True), "global_mask_block": global_index,
                "basis": "every block receives the observed global attention mask; RoPE unchanged",
            }))
            phase("hook_site_off_by_one", lambda: control(hook_control()))
            phase("entry_transform_omission", lambda: control({
                **loop(omit_entry=True),
                "basis": "raw stored embedding lookup bypassing the native entry transform",
            }))
        report["status"] = "measured"
        return report
    finally:
        references.clear()
        bundles.clear()
        entry = None


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
