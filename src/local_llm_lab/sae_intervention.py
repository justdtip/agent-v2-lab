"""Decoder-direction interventions independent of any SAE loader or model family.

Vectors use the derivation's column convention: ``decoder`` is [residual, feature],
or a linear, bias-free callable from feature vector to residual vector. ``bias``
is separate. An affine/nonlinear callable decoder violates this contract; the
wrapper cannot establish global linearity from a finite number of evaluations.
"""

from __future__ import annotations

from copy import deepcopy

import torch


def _vector(value, name, *, like=None):
    if not torch.is_tensor(value) or value.ndim != 1 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating vector")
    if like is not None and (value.dtype != like.dtype or value.device != like.device):
        raise ValueError(f"{name} must preserve residual dtype and device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


class SAEIntervention:
    """Callable ``h + D_F (target_values - encoder(h)_F)`` for TorchCapture.

    ``features`` orders ``target_values``; neither is inferred from an activation.
    The original vector is retained, including its SAE reconstruction error. Bias
    cancels from the additive edit, so it is checked but never added to the delta.
    The encoder may be nonlinear (including JumpReLU) and need not invert D.

    Callers choose dictionary precision explicitly: all inputs and outputs must
    match the residual's dtype and device. The replacement preserves autograd;
    diagnostics are evaluated without a graph and stored as plain numbers. Only
    the latest successful diagnostic is retained here, and a failed call clears
    it. Capture can snapshot it via ``diagnostic_record`` at each application.

    The diagnostic includes every unselected feature's re-encoding change. Its
    size scales with dictionary width; persisting it on every clamped token is a
    caller's storage decision, not a constant-memory tracing claim.
    """

    def __init__(self, encoder, decoder, bias, *, features, target_values):
        if not callable(encoder):
            raise ValueError("encoder must be callable")
        if not callable(decoder) and not torch.is_tensor(decoder):
            raise ValueError("decoder must be a matrix or a linear bias-free callable")
        self.features = tuple(features)
        if (
            not self.features
            or any(type(i) is not int or i < 0 for i in self.features)
            or len(set(self.features)) != len(self.features)
        ):
            raise ValueError("features must be distinct nonnegative integer indices")
        self.bias = _vector(bias, "bias")
        self.target_values = _vector(target_values, "target_values").clone()
        if len(self.target_values) != len(self.features):
            raise ValueError("target_values must have one value per selected feature")
        if torch.is_tensor(decoder) and (
            decoder.ndim != 2
            or not decoder.is_floating_point()
            or decoder.shape[0] != len(bias)
            or not bool(torch.isfinite(decoder).all())
        ):
            raise ValueError("decoder must be a finite [residual, feature] matrix")
        self.encoder, self.decoder = encoder, decoder
        self._diagnostic = None

    def _encode(self, h):
        # Protect the original even from an encoder that normalizes its input in place.
        return _vector(self.encoder(h.clone()), "encoder output", like=h)

    def __call__(self, h):
        self._diagnostic = None
        _vector(h, "residual")
        _vector(self.bias, "bias", like=h)
        _vector(self.target_values, "target_values", like=h)
        if self.bias.shape != h.shape:
            raise ValueError("bias must match the residual shape")
        z = self._encode(h)
        if max(self.features) >= len(z):
            raise ValueError("selected feature outside encoder output")
        indices = torch.tensor(self.features, dtype=torch.long, device=h.device)
        delta = torch.zeros_like(z)
        delta[indices] = self.target_values - z[indices]
        if torch.is_tensor(self.decoder):
            if (
                self.decoder.shape != (len(h), len(z))
                or self.decoder.dtype != h.dtype
                or self.decoder.device != h.device
            ):
                raise ValueError("decoder must match residual/feature shape, dtype and device")
            direction = self.decoder @ delta
        else:
            direction = self.decoder(delta)
        _vector(direction, "decoder output", like=h)
        if direction.shape != h.shape:
            raise ValueError("decoder output must match residual shape")
        replacement = h + direction
        _vector(replacement, "replacement", like=h)
        with torch.no_grad():
            achieved = self._encode(replacement)
            if achieved.shape != z.shape:
                raise ValueError("re-encoder output must preserve feature shape")
            mask = torch.ones_like(z, dtype=torch.bool)
            mask[indices] = False
            self._diagnostic = {
                "kind": "sae_decoder_reencoding",
                "basis": "measured-here",
                "features": list(self.features),
                "before": z[indices].detach().cpu().tolist(),
                "target": self.target_values.detach().cpu().tolist(),
                "achieved": achieved[indices].cpu().tolist(),
                "target_error": (achieved[indices] - self.target_values).cpu().tolist(),
                "off_target_features": torch.arange(len(z), device=h.device)[mask].cpu().tolist(),
                "off_target_change": (achieved[mask] - z[mask]).cpu().tolist(),
            }
        return replacement

    def diagnostic_record(self):
        """A defensive copy of the most recent successful numerical diagnostic."""
        if self._diagnostic is None:
            raise RuntimeError("no successful intervention diagnostic is available")
        return deepcopy(self._diagnostic)
