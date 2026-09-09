"""Emit the numerical non-attainment counterexample without any model or checkpoint."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from local_llm_lab.sae_intervention import SAEIntervention
from local_llm_lab.spawn import run
from research.acceptance.provenance import MEASURED_HERE, Measured


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    bias = torch.tensor([0.5, -0.5, 1.0])
    decoder = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    residual = torch.tensor([1.5, 1.5, 4.0])

    def encoder(h):
        pre = (h - bias)[:2]
        return pre * (pre > 0.25).to(pre.dtype)

    edit = SAEIntervention(encoder, decoder, bias, features=(0,), target_values=torch.tensor([3.0]))
    patched = edit(residual)
    replaced_code = encoder(residual).clone()
    replaced_code[0] = 3.0
    epsilon_before = residual - bias - decoder @ encoder(residual)
    epsilon_after = patched - bias - decoder @ replaced_code
    basis = "CPU float32 synthetic JumpReLU dictionary; no model; desktop not certified idle"
    values = {
        "base_residual": residual.tolist(),
        "patched_residual": patched.tolist(),
        "replaced_code": replaced_code.tolist(),
        "epsilon_before": epsilon_before.tolist(),
        "epsilon_after": epsilon_after.tolist(),
        "max_error_in_preservation": (epsilon_after - epsilon_before).abs().max().item(),
    }
    report = {
        "source_commit": run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "utc": datetime.now(UTC).isoformat(),
        "device": "cpu",
        "torch": torch.__version__,
        "checkpoint_sha256": None,
        "scope": basis,
        "input_convention": "decoder[residual, feature], bias separate, JumpReLU threshold 0.25",
        "measured": {
            key: Measured(value, MEASURED_HERE, basis).as_dict() for key, value in values.items()
        },
        "reencoding": edit.diagnostic_record(),
    }
    destination = Path(__file__).with_name("counterexample.json")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
