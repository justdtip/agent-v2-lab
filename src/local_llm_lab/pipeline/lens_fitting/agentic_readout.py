"""Actual-model readouts for the frozen paired population; no model loads on import."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def validated_tool_ids(manifest, tokenizer_path, configured):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    names = manifest["tools"]
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("capture must declare six unique tools")
    actual = {name: tokenizer.encode(name, add_special_tokens=False).ids[0] for name in names}
    ids = [actual[name] for name in names]
    if actual != manifest["tool_first_tokens"] or ids != configured:
        raise ValueError("tool token ids differ from capture and tokenizer")
    return ids


def summarize_logits(scores, tool_ids):
    import torch

    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("full vocabulary logits must be a finite vector")
    if (
        len(tool_ids) != 6
        or len(set(tool_ids)) != 6
        or not 0 <= min(tool_ids) <= max(tool_ids) < scores.numel()
    ):
        raise ValueError("six unique in-vocabulary tool ids required")
    top = torch.topk(scores.float(), 2)
    # The subset partition cannot exceed the full partition mathematically.
    # Clamp only their floating-point log-ratio at its exact upper bound.
    log_mass = torch.logsumexp(scores[tool_ids].float(), 0) - torch.logsumexp(scores.float(), 0)
    return {
        "six": torch.softmax(scores[tool_ids].float(), -1).tolist(),
        "mass": float(torch.exp(log_mass.clamp(max=0))),
        "top_token_id": int(scores.argmax()),
        "top_margin": float(top.values[0] - top.values[1]),
    }


def score_cells(capture_model, wrapped, cells, lenses, tool_ids, layers, progress=None):
    """Full original inputs, capture's own forward flags and kept-logit positions.

    Saved residuals must match byte for byte. Each lens reads those identical residuals
    through upstream's actual model unembed, never a gain-only vocabulary projection.
    """
    import torch

    from .upstream import load_upstream

    up = load_upstream()
    groups = defaultdict(list)
    for cell in cells:
        groups[cell["cell"]["row"]].append(cell)
    output = []
    for row, items in groups.items():
        first = items[0]
        if any(
            x["input_ids"] != first["input_ids"] or x["capture_forward"] != first["capture_forward"]
            for x in items
        ):
            raise ValueError("cell contexts or capture schedules disagree")
        ids = torch.tensor([first["input_ids"]], device=wrapped.input_device)
        positions = first["capture_forward"]["logits_to_keep"]
        with (
            up.fitting.ActivationRecorder(wrapped.layers, at=[i - 1 for i in layers]) as rec,
            torch.no_grad(),
        ):
            result = capture_model(
                input_ids=ids,
                logits_to_keep=torch.tensor(positions, device=ids.device),
                output_attentions=first["capture_forward"]["output_attentions"],
            )
        for item in items:
            cell = item["cell"]
            pos = cell["token_index"]
            model_readout = summarize_logits(result.logits[0, positions.index(pos)], tool_ids)
            for layer in layers:
                residual = rec.activations[layer - 1][0, pos].detach()
                array = residual.cpu().contiguous().numpy()
                if (
                    array.dtype != np.float32
                    or array.tobytes() != item["residuals"][layer].tobytes()
                ):
                    raise ValueError(f"capture residual mismatch: row {row} layer {layer}")
                entry = {
                    "row": row,
                    "position": cell["position"],
                    "layer": layer,
                    "episode": item["prompt_identity"]["task_id"],
                    "expert_token_id": cell["token_id"],
                    "tool_token_ids": tool_ids,
                    "model": model_readout,
                }
                for name, lens in lenses.items():
                    with torch.no_grad():
                        matrix = torch.tensor(lens.maps[layer], device=residual.device)
                        scores = wrapped.unembed(matrix @ residual)
                    entry[name] = summarize_logits(scores, tool_ids)
                output.append(entry)
        if progress:
            progress({"event": "scored", "row": row, "cell_layers": len(output)})
    return output


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = json.loads(parser.parse_args(argv).config.read_text())
    from local_llm_lab import runlock
    from local_llm_lab.probes import measure_pairings as P
    from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens

    from .agentic import atomic_json, sha
    from .agentic_compare import compare, validate_cell_pairings

    output = Path(args["output"])
    score_path = Path(args["scores"])
    if output.exists() or score_path.exists():
        raise ValueError("comparison and score outputs must be new")
    if args["layers"] != [18, 24]:
        raise ValueError("pilot comparison is layers 18 and 24")
    provenance, lenses = {}, {}
    for name in ("prose", "agentic"):
        paths = {
            k: Path(args[k])
            for k in ("model_snapshot", "checkpoint_manifest", "capture_dir", "positions", "corpus")
        }
        paths.update({k: Path(args[name][k]) for k in ("lens_archive", "lens_sidecar")})
        paths.update(
            capture_reference_archive=Path(args["prose"]["lens_archive"]),
            capture_reference_sidecar=Path(args["prose"]["lens_sidecar"]),
            layers=args["layers"],
        )
        cells, metadata, prov = P._admit(paths)
        lens, _ = load_admitted_device_lens(
            paths["lens_archive"],
            checkpoint=paths["model_snapshot"],
            model_base=metadata["model"]["base"],
        )
        lenses[name] = lens
        provenance[name] = {
            k: prov[k] for k in ("lens_sha256", "nu_sha256", "corpus_sha256", "positions_sha256")
        }
        validate_cell_pairings(
            json.loads(Path(args[name]["pairings"]).read_text()),
            cells,
            metadata,
            prov,
            args["layers"],
            args["maximum_pairing_relative"],
        )
    validated_tool_ids(
        prov["manifest"], Path(args["model_snapshot"]) / "tokenizer.json", args["tool_token_ids"]
    )
    runlock.hold_model_run_lock(command="agentic paired readout", session="codex-agentic")
    path, runtime = P._load_path(Path(args["model_snapshot"]), prov["manifest"])
    if runtime["checkpoint_sha256"] != prov["checkpoint_sha256"]:
        raise ValueError("loaded checkpoint differs from admitted pair")
    rows = score_cells(
        path.capture_model,
        path.wrapped,
        cells,
        lenses,
        args["tool_token_ids"],
        args["layers"],
        progress=lambda e: print(json.dumps(e), flush=True),
    )
    scores = {
        "readout_convention": "model-unembed-final-norm-and-softcap",
        "provenance": provenance,
        "runtime": runtime,
        "rows": rows,
    }
    score_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(score_path, scores)
    args["scores_sha256"] = sha(score_path)
    compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
