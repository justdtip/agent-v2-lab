"""One training step of the v2b 4B recipe at one row length, in its own process (R48d, issue 85).

Reproduces the trainer's step as ``pipeline.cli.stage_train`` runs it: the registry spec resolved
through ``evaluate.load_policy`` (which takes the model-run lock), LoRA rank 16 / scale 32 on the
resolved keys over every layer, the chunkwise gated-delta recurrence at chunk 64, gradient
checkpointing on, AdamW, ``mlx_lm``'s ``default_loss``. Two steps are run at the target length so
the reported step time is the second, past kernel compilation; the peak is the higher of the two.

Variant B swaps the loss for a chunked cross-entropy: the backbone runs once, and the vocabulary
projection plus the loss are computed per position chunk under ``mx.checkpoint``, so the full
(tokens x vocab) logits are never resident. The difference A - B is the vocabulary term.
"""
# RECORD, NOT A LAUNCHER (2026-09-05). This script produced the rows in summary.json. It loads the
# model, so it must not be started by hand: runs go through the package entry points under R47 and
# R48, in a window the Director declares, and take the model-run lock of issue 83.
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit("refusing to run: this file is a record of the 2026-09-05 training-cost probe, not a "
              "launcher; runs go through the package entry points under R47 and R48")

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/Users/daniel.tipton/Desktop/An app")
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MODEL_RUN_SESSION", "CRO train-cost probe")

GIB = 1024**3


def progress(**fields):
    print(json.dumps({"event": "progress", **fields}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="configs/agent_v2b_qwen35_4b.yaml")
    args = ap.parse_args()
    t_start = time.perf_counter()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from local_llm_lab.pipeline.cli import _effective_training_spec, load_config
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.training import install_chunkwise_gated_delta

    ws = mx.device_info()["max_recommended_working_set_size"]
    config = load_config(ROOT / args.config)
    train = config["train"]
    spec = _effective_training_spec(config)

    row = {
        "variant": args.variant,
        "tokens": args.tokens,
        "ce_chunk": args.ce_chunk if args.variant == "B" else None,
        "recipe": {
            "rank": train["rank"], "scale": train["scale"], "batch_size": 1,
            "grad_checkpoint": bool(train.get("grad_checkpoint", True)),
            "gated_delta_mode": train.get("gated_delta_mode"), "gated_delta_chunk": train.get("gated_delta_chunk"),
            "max_seq_length_in_config": train["max_seq_length"],
        },
        "working_set_gib": ws / GIB,
        "ok": False,
    }
    try:
        model, tokenizer, view, resolved = load_policy(spec, None)
        row["hf_id"] = spec.hf_id
        row["lora_keys"] = list(resolved.lora_keys)
        row["num_layers"] = resolved.num_layers
        row["load_peak_gib"] = mx.get_peak_memory() / GIB
        progress(stage="loaded", elapsed_s=round(time.perf_counter() - t_start, 1),
                 peak_gib=round(row["load_peak_gib"], 3), ws_share=round(mx.get_peak_memory() / ws, 3))

        model.freeze()
        linear_to_lora_layers(
            model, resolved.num_layers,
            {"keys": list(resolved.lora_keys), "rank": train["rank"], "scale": train["scale"],
             "dropout": train.get("dropout", 0.0)},
        )
        model.train()
        if row["recipe"]["grad_checkpoint"]:
            grad_checkpoint(model.layers[0])
        optimizer = optim.AdamW(learning_rate=train["learning_rate"])
        mx.set_wired_limit(ws)

        vocab = view.vocab_size
        T = args.tokens
        mx.random.seed(0)
        batch = mx.random.randint(0, vocab, (1, T + 1))
        lengths = mx.array([[0, T]])

        lm = model.language_model
        if lm.args.tie_word_embeddings:
            head = lambda x: lm.model.embed_tokens.as_linear(x)  # noqa: E731
        else:
            head = lm.lm_head

        def chunked_loss(model, batch, lengths):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = lm.model(inputs)

            total = mx.array(0.0, dtype=mx.float32)
            for s in range(0, inputs.shape[1], args.ce_chunk):
                t_chunk = targets[:, s:s + args.ce_chunk]

                # The targets are closed over, not passed: a checkpointed function is
                # differentiated with respect to every array it is given, and integer indices
                # have no VJP.
                def piece(h, t_chunk=t_chunk):
                    return nn.losses.cross_entropy(head(h), t_chunk).astype(mx.float32).sum()

                total = total + mx.checkpoint(piece)(hidden[:, s:s + args.ce_chunk])
            ntoks = mx.array(targets.size)
            return total / ntoks, ntoks

        loss = default_loss if args.variant == "A" else chunked_loss
        value_and_grad = nn.value_and_grad(model, loss)

        def one_step():
            (lvalue, ntoks), grad = value_and_grad(model, batch, lengths)
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state, lvalue)
            return float(lvalue), int(ntoks)

        mode = train.get("gated_delta_mode", "checkpointed")
        chunk = train.get("gated_delta_chunk", 64)
        assert mode == "chunkwise", mode
        with install_chunkwise_gated_delta(chunk):
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            loss1, n1 = one_step()
            t1 = time.perf_counter()
            peak1 = mx.get_peak_memory()
            progress(stage="step1", tokens=T, elapsed_s=round(t1 - t_start, 1), step_s=round(t1 - t0, 2),
                     peak_gib=round(peak1 / GIB, 3), ws_share=round(peak1 / ws, 3))
            t2 = time.perf_counter()
            loss2, n2 = one_step()
            t3 = time.perf_counter()
            peak2 = mx.get_peak_memory()
            progress(stage="step2", tokens=T, elapsed_s=round(t3 - t_start, 1), step_s=round(t3 - t2, 2),
                     peak_gib=round(peak2 / GIB, 3), ws_share=round(peak2 / ws, 3))

        row.update({
            "ok": True,
            "loss_step1": loss1, "loss_step2": loss2, "loss_tokens": n2,
            "first_step_s": t1 - t0, "step_s": t3 - t2,
            "step_tokens_per_s": T / (t3 - t2),
            "peak_gib": max(peak1, peak2) / GIB,
            "peak_ws_share": max(peak1, peak2) / ws,
            "active_after_gib": mx.get_active_memory() / GIB,
            "cache_after_gib": mx.get_cache_memory() / GIB,
            "total_s": time.perf_counter() - t_start,
        })
    except Exception as error:  # noqa: BLE001 - the row records the failure
        row["error"] = f"{type(error).__name__}: {error}"[:2000]
        row["traceback_tail"] = traceback.format_exc()[-1500:]
        try:
            import mlx.core as mx
            row["peak_gib_at_failure"] = mx.get_peak_memory() / GIB
        except Exception:  # noqa: BLE001
            pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "row", **{k: row.get(k) for k in ("variant", "tokens", "ok", "peak_gib", "step_s", "error")}}), flush=True)
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
