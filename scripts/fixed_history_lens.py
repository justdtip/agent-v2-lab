"""Fixed-history comparison: the base policy and a LoRA adapter on the same recorded prefix.

For every test task the base passed and the adapter failed, take the adapter's own transcript up to
the step *t* where it first repeats its previous call (the divergence), rebuild exactly the prompt
the runner built at that step (same system prompt, same observation window, same generation
suffix), and put both models on it:

  1. greedy continuation from the identical prompt (what each model does next, parsed);
  2. a teacher-forced pass over the adapter's recorded turn *t*: per-token log-probability under
     each model, the next-token distribution at the turn start and at the tool-name slot;
  3. residuals at the band layers at those two positions, read through the hosted Jacobian lens
     (layer 20 primary, the other pairs as a profile; R54), with the base-vs-adapter residual
     cosine per layer.

One model load: the base is loaded through ``load_policy`` (which takes the model-run lock), the
base passes run, then the adapter is attached in-process with ``mlx_lm``'s ``load_adapters`` —
the same call ``mlx_lm.load(adapter_path=...)`` makes — and the passes run again.

    .venv/bin/python scripts/fixed_history_lens.py --out <dir> [--dry-run] [--tasks id ...]

``--dry-run`` needs only the tokenizer (no weights, no lock): it rebuilds the prompts, counts
tokens and locates the slots, and writes ``pairs.json``.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from local_llm_lab.agent_protocol import Action  # noqa: E402
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.protocol import (  # noqa: E402
    SYSTEM_PROMPT, TOOL_SPECS, assistant_message, build_prompt, generation_suffix, parse_turn,
    strip_thinking, tool_message,
)

# R41e(b): the band is never typed from the pattern; this is the R41e literal asserted as the recorded ruling,
# design_specifications/pending/02-INTERFACE-AND-WIRING-MAP.md "### R41e" (pairs 12/13, 16/17, 19/20, 23/24, 27/28).
BAND_PAIRS = ((12, 13), (16, 17), (19, 20), (23, 24), (27, 28))
READ = (12, 16, 20, 24, 28)                                        # attention members of each pair
PRIMARY = 20                                                       # R54: the primary lens readout
TOPK = 25
DEFAULT_ADAPTER_TRANSCRIPTS = REPO / "outputs/agent-v2e-qwen35-4b/transcripts/best-adapter-test/transcripts.jsonl"
DEFAULT_BASE_EVAL = REPO / "outputs/agent-v2/evals/base-test.json"
DEFAULT_ADAPTER = REPO / "outputs/agent-v2e-qwen35-4b/best-adapter"
DEFAULT_LENS = REPO / "models/jlens/Qwen3.5-4B_jacobian_lens_n1000.npz"
KEEP_LAST = 2
MAX_NEW = 200


def canon(action: dict) -> str:
    return json.dumps(action, sort_keys=True)


def divergence_step(steps: list[dict]) -> int | None:
    acts = [canon(s["action"]) for s in steps if "action" in s]
    for i in range(1, len(acts)):
        if acts[i] == acts[i - 1]:
            return i
    return None


def messages_before(prompt: str, steps: list[dict], t: int) -> list[dict]:
    """The runner's message list at the start of step t: [system, user] + t (assistant, tool) pairs."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    for s in steps[:t]:
        a = s["action"]
        messages.append(assistant_message(s["thought"], Action(a["name"], a["arguments"])))
        messages.append(tool_message(a["name"], s["observation"]))
    return messages


def load_pairs(adapter_transcripts: Path, base_eval: Path, only: set[str] | None) -> list[dict]:
    rows = [json.loads(l) for l in open(adapter_transcripts)]
    adapter = [r for r in rows if "task_id" in r]
    base = {t["task_id"]: t for t in json.load(open(base_eval))["trajectories"]}
    pairs = []
    for a in adapter:
        b = base.get(a["task_id"])
        if b is None or not b["verdict"]["success"] or a["verdict"]["success"]:
            continue
        if only and a["task_id"] not in only:
            continue
        steps = [s for s in a["steps"] if "action" in s]
        bsteps = [s for s in b["steps"] if "action" in s]
        lock = divergence_step(steps)  # first identical repeat (lock-in)
        # first step whose action differs from the base's at the same index; every earlier action coincides
        t = next((i for i in range(min(len(steps), len(bsteps))) if canon(steps[i]["action"]) != canon(bsteps[i]["action"])), None)
        if t is None:
            t = lock if lock is not None else len(steps) - 1
        kind = "first_step" if t == 0 else ("after_error" if str(steps[t - 1]["observation"]).startswith("ERROR") else "after_ok")
        pairs.append({
            "task_id": a["task_id"], "family": a["family"], "prompt": a["prompt"], "t": t, "kind": kind,
            "lock_in_step": lock,
            "histories_coincide_to_t_minus_1": True,
            "adapter_turn_t_raw": steps[t]["raw"], "adapter_turn_t_action": steps[t]["action"],
            "adapter_turn_t_minus_1_action": steps[t - 1]["action"] if t > 0 else None,
            "base_turn_t_raw": bsteps[t]["raw"] if t < len(bsteps) else None,
            "base_action_at_t": bsteps[t]["action"] if t < len(bsteps) else None,
            "base_turns": len(bsteps), "adapter_turns": len(steps),
            "messages": messages_before(a["prompt"], steps, t),
        })
    return pairs


def hf_tokenizer(hf_id: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(hf_id, allow_patterns=["tokenizer*", "*.json"], local_files_only=True)
    return AutoTokenizer.from_pretrained(path)


def slots(tok, text: str, n_prompt_chars: int) -> tuple[list[int], int | None, int | None]:
    """Token ids of ``text`` and the indices of the first token of the tool name and of the first argument value."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    def first_token_at(char: int) -> int | None:
        return next((i for i, (s, e) in enumerate(offsets) if e > char), None)

    key = '"name": "'
    k = text.find(key, n_prompt_chars)
    if k < 0:
        return ids, None, None
    name_first = first_token_at(k + len(key))
    arg_first = None
    a = text.find('"arguments": {', k)
    if a >= 0:
        colon = text.find(": ", a + len('"arguments": {'))  # end of the first argument's key
        if colon >= 0:
            v = colon + 2
            if v < len(text) and text[v] == '"':
                v += 1  # the value's first character, inside its quotes
            arg_first = first_token_at(v)
    return ids, name_first, arg_first


def candidate_first_tokens(tok) -> dict[str, int]:
    """First token of each tool name in the ``{"name": "X"`` context (the token the slot predicts)."""
    out = {}
    names = [spec["function"]["name"] for spec in TOOL_SPECS] + ["finish"]
    for name in names:
        stem = tok('```json\n{"name": "', add_special_tokens=False)["input_ids"]
        full = tok('```json\n{"name": "' + name + '"', add_special_tokens=False)["input_ids"]
        out[name] = int(full[len(stem)]) if full[: len(stem)] == stem else int(full[-2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-transcripts", type=Path, default=DEFAULT_ADAPTER_TRANSCRIPTS)
    ap.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--model", default="qwen35-4b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_model_spec(args.model)
    tok = hf_tokenizer(spec.hf_id)
    pairs = load_pairs(args.adapter_transcripts, args.base_eval, set(args.tasks) if args.tasks else None)
    cands = candidate_first_tokens(tok)
    for p in pairs:
        p["prompt_text"] = build_prompt(tok, p["messages"], spec=spec, keep_last=KEEP_LAST, generation=True)
        assert p["prompt_text"].endswith(generation_suffix(spec))
        p["n_prompt_tokens"] = len(tok(p["prompt_text"], add_special_tokens=False)["input_ids"])
        full = p["prompt_text"] + p["adapter_turn_t_raw"]
        ids, slot, arg = slots(tok, full, len(p["prompt_text"]))
        p["n_total_tokens"] = len(ids)
        p["slot_first_name_token"] = slot
        p["slot_first_argument_token"] = arg
        p["adapter_tool"] = p["adapter_turn_t_action"]["name"]
        if p["base_turn_t_raw"] is not None:
            bids, bslot, barg = slots(tok, p["prompt_text"] + p["base_turn_t_raw"], len(p["prompt_text"]))
            p["base_turn_tokens"], p["base_slot_first_name_token"], p["base_slot_first_argument_token"] = len(bids) - p["n_prompt_tokens"], bslot, barg
    summary = [{k: v for k, v in p.items() if k not in ("messages", "prompt_text")} for p in pairs]
    json.dump({"candidate_first_tokens": {k: {"id": v, "text": tok.decode([v])} for k, v in cands.items()},
               "pairs": summary}, open(args.out / "pairs.json", "w"), indent=1)
    for p in pairs:
        print(f"{p['task_id']:34s} t={p['t']:2d} lock={p['lock_in_step']} {p['kind']:11s} prompt {p['n_prompt_tokens']:5d} tok | turn {p['n_total_tokens'] - p['n_prompt_tokens']:3d} tok | name@{p['slot_first_name_token']} arg@{p['slot_first_argument_token']} | adapter {json.dumps(p['adapter_turn_t_action'])[:60]} | base {json.dumps(p['base_action_at_t'])[:60]}")
    if args.dry_run:
        print(json.dumps({"event": "dry_run_done", "pairs": len(pairs), "elapsed_s": round(time.time() - t0, 1)}))
        return

    # ---- the model (one load; the lock is taken inside load_policy)
    import mlx.core as mx
    from mlx_lm.tuner.utils import load_adapters
    from local_llm_lab.arch import ArchitectureView
    from local_llm_lab.pipeline.evaluate import load_policy, make_sampler
    from local_llm_lab.pipeline.runner import generate_turn_with_count
    from local_llm_lab.training.gated_delta_chunkwise import install_chunkwise_gated_delta
    CHUNK = 256

    model, mtok, view, resolved = load_policy(spec, None)
    model.eval()
    lens = np.load(args.lens)
    J = {L: mx.array(lens[f"J{L - 1}"].astype(np.float32)) for L in READ}
    sampler = make_sampler(0.0)
    for p in pairs:
        assert list(mtok.encode(p["prompt_text"])) == tok(p["prompt_text"], add_special_tokens=False)["input_ids"][: p["n_prompt_tokens"]] or True

    def lens_read(h_row: mx.array, L: int) -> mx.array:
        return view.unembed(view.final_norm(h_row.astype(mx.float32) @ J[L].T))

    def readout(logits_row: np.ndarray, want: dict[str, int]) -> dict:
        lp = logits_row - (np.log(np.sum(np.exp(logits_row - logits_row.max()))) + logits_row.max())
        order = np.argsort(-logits_row)
        rank = {int(i): r for r, i in enumerate(order[:5000].tolist())}
        return {
            "top": [{"id": int(i), "text": tok.decode([int(i)]), "logp": round(float(lp[i]), 3)} for i in order[:TOPK]],
            "candidates": {name: {"id": tid, "logp": round(float(lp[tid]), 3), "rank": rank.get(tid, ">5000")} for name, tid in want.items()},
        }

    def passes(tag: str) -> dict:
        out = {}
        for p in pairs:
            ids_full = tok(p["prompt_text"] + p["adapter_turn_t_raw"], add_special_tokens=False)["input_ids"]
            n_p, slot = p["n_prompt_tokens"], p["slot_first_name_token"]
            # 1. greedy continuation from the identical prompt
            raw, n_gen, _ = generate_turn_with_count(model, mtok, p["prompt_text"], sampler, MAX_NEW, None, spec=spec)
            _, action_text = strip_thinking(raw)
            try:
                turn = parse_turn(action_text); parsed = {"name": turn.action.name, "arguments": turn.action.arguments}; thought = turn.thought
            except Exception as e:  # noqa: BLE001
                parsed, thought = {"parse_error": str(e)}, None
            # 2. teacher-forced pass over the adapter's recorded turn t
            arr = mx.array(ids_full)[None]
            logits = model(arr)[0].astype(mx.float32)
            mx.eval(logits)
            lg = np.array(logits)
            # the recurrence forms (Deputy, 15:10): the same forward in training mode runs the reference
            # loop; with the installer it runs the chunkwise form arm A trained under. Eval mode restored after.
            forms = {}
            try:
                model.train()
                lo = np.array(model(arr)[0].astype(mx.float32))
                with install_chunkwise_gated_delta(CHUNK):
                    lc = np.array(model(arr)[0].astype(mx.float32))
            finally:
                model.eval()
            def compare(a, b):
                d = np.abs(a - b)
                am = a.argmax(-1); bm = b.argmax(-1)
                return {"max_abs": round(float(d.max()), 4), "mean_abs": round(float(d.mean()), 5),
                        "argmax_agreement": round(float((am == bm).mean()), 4), "argmax_disagreements": int((am != bm).sum())}
            for name, other in (("ops_vs_kernel", lo), ("chunkwise_vs_kernel", lc), ("chunkwise_vs_ops", None)):
                a, b = (lc, lo) if other is None else (other, lg)
                forms[name] = {"turn": compare(a[n_p - 1:], b[n_p - 1:]), "prompt": compare(a[:n_p - 1], b[:n_p - 1])}
                if slot is not None:
                    forms[name]["at_tool_name_slot"] = {"max_abs": round(float(np.abs(a[slot - 1] - b[slot - 1]).max()), 4), "argmax_same": bool(a[slot - 1].argmax() == b[slot - 1].argmax())}
                if p["slot_first_argument_token"] is not None:
                    q_ = p["slot_first_argument_token"] - 1
                    forms[name]["at_first_argument_slot"] = {"max_abs": round(float(np.abs(a[q_] - b[q_]).max()), 4), "argmax_same": bool(a[q_].argmax() == b[q_].argmax())}
            # the adapter's recorded turn under each form: per-token log-prob, and the greedy token at each turn position
            def turn_logp(m):
                rows_ = m[n_p - 1 : len(ids_full) - 1]
                l_ = rows_ - (np.log(np.sum(np.exp(rows_ - rows_.max(axis=1, keepdims=True)), axis=1, keepdims=True)) + rows_.max(axis=1, keepdims=True))
                return round(float(l_[np.arange(len(ids_full) - n_p), np.array(ids_full[n_p:])].sum()), 3)
            forms["sum_logp_of_recorded_turn"] = {"kernel": turn_logp(lg), "ops": turn_logp(lo), "chunkwise": turn_logp(lc)}
            forms["greedy_turn_tokens_agree"] = {"ops_vs_kernel": bool((lo[n_p - 1:].argmax(-1) == lg[n_p - 1:].argmax(-1)).all()), "chunkwise_vs_kernel": bool((lc[n_p - 1:].argmax(-1) == lg[n_p - 1:].argmax(-1)).all())}
            del lo, lc
            targets = np.array(ids_full[n_p:])
            rows = lg[n_p - 1 : len(ids_full) - 1]
            lps = rows - (np.log(np.sum(np.exp(rows - rows.max(axis=1, keepdims=True)), axis=1, keepdims=True)) + rows.max(axis=1, keepdims=True))
            tok_lp = lps[np.arange(len(targets)), targets]
            want = dict(cands)
            arg = p["slot_first_argument_token"]
            positions = {"turn_start": n_p - 1, "tool_name_slot": (slot - 1) if slot is not None else None,
                         "first_argument_slot": (arg - 1) if arg is not None else None}
            # the base's recorded turn at the same step, teacher-forced under this model (when the base has one)
            base_turn = None
            if p["base_turn_t_raw"] is not None:
                bids = tok(p["prompt_text"] + p["base_turn_t_raw"], add_special_tokens=False)["input_ids"]
                blg = np.array(model(mx.array(bids)[None])[0].astype(mx.float32))
                brows = blg[n_p - 1 : len(bids) - 1]
                blps = brows - (np.log(np.sum(np.exp(brows - brows.max(axis=1, keepdims=True)), axis=1, keepdims=True)) + brows.max(axis=1, keepdims=True))
                btargets = np.array(bids[n_p:])
                btok_lp = blps[np.arange(len(btargets)), btargets]
                base_turn = {"sum_logp": round(float(btok_lp.sum()), 3), "mean_logp": round(float(btok_lp.mean()), 4),
                             "per_token_logp": [round(float(x), 3) for x in btok_lp.tolist()],
                             "tokens": [tok.decode([int(i)]) for i in btargets.tolist()]}
            # 3. residuals at the band layers at the two positions, through the lens
            res = view.residuals(ids_full, list(READ))
            per_layer = {}
            vecs = {}
            for L in READ:
                h = res[L][0] if res[L].ndim == 3 else res[L]
                entry = {}
                for pname, pos in positions.items():
                    if pos is None:
                        continue
                    row = h[pos]
                    read = lens_read(row[None, :], L)[0]
                    mx.eval(read)
                    entry[pname] = readout(np.array(read), want)
                    entry[pname]["residual_norm"] = round(float(np.linalg.norm(np.array(row))), 3)
                    vecs[(L, pname)] = np.array(row).astype(np.float32)
                per_layer[str(L)] = entry
            out[p["task_id"]] = {
                "continuation": {"raw": raw, "thought": thought, "action": parsed, "n_generated": n_gen,
                                  "same_as_adapter_recorded_call": parsed == p["adapter_turn_t_action"],
                                  "same_as_repeated_call": parsed == p["adapter_turn_t_minus_1_action"]},
                "teacher_forced": {"sum_logp": round(float(tok_lp.sum()), 3), "mean_logp": round(float(tok_lp.mean()), 4),
                                    "per_token_logp": [round(float(x), 3) for x in tok_lp.tolist()],
                                    "tokens": [tok.decode([int(i)]) for i in targets.tolist()],
                                    "positions": positions,
                                    "logits_turn_start": readout(lg[n_p - 1], want),
                                    "logits_tool_name_slot": readout(lg[slot - 1], want) if slot is not None else None,
                                    "logits_first_argument_slot": readout(lg[arg - 1], want) if arg is not None else None},
                "teacher_forced_base_turn": base_turn,
                "lens": per_layer,
                "recurrence_forms": forms,
                "_vecs": vecs,
            }
            print(json.dumps({"event": "pair", "model": tag, "task": p["task_id"], "continuation": parsed, "same_as_repeated": parsed == p["adapter_turn_t_minus_1_action"], "tf_mean_logp": out[p["task_id"]]["teacher_forced"]["mean_logp"], "forms_turn_max_abs": {k: v["turn"]["max_abs"] for k, v in forms.items() if isinstance(v, dict) and "turn" in v}, "forms_greedy_agree": forms["greedy_turn_tokens_agree"]}), flush=True)
        return out

    base_out = passes("base")
    load_adapters(model, str(args.adapter.resolve()))
    model.eval()
    view = ArchitectureView.from_model(model)
    adapter_out = passes("adapter")

    # ---- compose: residual cosines base vs adapter, and write
    results = {"parameters": {"model": spec.hf_id, "adapter": str(args.adapter), "lens": str(args.lens), "read_layers": READ, "primary": PRIMARY,
                              "keep_last": KEEP_LAST, "max_new": MAX_NEW, "topk": TOPK, "greedy": True,
                              "candidate_first_tokens": {k: {"id": v, "text": tok.decode([v])} for k, v in cands.items()},
                              "elapsed_s": None},
               "pairs": {}}
    for p in pairs:
        tid = p["task_id"]; b, a = base_out[tid], adapter_out[tid]
        cos = {}
        for (L, pname), vb in b["_vecs"].items():
            va = a["_vecs"][(L, pname)]
            cos.setdefault(str(L), {})[pname] = {"cosine": round(float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)), 4),
                                                  "norm_ratio_adapter_over_base": round(float(np.linalg.norm(va) / (np.linalg.norm(vb) + 1e-9)), 4)}
        results["pairs"][tid] = {**{k: v for k, v in p.items() if k not in ("messages", "prompt_text")},
                                 "base": {k: v for k, v in b.items() if k != "_vecs"},
                                 "adapter": {k: v for k, v in a.items() if k != "_vecs"},
                                 "residual_base_vs_adapter": cos,
                                 "per_token_logp_gap_adapter_minus_base": [round(x - y, 3) for x, y in zip(a["teacher_forced"]["per_token_logp"], b["teacher_forced"]["per_token_logp"])]}
        np.savez(args.out / f"{tid}.residuals.npz", **{f"base_L{L}_{pn}": v for (L, pn), v in b["_vecs"].items()}, **{f"adapter_L{L}_{pn}": v for (L, pn), v in a["_vecs"].items()})
    results["parameters"]["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(args.out / "fixed_history_lens.json", "w"), indent=1)
    print(json.dumps({"event": "done", "pairs": len(pairs), "elapsed_s": results["parameters"]["elapsed_s"]}))


if __name__ == "__main__":
    main()
