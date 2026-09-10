"""W-3b: the carrier masked in the attention. For each sampled decision: the ordinary forward, then
forwards with the carrier spans masked as KEYS for every query at or after the current turn's start
(task statement never masked), one arm per kind and one for all kinds, plus a random-span mask of
equal token count; and the current turn's note — the prose before the ```json fence, its trailing
newline included, cut as the previous-note spans are cut — masked as keys for every query from the
fence onward through P_act, alone and together with the carriers, so the action position reads
without the note the model has just written and without any relay through the syntax between them.
Reads at P_note and P_act: the model's six-tool distribution (with mass) and the lens readout per
layer. Implemented as a forward pre-hook on every attention module that adds the extra masking to the
layer's own attention_mask, so the sliding window is preserved.

Checks, each a hard stop (Codex 712f78c: a guard that only logs does not refuse): every masked arm
blocks at least one edge at P_act; the attention on every masked edge, over all queries, layers and
heads, is exactly zero; local layers place zero mass beyond the window at P_act; the leaky negative
control (one carrier key left open) receives attention; the current-note arm leaves P_note's readouts
bit-identical to the unmasked forward; the in-context tool-name token equals the standalone first
token the six-tool readout uses. Rows are written as they complete.

usage: workspace_w3b.py SNAPSHOT CORPUS_JSONL MAPS_NPZ CAPTURE_DIR OUT_DIR [--device cuda:0] [--max-sample N] [--seed 20260910]
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("corpus"); ap.add_argument("maps"); ap.add_argument("capture"); ap.add_argument("out")
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max-sample", type=int, default=None); ap.add_argument("--seed", type=int, default=20260910)
args = ap.parse_args(); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True); T0 = time.monotonic()
TOOLS = ["read_file", "finish", "replace_text", "calculate", "search_files", "list_files"]
KINDS = ["older_note", "previous_note", "older_call", "previous_call", "older_result", "latest_result", "hidden_result"]
FENCE = "```json"  # the current note is the completion text before this fence
def emit(event, /, **f):
    row = {"event": event, "elapsed_s": round(time.monotonic() - T0, 1), **f}; print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a") as fh: fh.write(json.dumps(row, default=str) + "\n")

from local_llm_lab import device  # noqa: E402
device.pin(seed=0)
import torch  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
from transformers import AutoTokenizer  # noqa: E402
from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import upstream_index_of_repo_layer  # noqa: E402
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
import importlib.util, sys as _sys
spec = importlib.util.spec_from_file_location("wcap", str(Path(__file__).with_name("workspace_capture.py")))
# reuse the span tagger by reading its source rather than importing the script (which runs a capture on import)
_src = Path(__file__).with_name("workspace_capture.py").read_text()
_ns = {"re": re}; exec(_src[_src.index("def spans(prompt):"): _src.index("# ---- the sample, by rule")], _ns); spans = _ns["spans"]

man = json.loads((Path(args.capture) / "manifest.json").read_text()); sample_ids = man["sample"]
# the capture and this pass must read the same corpus: its rows are addressed by index
corpus_sha = hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
if man.get("corpus_sha256"): assert man["corpus_sha256"] == corpus_sha, f"corpus digest {corpus_sha[:12]} is not the capture's {man['corpus_sha256'][:12]}"
else: assert man["corpus"] == args.corpus, f"corpus {args.corpus} is not the capture's {man['corpus']} (no digest in its manifest to compare)"
if args.max_sample: sample_ids = sample_ids[: args.max_sample]
idx = {r["i"]: r for r in (json.loads(l) for l in (Path(args.capture) / "index.jsonl").read_text().splitlines() if l.strip())}
rows_all, seen = [], set()
for line in open(args.corpus):
    r = json.loads(line); m = r.get("metadata", {})
    if "task_id" not in m: continue
    k = (m["task_id"], m["step"])
    if k in seen: continue
    seen.add(k); rows_all.append(r)

model, report = hf_text.load_text_causal_lm(args.snapshot, dtype="bfloat16", attn_implementation="eager", device=args.device)
for p in model.parameters(): p.requires_grad_(False)
model.eval(); model.to(torch.float32)
tok = AutoTokenizer.from_pretrained(args.snapshot); lens_model = upstream_hf.HFLensModel(model, tokenizer=tok)
n_layers = int(lens_model.n_layers); z = np.load(args.maps); repo_layers = sorted(int(k[1:]) for k in z.files)
maps = {r: torch.from_numpy(z[f"J{r}"]).to(args.device) for r in repo_layers}
record_at = sorted(set(upstream_index_of_repo_layer(r) for r in repo_layers) | {n_layers - 1})
tool_first = [tok(t, add_special_tokens=False).input_ids[0] for t in TOOLS]
assert len(set(tool_first)) == len(TOOLS), f"the six tools' first tokens are not distinct: {tool_first}"
tool_ids = torch.tensor(tool_first, device=args.device)
IS_GLOBAL = [not bool(lay.self_attn.is_sliding) for lay in lens_model.layers]; W = int(getattr(getattr(model.config, "text_config", model.config), "sliding_window", 0) or 0)
rng = np.random.default_rng(args.seed)

# ---- the extra mask, applied by pre-hook on every attention module
STATE = {"extra": None}  # [1, 1, q, k] boolean: True = block
def pre_hook(module, args_, kwargs):
    extra = STATE["extra"]
    if extra is None: return None
    am = kwargs.get("attention_mask")
    if am is None: return None
    if am.dtype == torch.bool:
        kwargs["attention_mask"] = am & ~extra.to(am.device)
    else:
        neg = torch.finfo(am.dtype).min
        kwargs["attention_mask"] = torch.where(extra.to(am.device), torch.full_like(am, neg), am)
    return args_, kwargs
handles = [lay.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True) for lay in lens_model.layers]

def six(logits):
    p = torch.softmax(logits.float(), dim=-1); q = p[..., tool_ids]; return (q / q.sum(-1, keepdim=True).clamp_min(1e-30)).tolist(), float(q.sum(-1))
def readouts(acts, pos):
    out = {}
    for r in repo_layers:
        h = acts[upstream_index_of_repo_layer(r)][pos]; s, m = six(lens_model.unembed(h @ maps[r].T)); out[str(r)] = {"six": s, "mass": m}
    return out

def run(ids, extra, want_attn):
    STATE["extra"] = extra
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        out = model(input_ids=ids, output_attentions=want_attn)
        acts = {k: rec.activations[k].detach()[0].float() for k in record_at}
    STATE["extra"] = None
    return out, acts

results_fh = (OUT / "w3b.jsonl").open("w"); n_written = 0
LEAKY_CHECKED = False
for n, i in enumerate(sample_ids):
    r = rows_all[i]; m = r["metadata"]; rec_row = idx[i]; prompt, comp = r["prompt"], r["completion"]
    enc_p = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); p_ids = enc_p.input_ids  # the rendered prompt carries its own <bos>
    assert p_ids[0] == tok.bos_token_id and p_ids[1] != tok.bos_token_id, "the prompt must begin with exactly one <bos>"
    enc_c = tok(comp, add_special_tokens=False, return_offsets_mapping=True); t_idx = rec_row["P_act"] - len(p_ids) + 1
    assert enc_c.input_ids[t_idx] == tool_first[rec_row["tool_idx"]], f"row {i}: in-context tool token {enc_c.input_ids[t_idx]} is not the standalone first token {tool_first[rec_row['tool_idx']]} of {rec_row['tool']}"
    ids = torch.tensor([p_ids + enc_c.input_ids[: t_idx + 1]], device=args.device); S = ids.shape[1]
    P_note, P_act = rec_row["P_note"], rec_row["P_act"]
    assert P_note == len(p_ids) - 1 and P_act == S - 2, f"row {i}: positions {P_note},{P_act} do not match the sequence ({len(p_ids)} prompt tokens, S={S}; the tool-name token is last)"
    kinds = ["format"] * S
    for k, (a, b) in enumerate(enc_p.offset_mapping):
        for name, s0, s1 in spans(prompt):
            if a >= s0 and b <= s1: kinds[k] = name
    # the completion prefix: the prose note (tokens starting before the fence) and the syntax after it, through P_act
    note_end = max(comp.find(FENCE), 0); q0 = None; straddle = 0
    for j in range(t_idx + 1):
        a, b = enc_c.offset_mapping[j]
        if a < note_end:
            kinds[len(p_ids) + j] = "note"; straddle += int(b > note_end)
        else:
            kinds[len(p_ids) + j] = "note_syntax"
            if q0 is None: q0 = len(p_ids) + j
    kinds[S - 1] = "tool_name"  # the target at P_act, supplied; a query position but never read
    note_keys = [k for k, kk in enumerate(kinds) if kk == "note"]
    turn_start = P_note  # carrier arms: queries from the decision position onward
    arms, skipped = {"unmasked": None}, {}
    for kind in KINDS + ["all_carriers"]:
        keys = [k for k, kk in enumerate(kinds) if (kk == kind if kind != "all_carriers" else kk in KINDS)]
        if not keys: continue
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, turn_start:, keys] = True; arms[kind] = e
    # the current note, masked as keys for every query from the fence onward through P_act
    if note_keys and q0 is not None and q0 <= P_act:
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, q0:, note_keys] = True; arms["current_note"] = e
        if "all_carriers" in arms:
            e2 = arms["all_carriers"].clone(); e2[0, 0, q0:, note_keys] = True; arms["all_carriers_and_current_note"] = e2
    else:
        skipped["current_note"] = f"no prose note before the fence (note_prose_tokens={len(note_keys)}, q0={q0}, P_act={P_act})"
    n_all = int(arms["all_carriers"][0, 0, turn_start].sum()) if "all_carriers" in arms else 0
    if n_all:
        candidates = [k for k, kk in enumerate(kinds) if kk not in ("task",) and k < turn_start]
        pick = rng.choice(candidates, size=min(n_all, len(candidates)), replace=False)
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, turn_start:, pick] = True; arms["random_equal_count"] = e
    row = {"i": i, "task_id": m["task_id"], "step": m["step"], "family": m["family"], "tool": rec_row["tool"], "tool_idx": rec_row["tool_idx"],
           "S": S, "P_note": P_note, "P_act": P_act, "note_prose_chars": note_end, "note_prose_tokens": len(note_keys), "note_syntax_tokens": kinds.count("note_syntax"),
           "note_fence_straddle_tokens": straddle, "q0": q0, "kind_token_counts": {k: kinds.count(k) for k in set(kinds)}, "skipped_arms": skipped, "arms": {}}
    if not LEAKY_CHECKED and "all_carriers" in arms:  # the cut verified with a deliberately leaky mask: one carrier key left open must receive attention
        leaky = arms["all_carriers"].clone(); keys = torch.nonzero(leaky[0, 0, P_act]).flatten().tolist(); leaky[0, 0, :, keys[0]] = False
        out_l, _ = run(ids, leaky, True); a = torch.stack([A[0, :, P_act, :].float().mean(0) for A in out_l.attentions]).sum(0)
        open_mass = float(a[keys[0]]); masked_mass = float(a[keys[1:]].sum()) if len(keys) > 1 else 0.0
        passes = open_mass > 0.0 and masked_mass == 0.0
        emit("leaky_mask_check", open_key_attention_sum_over_layers=open_mass, masked_keys_attention=masked_mass, passes=passes)
        assert passes, "leaky negative control failed: the open key received no attention or the masked keys received some"
        LEAKY_CHECKED = True
    for arm, extra in arms.items():
        out, acts = run(ids, extra, True)
        lg = out.logits[0].float()
        def margin(v):  # taken tool's logit minus the best other tool's logit, in logits
            t = v[tool_ids]; own = float(t[rec_row["tool_idx"]]); others = torch.cat([t[:rec_row["tool_idx"]], t[rec_row["tool_idx"] + 1:]]); return own - float(others.max())
        rec_arm = {"model_six_note": six(lg[P_note]), "model_six_act": six(lg[P_act]), "lens_note": readouts(acts, P_note), "lens_act": readouts(acts, P_act),
                   "margin_act_logits": margin(lg[P_act]), "margin_note_logits": margin(lg[P_note])}
        # the gate: the intended mask is non-empty at P_act; every masked edge receives exactly zero attention; local layers place zero mass beyond the window
        n_blk_act = int(extra[0, 0, P_act].sum()) if extra is not None else 0; n_blk_total = int(extra.sum()) if extra is not None else 0
        assert extra is None or n_blk_act > 0, f"row {i}: arm {arm} blocks no edge at P_act"
        masked_act, masked_all, far_mass = 0.0, 0.0, 0.0
        extra_dev = extra[0, 0].to(args.device) if extra is not None else None
        for li, A in enumerate(out.attentions):
            a = A[0, :, P_act, :].float().mean(0)
            if extra is not None:
                masked_act += float(a[extra_dev[P_act]].sum()); masked_all += float(A[0].float()[:, extra_dev].sum())
            if not IS_GLOBAL[li] and W and P_act - W + 1 > 0: far_mass += float(a[: P_act - W + 1].sum())
        rec_arm["gate"] = {"attention_on_masked_keys": masked_act, "attention_on_masked_edges_all_queries": masked_all, "local_mass_beyond_window": far_mass,
                           "n_blocked_edges_at_P_act": n_blk_act, "n_blocked_edges_total": n_blk_total}
        assert masked_all == 0.0 and far_mass == 0.0, f"row {i}: arm {arm} gate failed: {rec_arm['gate']}"
        if arm == "current_note":  # queries before the fence are untouched, so P_note's readouts must be bit-identical to the unmasked forward
            u = row["arms"]["unmasked"]
            same = all(json.dumps(rec_arm[f]) == json.dumps(u[f]) for f in ("model_six_note", "lens_note", "margin_note_logits"))
            rec_arm["p_note_identical_to_unmasked"] = same
            assert same, f"row {i}: the current-note mask changed P_note's readouts"
        row["arms"][arm] = rec_arm
    results_fh.write(json.dumps(row) + "\n"); results_fh.flush(); n_written += 1
    emit("row", n_done=n + 1, of=len(sample_ids), arms=list(arms), skipped=skipped, note_prose_tokens=len(note_keys), q0=q0,
         blocked_at_P_act={k: v["gate"]["n_blocked_edges_at_P_act"] for k, v in row["arms"].items()}, gate=row["arms"].get("all_carriers", {}).get("gate"))
for h in handles: h.remove()
results_fh.close()
(OUT / "manifest.json").write_text(json.dumps({"schema_version": 2, "seat": "chief", "checkpoint": args.snapshot, "maps": args.maps, "capture": args.capture, "sample": sample_ids,
    "kinds": KINDS, "queries_masked_from": "carrier arms: P_note (the decision position) onward; current-note arms: q0, the first completion token starting at or after the ```json fence, onward through P_act",
    "current_note": {"keys": "completion tokens whose character offset starts before the ```json fence (the prose note, trailing newline included, cut as the previous-note spans are cut)",
                     "queries": "every token from the fence onward through P_act, so no relay through the syntax between the note and the action is open",
                     "skipped_when": "the completion has no prose before the fence (recorded per row under skipped_arms)"},
    "checks_each_a_hard_stop": ["every masked arm blocks >= 1 edge at P_act", "attention on every masked edge over all queries, layers and heads == 0.0", "local layers: zero mass beyond the window at P_act",
                                "leaky negative control: one carrier key left open receives attention, the rest none", "current-note arm: P_note readouts bit-identical to the unmasked forward",
                                "in-context tool-name token == the standalone first token used by the six-tool readout; the six first tokens distinct"],
    "control": "random tokens of equal count (a token-count control only); the same-kind arms (older vs previous note, older vs previous call) are the role- and contiguity-matched comparisons; survival under a mask shows non-necessity of the masked edges under this intervention and does not date the decision; for the carrier arms the two-hop relay through unmasked earlier positions is an open route; for the current-note arm no such relay exists (every position after the note is a masked query)",
    "precision": "float32", "width": 1, "rows_written": n_written, "corpus": args.corpus, "corpus_sha256": corpus_sha,
    "device": args.device, "is_global_layer": IS_GLOBAL, "sliding_window": W, "design": "WORKSPACE-EXPERIMENTS-2026-09-10.md W-3b; repair of the empty current-note cut found by Codex 712f78c"}, indent=2) + "\n")
emit("done", n=n_written, minutes=round((time.monotonic() - T0) / 60, 1))
