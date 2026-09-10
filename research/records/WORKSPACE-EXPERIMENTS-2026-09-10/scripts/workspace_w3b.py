"""W-3b: the carrier masked in the attention (v3: placement certified, not only applied — Codex 63d0549).

For each sampled decision: the ordinary forward, then forwards with the carrier spans masked as KEYS
for every query at or after the current turn's start (task statement never masked), one arm per kind
and one for all kinds, plus a random mask of equal token count drawn from the non-carrier tokens (the task statement and the
format tokens; never position 0, the <bos> sink) — v3.2, after the 4B pass showed v3.1's pool (the non-task tokens) to be
mostly the carrier spans themselves — and, for the current-note arm, a second draw of the note's prose count from the
same pool masked from the fence onward; and the current turn's note — the
prose before the ```json fence, its trailing newline included, cut as the previous-note spans are cut —
masked as keys for every query from the fence onward through the last position S-1 (the supplied
tool-name token, a query that is never read), alone and together with the carriers, so the action
position P_act = S-2 reads without the note the model has just written and without any relay through
the syntax between them. Reads at P_note and P_act: the model's six-tool distribution (with mass) and
the lens readout per layer. Implemented as a forward pre-hook on every attention module that adds the
extra masking to the layer's own attention_mask, so the sliding window is preserved.

Placement is certified three ways, each a hard stop. (1) A self-test at start: the cut builder must
reproduce hand-labelled expected keys, q0 and straddle on Codex's synthetic tokenisations at four
prompt offsets, and the expected-edge oracle must reject six wrong masks (delayed start, wrong key,
missing key, extra key, future-only key, empty) while accepting the right one. (2) Per row, a text
oracle independent of the builder: the note keys decode to exactly the prose before the fence, and the
first query token begins the fence. (3) Per arm, a receipt: the mask's applied edges against the
expected edge set built from the declared keys and query interval (missing and unexpected edges), and
effective-mask checks read from the returned attentions — every masked edge receives exactly zero
over all queries, layers and heads; the native causal structure is intact (no mass above the diagonal);
local layers place zero mass beyond the window for every query; queries before the cut still attend to
the masked keys (the hook did not over-mask). The leaky negative control (one carrier key left open)
must receive attention; the current-note arm must leave P_note's readouts bit-identical to the unmasked
forward; the in-context tool-name token must equal the standalone first token the six-tool readout
uses, and the six must be distinct; the capture and this pass must read the same corpus by digest.
run.json names the requested sample at start; rows are written as they complete; the manifest closes.

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
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def emit(event, /, **f):
    row = {"event": event, "elapsed_s": round(time.monotonic() - T0, 1), **f}; print(json.dumps(row, default=str), flush=True)
    with (OUT / "progress.jsonl").open("a") as fh: fh.write(json.dumps(row, default=str) + "\n")

# ---- the cut builder, a pure function of the completion's token offsets, and the expected-edge oracle
def note_cut(offsets, note_end, n_prompt, t_idx):
    """Completion token offsets [(a, b)], the fence's char offset, the prompt length, the tool-name token's index.
    Returns global positions: note keys (tokens starting before the fence), q0 (first token starting at or after it),
    the count of tokens straddling the fence, and the syntax positions [q0, S-1)."""
    keys, q0, straddle = [], None, 0
    for j in range(t_idx + 1):
        a, b = offsets[j]
        if a < note_end: keys.append(n_prompt + j); straddle += int(b > note_end)
        elif q0 is None: q0 = n_prompt + j
    S = n_prompt + t_idx + 1
    return keys, q0, straddle, list(range(q0 if q0 is not None else S, S - 1))
def edge_set(S, start, keys): return {(q, k) for q in range(start, S) for k in keys}
def placement(actual, expected):
    return {"passes": actual == expected, "missing_edges": len(expected - actual), "unexpected_edges": len(actual - expected)}
def selftest():
    fixtures = [("ordinary", ["Go.", "\n", "```json", "\n", '{"name":"', "read_file"], [2, 3], 4, 0),
                ("one_note_token", ["Go.\n", "```json", "\n", '{"name":"', "read_file"], [2], 3, 0),
                ("straddling_fence", ["Go.", "\n```", "json", "\n", '{"name":"', "read_file"], [2, 3], 4, 1),
                ("no_prose", ["```json", "\n", '{"name":"', "read_file"], [], 2, 0)]
    n = 0
    for name, pieces, keys, q0, straddle in fixtures:
        comp = "".join(pieces); offsets, end = [], 0
        for part in pieces: offsets.append((end, end + len(part))); end += len(part)
        for n_prompt in (2, 47, 100, 1500):
            k, q, s, _ = note_cut(offsets, max(comp.find(FENCE), 0), n_prompt, len(pieces) - 1)
            exp_keys = [x + n_prompt - 2 for x in keys]; exp_q0 = q0 + n_prompt - 2
            assert (k, q, s) == (exp_keys, exp_q0, straddle), f"selftest {name}@{n_prompt}: got {(k, q, s)}, expected {(exp_keys, exp_q0, straddle)}"
            n += 1
    expected = edge_set(8, 4, [2, 3])
    wrong = {"delayed_query_start": edge_set(8, 6, [2, 3]), "wrong_keys": edge_set(8, 4, [4]), "missing_one_note_key": edge_set(8, 4, [2]),
             "extra_unrelated_key": edge_set(8, 4, [0, 2, 3]), "future_only_key": edge_set(8, 4, [7]), "empty": set()}
    assert placement(expected, expected)["passes"], "selftest: the correct mask must pass placement"
    for name, cells in wrong.items():
        assert not placement(cells, expected)["passes"], f"selftest: the wrong mask {name} passed placement"
    return {"builder_cases": n, "wrong_masks_rejected": len(wrong)}
emit("selftest", **selftest())

from local_llm_lab import device  # noqa: E402
device.pin(seed=0)
import torch  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
from transformers import AutoTokenizer  # noqa: E402
from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.upstream import upstream_index_of_repo_layer  # noqa: E402
import jlens.hf as upstream_hf  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
# reuse the span tagger by reading its source rather than importing the script (which runs a capture on import)
_src = Path(__file__).with_name("workspace_capture.py").read_text()
_ns = {"re": re}; exec(_src[_src.index("def spans(prompt):"): _src.index("# ---- the sample, by rule")], _ns); spans = _ns["spans"]

man = json.loads((Path(args.capture) / "manifest.json").read_text()); sample_ids = man["sample"]
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
QUERIES = {"carrier_arms": "P_note (the decision position) through S-1", "current_note_arms": "q0 (the first token starting at or after the ```json fence) through S-1; S-1 is the supplied tool-name token, a query never read; P_act = S-2"}
CONTROL = ("random_equal_count: the all-carriers count of tokens drawn without replacement from the positions 1..P_note-1 in no carrier span (the task statement and the format tokens; never position 0, the <bos> sink), masked from P_note; random_equal_count_note: the current note's prose count from the same pool, masked from q0 — the count-matched control for the current-note arm; all_carriers_matched: the carrier keys subsampled to the pool's size where the carriers outnumber it, else all of them — the carrier arm that is count-matched to random_equal_count in every row (v3.3; v3.2 added the pool and the note control; v3.1's pool, the non-task tokens before P_note, was mostly the carrier spans themselves — degenerate, see the 4B record); the same-kind arms (older vs previous note, older vs previous call) are the role- and contiguity-matched comparisons; "
           "survival under a mask shows non-necessity of the masked edges under this intervention and does not date the decision; for the carrier arms the two-hop relay through unmasked earlier positions is an open route; "
           "for the current-note arm no such relay exists (every position after the note is a masked query)")
(OUT / "run.json").write_text(json.dumps({"schema_version": 3, "script_version": "3.3", "seat": "chief", "script_sha256": SCRIPT_SHA, "checkpoint": args.snapshot, "maps": args.maps, "maps_sha256": hashlib.sha256(Path(args.maps).read_bytes()).hexdigest(),
    "capture": args.capture, "corpus": args.corpus, "corpus_sha256": corpus_sha, "sample": sample_ids, "requested": len(sample_ids), "kinds": KINDS, "queries_masked_from": QUERIES, "control": CONTROL, "device": args.device, "seed": args.seed,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")

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
STATE = {"extra": None, "hook_calls": 0, "no_mask_calls": 0}  # extra: [1, 1, q, k] boolean, True = block
def pre_hook(module, args_, kwargs):
    extra = STATE["extra"]
    if extra is None: return None
    STATE["hook_calls"] += 1
    am = kwargs.get("attention_mask")
    if am is None: STATE["no_mask_calls"] += 1; return None
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
def run(ids, extra, want_attn, keep):
    STATE["extra"] = extra; STATE["hook_calls"] = 0; STATE["no_mask_calls"] = 0
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        out = model(input_ids=ids, output_attentions=want_attn, logits_to_keep=keep)  # logits only at P_note and P_act, where they are read
        acts = {k: rec.activations[k].detach()[0].float() for k in record_at}
    calls = (STATE["hook_calls"], STATE["no_mask_calls"]); STATE["extra"] = None
    if extra is not None:
        assert calls[0] == n_layers and calls[1] == 0, f"the hook ran on {calls[0]} of {n_layers} layers and found no attention_mask on {calls[1]}"
    return out, acts

def effective_checks(attn, extra_dev, P_act, cut_queries_from, keys):
    """Read from the returned attentions: masked mass on the applied set (all queries; at P_act), mass above the diagonal,
    local mass beyond the window (all queries; at P_act), and the mass from queries before the cut onto the masked keys."""
    masked_all = masked_act = above = far_all = far_act = pre = 0.0
    for li, A in enumerate(attn):
        a = A[0].float()  # [heads, S, S]
        above += float(torch.triu(a, diagonal=1).sum())
        if not IS_GLOBAL[li] and W:
            far_all += float(torch.tril(a, diagonal=-W).sum())
            if P_act - W + 1 > 0: far_act += float(a[:, P_act, : P_act - W + 1].sum())
        if extra_dev is not None:
            masked_all += float(a[:, extra_dev].sum()); masked_act += float(a[:, P_act, :][:, extra_dev[P_act]].sum())
            if keys and cut_queries_from > 0: pre += float(a[:, :cut_queries_from, :][:, :, keys].sum())
    return {"attention_on_masked_keys": masked_act, "attention_on_masked_edges_all_queries": masked_all, "attention_above_diagonal": above,
            "local_mass_beyond_window": far_act, "local_mass_beyond_window_all_queries": far_all, "pre_cut_queries_mass_on_masked_keys": pre}

results_fh = (OUT / "w3b.jsonl").open("w"); n_written = 0
LEAKY_CHECKED = False
for n, i in enumerate(sample_ids):
    r = rows_all[i]; m = r["metadata"]; rec_row = idx[i]; prompt, comp = r["prompt"], r["completion"]
    enc_p = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); p_ids = enc_p.input_ids  # the rendered prompt carries its own <bos>
    assert p_ids[0] == tok.bos_token_id and p_ids[1] != tok.bos_token_id, "the prompt must begin with exactly one <bos>"
    enc_c = tok(comp, add_special_tokens=False, return_offsets_mapping=True); t_idx = rec_row["P_act"] - len(p_ids) + 1
    assert enc_c.input_ids[t_idx] == tool_first[rec_row["tool_idx"]], f"row {i}: in-context tool token {enc_c.input_ids[t_idx]} is not the standalone first token {tool_first[rec_row['tool_idx']]} of {rec_row['tool']}"
    ids = torch.tensor([p_ids + enc_c.input_ids[: t_idx + 1]], device=args.device); S = ids.shape[1]
    P_note, P_act = rec_row["P_note"], rec_row["P_act"]; keep = torch.tensor([P_note, P_act], device=args.device)
    assert P_note == len(p_ids) - 1 and P_act == S - 2, f"row {i}: positions {P_note},{P_act} do not match the sequence ({len(p_ids)} prompt tokens, S={S}; the tool-name token is last)"
    kinds = ["format"] * S
    for k, (a, b) in enumerate(enc_p.offset_mapping):
        for name, s0, s1 in spans(prompt):
            if a >= s0 and b <= s1: kinds[k] = name
    note_end = max(comp.find(FENCE), 0)
    note_keys, q0, straddle, syntax_pos = note_cut(enc_c.offset_mapping, note_end, len(p_ids), t_idx)
    for k in note_keys: kinds[k] = "note"
    for k in syntax_pos: kinds[k] = "note_syntax"
    kinds[S - 1] = "tool_name"  # the target at P_act, supplied; a query position but never read
    # the text oracle, independent of the builder: the keys decode to the prose, the first query token begins the fence
    dec_keys = tok.decode([enc_c.input_ids[k - len(p_ids)] for k in note_keys]); first_q = tok.decode([enc_c.input_ids[q0 - len(p_ids)]]) if q0 is not None else ""
    oracle = (dec_keys == comp[:note_end]) and (q0 is not None) and comp[note_end:].startswith(first_q.lstrip())
    assert oracle, f"row {i}: placement oracle failed: keys decode to {dec_keys[-40:]!r} vs prose {comp[:note_end][-40:]!r}; first query token {first_q!r}"
    turn_start = P_note  # carrier arms: queries from the decision position onward
    arms, expected, cut_from, arm_keys, skipped = {"unmasked": None}, {}, {}, {}, {}
    for kind in KINDS + ["all_carriers"]:
        keys = [k for k, kk in enumerate(kinds) if (kk == kind if kind != "all_carriers" else kk in KINDS)]
        if not keys: continue
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, turn_start:, keys] = True; arms[kind] = e
        expected[kind] = edge_set(S, turn_start, keys); cut_from[kind] = turn_start; arm_keys[kind] = keys
    if note_keys and q0 is not None and q0 <= P_act:
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, q0:, note_keys] = True; arms["current_note"] = e
        expected["current_note"] = edge_set(S, q0, note_keys); cut_from["current_note"] = q0; arm_keys["current_note"] = note_keys
        if "all_carriers" in arms:
            e2 = arms["all_carriers"].clone(); e2[0, 0, q0:, note_keys] = True; arms["all_carriers_and_current_note"] = e2
            expected["all_carriers_and_current_note"] = expected["all_carriers"] | expected["current_note"]
            cut_from["all_carriers_and_current_note"] = turn_start; arm_keys["all_carriers_and_current_note"] = arm_keys["all_carriers"] + note_keys
    else:
        skipped["current_note"] = f"no prose note before the fence (note_prose_tokens={len(note_keys)}, q0={q0}, P_act={P_act})"
    # v3.2 control pool: every position before the decision position that lies in no carrier span — the task statement and the
    # format tokens — and never position 0 (the <bos> sink). v3.1 drew from the non-task tokens, which are the carrier spans plus
    # the format tokens, so its draw of the all-carriers count was three quarters the all-carriers mask itself (4B pass, 187 rows).
    pool = [k for k, kk in enumerate(kinds) if kk not in KINDS and 0 < k < turn_start]
    def kind_counts(ks): return {kk: sum(1 for k in ks if kinds[k] == kk) for kk in sorted({kinds[k] for k in ks})}
    random_info = {"pool": "positions 1..P_note-1 in no carrier span (task statement and format tokens); <bos> never", "pool_size": len(pool), "pool_kinds": kind_counts(pool)}
    n_all = int(arms["all_carriers"][0, 0, turn_start].sum()) if "all_carriers" in arms else 0
    if n_all and pool:
        pick = sorted(int(x) for x in rng.choice(pool, size=min(n_all, len(pool)), replace=False))
        assert all(kinds[k] not in KINDS and k > 0 for k in pick), "the random control drew a carrier token or position 0"
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, turn_start:, pick] = True; arms["random_equal_count"] = e
        expected["random_equal_count"] = edge_set(S, turn_start, pick); cut_from["random_equal_count"] = turn_start; arm_keys["random_equal_count"] = pick
        random_info["random_equal_count"] = {"requested": n_all, "picked": len(pick), "picked_kinds": kind_counts(pick), "queries_from": "P_note"}
    if "current_note" in arms and pool:  # the count-matched control for the current-note arm: the note's prose count, the same pool, the same queries (from the fence)
        pick2 = sorted(int(x) for x in rng.choice(pool, size=min(len(note_keys), len(pool)), replace=False))
        assert all(kinds[k] not in KINDS and k > 0 for k in pick2), "the note control drew a carrier token or position 0"
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, q0:, pick2] = True; arms["random_equal_count_note"] = e
        expected["random_equal_count_note"] = edge_set(S, q0, pick2); cut_from["random_equal_count_note"] = q0; arm_keys["random_equal_count_note"] = pick2
        random_info["random_equal_count_note"] = {"requested": len(note_keys), "picked": len(pick2), "picked_kinds": kind_counts(pick2), "queries_from": "q0"}
    # v3.3: a carrier arm that is count-matched to the control in every row. When the carriers outnumber the pool (55 of 187
    # rows in the 4B v3.2 pass, long episodes), the control could remove only pool-many keys; this arm removes a random subset of
    # the carrier keys of exactly that size, so all_carriers_matched vs random_equal_count is matched everywhere, and equals
    # all_carriers where no subsampling was needed. Drawn after the two control draws so v3.2's picks are unchanged.
    if "all_carriers" in arms and pool:
        car_keys = list(arm_keys["all_carriers"]); short = n_all > len(pool)
        sub = sorted(int(x) for x in rng.choice(car_keys, size=len(pool), replace=False)) if short else car_keys
        e = torch.zeros(1, 1, S, S, dtype=torch.bool); e[0, 0, turn_start:, sub] = True; arms["all_carriers_matched"] = e
        expected["all_carriers_matched"] = edge_set(S, turn_start, sub); cut_from["all_carriers_matched"] = turn_start; arm_keys["all_carriers_matched"] = sub
        random_info["all_carriers_matched"] = {"carrier_keys": n_all, "picked": len(sub), "subsampled": short, "picked_kinds": kind_counts(sub), "queries_from": "P_note"}
    row = {"i": i, "task_id": m["task_id"], "step": m["step"], "family": m["family"], "tool": rec_row["tool"], "tool_idx": rec_row["tool_idx"],
           "S": S, "P_note": P_note, "P_act": P_act, "note_prose_chars": note_end, "note_prose_tokens": len(note_keys), "note_syntax_tokens": len(syntax_pos),
           "note_fence_straddle_tokens": straddle, "q0": q0, "placement_oracle": bool(oracle), "kind_token_counts": {k: kinds.count(k) for k in set(kinds)}, "skipped_arms": skipped, "random_control": random_info, "arms": {}}
    if not LEAKY_CHECKED and "all_carriers" in arms:  # the cut verified with a deliberately leaky mask: one carrier key left open must receive attention
        leaky = arms["all_carriers"].clone(); keys = torch.nonzero(leaky[0, 0, P_act]).flatten().tolist(); leaky[0, 0, :, keys[0]] = False
        out_l, _ = run(ids, leaky, True, keep); a = torch.stack([A[0, :, P_act, :].float().mean(0) for A in out_l.attentions]).sum(0)
        open_mass = float(a[keys[0]]); masked_mass = float(a[keys[1:]].sum()) if len(keys) > 1 else 0.0
        passes = open_mass > 0.0 and masked_mass == 0.0
        emit("leaky_mask_check", open_key_attention_sum_over_layers=open_mass, masked_keys_attention=masked_mass, passes=passes)
        assert passes, "leaky negative control failed: the open key received no attention or the masked keys received some"
        LEAKY_CHECKED = True
    for arm, extra in arms.items():
        out, acts = run(ids, extra, True, keep)
        lg = out.logits[0].float(); assert lg.shape[0] == 2, f"logits_to_keep returned {tuple(lg.shape)}"; lg_note, lg_act = lg[0], lg[1]
        def margin(v):  # taken tool's logit minus the best other tool's logit, in logits
            t = v[tool_ids]; own = float(t[rec_row["tool_idx"]]); others = torch.cat([t[:rec_row["tool_idx"]], t[rec_row["tool_idx"] + 1:]]); return own - float(others.max())
        rec_arm = {"model_six_note": six(lg_note), "model_six_act": six(lg_act), "lens_note": readouts(acts, P_note), "lens_act": readouts(acts, P_act),
                   "margin_act_logits": margin(lg_act), "margin_note_logits": margin(lg_note)}
        extra_dev = extra[0, 0].to(args.device) if extra is not None else None
        chk = effective_checks(out.attentions, extra_dev, P_act, cut_from.get(arm, 0), arm_keys.get(arm, []))
        n_blk_act = int(extra[0, 0, P_act].sum()) if extra is not None else 0; n_blk_total = int(extra.sum()) if extra is not None else 0
        rec_arm["gate"] = {**chk, "n_blocked_edges_at_P_act": n_blk_act, "n_blocked_edges_total": n_blk_total}
        if extra is not None:
            actual = {(int(q), int(k)) for q, k in torch.nonzero(extra[0, 0]).tolist()}
            pl = placement(actual, expected[arm])
            rec_arm["receipt"] = {**pl, "expected_keys": arm_keys[arm], "query_interval": [cut_from[arm], S - 1], "endpoint": "S-1, the supplied tool-name token (an unread query); P_act = S-2",
                                  "applied_edges": len(actual), "expected_edges": len(expected[arm]), "causality_preserved": chk["attention_above_diagonal"] == 0.0,
                                  "pre_cut_queries_still_attend": chk["pre_cut_queries_mass_on_masked_keys"] > 0.0, "window_preserved_all_queries": chk["local_mass_beyond_window_all_queries"] == 0.0}
            assert pl["passes"], f"row {i}: arm {arm} placement receipt: {pl}"
            assert n_blk_act > 0, f"row {i}: arm {arm} blocks no edge at P_act"
            assert chk["attention_on_masked_edges_all_queries"] == 0.0, f"row {i}: arm {arm}: attention on masked edges {chk['attention_on_masked_edges_all_queries']}"
            assert rec_arm["receipt"]["pre_cut_queries_still_attend"], f"row {i}: arm {arm}: queries before the cut place no mass on the masked keys (over-masking?)"
        assert chk["attention_above_diagonal"] == 0.0, f"row {i}: arm {arm}: mass above the diagonal {chk['attention_above_diagonal']} (causal structure broken)"
        assert chk["local_mass_beyond_window_all_queries"] == 0.0, f"row {i}: arm {arm}: local mass beyond the window {chk['local_mass_beyond_window_all_queries']}"
        if cut_from.get(arm, 0) > P_note:  # queries before the cut (P_note among them) are untouched, so P_note's readouts must be bit-identical to the unmasked forward
            u = row["arms"]["unmasked"]
            same = all(json.dumps(rec_arm[f]) == json.dumps(u[f]) for f in ("model_six_note", "lens_note", "margin_note_logits"))
            rec_arm["p_note_identical_to_unmasked"] = same
            assert same, f"row {i}: the current-note mask changed P_note's readouts"
        row["arms"][arm] = rec_arm
    results_fh.write(json.dumps(row) + "\n"); results_fh.flush(); n_written += 1
    emit("row", n_done=n + 1, of=len(sample_ids), arms=list(arms), skipped=skipped, note_prose_tokens=len(note_keys), q0=q0,
         blocked_at_P_act={k: v["gate"]["n_blocked_edges_at_P_act"] for k, v in row["arms"].items()}, receipts_pass=all(v.get("receipt", {}).get("passes", True) for v in row["arms"].values()))
for h in handles: h.remove()
results_fh.close()
(OUT / "manifest.json").write_text(json.dumps({"schema_version": 3, "script_version": "3.3", "seat": "chief", "script_sha256": SCRIPT_SHA, "checkpoint": args.snapshot, "maps": args.maps, "capture": args.capture, "sample": sample_ids, "requested": len(sample_ids),
    "kinds": KINDS, "queries_masked_from": QUERIES,
    "current_note": {"keys": "completion tokens whose character offset starts before the ```json fence (the prose note, trailing newline included, cut as the previous-note spans are cut)",
                     "queries": "every token from the fence onward through S-1 (the supplied tool-name token, never read), so no relay through the syntax between the note and the action is open",
                     "skipped_when": "the completion has no prose before the fence (recorded per row under skipped_arms)"},
    "checks_each_a_hard_stop": ["self-test: the builder reproduces hand-labelled keys/q0/straddle on synthetic tokenisations at four prompt offsets; the expected-edge oracle rejects six wrong masks",
                                "text oracle per row: note keys decode to the prose before the fence; the first query token begins the fence",
                                "receipt per arm: applied edges == expected edges (no missing, no unexpected); >= 1 edge at P_act",
                                "attention on every masked edge over all queries, layers and heads == 0.0", "no attention above the diagonal (causal structure intact)",
                                "local layers: zero mass beyond the window for every query", "queries before the cut still place mass on the masked keys (no over-masking)",
                                "the hook ran on every layer and found the native attention_mask on each",
                                "leaky negative control: one carrier key left open receives attention, the rest none", "current-note arm: P_note readouts bit-identical to the unmasked forward",
                                "in-context tool-name token == the standalone first token used by the six-tool readout; the six first tokens distinct", "capture and pass read the same corpus by digest"],
    "control": CONTROL, "precision": "float32", "width": 1, "rows_written": n_written, "corpus": args.corpus, "corpus_sha256": corpus_sha,
    "device": args.device, "is_global_layer": IS_GLOBAL, "sliding_window": W, "design": "WORKSPACE-EXPERIMENTS-2026-09-10.md W-3b; repair of the empty current-note cut (Codex 712f78c); placement certified (Codex 63d0549)"}, indent=2) + "\n")
emit("done", n=n_written, minutes=round((time.monotonic() - T0) / 60, 1))
