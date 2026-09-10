"""In-domain control for the A2 shares: the same dictionaries, the same admitted lens, the same error budget
and the same decomposition function as the device A2 runs, applied to residuals from the lens's OWN fit
domain -- the fit split's prose prompts, 128 tokens, positions 16..126 -- captured at width 1 in float32
with the capture tool's conventions (bf16 checkpoint cast to float32, eager attention, block output at
repository layer L = ActivationRecorder at block index L-1). Five fixed positions per prompt.

Question it answers: is the 5-11x amplification of the dictionary's residual by the readout a property of
the out-of-domain agent-transcript positions, or of these dictionaries and this readout wherever they are
applied? Not an admission run: no pairing is registered for these cells (fit path vs capture path at the fit
domain is what the pairing tables measure, and every registered term is below 5e-5), and the runner's
capture-format admission does not apply. Reported as a control beside the A2 results, never as an A2 result.

usage: indomain_control.py CONFIG LAYERS(comma) OUT_JSON   (paths for the 4B are fixed below)
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/chief/lab-a2/src")
from local_llm_lab import hf_text
from local_llm_lab.probes import sae_bridge as B, device_bridge as DB
from local_llm_lab.probes.device_lens_admission import load_admitted_device_lens
from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus
from local_llm_lab.pipeline.lens_fitting.upstream import load_upstream, upstream_index_of_repo_layer
from jlens.hooks import ActivationRecorder
import jlens.hf as upstream_hf

conf = json.loads(Path(sys.argv[1]).read_text()); layers = [int(x) for x in sys.argv[2].split(",")]; out = Path(sys.argv[3])
if out.exists(): sys.exit(f"refused: {out} exists")
S4 = Path("/workspace/.hf_home/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
CORPUS = Path("/workspace/lens-corpus/prose-gemma3-4b-cuda-bf16.json"); MAX_SEQ = 128; POSITIONS = [16, 44, 71, 99, 126]
DICT_ROOT = Path("/workspace/.hf_home/hub/models--google--gemma-scope-2-4b-it/snapshots/3e94b68be95290aada5b7525cf431d3040f81bb1/resid_post_all")
REC_ROOT = Path("/workspace/.hf_home/dictionaries/google/gemma-scope-2-4b-it/resid_post_all")
budget = B.ErrorBudget.from_dict(conf["error_budget"]); t0 = time.time()
lens, metadata = load_admitted_device_lens(Path("/workspace/chief/out/lens4b-f32/admitted-maps.npz"), checkpoint=S4, model_base=conf["model_base"], expected_sha256=conf["lens_sha256"])
rows = [r for r in read_corpus(CORPUS) if r.get("split") == "fit"]
assert len(rows) == 201, len(rows)
model, report = hf_text.load_text_causal_lm(S4, dtype="bfloat16", attn_implementation="eager", device="cuda:0")
model.eval(); model.to(torch.float32)
for p in model.parameters(): p.requires_grad_(False)
load_upstream(); lm = upstream_hf.HFLensModel(model, tokenizer=None)
record_at = [upstream_index_of_repo_layer(L) for L in layers]
resid = {L: np.zeros((len(rows), len(POSITIONS), lens.hidden_size), np.float32) for L in layers}; emitted = np.zeros((len(rows), len(POSITIONS)), np.int64)
with torch.no_grad():
    for i, r in enumerate(rows):
        ids = torch.tensor([r["ids"][:MAX_SEQ]], device="cuda:0")
        with ActivationRecorder(lm.layers, at=record_at) as rec:
            model(input_ids=ids)
        for L, u in zip(layers, record_at):
            a = rec.activations[u].detach()[0].float().cpu().numpy()
            for j, p in enumerate(POSITIONS): resid[L][i, j] = a[p]
        for j, p in enumerate(POSITIONS): emitted[i, j] = int(r["ids"][p + 1])
print(json.dumps({"forward_done_s": round(time.time() - t0, 1), "prompts": len(rows), "dtype": str(next(model.parameters()).dtype), "attn": report["attn_implementation"]}), flush=True)
del model; torch.cuda.empty_cache()
unemb = B.load_unembedding(S4); result = {"basis": __doc__, "config_sha256": DB._hash(Path(sys.argv[1])), "lens_sha256": lens.sha256, "positions": POSITIONS, "prompts": len(rows), "max_seq_len": MAX_SEQ, "corpus": str(CORPUS), "layers": {}}
def q(xs): xs = np.asarray([x for x in xs if x is not None], float); return {"n": int(xs.size), "min": float(xs.min()), "median": float(np.median(xs)), "p90": float(np.quantile(xs, .9)), "max": float(xs.max())} if xs.size else {"n": 0}
for L in layers:
    folder = f"layer_{L - 1}_width_16k_l0_small"; receipt = json.loads((REC_ROOT / folder / "DIGEST.json").read_text())
    dictionary = B.load_dictionary(DICT_ROOT / folder / "params.safetensors", DICT_ROOT / folder / "config.json", source=f"{receipt['repo']}/{receipt['folder']}")
    assert B.layer_for_hook(dictionary.hook_point) == L
    J = B.lens_map_for_layer(lens, L); cells = []
    for i in range(len(rows)):
        for j, p in enumerate(POSITIONS):
            o = B.decompose_position(dictionary, unemb, J, resid[L][i, j], int(emitted[i, j]), error_budget=budget, k=conf["a2"]["k"], tolerance=conf["a2"]["identity_tolerance"])
            cells.append({"prompt": i, "position": p, "emitted": int(emitted[i, j]), "raw_share": o["raw_reconstruction_share"], "score_share": o["lens_score_error_share"], "ratio": (o["lens_score_error_share"] / o["raw_reconstruction_share"]) if o["raw_reconstruction_share"] else None, "active": o["active_features"], "ranked": bool(o["ranked"]), "identity_gap": o["identity_gap"], "score": o["score"]})
    result["layers"][str(L)] = {"dictionary_folder": receipt["folder"], "cells": len(cells), "raw_share": q([c["raw_share"] for c in cells]), "score_share": q([c["score_share"] for c in cells]), "ratio": q([c["ratio"] for c in cells]), "active": q([c["active"] for c in cells]), "ranked": sum(c["ranked"] for c in cells), "identity_gap_max": max(c["identity_gap"] for c in cells), "by_position": {str(p): {"raw_share_median": float(np.median([c["raw_share"] for c in cells if c["position"] == p])), "score_share_median": float(np.median([c["score_share"] for c in cells if c["position"] == p])), "ratio_median": float(np.median([c["ratio"] for c in cells if c["position"] == p]))} for p in POSITIONS}, "rows": cells}
    print(json.dumps({"layer": L, "ranked": result["layers"][str(L)]["ranked"], "raw": result["layers"][str(L)]["raw_share"]["median"], "score": result["layers"][str(L)]["score_share"]["median"], "ratio": result["layers"][str(L)]["ratio"]["median"]}), flush=True)
result["elapsed_s"] = round(time.time() - t0, 1); out.write_text(json.dumps(result, indent=1) + "\n"); print(json.dumps({"event": "done", "elapsed_s": result["elapsed_s"]}))
