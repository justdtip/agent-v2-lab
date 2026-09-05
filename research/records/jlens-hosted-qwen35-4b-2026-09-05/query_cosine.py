"""R3d: median pairwise cosine between a head's queries at different positions, long-gate heads
against the rest, on the same two contexts. Also the cosine restricted to the four read positions
the overlap test used, and the overlap the Head's calibration predicts from that cosine."""
from __future__ import annotations
# RECORD, NOT A LAUNCHER (2026-09-05). This script produced figures cited in
# design_specifications/pending/WP3-TRANSPORT-WEIGHTS-DESIGN-2026-09-05.md and the hosted-lens results memo.
# It loads the model directly with an ad hoc process check and must not be started by hand again: the
# one operating constraint is one model load at a time, enforced by the loader wrapper and model-run lock
# of issue 83. Runnable versions arrive as package entry points under issues 79 (hosted lens) and 81 (transport).
import sys as _sys
if __name__ == "__main__" and "--i-am-a-record" not in _sys.argv:
    _sys.exit("refusing to run: this file is a record of the 2026-09-05 measurements, not a launcher; "
              "re-run through the package entry points of issues 79 and 81 once the issue 83 lock has landed")

import json, sys, time, subprocess
from pathlib import Path
import numpy as np
REPO = Path("/Users/daniel.tipton/Desktop/An app"); sys.path.insert(0, str(REPO / "src"))
import mlx.core as mx
mx.set_cache_limit(2 * 1024 ** 3)
import os as _os
def _other_model_processes():
    """Pids running a model-loading entry point, excluding this process and the shell that launched it."""
    pids = subprocess.run(["pgrep", "-f", "agent-v2-|run_hosted_lens|contrast_reactions|calibrate_alpha|overlap_test|query_cosine|mlx_lm convert"], capture_output=True, text=True).stdout.split()
    mine = {str(_os.getpid()), str(_os.getppid())}
    found = []
    for pid in pids:
        if pid in mine: continue
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout.strip().replace("\n", " ")
        if not cmd or "pgrep" in cmd or "ps -o" in cmd: continue
        if any(tag in cmd.split("<<")[0] for tag in ("agent-v2-", "run_hosted_lens", "contrast_reactions", "calibrate_alpha.py", "overlap_test.py", "query_cosine.py", "mlx_lm convert")):
            found.append(f"{pid}: {cmd[:120]}")
    return found
others = _other_model_processes()
if others: print("REFUSED: another model-loading process is alive:\n" + "\n".join(others)); sys.exit(2)
from mlx_lm.models import qwen3_5 as q35
from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.evaluate import load_policy
from local_llm_lab.pipeline.jlens import encode
from local_llm_lab.pipeline.tasks import make_jspace_tasks
from local_llm_lab.probes.jspace_sweep import select_cases
captures = []
_orig = q35.gated_delta_update
def hooked(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
    out, st = _orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel); mx.eval(out)
    captures.append({"q": q, "k": k}); return out, st
q35.gated_delta_update = hooked
t0 = time.time()
spec = load_model_spec("qwen35-4b"); model, tokenizer, view, resolved = load_policy(spec, None)
tasks = make_jspace_tasks("jsweep", 720, 20260902); cases = select_cases(tasks, tokenizer, spec=spec, probe_step=3)
contexts = [encode(tokenizer, cases[i]["prompt"]) for i in (0, 21)]
calib = json.load(open(REPO / "outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/alpha_calibration.json"))
tau = {0: np.zeros((24, 32)), 1: np.zeros((24, 32))}
for hi in calib["head_info"]: tau[hi["context"]][hi["block"]] = np.array(hi["tau_head"])
long_mask = (tau[0] > 1000) & (tau[1] > 1000)          # (24, 32) value heads
rng = np.random.default_rng(0)
rows = []
for ci, ids in enumerate(contexts):
    captures.clear(); view.residuals(ids, [view.num_layers]); T = len(ids)
    read_pos = sorted(set([T - 1] + [int(round(T * f)) - 1 for f in (0.5, 0.75, 0.9)]))
    many = np.linspace(300, T - 1, 60).astype(int)
    for b in range(24):
        q = np.array(captures[b]["q"].astype(mx.float32))[0]      # (T, Hk, Dk)
        k = np.array(captures[b]["k"].astype(mx.float32))[0]
        Hk = q.shape[1]
        for hk in range(Hk):
            Q = q[:, hk]; Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            def med_cos(idx):
                C = Qn[idx] @ Qn[idx].T; iu = np.triu_indices(len(idx), 1); return float(np.median(C[iu]))
            c_read = med_cos(np.array(read_pos)); c_many = med_cos(many)
            # keys' own stability for reference (a static key direction would be a different story)
            Kn = k[:, hk] / (np.linalg.norm(k[:, hk], axis=1, keepdims=True) + 1e-12)
            Ck = Kn[many] @ Kn[many].T; iu = np.triu_indices(len(many), 1); c_key = float(np.median(Ck[iu]))
            for hv in (2 * hk, 2 * hk + 1):                        # value heads sharing this key head (h // 2)
                rows.append({"context": ci, "block": b, "key_head": hk, "value_head": hv, "long_gate": bool(long_mask[b, hv]),
                             "median_query_cosine_read_positions": c_read, "median_query_cosine_60_positions": c_many, "median_key_cosine_60_positions": c_key})
    print(f"context {ci} done, {time.time()-t0:.0f}s")
lg = [r for r in rows if r["long_gate"]]; ot = [r for r in rows if not r["long_gate"]]
def q(vals, p): return float(np.percentile(vals, p))
summary = {"long_gate_entries": len(lg), "other_entries": len(ot),
           "query_cosine_read_positions": {"long_gate": {p: q([r["median_query_cosine_read_positions"] for r in lg], p) for p in (10, 50, 90)},
                                           "other": {p: q([r["median_query_cosine_read_positions"] for r in ot], p) for p in (10, 50, 90)}},
           "query_cosine_60_positions": {"long_gate": {p: q([r["median_query_cosine_60_positions"] for r in lg], p) for p in (10, 50, 90)},
                                         "other": {p: q([r["median_query_cosine_60_positions"] for r in ot], p) for p in (10, 50, 90)}},
           "key_cosine_60_positions": {"long_gate": {p: q([r["median_key_cosine_60_positions"] for r in lg], p) for p in (10, 50, 90)},
                                       "other": {p: q([r["median_key_cosine_60_positions"] for r in ot], p) for p in (10, 50, 90)}},
           "long_gate_entries_with_query_cosine_over_0.6": int(sum(1 for r in lg if r["median_query_cosine_read_positions"] > 0.6)),
           "long_gate_entries_with_query_cosine_under_0.3": int(sum(1 for r in lg if r["median_query_cosine_read_positions"] < 0.3))}
json.dump({"summary": summary, "rows": rows}, open(REPO / "outputs/probes/jlens-hosted-qwen35-4b-2026-09-05/query_cosine.json", "w"), indent=1)
print(json.dumps(summary, indent=1)); print("elapsed", round(time.time() - t0, 1), "s")
