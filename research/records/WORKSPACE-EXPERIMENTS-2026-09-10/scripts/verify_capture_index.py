"""After a capture closes: its own index.jsonl against the deterministic reconstruction, every row.
usage: verify_capture_index.py CAPTURE_DIR CORPUS SNAPSHOT  (runs reconstruct_index.py to a temp file, compares the deterministic fields, counts)"""
import json, subprocess, sys, tempfile
from pathlib import Path
cap, corpus, snap = sys.argv[1:4]
tmp = tempfile.mktemp(suffix=".jsonl")
out = subprocess.run([sys.executable, str(Path(__file__).with_name("reconstruct_index.py")), cap, corpus, snap, tmp], capture_output=True, text=True)
print(out.stdout.strip().splitlines()[-1][:200] if out.stdout.strip() else out.stderr[-400:])
a = [json.loads(l) for l in open(Path(cap) / "index.jsonl") if l.strip()]; b = [json.loads(l) for l in open(tmp) if l.strip()]
det = ("i", "task_id", "step", "family", "variant", "recovery", "tool", "tool_idx", "P_note", "P_act", "n_prompt_tokens", "n_note_tokens", "in_sample")
diffs = [(r["i"], k, r[k], q[k]) for r, q in zip(a, b) for k in det if r[k] != q[k]]
man = json.loads((Path(cap) / "manifest.json").read_text())
print(json.dumps({"event": "verified", "index_rows": len(a), "reconstructed_rows": len(b), "manifest_decisions": man.get("decisions"), "sample_n": len(man.get("sample", [])),
                  "deterministic_diffs": len(diffs), "first_diffs": diffs[:5], "unresolved_model_act_below_1e-3": sum(1 for r in a if r["model_mass_act"] < 1e-3),
                  "verdict": "index consistent with the corpus" if len(a) == len(b) == man.get("decisions") and not diffs else "SEE ABOVE"}))
