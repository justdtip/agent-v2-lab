"""Merge a tail re-capture (workspace_capture_v2.py --rows-from N --rows-to M) into a main capture that died at row N.

The main capture's memmaps hold rows [0, N) (zeros beyond); the tail's hold [N, M) at memmap rows 0..M-N-1.
Copies the tail rows into the main memmaps in place, appends the tail's index rows to the main index
(a reconstructed one if the main pass never wrote its own), appends the tail's sample rows and copies its
attn_*.npy files. Run only when no process holds the main memmaps open. Refuses if the main rows the tail
would overwrite are non-zero (they were captured, not lost) or if the tail's manifest range disagrees.

usage: merge_tail_capture.py MAIN_DIR TAIL_DIR MAIN_INDEX_JSONL
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import numpy as np
main, tail, main_index = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tm = json.loads((tail / "manifest.json").read_text()); lo, hi = int(tm["rows_from"]), int(tm["rows_to"])
assert tm.get("memmap_row") == "global index minus rows_from", "the tail is not a ranged v2 capture"
mrows = [json.loads(l) for l in main_index.read_text().splitlines() if l.strip()]
trows = [json.loads(l) for l in (tail / "index.jsonl").read_text().splitlines() if l.strip()]
assert [r["i"] for r in trows] == list(range(lo, lo + len(trows))), "the tail index is not contiguous from rows_from"
assert all(r["i"] < lo for r in mrows), f"the main index already holds rows at or beyond {lo}"
for name in ("residual_note", "residual_act"):
    M = np.load(main / f"{name}.npy", mmap_mode="r+"); T = np.load(tail / f"{name}.npy", mmap_mode="r")
    assert M.shape[1:] == T.shape[1:] and T.shape[0] == len(trows) <= hi - lo, (M.shape, T.shape)
    assert not np.any(M[lo: lo + T.shape[0], 0, :] != 0.0), f"{name}: main rows [{lo}, {lo + T.shape[0]}) are not zeros — captured, not lost"
    M[lo: lo + T.shape[0]] = T; M.flush(); del M
merged = sorted(mrows + trows, key=lambda r: r["i"])
assert [r["i"] for r in merged] == list(range(len(merged))), "merged index is not contiguous from 0"
main_index.write_text("\n".join(json.dumps(r) for r in merged) + "\n")
ts = tail / "sample.jsonl"
if ts.exists() and ts.read_text().strip():
    with (main / "sample.jsonl").open("a") as fh: fh.write(ts.read_text() if ts.read_text().endswith("\n") else ts.read_text() + "\n")
n_attn = 0
for f in tail.glob("attn_*.npy"):
    shutil.copy2(f, main / f.name); n_attn += 1
print(json.dumps({"event": "merged", "main_rows_before": len(mrows), "tail_rows": len(trows), "merged_rows": len(merged), "attn_copied": n_attn, "range": [lo, lo + len(trows)]}))
