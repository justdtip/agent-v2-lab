"""CPU-only read of an artefact already on disk: no GPU, no window, no model."""
import json
import numpy as np

fd = np.load("/workspace/wsd/out/golden-4b/fd-maps.npz")
ex = np.load("/workspace/wsd/out/golden-4b/exact-maps.npz")
print("layer  zero_cols  frac_zero   median_col_norm_fd   median_col_norm_exact   fd/exact")
rows = []
for key in sorted(fd.files, key=lambda k: int(k[1:])):
    layer = int(key[1:]) + 1
    a = np.asarray(fd[key], np.float64)
    b = np.asarray(ex[key], np.float64)
    # A direction whose response was exactly zero at both selected positions leaves its column
    # exactly zero: the accumulator is only ever incremented by that direction's own delta.
    col_fd = np.linalg.norm(a, axis=0)
    col_ex = np.linalg.norm(b, axis=0)
    zeros = int((col_fd == 0.0).sum())
    rows.append({
        "repo_layer": layer, "zero_columns": zeros, "fraction_zero": zeros / a.shape[1],
        "median_col_norm_fd": float(np.median(col_fd)),
        "median_col_norm_exact": float(np.median(col_ex)),
        "frobenius_fd": float(np.linalg.norm(a)), "frobenius_exact": float(np.linalg.norm(b)),
    })
    if layer in (1, 2, 5, 9, 13, 17, 21, 25, 29, 32, 33):
        print("%5d  %9d  %9.4f   %18.6g   %21.6g   %.4f" % (
            layer, zeros, zeros / a.shape[1], np.median(col_fd), np.median(col_ex),
            np.linalg.norm(a) / np.linalg.norm(b)))
with open("/workspace/wsd/out/golden-4b/zero-response.json", "w") as f:
    json.dump({"rule": "a column is exactly zero iff that direction's response was exactly zero at "
                       "both selected positions", "per_layer": rows}, f, indent=1)
tot = sum(r["zero_columns"] for r in rows)
print("\ntotal exactly-zero columns across 33 layers: %d of %d (%.4f)" % (
    tot, 33 * 2560, tot / (33 * 2560)))
print("norm ratio fd/exact, min %.4f  max %.4f" % (
    min(r["frobenius_fd"] / r["frobenius_exact"] for r in rows),
    max(r["frobenius_fd"] / r["frobenius_exact"] for r in rows)))
