"""Where did depth-restricted training put its changes, and did it learn the same thing?

Three measurements on LoRA adapters, none of which loads a model, so this runs beside a job.

    .venv/bin/python research/records/DEPTH-WHY-2026-09-08/adapter_geometry.py

1. Per-layer relative perturbation: ||dW_layer||_F / ||W_layer||_F, aggregating numerator and
   denominator separately in quadrature over the layer's adapted modules. NOT the quadrature sum
   of per-module ratios -- that inflates by roughly sqrt(number of modules) and was the author's
   first mistake here.
2. Effective rank of each delta, exp of the entropy of its normalised squared singular values.
   Computed from the r x r core, never materialising the d x d product.
3. Cross-adapter cosine in weight space between two adapters' deltas at the same layer and
   module, with two references without which the number means nothing: a null from independent
   random rank-r deltas of the same shapes, and a control from one adapter against its own later
   checkpoint.

The LoRA convention is read from the library rather than assumed: mlx_lm's LoRALinear.fuse
computes ``delta = (scale * lora_b.T) @ lora_a.T`` and adds it to the base weight, and
``linear_to_lora_layers`` passes ``config["scale"]`` through undivided, so the multiplier is the
recorded scale itself and not scale/rank.
"""
import collections, glob, json, re, struct, sys
import numpy as np

ROOT = "/Users/daniel.tipton/Desktop/An app"
BF16 = glob.glob("/Users/daniel.tipton/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-bf16/snapshots/*/")[0]
SCALE = 32.0

def st_open(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n

def st_get(f, hdr, base, key):
    m = hdr[key]
    s, e = m["data_offsets"]
    f.seek(base + s)
    buf = f.read(e - s)
    if m["dtype"] == "F32":
        a = np.frombuffer(buf, dtype=np.float32)
    elif m["dtype"] == "BF16":
        a = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif m["dtype"] == "F16":
        a = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    else:
        raise ValueError(m["dtype"])
    return a.reshape(m["shape"])

def factors(path):
    f, hdr, base = st_open(path)
    out = {}
    for stem in sorted({k.rsplit(".lora_", 1)[0] for k in hdr if k != "__metadata__"}):
        out[stem] = (st_get(f, hdr, base, stem + ".lora_a").astype(np.float64),
                     st_get(f, hdr, base, stem + ".lora_b").astype(np.float64))
    f.close()
    return out

def dnorm(a, b):
    """||scale * a @ b||_F from the r x r core."""
    return SCALE * float(np.sqrt(np.trace((a.T @ a) @ (b @ b.T))))

def dcos(x, y):
    (a1, b1), (a2, b2) = x, y
    num = float(np.trace((a1.T @ a2) @ (b2 @ b1.T)))
    return SCALE * SCALE * num / (dnorm(a1, b1) * dnorm(a2, b2))

def effective_rank(a, b):
    qa, ra = np.linalg.qr(a)
    qb, rb = np.linalg.qr(b.T)
    sv = np.linalg.svd(ra @ rb.T, compute_uv=False)
    p = sv ** 2
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))

layer_of = lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1))

def base_norms(stems):
    index = json.load(open(BF16 + "model.safetensors.index.json"))["weight_map"]
    want = {s + ".weight" for s in stems}
    out = {}
    for shard in sorted({index[k] for k in want if k in index}):
        f, hdr, base = st_open(BF16 + shard)
        for k in want:
            if index.get(k) == shard and k in hdr:
                out[k] = float(np.linalg.norm(st_get(f, hdr, base, k)))
        f.close()
    return out

def main():
    runs = {
        "armA@400": f"{ROOT}/outputs/agent-v2e-qwen35-4b/adapters/0000400_adapters.safetensors",
        "arm1@400": f"{ROOT}/outputs/agent-v2e-qwen35-4b-top8/adapters/0000400_adapters.safetensors",
        "arm1@800": f"{ROOT}/outputs/agent-v2e-qwen35-4b-top8/adapters/0000800_adapters.safetensors",
    }
    F = {k: factors(p) for k, p in runs.items()}
    W = base_norms({s for f in F.values() for s in f})

    print("1. per-layer relative perturbation ||dW||/||W||")
    print(f"{'layer':>5} {'kind':>10} " + " ".join(f"{k:>9}" for k in runs))
    cols = {}
    for name, f in F.items():
        num, den = collections.defaultdict(float), collections.defaultdict(float)
        for s, (a, b) in f.items():
            num[layer_of(s)] += dnorm(a, b) ** 2
            den[layer_of(s)] += W[s + ".weight"] ** 2
        cols[name] = {L: num[L] ** 0.5 / den[L] ** 0.5 for L in num}
    for L in range(32):
        kind = "attention" if L % 4 == 3 else "recurrent"
        cells = [f"{cols[n][L]:9.4f}" if L in cols[n] else "        -" for n in runs]
        print(f"{L:5d} {kind:>10} " + " ".join(cells))

    print("\n2. effective rank of the delta (max = LoRA rank)")
    for name, f in F.items():
        by = collections.defaultdict(list)
        for s, (a, b) in f.items():
            by[layer_of(s)].append(effective_rank(a, b))
        vals = [np.mean(v) for _, v in sorted(by.items())]
        print(f"  {name:9} mean {np.mean(vals):.2f} over {len(vals)} layers")

    print("\n3. cross-adapter cosine in weight space, same layer and module")
    for x, y, label in [("armA@400", "arm1@400", "matched iterations"),
                        ("armA@400", "arm1@800", "each at its evaluated checkpoint"),
                        ("arm1@400", "arm1@800", "CONTROL: one adapter vs its own later step")]:
        v = [dcos(F[x][s], F[y][s]) for s in sorted(set(F[x]) & set(F[y]))]
        print(f"  {label:42} {np.mean(v):+.4f}   n={len(v)}")
    rng = np.random.default_rng(0)
    a, b = next(iter(F["arm1@800"].values()))
    null = [dcos((rng.normal(size=a.shape), rng.normal(size=b.shape)),
                 (rng.normal(size=a.shape), rng.normal(size=b.shape))) for _ in range(30)]
    print(f"  {'NULL: independent random deltas, same shapes':42} {np.mean(null):+.4f}   sd {np.std(null):.4f}")

if __name__ == "__main__":
    sys.exit(main())
