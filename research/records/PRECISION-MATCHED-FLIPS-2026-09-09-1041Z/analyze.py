"""Join frozen precision-control reports and draw figures without model imports."""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_SHA256 = '84b0e3ef66543f6eff70426c5f23f6132e99c6d697b92bd816e0fb24f45aeb39'
_original_import = builtins.__import__


def file_only_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'transformers'}:
        raise ImportError('model imports forbidden: this analysis reads files only')
    return _original_import(name, *args, **kwargs)


builtins.__import__ = file_only_import
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-gpu-analysis-mpl')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scalar(obj, key):
    require(obj[key]['basis'] == 'measured-here', f'{key}: unexpected basis')
    return obj[key]['value']


def key(row):
    return row['episode'], row['turn'], row['position']


def index(rows):
    result = {key(row): row for row in rows}
    require(len(result) == len(rows), 'duplicate position identity')
    return result


def load_sources(root=ROOT):
    raw = (root / 'SOURCES.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, 'manifest hash mismatch')
    manifest = json.loads(raw)
    data = {}
    for name, entry in manifest['files'].items():
        raw = (root / 'source' / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == entry['sha256'], f'source hash mismatch: {name}')
        if name.endswith('.json'):
            data[name] = json.loads(raw)
        elif name.endswith('.jsonl'):
            require(raw.endswith(b'\n'), f'partial JSONL: {name}')
            data[name] = [json.loads(line) for line in raw.splitlines()]
    return manifest, data


def joined_data(data):
    flips = data['confident-flips.json']
    arm = data['mlx-bf16-arm.json']
    margins = data['flip-margins.json']
    reference = data['mlx-bf16-reference.json']
    before, matched, margin = (index(d['positions']) for d in (flips, arm, margins))
    require(flips['count'] == len(before) == 24, 'expected frozen set of 24')
    require(before.keys() == matched.keys(), 'precision arm position set differs')
    require(arm['positions'] == data['mlx-bf16-arm.jsonl'], 'arm JSONL differs')
    full_lines = data['mlx-bf16-reference.jsonl']
    require(len({r['episode'] for r in full_lines}) == len(full_lines), 'duplicate reference episode')
    require({r['episode']: r['positions'] for r in full_lines} == reference['episodes'], 'reference JSONL differs')
    full = index([{'episode': e, **r} for e, rows in reference['episodes'].items() for r in rows])
    require(len(reference['episodes']) == 15, 'reference episode count differs')
    require(len(full) == scalar(reference, 'positions') == 5245, 'reference denominator differs')
    require(before.keys() <= full.keys(), 'reference misses selected positions')
    require(sum(r['gap_ulps'] <= 2 for r in full.values()) == scalar(reference, 'within_2_ulp'), 'reference tie count differs')
    for row in full.values():
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ('gap', 'gap_ulps')), 'invalid reference gap')
    gpu = data['cuda-all15-v2.json']
    require(not gpu['head']['smoke'], 'fixture data not allowed')
    require(gpu['head']['port_precision'] == 'cuda:0 bfloat16', 'GPU precision differs')
    require(gpu['head']['checkpoint_sha256'] == data['cuda-all15.json']['head']['checkpoint_sha256'], 'GPU checkpoint differs')
    extracted = index([{'episode': e['label'], **r} for e in gpu['episodes'] for r in scalar(e, 'hard_flip_positions')])
    require(extracted == before, '24 positions differ from canonical CUDA recapture')
    require(scalar(gpu, 'hard_flips') == len(before), 'CUDA flip denominator differs')
    rows = []
    for ident, original in before.items():
        a, ref = matched[ident], full[ident]
        require(all(a[k] == v for k, v in original.items()), 'arm token/probability identity differs')
        p = original['recorded_probability']
        require(math.isfinite(p) and .99 <= p <= 1, 'original probability outside frozen gate')
        require(original['recorded_token'] != original['produced_token'], 'original row is not a flip')
        require(ref['token'] == a['mlx_bf16_token'], 'later MLX reference changed selected token')
        agrees = a['mlx_bf16_token'] == original['produced_token']
        expected = 'quantisation' if agrees else ('port' if a['mlx_bf16_token'] == original['recorded_token'] else 'unresolved')
        require(a['verdict'] == expected, 'producer verdict does not follow token equality')
        rows.append({**original, 'mlx_bf16_token': a['mlx_bf16_token'],
                     'bf16_cuda_agrees': agrees, 'source_verdict': a['verdict'],
                     'reference_gap': ref['gap'], 'reference_gap_ulps': ref['gap_ulps'],
                     'reference_within_2_ulp': ref['gap_ulps'] <= 2,
                     'outcome_basis': 'card CUDA bf16 joined to laptop MLX bf16',
                     'probability_basis': 'original laptop MLX 4-bit reference',
                     'gap_basis': 'later laptop MLX bf16 reference; actual-top1-spacing producer'})
    counts = Counter(r['source_verdict'] for r in rows)
    require(dict(counts) == {k: scalar(arm['counts'], k) for k in arm['counts']}, 'arm counts differ')
    survivors = {key(r) for r in rows if not r['bf16_cuda_agrees']}
    require(margin.keys() == survivors, 'margin subset differs from surviving positions')
    diagnostics = []
    for ident, m in margin.items():
        require(all(m[k] == v for k, v in matched[ident].items()), 'margin provenance row differs')
        candidates = {m['recorded_token'], m['produced_token']}
        d = {'episode': m['episode'], 'turn': m['turn'], 'position': m['position'],
             'reference_gap_ulps_actual_spacing': full[ident]['gap_ulps'],
             'old_assumed_spacing': m['bf16_ulp']}
        for backend in ('mlx', 'torch'):
            require({m[backend+'_top1'], m[backend+'_top2']} == candidates, 'margin candidate pair changed')
            p1, p2 = m[backend+'_p1'], m[backend+'_p2']
            require(0 < p2 <= p1 <= 1, 'invalid margin probabilities')
            gap = math.log(p1 / p2)
            require(math.isclose(gap, m[backend+'_logit_gap'], abs_tol=1e-12), 'logit gap reconstruction differs')
            require(math.isclose(gap / m['bf16_ulp'], m[backend+'_gap_ulps'], abs_tol=1e-12), 'reported assumed-scale division differs')
            sign = 1 if m[backend+'_top1'] == m['recorded_token'] else -1
            d[backend+'_signed_gap'] = sign * gap
            d[backend+'_reported_assumed_ulps'] = m[backend+'_gap_ulps']
            d[backend+'_top1'] = m[backend+'_top1']
        d['torch_device_basis'] = 'laptop CPU, per canonical README; not original CUDA margin'
        diagnostics.append(d)
    return rows, diagnostics, reference


def save_figure(fig, out, stem):
    fig.savefig(out / (stem+'.png'), dpi=150, facecolor='white')
    svg = out / (stem+'.svg')
    fig.savefig(svg, facecolor='white', metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def label(row):
    return row['episode'].removeprefix('agentic-d2-').replace('_', ' ') + f"  · T{row['turn']} P{row['position']}"


def plots(rows, diagnostics, out):
    blue, gold, ink = '#247AA5', '#B77818', '#25323A'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': ink, 'axes.labelcolor': ink, 'svg.hashsalt': '24-precision-matched'})
    ordered = sorted(rows, key=lambda r: -r['recorded_probability'])
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.subplots_adjust(left=.36, right=.76, top=.80, bottom=.13)
    fig.text(.045, .955, 'Precision matching resolves 21 of the 24 original flips', fontsize=20, weight='bold')
    fig.text(.045, .919, 'Every original position, joined by episode + turn + token position', fontsize=12)
    fig.text(.045, .887, 'Original probability: MLX 4-bit  |  Comparison: card CUDA bf16 versus laptop MLX bf16', fontsize=10)
    fig.text(.045, .851, '21 bf16 agreements', color=blue, weight='bold', fontsize=12)
    fig.text(.28, .851, '3 remaining CUDA disagreements', color=gold, weight='bold', fontsize=12)
    ax.set_ylim(23.8, -.8)
    ax.set_xlim(99, 100.025)
    ax.set_yticks(range(len(ordered)), [label(r) for r in ordered])
    ax.set_xticks([99, 99.25, 99.5, 99.75, 100])
    ax.set_xlabel('Probability assigned to the original token by MLX 4-bit (%)', labelpad=14)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.grid(axis='x', color='#DFE5E8', linewidth=.7)
    ax.set_axisbelow(True)
    for spine in ('top', 'left', 'right'):
        ax.spines[spine].set_visible(False)
    for y, r in enumerate(ordered):
        color, marker = (blue, 'o') if r['bf16_cuda_agrees'] else (gold, 's')
        ax.axhline(y, color='#F1F4F5', linewidth=.5, zorder=0)
        ax.scatter(100*r['recorded_probability'], y, c=color, marker=marker, s=38, zorder=3)
        status = 'agree' if r['bf16_cuda_agrees'] else f"remains · reference {r['reference_gap_ulps']:g} ULP"
        ax.text(1.035, y, status, transform=ax.get_yaxis_transform(), va='center', color=color, fontsize=9.5)
    fig.text(.045, .068, 'Two remaining rows fall within the new reference-side 2-ULP rule; read-0108 P521 is at 3 ULP.', fontsize=11, weight='bold')
    fig.text(.045, .038, 'Selected set only, not a full-corpus pass. “Remains” describes token inequality, not a proven port defect.', fontsize=10)
    fig.text(.045, .016, 'Canonical source: WSB-DEVICE-2026-09-10 · original arm df2f0a3 · later reference 00e88b0 · file-only analysis', fontsize=9, color='#61717A')
    save_figure(fig, out, 'precision-matched-24')

    diagnostics = sorted(diagnostics, key=lambda r: r['episode'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.subplots_adjust(left=.23, right=.955, top=.68, bottom=.27, wspace=.34)
    fig.text(.045, .94, 'The remaining three: reference spacing and CPU margins', fontsize=18, weight='bold')
    fig.text(.045, .879, 'Later reference uses actual winning-logit spacing; the earlier margin arm ran torch on the laptop CPU.', fontsize=10.5)
    axes[0].set_title('MLX bf16 reference gap\n00e88b0 · actual-spacing producer', fontsize=11, pad=16)
    axes[1].set_title('Signed logit gap in the earlier margin arm\ndf2f0a3 · MLX and torch laptop CPU', fontsize=11, pad=16)
    axes[0].axvspan(0, 2, color='#EAF2F4')
    axes[0].axvline(2, color='#607C86', linestyle='--', linewidth=1)
    for i, r in enumerate(diagnostics):
        axes[0].scatter(r['reference_gap_ulps_actual_spacing'], i, color=gold, marker='s', s=60)
        axes[0].text(r['reference_gap_ulps_actual_spacing']+.12, i, f"{r['reference_gap_ulps_actual_spacing']:g}", va='center')
        for backend, color, marker, shift in [('mlx', blue, 'o', -.1), ('torch', gold, 's', .1)]:
            v = r[backend+'_signed_gap']
            axes[1].scatter(v, i+shift, color=color, marker=marker, s=55, label=backend if i == 0 else None)
            axes[1].text(v+.06, i+shift, f'{v:+.2f}', va='center', fontsize=9)
    for ax in axes:
        ax.set_ylim(2.6, -.6)
        ax.grid(axis='x', color='#DFE5E8', linewidth=.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_xlim(0, 3.65)
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_yticks(range(3), [label(r) for r in diagnostics])
    axes[0].set_xlabel('Gap in reference ULPs\nShaded: rule counts ≤2 as a tie', labelpad=12)
    axes[1].set_xlim(-.75, 1.8)
    axes[1].set_yticks([])
    axes[1].axvline(0, color='#74868F', linewidth=1)
    axes[1].set_xlabel('logit(original token) − logit(original CUDA token)\nPositive favors the original token', labelpad=12)
    axes[1].legend(labels=['MLX bf16', 'torch CPU bf16'], loc='lower right', bbox_to_anchor=(1, 1.29), frameon=False, ncol=2, fontsize=9)
    fig.text(.045, .135, 'The old margin script assumed spacing 0.25. Its 8-ULP label is neither measured spacing nor the new 2-ULP rule.', fontsize=10)
    fig.text(.045, .086, 'Only read-0108 has equal saved torch-CPU top probabilities. Nonzero gaps remain distinct representable values.', fontsize=10)
    fig.text(.045, .035, 'No original-CUDA margins are supplied. These data do not prove that all three discrepancies are rounding artifacts.', fontsize=10, weight='bold')
    save_figure(fig, out, 'remaining-three')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT)
    args = p.parse_args()
    manifest, data = load_sources()
    rows, diagnostics, reference = joined_data(data)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names = ['joined.csv', 'margins.csv', 'analysis.json', 'README.md', 'precision-matched-24.png', 'precision-matched-24.svg', 'remaining-three.png', 'remaining-three.svg']
    require(not any((out/n).exists() for n in names), 'use a fresh output directory')
    remaining = [r for r in rows if not r['bf16_cuda_agrees']]
    result = {
        'scope': 'selected-position instrument comparability; not a full-corpus acceptance or model-behavior finding',
        'counts': {
            'selected': {'value': len(rows), 'basis': 'derived from original card-CUDA flip identities'},
            'bf16_agrees': {'value': len(rows)-len(remaining), 'basis': 'joined card-CUDA and laptop-MLX readings'},
            'bf16_disagrees': {'value': len(remaining), 'basis': 'joined card-CUDA and laptop-MLX readings'},
            'remaining_within_reference_2_ulp': {'value': sum(r['reference_within_2_ulp'] for r in remaining), 'basis': 'derived from later laptop-MLX reference'},
            'remaining_outside_reference_2_ulp': {'value': sum(not r['reference_within_2_ulp'] for r in remaining), 'basis': 'derived from later laptop-MLX reference'},
        },
        'remaining_positions': remaining,
        'source_commits': [manifest['source_commit'], manifest['supplemental_reference_commit']],
        'canonical_record': manifest['source_record'],
        'limits': [
            'The 24 rows were selected by original MLX 4-bit P >= .99 and CUDA disagreement; not a random or exhaustive precision-matched sample.',
            'Matched agreement supports the 4-bit reference as the outlier, conditional on source-declared checkpoint and input identity.',
            'MLX arm and margin reports omit complete checkpoint/input digests and do not seal raw logits. File hashes identify saved evidence, not the omitted execution inputs.',
            'Earlier margins use torch laptop CPU, not the original CUDA run; two CPU top1 choices differ from their original CUDA produced_token.',
            'The original 0.25 ULP denominator was assumed at magnitude 32. Multiples of a chosen spacing cannot establish the true spacing.',
            'The later reference computes spacing from its actual top1; its 1,3,1 ULP values supersede the original MLX assumed-scale 1,6,2 for this comparison.',
            'The later artifact stores gaps and ratios, not the raw top1 logits; spacing provenance rests on the pinned producer.',
            'A rule-defined near tie is not proof of an arithmetic cause. One to several ULPs are not exact ties.',
            'log(p1/p2) reconstructs gaps approximately from saved float probabilities; no claim of exact real arithmetic.',
            'No model loads, GPU operations, run changes, or timing claims occur in this analysis.',
        ],
        'versions': {'matplotlib': matplotlib.__version__},
    }
    write_csv(out/'joined.csv', rows)
    write_csv(out/'margins.csv', diagnostics)
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    plots(rows, diagnostics, out)
    (out/'README.md').write_text('''# Which original flips survive precision matching?

**Twenty-one of the original 24 disappear:** laptop MLX bf16 selects the card CUDA bf16 token, while the old MLX 4-bit token is the outlier. **Three remain** in that original CUDA comparison. A later MLX bf16 reference confirms the same tokens at all 24 positions and measures its gaps using the actual winning-logit spacing.

| remaining position | original 4-bit P | later MLX reference gap | reference-side ≤2 ULP rule |
|---|---:|---:|---|
| list-0149, turn 0, position 522 | 0.99150777 | 1 ULP | within tie band |
| read-0108, turn 0, position 521 | 0.99973875 | 3 ULP | outside tie band |
| update-0028, turn 0, position 556 | 0.99987268 | 1 ULP | within tie band |

These are **two policy-defined ties and one remaining position outside that band**, not a completed acceptance verdict. No new card acceptance run is represented here. In particular, “outside” does not itself prove a port defect: original-CUDA margins are missing, and cross-framework numerical error has not been bounded by these records.

![All 24 positions](precision-matched-24.png)

## What the margin evidence does and does not say

The original three-position margin arm ran torch on the **laptop CPU**, as the canonical README states. Its top1 agrees with MLX at list-0149; at read-0108 the saved torch top probabilities are exactly equal; at update-0028 the frameworks favor opposite candidates by about 0.5 logit. These CPU readings are not CUDA margins.

The producer at df2f0a3 divides every recovered gap by `bf16_ulp(32.0)`, giving 0.25, without retaining the actual logits. Its MLX readings of approximately 1, 6 and 2 ULP therefore use an **assumed scale**. The later producer at 00e88b0 uses `gap / bf16_ulp(best)` and reports **1, 3 and 1** for the same three positions. Same gaps, different spacing at two positions. The later record directly demonstrates why checking that gaps are multiples of 0.25 did not verify the original assumption.

The earlier report also used an eight-ULP cutoff, while the Chief's subsequent rule names **two ULPs on the reference side**. Those conventions cannot be substituted. Finite nonzero gaps spanning several representable values are not exact ties; a chosen near-tie band is an operational tolerance, not proof of harmless rounding. The probability-ratio calculation recovers logit differences approximately from finite-precision probabilities, not exactly in real arithmetic.

![Remaining three positions](remaining-three.png)

## Finding, technique and implementation

**Finding:** the confident-flip rule fired mostly on a change of precision. At 21 selected positions both bf16 readings agree and the 4-bit reference dissents, including batch_update-0166 T1 P735 at recorded P=0.9999998807907104. These data support rebuilding a precision-matched reference. They do not establish that all three residual disagreements are rounding artifacts or certify the port globally.

**Technique:** freeze the original selection; join on episode, turn and absolute token position; require the original tokens and probabilities to match exactly; recompute agreement from token IDs; preserve the device used for every later margin. Measure spacing from the actual stored values, distinguish an exact tie from a chosen tolerance, and apply the stated tolerance to the specified reference. This procedure transfers across frameworks without relying on a particular model or our implementation.

**Implementation:** `analyze.py` rejects changed source and manifest hashes and model imports. It checks complete JSONL and equality with the JSON summaries, unique/full joins, the canonical CUDA recapture's exact 24-position set, all source classifications, full-reference totals and all probability-ratio reductions. Producer labels such as `port` are retained as source fields but are not adopted as causal verdicts. `joined.csv` has all original probabilities and token IDs; `margins.csv` keeps signed CPU gaps separate from actual-spacing MLX reference gaps.

## Sources, bases and limits

The canonical source is **`research/records/WSB-DEVICE-2026-09-10/`**, not the rental filesystem. The original CUDA summaries came from `/workspace/wsb/out/`; the preserved canonical record survives the rental. `SOURCES.json` pins each file to df2f0a3 (original selection, matched arm and margin arm) or 00e88b0 (the later full MLX reference and its producer) and records SHA-256. Sources are copied directly from those Git objects. This new directory supplements the approved `GPU-COMPARABILITY-2026-09-09-0955Z` record, which remains unchanged: position IDs absent from that earlier snapshot are present in the later canonical evidence.

Original confidence is **laptop MLX 4-bit**; original produced tokens are **card CUDA bf16**; matched outcomes and later reference gaps are **laptop MLX bf16**; the auxiliary torch margins are **laptop CPU bf16**. Derived comparisons combine those explicitly identified readings, never their timing or memory. Reports declare checkpoint paths but the new MLX reports omit complete checkpoint/input digests and raw logits. Hashes seal the provided files, not execution metadata they did not retain.

The 24 were selected because the old reference was confident and CUDA disagreed. They cannot estimate whole-corpus precision-matched agreement or prove all confident positions are safe. The full reference is read for identity and gap checks, not treated as a new CUDA result. The original gate's failure remains part of the record; rebasing and a later acceptance verdict belong to SWE-1 and the Chief.

## Reproduction

Run `analyze.py --output-dir <fresh-directory>` with Matplotlib and the standard library. It reads its own immutable sources, refuses existing outputs, and needs no repository imports or model libraries. Plot environment: Matplotlib 3.11.1, installed separately from the shared research environment. The script writes this README, both figures (PNG/SVG), CSVs and analysis JSON. `verify.py` tests the joins with corrupted copies and compares a fresh reproduction byte for byte.
''')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
