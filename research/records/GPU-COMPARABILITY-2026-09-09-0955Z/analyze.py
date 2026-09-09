"""Reproduce a CPU/GPU comparability figure from frozen reports; no model imports."""
from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import json
import math
import os
from pathlib import Path

_original_import = builtins.__import__


def _read_only_analysis_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'transformers'}:
        raise ImportError('model libraries are forbidden in this file-only analysis')
    return _original_import(name, *args, **kwargs)


builtins.__import__ = _read_only_analysis_import
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-gpu-analysis-mpl')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from research.acceptance.provenance import MEASURED_HERE, Measured  # noqa: E402

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'cuda-all15.json': 'a29d3b1dc919fa3d9508f628e154e51fa453469925bdcda42e3c2d7a9d167b42',
    'cpu-all15.json': '1ea5f8db9c317abca5042f61705e3b40ae0c997a97da8cd79a9bf2f2bf7a8225',
    'cuda-all15.jsonl': 'a024bc94e3c0b498662552a83505c7d924d1e024340b125f07a052e0ac9eb4af',
    'cpu-all15.jsonl': '4f95913c11c4d3dcc2990627df1eb4b06a6b05d7c0e3dd4014565a415e7a39ce',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def value(row, key, expected_basis='measured-here'):
    entry = row[key]
    require(entry['basis'] == expected_basis, f'{key}: unexpected basis')
    return entry['value']


def load_reports():
    reports = {}
    for name, digest in EXPECTED.items():
        raw = (ROOT / 'source' / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, f'source hash mismatch: {name}')
    for mode in ('cuda', 'cpu'):
        report = json.loads((ROOT / 'source' / f'{mode}-all15.json').read_text())
        raw = (ROOT / 'source' / f'{mode}-all15.jsonl').read_bytes()
        require(raw.endswith(b'\n'), f'{mode}: partial JSONL line')
        rows = [json.loads(line) for line in raw.splitlines()]
        require(rows == report['episodes'], f'{mode}: summary/JSONL disagree')
        require(len(rows) == len({r['label'] for r in rows}) == 15, 'episode count or duplicate')
        for key in ('compared', 'agreed', 'hard_flips'):
            require(sum(value(r, key) for r in rows) == value(report, key), f'{mode}: {key} sum')
        for row in rows:
            n, agreed, hard = (value(row, k) for k in ('compared', 'agreed', 'hard_flips'))
            require(all(type(x) is int for x in (n, agreed, hard)), 'non-integer count')
            require(0 <= hard <= n - agreed <= n and n > 0, 'invalid count range')
            confidences = value(row, 'flip_confidences', 'laptop-basis')
            require(all(math.isfinite(p) and 0 <= p <= 1 for p in confidences), 'invalid probability')
            require(len(confidences) == n - agreed, 'missing flip confidences')
            require(sum(p >= .99 for p in confidences) == hard, 'hard-flip reduction differs')
            confident = value(row, 'confident_positions', 'laptop-basis')
            require(0 <= hard <= confident <= n, 'confidence denominator range')
        require(not report['head']['smoke'], 'fixture/smoke result cannot enter this comparison')
        reports[mode] = report
    cpu, gpu = (reports[m]['head'] for m in ('cpu', 'cuda'))
    for key in ('source_commit', 'tree_hash', 'checkpoint_sha256', 'records', 'records_precision'):
        require(cpu[key] == gpu[key], f'comparison identity mismatch: {key}')
    require(cpu['records_precision'] == 'mlx 4-bit', 'reference precision changed')
    for mode in ('cpu', 'cuda'):
        head = reports[mode]['head']
        device = json.loads(head['device'])
        require(device['device'] == ('cuda:0' if mode == 'cuda' else 'cpu'), 'device mismatch')
        require(device['determinism'] == 'pinned', 'determinism not pinned')
        require(head['port_precision'].endswith(' bfloat16'), 'port precision mismatch')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    outputs = ['comparison.png', 'comparison.svg', 'metrics.csv', 'analysis.json', 'README.md']
    require(not any((out / name).exists() for name in outputs), 'use a new output directory; records are append-only')
    reports = load_reports()
    indexed = {m: {r['label']: r for r in d['episodes']} for m, d in reports.items()}
    require(indexed['cpu'].keys() == indexed['cuda'].keys(), 'episode join mismatch')
    rows = []
    for label in indexed['cuda']:
        gpu, cpu = (indexed[m][label] for m in ('cuda', 'cpu'))
        require(value(gpu, 'compared') == value(cpu, 'compared'), 'compared denominator mismatch')
        require(value(gpu, 'confident_positions', 'laptop-basis') == value(cpu, 'confident_positions', 'laptop-basis'), 'reference population mismatch')
        for mode, row in (('cuda', gpu), ('cpu', cpu)):
            n, agreed = value(row, 'compared'), value(row, 'agreed')
            rows.append({
                'episode': label, 'port_device': mode, 'reference_precision': 'MLX 4-bit',
                'port_precision': 'bf16', 'compared': n, 'agreed': agreed,
                'disagreed': n - agreed, 'disagreement_percent': 100 * (n - agreed) / n,
                'reference_confident_positions': value(row, 'confident_positions', 'laptop-basis'),
                'reference_confidence_gated_flips': value(row, 'hard_flips'),
                'measurement_basis': 'device report; measured-here',
                'confidence_basis': 'laptop-basis; recorded MLX probability',
                'source_file': f'{mode}-all15.json',
            })
    with (out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    totals = {}
    for mode, report in reports.items():
        n, agreed = value(report, 'compared'), value(report, 'agreed')
        note = f'Derived locally from the card {mode} report; shared MLX reference, no new model run'
        raw = {'compared': n, 'agreed': agreed, 'disagreed': n - agreed,
               'agreement_percent': 100 * agreed / n,
               'reference_confidence_gated_flips': value(report, 'hard_flips')}
        totals[mode] = {k: Measured(v, MEASURED_HERE, note).as_dict() for k, v in raw.items()}
    changed = [label for label in indexed['cpu'] if value(indexed['cpu'][label], 'hard_flips') != value(indexed['cuda'][label], 'hard_flips')]
    analysis = {
        'scope': 'instrument comparability; not a model-behavior finding',
        'source_commit': reports['cpu']['head']['source_commit'],
        'source_checkpoint_sha256': reports['cpu']['head']['checkpoint_sha256'],
        'source_sha256': EXPECTED, 'totals': totals,
        'episodes_with_changed_gated_counts': changed,
        'checks': ['transfer hashes', 'complete JSONL', 'JSON/JSONL equality', 'unique episode join', 'totals reconcile', 'confidence counts reproduce', 'matching tree/checkpoint/denominators', 'declared devices and bf16 precision'],
        'limits': [
            'hard_flips uses MLX reference P >= .99, not both machines confidence',
            'per-position flip identities and port probabilities are absent',
            'one net agreement difference is not one CPU/GPU token difference',
            'input_digest is none; matched paths/counts do not independently hash the replay inputs',
            'MLX 4-bit and torch bf16 remain different precision paths',
            'positions are repeated/correlated observations, not independent samples',
            'no performance or memory inference from these potentially shared-card runs',
        ],
        'inference': 'A CPU run retains the aggregate gated mismatch count, weakening an exclusively GPU-specific account; these summaries do not isolate the port from precision and reference differences.',
        'versions': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
    }
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
    make_plot(rows, indexed, out)
    readme = f'''# CPU/GPU comparability snapshot

The card-CPU rerun retains **24 reference-confidence-gated mismatches**, the same
aggregate as the GPU run. Two episode counts change: `list-0149` is 2 → 1 and
`read-0108` is 2 → 3 (GPU → CPU). Equal totals do not mean identical flipped positions.

GPU agrees on **4,999/5,245** recorded positions ({totals['cuda']['agreement_percent']['value']:.3f}%);
CPU agrees on **4,998/5,245** ({totals['cpu']['agreement_percent']['value']:.3f}%). This is one
net agreement difference against a common reference, not a direct one-token
CPU-versus-GPU difference. The replay is teacher-forced; these are not two free-running
trajectories. No independent-sample count, interval or significance claim is made.

## Definition and limits

The producer defines a hard flip using the **MLX reference's probability ≥0.99**.
It does not test joint confidence. Neither the summaries nor their per-episode
JSONL copies store port probabilities or per-position flip identities. Therefore
this record cannot substantiate “both machines were confident” or identify which
specific positions persist. The reference is **MLX 4-bit**, while both card runs
are **torch bf16**. The CPU result weakens a GPU-only account but does not isolate
the port from precision or reference-path differences. This is instrument evidence,
not a model-behavior finding.

## Technique and data quality

Join episodes by their full label, reject duplicates, match denominators and
checkpoint/source identities, reconcile aggregate counts, and reconstruct gated
counts from the saved reference probabilities before plotting. Count differences
per episode before interpreting aggregate equality. Thirty condition-episode rows
come from fifteen distinct episodes; the two conditions are never pooled.

The four downloaded files match the SHA-256 values read on the server. Each JSONL
has a complete final line and exactly reproduces its JSON report's episode rows.
Both source heads name commit `{analysis['source_commit']}` and the same checkpoint
digest. `input_digest` is `none`; input byte identity is not independently sealed
by these heads, despite matching paths and per-episode counts. No input rows were
regenerated, and no source records or remote jobs were changed.

## Provenance and reproduction

Sources: `/workspace/wsb/out/{{cuda,cpu}}-all15.{{json,jsonl}}`, copied read-only on
2026-09-09. `source/` holds their bytes and the producer definitions from the named
commit; `analysis.json` holds hashes, bases, checks and limitations. Wall-clock and
memory values are deliberately not plotted. Chart design is in `CHART-CONTRACT.md`.

Run `analyze.py --output-dir <new-directory>` with NumPy and Matplotlib, with the
repository root and `src` on PYTHONPATH. It refuses changed source hashes and
existing output files. No model library may be imported. Tested versions:
Matplotlib {matplotlib.__version__}, NumPy {np.__version__}. A separate manifest
records the WS-D native-capture check and its declared one-row budget departure;
that in-flight comparison is not included in this chart.

![CPU/GPU comparison](comparison.png)
'''
    (out / 'README.md').write_text(readme)
    print(json.dumps({'output': str(out), 'totals': totals, 'changed_episodes': changed}))


def make_plot(rows, indexed, out):
    blue, gold, ink, gray = '#2877B5', '#C28B25', '#242B34', '#DDE2E6'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': ink, 'axes.labelcolor': ink,
                         'xtick.color': ink, 'ytick.color': ink,
                         'svg.hashsalt': 'cpu-gpu-comparability-20260909'})
    labels = sorted(indexed['cuda'], key=lambda k: (-max(value(indexed[m][k], 'hard_flips') for m in ('cpu','cuda')), k))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 9.3), sharey=True,
                             gridspec_kw={'width_ratios': [1.15, 1]})
    fig.subplots_adjust(left=.275, right=.955, top=.79, bottom=.18, wspace=.16)
    fig.text(.055, .944, 'CPU and GPU agreement with the saved MLX reference', fontsize=19, weight='bold')
    fig.text(.055, .905, 'Teacher-forced comparison • 15 episodes • 5,245 positions per condition', fontsize=12)
    fig.text(.055, .874, 'Reference: MLX 4-bit   |   Card runs: torch bf16   |   Instrument comparability, not model behavior', fontsize=10, color='#59636D')
    fig.text(.055, .826, 'Overall agreement: GPU 4,999 / 5,245     CPU 4,998 / 5,245', fontsize=11, weight='bold')
    for mode, color, marker, shift in [('cuda', blue, 'o', -.12), ('cpu', gold, 's', .12)]:
        chosen = {r['episode']: r for r in rows if r['port_device'] == mode}
        ys = np.arange(len(labels)) + shift
        for ax, metric in zip(axes, ['disagreement_percent', 'reference_confidence_gated_flips'], strict=True):
            ax.scatter([chosen[k][metric] for k in labels], ys, c=color, marker=marker,
                       s=38, linewidths=.7, edgecolors=ink, label='GPU' if mode=='cuda' else 'Card CPU', zorder=3)
    short = [k.replace('agentic-d2-', '').replace('chat-', 'chat: ').replace('_',' ') for k in labels]
    axes[0].set_yticks(range(len(labels)), short)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Disagreement with reference (%)', labelpad=11)
    axes[1].set_xlabel('Mismatches at reference P ≥ 0.99', labelpad=11)
    axes[0].set_xlim(-.08, max(r['disagreement_percent'] for r in rows)*1.13)
    axes[1].set_xlim(-.12, 4.55)
    axes[1].set_xticks(range(5))
    for ax in axes:
        ax.grid(axis='x', color=gray, linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#AEB7BF')
        for y in range(len(labels)):
            ax.axhline(y, color='#F1F3F5', linewidth=.6, zorder=0)
    axes[1].legend(loc='lower right', bbox_to_anchor=(1,1.035), frameon=False, ncol=2)
    fig.text(.055, .10, 'The CPU run retains 24 gated mismatches, but the list/read episode counts change.', fontsize=11, weight='bold')
    fig.text(.055, .067, '“Gated” refers only to the saved MLX probability. Port confidences and flip-position identities are absent.', fontsize=9, color='#59636D')
    fig.text(.055, .038, 'Source: card reports at a5a977b • Read-only snapshot, 9 Sep 2026 • No new model run; no timing inference', fontsize=9, color='#59636D')
    fig.savefig(out / 'comparison.png', dpi=140, facecolor='white')
    fig.savefig(out / 'comparison.svg', facecolor='white', metadata={'Date': None})
    svg = out / 'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
