"""Locate specific surviving claims and verify the corrected numerical table. No model imports."""
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# This is a pinned passage inventory, not a general semantic document linter.
PASSAGES = {
    'concluding_cross_precision_claim': 'is benign at layer 33, where the amplification is 1.4×, and is not a small correction at layer 1,\nwhere it is 539×.',
    'same_partial_input_used_in_explanation': 'A map whose derivative changes by 100% under a 0.19% move of its input',
    'common_step_refutation_repeated': 'every layer is refuted — the two minima are a factor of 64 apart.',
    'sampled_depth_scope_in_conclusion': 'Coherent float32 is width-stable at 12B scale at every depth, and a float32 12B exact fit at',
}
CORRECTIONS = {
    'repeat_unexecuted': 'The repeat gate is **not**:',
    'metric_relabelled': 'the **relative Frobenius error of the whole map**',
    'amplification_withdrawn': '**withdrawn**, correctly, on Codex\'s third audit.',
    'one_draw': '**On the random control:** one draw is one draw.',
}


def locate(text, needle):
    starts = [m.start() for m in re.finditer(re.escape(needle), text)]
    return [{'line': text.count('\n', 0, start) + 1, 'text': needle} for start in starts]


def inventory(text):
    return {key: locate(text, phrase) for key, phrase in PASSAGES.items()}


def check(root):
    manifest = json.loads((root / 'sources.json').read_text())
    for item in manifest['sources']:
        raw = (root / item['snapshot']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256'] or len(raw) != item['bytes']:
            raise ValueError('source changed: ' + item['snapshot'])
    producer = (root / 'producer.md').read_text()
    data = [json.loads(line) for line in (root / 'displacement.jsonl').read_text().splitlines()]
    assert len(data) == 108
    # The newly added table's own tokens define their reporting precision.
    table = producer.split('Over the eighteen\ndirection-and-cotangent pairs:', 1)[1].split('So **', 1)[0]
    verified = []
    for line in table.splitlines():
        if not line.startswith(('| native bf16 |', '| float32 |')):
            continue
        cells = [s.strip().replace('**', '') for s in line.strip('|').split('|')]
        label, layer, *reported = cells
        phase = 'native' if label == 'native bf16' else 'float32'
        part = [r for r in data if r['phase'] == phase and r['repo_layer'] == int(layer)]
        assert len(part) == 18
        values = [abs(r['a_plus_delta'] - r['a_base']) / abs(r['a_base']) for r in part]
        computed = [min(values), statistics.median(values), max(values)]
        for token, value in zip(reported, computed):
            decimals = len(token.split('.')[1]) if '.' in token else 0
            assert f'{value:.{decimals}f}' == token, (label, layer, token, value)
        verified.append({'precision': phase, 'repo_layer': int(layer), 'computed': computed,
                         'reported': reported, 'rounded_values_match': True})
    assert len(verified) == 6
    return {'schema_version': 1, 'basis': 'file-only comparison of accepted rulings and corrected prose',
            'producer_commit': manifest['producer_commit'], 'order_commit': manifest['order_commit'],
            'sources_verified': len(manifest['sources']), 'corrected_table': verified,
            'corrections_present': {key: locate(producer, text) for key, text in CORRECTIONS.items()},
            'surviving_passages': inventory(producer),
            'new_experiments': False, 'prior_scientific_findings_changed': False,
            'preregistration_P1_P6_review': 'still pending producer amendment'}


if __name__ == '__main__':
    print(json.dumps(check(ROOT), indent=2, sort_keys=True) + '\n', end='')
