"""Recompute raw-array comparisons and native block accounting without MLX."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'position-step-01'
report = json.loads((RUN / 'diagnostic.json').read_text())
assert report['status'] != 'incomplete'
assert report['coefficient_grid'] == [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
assert report['row'] == 0 and report['position'] == 438
assert report['provenance']['concurrent_authorized'] is True
measurements = report['measurements']
arrays = {}
for m in measurements:
    path = RUN / m['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == m['file_sha256']
    arrays[m['file']] = np.load(path, allow_pickle=False)
    assert arrays[m['file']].shape == (16, 2560)
assert len(arrays) == len(measurements)

def matching(layer, coefficient, mode):
    return [arrays[m['file']] for m in measurements
            if (m['layer'], m['coefficient'], m['mode']) == (layer, coefficient, mode)]

recomputed = []
for point in report['stability']:
    a = matching(point['layer'], point['coefficient'], 'reference')[0].astype(np.float64)
    b = matching(point['layer'], point['coefficient']/2, 'reference')[0].astype(np.float64)
    passed = np.isclose(b, a, atol=.003, rtol=.03) & np.isfinite(a) & np.isfinite(b)
    failed = int((~passed).sum())
    maximum = float(np.max(np.abs(a-b)))
    assert failed == point['failed_coordinates']
    assert maximum == point['max_error']
    recomputed.append(dict(layer=point['layer'], coefficient=point['coefficient'],
                           failed_coordinates=failed, max_error=maximum))
assert len(recomputed) == 18
eligible = {}
for layer in (1, 16, 31):
    points = [p for p in recomputed if p['layer'] == layer]
    eligible[str(layer)] = [b['coefficient'] for a, b in zip(points, points[1:], strict=False)
                           if a['failed_coordinates'] == b['failed_coordinates'] == 0]
assert eligible == report['eligible']
common = set.intersection(*(set(v) for v in eligible.values()))
assert (max(common) if common else None) == report['selected_common_coefficient']
head, pending = '0'*64, {}
started_calls = returned_calls = materialized = 0
for line in (RUN/'block-ledger.jsonl').open():
    event = json.loads(line)
    claimed = event.pop('sha256')
    assert event['previous_sha256'] == head
    head = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert head == claimed
    if event['event'] == 'block_started':
        started_calls += 1
    elif event['event'] == 'block_returned':
        returned_calls += 1
        assert event['block_tokens'] == event['batch']*event['sequence']
        pending[event['call']] = event['block_tokens']
    elif event['event'] == 'materialized_workload':
        materialized += sum(pending.pop(n) for n in event['calls'])
assert not pending
assert head == report['block_accounting']['ledger_sha256']
assert materialized == report['block_accounting']['materialized_block_tokens']
# Each prepared layer: L full-sequence blocks, 32 prefix blocks, L singleton blocks.
# Each reference response: +/- for 16 directions through 32-L blocks at all 439 positions.
expected_calls = expected_tokens = 0
for primal in report['primals']:
    layer = primal['layer']
    expected_calls += 2*layer+32
    expected_tokens += layer*439+32*438+layer
for m in measurements:
    tail = 32-m['layer']
    if m['mode'] == 'reference':
        expected_calls += 32*tail
        expected_tokens += 32*tail*439
    else:
        expected_calls += (32 if m['mode'] == 'restore' else 4)*tail
        expected_tokens += 32*tail
assert started_calls == returned_calls == expected_calls
assert materialized == expected_tokens
out = dict(status='passed', run_status=report['status'], measurements=len(arrays),
           stability_comparisons=recomputed, eligible=eligible,
           block_calls=expected_calls, materialized_block_tokens=expected_tokens,
           ledger_sha256=head, pending_block_calls=0,
           scope='All raw response hashes, stability comparisons, plateau selection and native ledger; no model import')
with (ROOT/'POSITION-STEP-01-AUDIT.json').open('x') as stream:
    json.dump(out, stream, indent=2)
    stream.write('\n')
print(json.dumps(out, indent=2))
