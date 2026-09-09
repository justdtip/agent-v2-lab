"""Reproduce small-model evidence and a metadata-only full-shape check; never load weights."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import runpy
from unittest.mock import patch

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from local_llm_lab.arch_torch import TorchArchitectureView
from research.acceptance.torch_seam import decode_readout_probe, structural_report

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).resolve().parent
SNAPSHOT = Path('/Users/daniel.tipton/Desktop/An app/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767')

def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    make_model = runpy.run_path(str(ROOT / 'tests/test_arch_torch.py'))['make_model']
    view = TorchArchitectureView.from_model(make_model())
    observed = []
    for length in (64, 1400):
        ids = torch.arange(length) % view.vocab_size
        normal = view.residual_source_agreement(ids, range(view.num_layers + 1))
        original = view._observe_forward
        def broken(ids, cache=None):
            entry, masks = original(ids, cache)
            for index in masks:
                masks[index] = {**masks[index], 'attention_mask': masks[1]['attention_mask']}
            return entry, masks
        with patch.object(view, '_observe_forward', broken):
            control = view.residual_source_agreement(ids, range(view.num_layers + 1))
        assert max(normal.values()) == 0.0
        assert (max(control.values()) > 0.0) == (length > view.config.sliding_window)
        observed.append({'tokens': length, 'normal_max_abs_by_layer': normal,
                         'broken_mask_max_abs_by_layer': control})
    baseline = decode_readout_probe(view, [1, 5, 9, 3], steps=3)
    tolerance = max(baseline['readout_errors'])
    controls = [decode_readout_probe(view, [1, 5, 9, 3], steps=3, control=control)
                for control in ('skew_after_prefill', 'corrupt_residual')]
    assert not baseline['identity_failures']
    assert controls[0]['readout_errors'][0] <= tolerance
    assert min(controls[0]['readout_errors'][1:]) > tolerance
    assert min(controls[1]['readout_errors']) > tolerance
    small_structure = structural_report(view)
    del view
    config_data = json.loads((SNAPSHOT / 'config.json').read_text())
    config = Gemma3TextConfig(**config_data['text_config'])
    with torch.device('meta'):
        meta_model = Gemma3ForCausalLM(config)
    full_structure = structural_report(TorchArchitectureView.from_model(meta_model))
    assert (full_structure['num_layers'], full_structure['hidden_size']) == (34, 2560)
    full_structure['device'] = 'meta'
    full_structure['weights_loaded'] = False
    parameters = {'text': 0, 'vision_and_projector': 0}
    headers = {}
    for path in sorted(SNAPSHOT.glob('*.safetensors')):
        with path.open('rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            raw = stream.read(size)
        headers[path.name] = hashlib.sha256(raw).hexdigest()
        for key, tensor in json.loads(raw).items():
            if key == '__metadata__':
                continue
            count = 1
            for dimension in tensor['shape']:
                count *= dimension
            category = 'text' if key.startswith('language_model.') else 'vision_and_projector'
            parameters[category] += count
    report = {
        'source_commit': 'a639cae',
        'scope': 'small randomly initialized real HF model; not full-checkpoint acceptance',
        'device': 'cpu', 'dtype': 'float32', 'attention': 'eager',
        'threads': 1, 'deterministic_algorithms': True,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'jlens', 'numpy', 'pytest')},
        'small_model_structure': small_structure,
        'development_gate_2': observed,
        'development_gates_3_and_4': {'baseline': baseline, 'controls': controls,
            'observed_max_error': tolerance,
            'threshold_scope': 'observed development distribution only; no checkpoint/CUDA bound'},
        'missing_readout': decode_readout_probe(TorchArchitectureView.from_model(make_model()),
                            [1, 5, 9, 3], steps=3, readout=False),
        'full_model_structural_metadata': full_structure,
        'checkpoint_header_audit': {'path': str(SNAPSHOT), 'header_sha256': headers,
            'parameters': parameters,
            'fp32_tensor_gib': {k: value * 4 / 2**30 for k, value in parameters.items()},
            'basis': 'tensor shapes from safetensors headers only; tensor bytes were not read'},
        'full_checkpoint_gates_2_3_4': 'unexecuted', 'cuda': 'unexecuted',
        'estimator_rewrite': 'not started; gated by full-checkpoint gates 1-4',
    }
    (RECORD / 'development-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'small_model_gate2': observed, 'readout_max_error': tolerance,
                      'full_model_meta': full_structure, 'weights_loaded': False}))

if __name__ == '__main__':
    main()
