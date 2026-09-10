"""Execute only the producer's mask-building AST with a set-backed array; no ML imports."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


class Mask:
    def __init__(self, shape):
        assert shape[:2] == (1, 1)
        self.size = shape[-1]
        self.cells = set()

    def __setitem__(self, key, value):
        assert key[:2] == (0, 0) and value is True
        queries = range(*key[2].indices(self.size))
        self.cells.update((q, k) for q in queries for k in key[3])

    def clone(self):
        copy = Mask((1, 1, self.size, self.size))
        copy.cells = set(self.cells)
        return copy


def main():
    provenance = json.loads((ROOT / 'sources.json').read_text())
    for row in provenance['files']:
        assert hashlib.sha256((ROOT / row['file']).read_bytes()).hexdigest() == row['sha256']
    tree = ast.parse((ROOT / 'source/workspace_w3b.py').read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
    start = next(i for i, n in enumerate(outer.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'note_keys')
    nodes = outer.body[start:start + 2]
    assert isinstance(nodes[1], ast.If)
    code = compile(ast.Module(body=nodes, type_ignores=[]), '<producer note-mask AST>', 'exec')
    outcomes = []
    for prompt in [2, 47, 100, 1500]:
        for tool_index in [0, 1, 5, 30]:
            size = prompt + tool_index + 1
            kinds = ['format'] * prompt + ['note'] * (tool_index + 1)
            carrier = Mask((1, 1, size, size))
            carrier[0, 0, slice(prompt - 1, None), [0]] = True
            env = {'kinds': kinds, 'S': size, 'arms': {'all_carriers': carrier},
                   'torch': SimpleNamespace(zeros=lambda *s, dtype: Mask(s), bool=bool)}
            exec(code, env)
            assert env['note_keys'][-1] + 1 == size
            assert not env['arms']['current_note'].cells
            assert env['arms']['all_carriers_and_current_note'].cells == carrier.cells
            outcomes.append({'prompt_tokens': prompt, 'tool_index': tool_index,
                             'sequence_length': size, 'P_act': size - 2,
                             'current_note_masked_edges': 0, 'combined_equals_carrier': True})
    # Deliberately change the tagging boundary in the fixture: a real suffix remains.
    # The same producer mask AST must now block edges at the measured action position.
    env = {'kinds': ['format'] * 4 + ['note'] * 2 + ['format'] * 4, 'S': 10,
           'arms': {}, 'torch': SimpleNamespace(zeros=lambda *s, dtype: Mask(s), bool=bool)}
    exec(code, env)
    assert (8, 4) in env['arms']['current_note'].cells
    result = {'scope': 'Python slice semantics and producer AST, not torch or a model run',
              'source_hashes_verified': len(provenance['files']), 'cases': outcomes,
              'negative_control': 'nonempty note boundary creates blocked edges at P_act; passes',
              'conclusion': 'current_note is empty; combined arm equals all_carriers for the shipped tagging rule'}
    (ROOT / 'check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cases': len(outcomes), 'negative_control': 'passed', 'model_imports': 0}))


if __name__ == '__main__':
    main()
