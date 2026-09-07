"""Reconstruct sampled observational scores from retained JSONL rows, with no model."""
from pathlib import Path
import json
from local_llm_lab.pipeline.live_lens.session import read_record
root=Path(__file__).parent
errors=[]
for row in read_record(root/'short-capture-1.jsonl'):
    if row['kind']!='head' or 'audit' not in row:
        continue
    audit=row['audit']; tokens=set(row['written_top'])
    score=sum(weight*(len(tokens.intersection(source))/len(tokens))
              for pos,source,weight in zip(audit['positions'],audit['source_top'],audit['attention'],strict=True)
              if pos<row['position'])
    errors.append(abs(score-row['score']))
assert errors and max(errors)==0., errors
print(json.dumps({'audited_rows':len(errors),'max_abs_error':max(errors)}))
