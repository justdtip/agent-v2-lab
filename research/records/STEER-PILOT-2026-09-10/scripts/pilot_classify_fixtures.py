"""Model-free fixtures for steer_pilot.py's parse() and classify() (Codex 59e6b11 F1, F3, F4), executed on the
script's own source segments. Run from the scripts directory: python3 pilot_classify_fixtures.py"""
import re, json
from pathlib import Path
src = Path(__file__).with_name("steer_pilot.py").read_text()
seg = src[src.index("def parse(text"): src.index("results_fh = ")]
ns = {"re": re, "json": json, "name_pat": re.compile(r'"name":\s*"([a-z_]+)"'), "path_pat": re.compile(r'"path":\s*"([^"]+)"')}
exec(seg, ns); parse, classify = ns["parse"], ns["classify"]
rec = {"tool": "read_file", "path": "lab/test/0011/metrics/metric-1.txt"}
calc_donor = {"tool": "calculate", "path": None}; opp_donor = {"tool": "read_file", "path": "lab/test/0023/metrics/metric-1.txt"}
pend = {"tool": "read_file", "path": "lab/test/0011/metrics/metric-2.txt"}
base_p = parse('read_file", "arguments": {"path": "lab/test/0011/metrics/metric-1.txt"}}\n```'); assert base_p["valid_json"]
cases = []
c = classify(base_p, rec, calc_donor, base_p); cases.append(("F1 pathless donor, complete unchanged read call: no crash, path outcomes inapplicable, tool unchanged", c["directory_switched_to_donor"] is None and c["path_state"] == "unchanged" and c["tool_state"] == "unchanged" and not c["switched_to_donor_tool"]))
c = classify(parse('calculate", "arguments": {"expression": "10 + 10"}}\n```'), rec, calc_donor, base_p); cases.append(("operation switch: tool switched_to_donor", c["tool_state"] == "switched_to_donor" and c["switched_to_donor_tool"]))
donor_call = parse('read_file", "arguments": {"path": "lab/test/0023/metrics/metric-1.txt"}}\n```')
c = classify(donor_call, rec, opp_donor, donor_call); cases.append(("F3 baseline already at the donor path: unchanged, donor agreement descriptive only", c["path_state"] == "unchanged" and not c["switched_to_donor_path"] and c["path_class"] == "donor"))
c = classify(donor_call, rec, opp_donor, base_p); cases.append(("real directory switch", c["switched_to_donor_path"] and c["directory_switched_to_donor"] and not c["file_switched_same_directory"]))
trunc = parse('read_file", "arguments": {"path": "lab/test/0023/metrics/metric-1.txt"'); cases.append(("F4 truncated call carries the donor path but is not valid", (not trunc["valid_json"]) and trunc["path"] == opp_donor["path"]))
c = classify(trunc, rec, opp_donor, base_p); cases.append(("F4 truncated arm: not jointly valid, never a switch", c["jointly_valid"] is False and c["path_state"] is None and not c["switched_to_donor_path"]))
c = classify(parse('read_file", "arguments": {"path": "lab/test/0011/metrics/metric-2.txt"}}\n```'), rec, pend, base_p); cases.append(("pending-file switch: file switched, directory kept", c["switched_to_donor_path"] and c["file_switched_same_directory"] and not c["directory_switched_to_donor"]))
c = classify(donor_call, rec, opp_donor, parse('read_file", "arguments": {"path": "lab/test/0011/metrics/metric-1.txt"')); cases.append(("invalid baseline: never a switch", c["jointly_valid"] is False and not c["switched_to_donor_path"]))
for name, ok in cases: print(("PASS " if ok else "FAIL ") + name)
assert all(ok for _, ok in cases), "a classifier fixture failed"
print(json.dumps({"event": "classify_fixtures", "cases": len(cases), "all_pass": True}))
