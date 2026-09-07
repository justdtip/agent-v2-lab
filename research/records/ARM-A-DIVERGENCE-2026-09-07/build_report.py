"""Build the divergence report (HTML) from descriptive.json and, when present, fixed_history_lens.json."""
import html, json, sys
from pathlib import Path

S = Path(__file__).resolve().parent
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/daniel.tipton/Desktop/An app/research/records/ARM-A-DIVERGENCE-2026-09-07")
desc = json.load(open(R / "descriptive.json"))
fh_path = R / "run" / "fixed_history_lens.json"
fh = json.load(open(fh_path)) if fh_path.is_file() else None
forms_f32 = json.load(open(R / "recurrence_forms_f32.json")) if (R / "recurrence_forms_f32.json").is_file() else None
forms_bf16 = json.load(open(R / "recurrence_forms_bf16.json")) if (R / "recurrence_forms_bf16.json").is_file() else None
cache_path = R / "cache" / "cache_split_diagnostic.json"
cache = json.load(open(cache_path)) if cache_path.is_file() else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else S / "report.html"
E = html.escape

def num(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

def tok(s):
    return f'<code class="tk">{E(s)}</code>'

CSS = """
:root{--bg:#f4f6f8;--surface:#ffffff;--ink:#1b2127;--muted:#5b6670;--rule:#d5dbe1;--base:#0e6b70;--adapter:#b04e1a;--base-soft:#e3f1f1;--adapter-soft:#f8e8dd;--code:#eef1f4;--good:#2f7d32;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121619;--surface:#1a2026;--ink:#e6eaee;--muted:#98a3ad;--rule:#2a323a;--base:#4fb3b8;--adapter:#e58a52;--base-soft:#15302f;--adapter-soft:#3a261a;--code:#222a31;--good:#7fc984;--bad:#f28b82}}
:root[data-theme="dark"]{--bg:#121619;--surface:#1a2026;--ink:#e6eaee;--muted:#98a3ad;--rule:#2a323a;--base:#4fb3b8;--adapter:#e58a52;--base-soft:#15302f;--adapter-soft:#3a261a;--code:#222a31;--good:#7fc984;--bad:#f28b82}
body{background:var(--bg);color:var(--ink);font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:17px;line-height:1.55;margin:0}
main{max-width:76ch;margin:0 auto;padding:3rem 1.25rem 5rem}
h1{font-size:2.1rem;line-height:1.15;margin:0 0 .5rem;text-wrap:balance;font-weight:600}
h2{font-size:1.35rem;margin:2.6rem 0 .8rem;font-weight:600;text-wrap:balance}
h3{font-size:1.05rem;margin:1.8rem 0 .5rem;font-weight:600}
p{margin:0 0 1rem}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:.6rem}
.meta{color:var(--muted);font-size:.95rem;margin-bottom:2rem}
.abstract{border-left:3px solid var(--rule);padding:.2rem 0 .2rem 1.1rem;margin:1.5rem 0 2rem;font-size:1.02rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums;margin:.6rem 0 1.2rem}
th,td{border-bottom:1px solid var(--rule);padding:.4rem .55rem;text-align:left;vertical-align:top}
th{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:500}
td.n,th.n{text-align:right}
.wrap{overflow-x:auto}
.tk,code{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.84em;background:var(--code);padding:.05em .3em;border-radius:3px;white-space:pre-wrap}
.pass{color:var(--good);font-weight:600}.fail{color:var(--bad);font-weight:600}
.base{color:var(--base);font-weight:600}.adapter{color:var(--adapter);font-weight:600}
.pair{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:1rem 1.1rem;margin:1.2rem 0}
.pair h3{margin-top:0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media (max-width:640px){.grid{grid-template-columns:1fr}}
.col{padding:.6rem .8rem;border-radius:4px}
.col.b{background:var(--base-soft)}.col.a{background:var(--adapter-soft)}
.col h4{margin:0 0 .4rem;font-size:.95rem}
.small{font-size:.86rem;color:var(--muted)}
.note{background:var(--surface);border:1px solid var(--rule);padding:.8rem 1rem;border-radius:4px;font-size:.95rem}
ol,ul{padding-left:1.3rem}li{margin-bottom:.35rem}
figure{margin:1rem 0}figcaption{font-size:.86rem;color:var(--muted)}
"""

parts = []
add = parts.append
add(f'<title>Arm A Divergence Report</title><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap"><style>{CSS}</style>')
add('<main>')
add('<div class="eyebrow">Unified run · arm A · 2026-09-07</div>')
add('<h1>Where the 800-row adapter departs from the base model, and why</h1>')
add(f'<div class="meta">Chief AI Research Scientist, for the Research Director. Evaluation stopped at task {desc["n_tasks"]} of 180 on the test split; the fixed-history comparison {"is included" if fh else "is pending the model"}.</div>')

n, ap, bp = desc["n_tasks"], desc["adapter_passes"], desc["base_passes"]
add('<div class="abstract"><p><strong>Abstract.</strong> The unified run\'s deliverable checkpoint (a LoRA adapter after 800 training rows, validation loss 0.092) was evaluated greedily on the held-out test tasks beside the untrained base model until the evaluation was stopped to free the machine. '
    f'On the {n} tasks both completed, the adapter passed {ap} and the base {bp}. Every adapter failure but two ended in a verbatim repeat of the previous turn until the step budget ran out. '
    'Reading the first wrong step of each failure gives six defect classes; the three commonest (a call re-issued after a successful observation, a malformed argument, an abbreviated path) recur on both test instances of their task family and occur nowhere in the training rows. '
    'The adapter learned the note templates of the training data and lost the construction of the call that follows them. '
    + ('A fixed-history comparison gives both models the adapter\'s own recorded prefix at its first departure from the base\'s trajectory: the base continues correctly in all eleven pairs and the adapter reproduces its wrong call in all eleven, assigning it near-certainty; the wrong arguments are not near-ties. The same turns scored under the chunkwise recurrence the adapter was trained with keep every greedy token, so a train/eval mismatch is ruled out. At the decision positions the layer-20 lens reads the JSON format for both models; the adapter\'s residual there has about twice the base\'s norm.' if fh else 'A fixed-history comparison on the same recorded prefixes, reading the residual stream through the hosted Jacobian lens at the point of departure, will be added when the machine is free.')
    + '</p></div>')

add('<h2>1. Background</h2>')
add('<p>The overnight order was to compose the evidence into one training run, obtain the best model under the coherence standard, and compare across runs. Arm A trained a rank-16 LoRA on all 32 blocks (attention, gated-delta and MLP projections) of Qwen3.5-4B on 6,648 rows capped at 2,688 tokens. After two memory failures the run was resumed from its 400-iteration checkpoint and died again at 780; the Director ruled that the resumed run\'s 400-iteration checkpoint, 800 rows or 200 optimizer updates, is the deliverable and is not to be restarted.</p>')
add('<p>The capability standard is coherence to completion: the fraction of tasks finished without an integrity, loop, invalid-action or tool-error event, with the four causes treated as competing risks. The base model had already been measured on the full test split: 123 of 180 passed, 115 of 180 coherent to completion, and integrity events preceded every loop.</p>')

add('<h2>2. Methods</h2>')
add('<h3>2.1 Evaluation</h3>')
add('<p>The adapter was evaluated by the pipeline\'s evaluation stage: greedy decoding (temperature 0), a 24-step budget, 200 generated tokens per turn, and the two-observation window of the training data. The task list and order are the base evaluation\'s, so every adapter task has a paired base trajectory. The run was stopped after 19 tasks at the Director\'s instruction; the 19 are the first 19 of the split in its recorded order, not a sample chosen by outcome.</p>')
add('<h3>2.2 Reading the failures</h3>')
add('<p>For each failed adapter trajectory two steps were located mechanically: the <em>lock-in step</em>, the first step whose call is identical to the previous step\'s, and the <em>divergence step</em>, the first step whose call differs from the base\'s call at the same index on the coinciding history. The first wrong step of each failure was then read from the transcript and classified by hand; the classes are given in §3.2 with the reading that supports each.</p>')
add('<h3>2.3 The training rows</h3>')
add('<p>The 6,648 training rows were searched for the three commonest defects: consecutive assistant turns with the same call, split by whether the preceding observation was an error and by whether the repeat is the trained target turn or only visible history; list_files calls in list-family rows with a relative directory; calculate calls whose expression is the literal word "calculate".</p>')
add('<h3>2.4 Fixed-history comparison</h3>')
add('<p>For each task the base passed and the adapter failed, the runner\'s exact prompt at the divergence step was rebuilt from the adapter\'s own transcript (system prompt, the task, the recorded turns and observations with the two-observation window, and the generation suffix). The base model was loaded once through the pipeline\'s locked loader; the adapter was then attached in-process by the same call the loader uses. On the identical prompt each model produced (i) a greedy continuation, parsed as a tool call; (ii) a teacher-forced pass over the adapter\'s recorded turn and, where the base has one at that step, over the base\'s recorded turn, giving the log-probability of every token of each turn under each model; (iii) residuals at the five attention members of the band (layers 12, 16, 20, 24, 28) at three positions, the turn start, the token before the tool name, and the token before the first argument value, read through the hosted Jacobian lens (layer 20 primary, the other layers a profile, never pooled) and compared between models by cosine.</p>')

add('<h3>2.5 The recurrence forms</h3>')
add('<p>The Deputy raised, at 15:10, that the adapter was trained under one recurrence and is evaluated under another: the training stage installs a chunkwise gated-delta form (chunk 256), while inference runs the library\'s Metal kernel, and the stock training path runs a sequential reference loop. Two measurements: at the operator, random inputs at the model\'s own dimensions (16 key heads, 32 value heads, 128/128) and lengths up to 3,072 tokens, comparing the three forms\' outputs and final states in bfloat16 and float32; at the model, the adapter\'s teacher-forced logits over each pair\'s recorded turn under the kernel (eval mode), the reference loop (training mode) and the chunkwise form (training mode with the installer), compared position by position.</p>')
add('<h3>2.6 The snapshot cache</h3>')
add('<p>The Director\'s run of the cache-equivalence script found four of six greedy trajectories diverging, two at step 0, with no speedup. Astra\'s source reading attributes step-0 divergence to the execution schedule: the snapshot strategy processes the system-and-task prefix in one model call before the library\'s prefill loop sees the rest, so the forward passes differ in shape. The isolation table runs on fixed token ids with the continuation teacher-forced: the ordinary schedule against the split, the split with and without restored state, restoration after the live cache has advanced, and a control split at half the prompt; per comparison the largest logit difference at the last prompt position, whether the greedy token agrees, the winning margin, the first continuation position whose greedy token differs, and the per-layer residual difference.</p>')
add('<h2>3. Results</h2>')
add('<h3>3.1 Outcomes on the 19 tasks</h3>')
add('<div class="wrap"><table><tr><th>family</th><th class="n">tasks</th><th class="n">adapter passes</th><th class="n">base passes</th></tr>')
for f, v in desc["by_family"].items():
    add(f'<tr><td>{E(f)}</td><td class="n">{v["n"]}</td><td class="n adapter">{v["adapter"]}</td><td class="n base">{v["base"]}</td></tr>')
add(f'<tr><td><strong>all</strong></td><td class="n"><strong>{n}</strong></td><td class="n adapter"><strong>{ap}</strong></td><td class="n base"><strong>{bp}</strong></td></tr></table></div>')
add(f'<p>Of the adapter\'s {desc["adapter_failures"]} failures, {desc["adapter_failures_locked_in"]} contain an identical repeat of the previous call and, once repeating, every one of them repeats the note and the call verbatim to step 24. The two failures without an identical repeat are the cross-reference task, which cycles among records already read, and the conditional update, which finishes in 9 turns with the wrong service throttled. {desc["pairs_base_pass_adapter_fail"]} tasks form base-pass/adapter-fail pairs.</p>')

add('<h3>3.2 The first wrong step</h3>')
add('<div class="wrap"><table><tr><th>task</th><th>adapter</th><th>base</th><th class="n">diverges at</th><th class="n">locks in at</th><th>first wrong step</th></tr>')
for t in desc["tasks"]:
    a = '<span class="pass">pass</span>' if t["adapter_success"] else f'<span class="fail">fail</span> <span class="small">{t["adapter_turns"]} turns</span>'
    b = '<span class="pass">pass</span>' if t["base_success"] else f'<span class="fail">fail</span> <span class="small">{t["base_turns"]} turns</span>'
    if t["base_success"]:
        b += f' <span class="small">{t["base_turns"]} turns</span>'
    d = f'<strong>{E(t["defect"])}</strong> <span class="small">{E(t["defect_note"])}</span>' if t["defect"] else ''
    add(f'<tr><td>{E(t["task_id"].replace("test-","").replace("-clean",""))}</td><td>{a}</td><td>{b}</td><td class="n">{num(t["first_divergence_step"])}</td><td class="n">{num(t["lock_in_step"])}</td><td>{d}</td></tr>')
add('</table></div>')
add('<div class="wrap"><table><tr><th>class</th><th class="n">failures</th><th>reading</th></tr>')
READING = {
 "re-issued call after success": "the note is coherent (a correct plan, or the matched file named) and the call repeats the previous successful call; twice the note itself is copied verbatim",
 "malformed argument": "the note carries the right expression and the call carries a wrong one (the tool's own name, or an extra parenthesis); the syntax error is then annotated with the transient-failure template",
 "path abbreviation": "the prompt names a workspace path and the call sends its last component; the empty listing is then narrated as a listing of one file",
 "wrong hop": "an already-read record is re-read and the chain cycles",
 "wrong file": "one metric file is skipped for the next",
 "state-note comparison error": "the running maximum in the note is overwritten by the most recent value",
}
for k, v in sorted(desc["defects"].items(), key=lambda kv: -kv[1]):
    add(f'<tr><td>{E(k)}</td><td class="n">{v}</td><td>{E(READING.get(k, ""))}</td></tr>')
add('</table></div>')

tr = desc["training_rows"]
add('<h3>3.3 The training rows do not contain these defects</h3>')
add('<div class="wrap"><table><tr><th>quantity</th><th class="n">count</th></tr>'
    f'<tr><td>training rows</td><td class="n">{tr["rows"]:,}</td></tr>'
    f'<tr><td>target turns repeating the previous call after an ERROR observation (all transient-fault rows)</td><td class="n">{tr["target_repeat_after_error"]}</td></tr>'
    f'<tr><td>target turns repeating the previous call after a non-error observation</td><td class="n">{tr["target_repeat_after_ok"]}</td></tr>'
    f'<tr><td>target turns changing the call after an ERROR observation</td><td class="n">{tr["changed_call_after_error"]:,}</td></tr>'
    f'<tr><td>history turns showing "retry it unchanged" after a hidden (windowed) observation</td><td class="n">{tr["history_retry_after_hidden"]}</td></tr>'
    f'<tr><td>list_files calls in list-family rows / with a relative directory</td><td class="n">{tr["list_files_calls_in_list_rows"]} / {tr["list_files_relative"]}</td></tr>'
    f'<tr><td>calculate calls / with the literal "calculate" as the expression</td><td class="n">{tr["calculate_calls"]:,} / {tr["calculate_literal_expression"]}</td></tr>'
    f'<tr><td>update-family rows going read_file → replace_text / read_file → read_file (recovery rows)</td><td class="n">{tr["update_read_then_replace"]} / {tr["update_read_then_read"]}</td></tr>'
    '</table></div>')
add('<p>The pre-registered hypothesis for this run was that doubling the transient-fault repeats (from one to two per row) taught retry-after-any-error. The rows do carry 488 trained retries after an error, and the calculator failures fit that reading exactly: the note applies the transient-failure template to a syntax error. But five of the fourteen failures repeat a call after a <em>successful</em> observation, a pattern with no trained target at all, and the argument and path defects that start most failures are absent from the rows. The retry hypothesis explains the lock-in, not the first wrong step.</p>')

add('<h3>3.4 Fixed-history comparison</h3>')
if not fh:
    add('<div class="note">Pending. The script is written and dry-run on all eleven pairs (prompts of 417 to 1,859 tokens; both slots located in every pair). It runs once the machine is released and this section is filled from its output.</div>')
else:
    prm = fh["parameters"]
    add(f'<p class="small">{E(prm["model"])} with and without the adapter; lens {E(Path(prm["lens"]).name)}; greedy; read layers {", ".join(map(str, prm["read_layers"]))} with layer {prm["primary"]} primary.</p>')
    # summary table
    add('<div class="wrap"><table><tr><th>pair</th><th class="n">t</th><th>adapter\'s recorded call</th><th>base\'s recorded call</th><th>base continues with</th><th>adapter continues with</th><th class="n">log p(adapter turn): base / adapter</th><th class="n">log p(base turn): base / adapter</th></tr>')
    for tid, p in fh["pairs"].items():
        b, a = p["base"], p["adapter"]
        def call(x):
            return E(json.dumps(x)[:70]) if x else "—"
        lpa = f'{num(b["teacher_forced"]["sum_logp"],1)} / {num(a["teacher_forced"]["sum_logp"],1)}'
        lpb = (f'{num(b["teacher_forced_base_turn"]["sum_logp"],1)} / {num(a["teacher_forced_base_turn"]["sum_logp"],1)}' if b.get("teacher_forced_base_turn") else "—")
        add(f'<tr><td>{E(tid.replace("test-","").replace("-clean",""))}</td><td class="n">{p["t"]}</td><td><code>{call(p["adapter_turn_t_action"])}</code></td><td><code>{call(p["base_action_at_t"])}</code></td>'
            f'<td class="base"><code>{call(b["continuation"]["action"])}</code></td><td class="adapter"><code>{call(a["continuation"]["action"])}</code></td><td class="n">{lpa}</td><td class="n">{lpb}</td></tr>')
    add('</table></div>')
    # per pair cards
    for tid, p in fh["pairs"].items():
        b, a = p["base"], p["adapter"]
        add(f'<div class="pair"><h3>{E(tid)} · divergence at step {p["t"]} ({E(p["kind"])}), lock-in at {num(p["lock_in_step"])}</h3>')
        add(f'<p class="small">{E(p["prompt"])}</p>')
        add('<div class="grid">')
        for lab, cls, m in (("base", "b", b), ("adapter", "a", a)):
            c = m["continuation"]
            add(f'<div class="col {cls}"><h4 class="{lab}">{lab} on the same prompt</h4>')
            add(f'<p class="small">continues: <code>{E(json.dumps(c["action"]))}</code>{" — the adapter\'s recorded call" if c["same_as_adapter_recorded_call"] else ""}{" — the repeated call" if c["same_as_repeated_call"] else ""}</p>')
            tf = m["teacher_forced"]
            for pname in ("tool_name_slot", "first_argument_slot", "turn_start"):
                key = f"logits_{pname}"
                if tf.get(key):
                    top = ", ".join(f'{tok(x["text"])} {x["logp"]:.2f}' for x in tf[key]["top"][:5])
                    add(f'<p class="small"><strong>{pname.replace("_"," ")}</strong> next token: {top}</p>')
            L = str(prm["primary"])
            lens = m["lens"].get(L, {})
            for pname in ("tool_name_slot", "first_argument_slot"):
                if lens.get(pname):
                    top = ", ".join(f'{tok(x["text"])}' for x in lens[pname]["top"][:8])
                    add(f'<p class="small"><strong>lens L{L} at {pname.replace("_"," ")}</strong>: {top}</p>')
            add('</div>')
        add('</div>')
        # per-token gap: the tokens of the adapter's turn with the largest adapter-minus-base gap
        gaps = p["per_token_logp_gap_adapter_minus_base"]; toks = a["teacher_forced"]["tokens"]
        top = sorted(range(len(gaps)), key=lambda i: -abs(gaps[i]))[:6]
        add('<p class="small">largest per-token log-probability gaps, adapter minus base, over the adapter\'s recorded turn: ' + "; ".join(f'{tok(toks[i])} {gaps[i]:+.2f}' for i in sorted(top)) + '</p>')
        cos = p["residual_base_vs_adapter"]
        add('<p class="small">residual cosine base·adapter by layer at the tool-name slot: ' + ", ".join(f'L{L} {v["tool_name_slot"]["cosine"]:.3f}' for L, v in cos.items() if "tool_name_slot" in v) + '</p>')
        add('</div>')

add('<h3>3.5 The recurrence forms agree to rounding at the operator; the model-level comparison</h3>')
if forms_f32 and forms_bf16:
    add('<div class="wrap"><table><tr><th>tokens</th><th class="n">chunks at 256</th><th class="n">kernel vs loop, bf16</th><th class="n">chunkwise-256 vs loop, bf16</th><th class="n">chunkwise-256 vs kernel, bf16</th><th class="n">chunkwise-256 vs loop, f32</th><th>R32 tolerance, f32</th></tr>')
    f32 = {r["tokens"]: r for r in forms_f32["rows"]}
    for r in forms_bf16["rows"]:
        c = r["chunkwise"]["256"]; g = f32.get(r["tokens"])
        g32 = f'{g["chunkwise"]["256"]["vs_ops"]["y"]["max_abs"]:.2e}' if g else "—"
        gpass = "pass" if g and g["chunkwise"]["256"]["vs_ops"]["y"]["allclose_r32"] else "—"
        add(f'<tr><td>{r["tokens"]:,}</td><td class="n">{c["n_chunks"]}</td><td class="n">{r["kernel_vs_ops"]["y"]["max_abs"]:.2e}</td><td class="n">{c["vs_ops"]["y"]["max_abs"]:.2e}</td><td class="n">{c["vs_kernel"]["y"]["max_abs"]:.2e}</td><td class="n">{g32}</td><td>{gpass}</td></tr>')
    add('</table></div><p>Largest absolute difference in the block output; bfloat16 differences are one unit in the last place of the output. The three forms are the same function to rounding at the operator, at 11 and 12 chunks.</p>')
if fh:
    add('<div class="wrap"><table><tr><th>pair</th><th>model</th><th class="n">max |Δlogit| over the turn: loop−kernel / chunkwise−kernel / chunkwise−loop</th><th class="n">greedy agreement over the turn, chunkwise vs kernel</th><th>tool-name slot argmax same</th><th>argument slot argmax same</th><th class="n">log p(recorded turn): kernel / loop / chunkwise</th></tr>')
    for tid, p in fh["pairs"].items():
        for lab in ("base", "adapter"):
            f = p[lab].get("recurrence_forms")
            if not f: continue
            ck = f["chunkwise_vs_kernel"]; lp = f["sum_logp_of_recorded_turn"]
            add(f'<tr><td>{E(tid.replace("test-","").replace("-clean",""))}</td><td class="{lab}">{lab}</td><td class="n">{f["ops_vs_kernel"]["turn"]["max_abs"]:.2f} / {ck["turn"]["max_abs"]:.2f} / {f["chunkwise_vs_ops"]["turn"]["max_abs"]:.2f}</td><td class="n">{ck["turn"]["argmax_agreement"]:.3f} ({ck["turn"]["argmax_disagreements"]} of {len(p[lab]["teacher_forced"]["tokens"])})</td><td>{ck.get("at_tool_name_slot",{}).get("argmax_same","—")}</td><td>{ck.get("at_first_argument_slot",{}).get("argmax_same","—")}</td><td class="n">{lp["kernel"]:.1f} / {lp["ops"]:.1f} / {lp["chunkwise"]:.1f}</td></tr>')
    add('</table></div>')
add('<h3>3.6 The snapshot cache: the schedule, not the restore</h3>')
if not cache:
    add('<div class="note">Pending: the isolation script is written and runs after the fixed-history job releases the model.</div>')
if cache:
    add('<p>Read with one caveat. The isolation drove the blocks through the architecture view, which promotes activations to float32; the runner\'s path is bfloat16 end to end and the greedy choice is taken on bfloat16 log-probabilities, whose unit in the last place at these magnitudes is 0.06 to 0.125, wider than the 0.07 first-token margin on the ledger task that flipped in the Director\'s run. Restoration is exact and the forward deterministic; the schedule perturbation is rounding-sized in float32 and is being re-measured on the library\'s own bfloat16 path with and without a float32 up-cast before the sampler. The uncached trajectories reproduce the earlier evaluation token for token, so the flips are a property of the cached schedule under bfloat16 rounding, not of a wrong state.</p>')
else:
    add('<div class="wrap"><table><tr><th>task</th><th class="n">prompt / prefix tokens</th><th class="n">1 native vs split: max |Δ|, argmax same, margins</th><th class="n">4 native vs half-split: max |Δ|</th><th class="n">2 fresh vs restored split: max |Δ|</th><th class="n">3 restore after advance: max |Δ|</th><th class="n">first differing greedy position (of N)</th><th class="n">residual Δ/‖h‖ at L12 … L32</th></tr>')
    for c in cache["cases"]:
        a = c["1_native_vs_split"]; pl = c["per_layer_residual_native_vs_split_at_last_position"]
        add(f'<tr><td>{E(c["task_id"].replace("test-","").replace("-clean",""))}</td><td class="n">{c["prompt_tokens"]} / {c["prefix_tokens"]}</td><td class="n">{a["max_abs"]:.3f}, {a["argmax_same"]}, {a["margin"][0]:.2f} / {a["margin"][1]:.2f}</td><td class="n">{c["4_native_vs_half_split"]["max_abs"]:.3f}</td><td class="n">{c["2_split_fresh_vs_split_restored"]["max_abs"]:.3f}</td><td class="n">{c["3_restore_after_advance_vs_restore_again"]["max_abs"]:.3f}</td><td class="n">{c["continuation_first_differing_greedy_position"]} ({c["continuation_positions_compared"]})</td><td class="n">' + ", ".join(f'{v["rel_to_norm"]:.1e}' for v in pl.values()) + '</td></tr>')
    add('</table></div>')
add('<h2>4. Discussion</h2>')
if fh:
    add('<p><strong>The fixed-history comparison settles the attribution.</strong> Given the adapter\'s own history, the base writes the full path, the arithmetic expression, the replacement or the finish in every pair, and the adapter writes what it wrote before in every pair. The adapter\'s log-probability for its wrong turn is between −0.1 and −4.3 nats; for the base\'s correct turn it is between −12 and −83. The base\'s numbers run the other way. The divergence is in the weights, and it is not marginal.</p>')
    add('<p><strong>The recurrence hypothesis is refuted.</strong> At the operator the kernel, the reference loop and the chunkwise form agree to rounding at production shapes and lengths. At the model, the adapter\'s wrong turns keep every greedy token and their log-probability within 0.1 nats under all three forms. What remains is the ordinary gap between any training-mode forward and the inference kernel, which the base shows equally, and which every adapter trained on this model with this library shares.</p>')
    add('<p><strong>What the lens says at the decision.</strong> At the tool-name and first-argument slots the layer-20 lens returns quote and punctuation pieces for both models: inside a JSON call the residual, read through a lens fitted on prose against the final layer, encodes the format that follows rather than the argument\'s content. The content decision is visible only in the final logits. What the residual geometry does show is the size of the adapter\'s change: cosine 0.50 to 0.75 with the base across the band and a norm about twice the base\'s at layer 20 on every pair. Two hundred updates of a rank-16 adapter on every projection of every block moved the stream far, uniformly, and in a direction the lens does not resolve into words at these positions.</p>')
add('<p>Three things are established on the 19 tasks. First, the adapter is worse than the base on the same tasks by a wide margin, and the pairing removes task difficulty as an explanation. Second, its failures have a two-stage shape: a first wrong step that is a construction error (an argument, a path, a call re-issued after the note that should have led elsewhere), followed by a verbatim lock-in that the trained retry-after-error behaviour makes permanent. Third, the first-stage errors were not copied from the data; they are what the model now produces after the note templates it did copy. The most economical reading is that 200 updates of a rank-16 adapter over every projection of every block, at a learning rate sized for a longer run, moved the model far enough to reproduce the surface of the rows (validation loss 0.092) while damaging the base\'s ability to build the call from the note.</p>')
add('<p>For the deliverable this means the checkpoint is a measurement, not a model: it answers the question the standard asks (coherence to completion collapses under this recipe) and it does not replace the base. For the next run the recipe changes are on record: transient repeats back to run D\'s value or below, more correction variants than retry variants, a doomed-trajectory stop as a rollout flag, and an early evaluation at the first checkpoint rather than at the end.</p>')
add('<h2>5. Limitations</h2>')
add('<ul><li>Nineteen tasks, one or two per family: the per-family readings are exact for these instances and are not rates.</li><li>The defect classes are a hand reading of the transcripts; the divergence and lock-in steps are mechanical.</li><li>The training-row search covers the three commonest defects, not every class.</li>' + ('' if fh else '<li>The fixed-history comparison is pending.</li>') + '</ul>')
add('<p class="small">Sources: outputs/agent-v2e-qwen35-4b/transcripts/best-adapter-test/transcripts.jsonl; outputs/agent-v2/evals/base-test.json; data/agent_v2e-qwen35-4b-cap2688/train.jsonl; scripts/fixed_history_lens.py; heartbeat entries 14:30–15:00, 2026-09-07.</p>')
add('</main>')
OUT.write_text("\n".join(parts))
print("wrote", OUT, OUT.stat().st_size, "bytes; fixed-history", "included" if fh else "pending", "; cache", "included" if cache else "pending")
