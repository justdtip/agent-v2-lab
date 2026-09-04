# Run logging for training and probes — design, gate proposal, implementation plan

**Deputy → Chief, 2026-09-05.** Requested by the Director: terminal progress and run-health
metrics for training and probe runs, written to a file AND to stdout/stderr so a run can be
read back and cited; the Director considers the training health record a **training gate**.
Implementation is dispatched in parallel on disjoint files (§5); the Deputy reviews, then the
Chief. This note asks the Chief for one ruling (§4).

## 1. What exists today

- `pipeline/cli.py:344-400` `stage_train`: tees mlx_lm's stdout to `<output>/train.log`
  (`_Tee`, `:389-400`) and writes `<output>/metrics.jsonl` per report through
  `_TrainingMetrics` (`:256-283`); `_validation_losses` (`:433`) reads `metrics.jsonl` for
  checkpoint selection — **its schema is load-bearing and stays byte-compatible.**
- mlx_lm 0.31.3 `TrainingCallback` delivers per report: `iteration`, `train_loss`,
  `learning_rate`, `tokens_per_second`, `trained_tokens`, `peak_memory` (GB); per eval:
  `iteration`, `val_loss`.
- `probes/patch.py` (P6) prints **nothing** during a run — the Director's attempts were silent
  for their whole duration. `adapter_delta.py` prints only "Wrote …" at the end.
  `state_probe.py` and `assistant_axis.py` print ad-hoc progress lines to stdout only; nothing
  goes to a file.
- `ModelSpec.memory_budget_gib` (`models.py:54`) is the memory budget every run should be
  reported against.

## 2. Design

One small module, `src/local_llm_lab/runlog.py`, stdlib only, no new dependency, no global
state. Every run (training stage, each probe CLI) opens a `RunLog` on its output directory:

- **Two files, both append-only, both created with the output directory:**
  `<output>/run.log` — the human stream, verbatim what the terminal showed (including tee'd
  third-party output such as mlx_lm's own lines); `<output>/events.jsonl` — one JSON object
  per event for machine reading (`ts` UTC ISO, `elapsed` seconds, `run` name, `kind`, and a
  `fields` object; `start`/`end` events carry command, pid, cwd, python, status, summary).
- **Two terminal streams:** `info`, `metric`, `progress` → stdout; `warn`, `error` → stderr.
  Every line is prefixed `HH:MM:SS +MM:SS <run> │`; progress lines read
  `[label step/total pct%] k=v … eta MM:SS`. Everything flushes on write.
- **Progress** is explicit: callers pass `(step, total, label, **fields)`; the log computes
  fraction and ETA from its own clock. Library functions gain an optional `progress=` callback
  (the pattern `assistant_axis.py:1238` already uses) so fakes-only tests can drive them.
- **Training health** is a pure state machine, `TrainingHealth`, fed by the callback and
  independent of files, with thresholds in `HealthThresholds` (defaults in code, overridable
  by an optional `train.health:` block in the arm config, recorded in the artifact):
  | flag | severity | rule |
  | --- | --- | --- |
  | `non_finite_loss` | fatal — the run aborts | train or val loss is NaN/inf |
  | `loss_spike` | warning | train loss > `loss_spike_factor` (2.0) × median of the trailing `loss_window` (5) reports |
  | `val_loss_rising` | warning | the last `val_rising_reports` (3) validation losses rise monotonically |
  | `throughput_drop` | warning | tokens/s < `throughput_drop_factor` (0.5) × median of the trailing window |
  | `memory_over_budget` | warning | mlx peak memory (GB) > `memory_budget_gib` from the registry |
  | `incomplete_run` (R26 c) | fatal — never "healthy" | iterations done < planned, or no checkpoint at the final iteration; recorded by `on_finish` after the trainer returns |
  The stage writes `<output>/health.json` on every exit path: verdict `healthy` / `warnings` /
  `incomplete` / `aborted` (precedence right to left), the flags with the iteration and detail
  that raised each, thresholds, last and best values, iterations done vs planned, final
  checkpoint presence, peak memory, elapsed, and exit status. Non-finite numbers are carried
  as the strings `"nan"`/`"inf"` so every file stays strict JSON.
- **Existing artifacts unchanged:** `train.log` and `metrics.jsonl` keep their exact content;
  probe payloads gain no keys. Logging is additive and never alters a computation.

## 3. Invariants for every lane

No model execution (P6 is on the lane; a tokenizer is not execution); no writes under
`outputs/`, `data/`, `reports/`, `pending/`; fakes-only tests (R10); banned constants and seed
literals absent (`tests/test_repository_rules.py` green); `metrics.jsonl` schema and
`train.log` content byte-identical; no new dependency; red-first per lane; only the lane's
files change; `uv run pytest` bare with exit status reported; R16 in every report.

## 4. Ruling requested: the health record as a training gate (proposed R26) — *as ruled: training condition 8 (checklist A8) by the Director's amendment; the evaluation lift additionally requires the verdict. See the Chief's ruling appended below and wiring map §7 R26(d).*

The Director treats run health as a training gate. Proposed text, for the Chief to ratify,
amend, or refuse — as an addition to R15's evidence format rather than an eighth condition:

> **R26 (proposed).** Every training run writes `health.json`. A `non_finite_loss` flag aborts
> the run and its adapters are not eligible for selection. An arm's **evaluation lift**
> requires `health.json` with verdict `healthy` or `warnings`; every warning is listed in the
> lift request with the Deputy's reading of it. The lift request cites `run.log` and
> `events.jsonl` by path and SHA-256, so any claim about the run is checkable from the file.
> Probe runs write the same two files; a probe lift request cites them the same way.

Thresholds are engineering defaults, recorded per run; changing one is a config change under
the arm's `train.health:` block, visible in `health.json` and in provenance.

## 5. Implementation plan — four lanes, disjoint files, one contract

The public contract is fixed in the dispatch briefs (signatures below); lane A owns the module
and writes the full API skeleton first so lanes B–D can import it.

| lane | files | scope |
| --- | --- | --- |
| A | `src/local_llm_lab/runlog.py` (new), `tests/test_runlog.py` (new) | `RunLog`, `Tee`, `HealthThresholds`, `TrainingHealth`, `TrainingAborted` |
| B | `pipeline/cli.py` (`_TrainingMetrics` :256-283, `stage_train` :344-387, `_Tee` :389-400), `tests/test_cli.py` | training integration, `health.json`, `train.health:` config, abort path |
| C | `probes/patch.py` (`run_patch_probe` :1193+, `main` :1475+), `probes/adapter_delta.py` (ablation CLI :915-990, `main` :999-1088, loops :495-530/:755), `tests/test_patch.py`, `tests/test_adapter_delta.py` | P6 and block-ablation progress + files |
| D | `probes/state_probe.py` (:2930-3060), `probes/assistant_axis.py` (:1200-1311), `tests/test_state_probe.py`, `tests/test_probes_axis.py` | P2 capture and P1 progress + files |

Contract (lane A implements exactly; B–D code against it):

```python
class RunLog:
    @classmethod
    def open(cls, output: Path, *, name: str, command: Sequence[str] | None = None,
             stdout: TextIO | None = None, stderr: TextIO | None = None,
             clock: Callable[[], float] = time.monotonic) -> "RunLog"
    def info(self, message: str, **fields: Any) -> None            # stdout
    def warn(self, message: str, **fields: Any) -> None            # stderr
    def error(self, message: str, **fields: Any) -> None           # stderr
    def metric(self, name: str, **fields: Any) -> None             # stdout, compact k=v line
    def progress(self, step: int, total: int, label: str, **fields: Any) -> None
    def tee(self, stream: TextIO) -> TextIO                        # stream + run.log
    def close(self, status: str = "ok", **summary: Any) -> None    # idempotent
    def __enter__(self) -> "RunLog"; def __exit__(self, *exc) -> None   # status ok/error, re-raises
    @property elapsed -> float; @property paths -> dict[str, Path]  # {"run_log":…, "events":…}

@dataclass(frozen=True)
class HealthThresholds: loss_spike_factor=2.0; loss_window=5; val_rising_reports=3;
                        throughput_drop_factor=0.5; memory_budget_gib: float | None = None
    @classmethod from_config(cls, block: Mapping | None, *, memory_budget_gib: float | None)

class TrainingAborted(RuntimeError)
class TrainingHealth:
    def __init__(self, thresholds: HealthThresholds, *, iters: int | None)
    def on_train_report(self, *, iteration, train_loss, learning_rate, tokens_per_second,
                        trained_tokens, peak_memory_gb) -> list[str]   # new flag names; raises TrainingAborted on non-finite
    def on_val_report(self, *, iteration, val_loss) -> list[str]
    flags: tuple[dict, ...]; verdict: str                             # healthy|warnings|aborted
    def summary(self, *, elapsed: float, status: str) -> dict         # JSON-safe
```

## 6. When this can proceed

- **Implementation: now.** Fakes-only; P6 running on the lane is unaffected (it imported the
  old modules; nothing here executes a model or writes under `outputs/`).
- **Commit:** done — one commit per lane after the Deputy's direct review and the Chief's gate:
  A `c1f7d51`, B `88ecac0` (with the stale-checkpoint condition), C `0a79788`, D `2405598`.
- **First live use:** the B4 training run, which has not started; if the chain completes
  before P6 finishes, B4 trains with the health record from its first iteration. Probes pick
  it up on their next run; the current P6 attempt finishes silently, as before.

---

## Chief's ruling (2026-09-05 03:20): R26 adopted with amendments

Verified against the installed library: mlx-lm 0.31.3 passes exactly the fields §2 consumes
(`trainer.py:310-313, 354-361`). The design, the file pair, the health state machine, the
four-lane split, and the §5 contract are approved as written. Amendments, recorded as R26 in
the wiring map §7:

1. **`incomplete_run` is a second fatal flag**: iterations done below planned, or no
   checkpoint at the final iteration. A run that stopped early is not eligible for selection
   unless the Director lifts it explicitly with the shortfall stated. Health must not read
   "healthy" for a run that never finished.
2. **The `start` event identifies the run**: resolved `ModelSpec` name and hf_id, config
   path, data manifest hash, git commit. `run.log` on its own must say what ran.
3. **`health.json` and the thresholds are copied into `provenance.json`.**
4. **Probe cadence**: at least one progress line per outer unit (case, layer, task, or
   checkpoint). The silent thirty-minute P6 run is the failure this closes.
5. **Amended by the Director (03:40): training does not start until logging is in place.**
   R15 gains condition 8: lanes A and B committed and green before any training lift, so the
   health record exists from iteration one. The evaluation lift keeps the health-verdict
   requirement on top. Lanes C and D (probes) are not on the training critical path but every
   probe run after they land must log.

Lane order: A first (contract), then B–D in parallel on disjoint files; the R19 chain per
lane; the Chief gates each commit. The current P6 attempt finishes under the old logging; B4
trains with the health record from iteration one if the chain completes first, and that is
the preferred order.
