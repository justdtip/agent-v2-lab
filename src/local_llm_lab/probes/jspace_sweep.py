"""J-space sweep: does a mid-layer J-lens know the hidden filename, or just like some digits?

The single-task probe in ``research/jspace_probe.md`` left one result unexplained: at two of
three probed layers the Jacobian lens ranked the correct suffix above a previously-seen suffix
above an unrelated one, *including* in the condition where the model's own output shows it does
not know the answer.

Two explanations, and they make opposite predictions under averaging:

- **Weak internal trace.** The ordering tracks the task, so it survives averaging over many
  tasks whose correct and incorrect digits differ.
- **Token-frequency artifact.** The ordering tracks the digits themselves (some digit tokens
  are simply likelier in this readout), so it washes out once the digit assignment varies, and
  it persists when the candidates are paired with a context from a *different* task.

So this runs a paired sign test. For each ledger task we take the probe point where the
directory listing has fallen out of the window, strip the ``pending:`` lists so the filename is
genuinely unavailable, force the context to end mid-note at ``Reading invoice-N-``, and compare
the J-lens probability of the true suffix's first token against a wrong-but-previously-seen
suffix from the same task. Then we repeat the identical comparison with each candidate pair
scored against a *mismatched* context taken from another task, which is the null: under the
artifact hypothesis the win rate is unchanged.

EXP-001 (issue #54) turned that research script into this installed CLI. What changed, and why:

- **Nothing is a literal any more.** Model, policy, layers, corpus size and length, source
  positions, task count, split, data seed, generator version and derivative method all come
  from flags or from the registry (R17). The layer list is resolved through
  ``probes.policies.resolve_layers`` against the loaded model's actual depth.
- **The model is loaded through ``pipeline.evaluate.load_policy``**, never ``mlx_lm.load``, so
  the resolved specification, the preflight gate (SPEC-001 §10) and the registry's policy table
  apply to this probe as they do to every other stage.
- **Prompts render through ``protocol.build_prompt`` with the registry's template kwargs**
  (B2). A bare ``apply_chat_template`` on a thinking model leaves the think block open, so the
  probe would read a distribution taken inside it.
- **Three readouts, one JVP** (B3): ``self``, ``future`` and ``all``; ``all`` primary. An
  empty future window raises rather than reading as a structural zero (A1).
- **One sweep over the kind-matched layer family** (B5). With ``--layers`` omitted the layer
  list is the registry fractions *plus* the kind-matched partners the config's
  ``full_attention_interval`` implies, because §2's predictions compare an attention layer
  against a linear-attention layer at comparable depth and the fractions alone guarantee no
  such pair. Holm corrects across that whole family, per readout.
- **R34 conformance and R35 comparability blocks** are written into the artifact and stated in
  this docstring; **R26** logging (``run.log``, ``events.jsonl``, one progress line per probe
  point) covers the whole run.

Conformance statement (R34), for the estimator this module computes. Layer ``L`` is the
residual after block ``L - 1`` and a probe layer's kind is the kind of the block that **wrote**
it (block ``L - 1``), not the block about to read it. The layer list is the one EXP-001 §3.5
pre-registers: the registry fractions plus, for each in-band fraction that landed on a
recurrent block, the nearest layer written by an attention block, derived from the loaded
model's own ``full_attention_interval`` and recorded with the period, the pairs and each
layer's role. Sources are placed at the fractions given
by ``--source-positions`` of each corpus context; ``self`` reads the output tangent at the
source, ``future`` sums it over every strictly later position, ``all`` is their sum and is
primary; the ``self`` readout alone is the source paper's self-only limiting case, the variant
every J-lens number recorded here before EXP-001 was drawn under. Corpus size and minimum
context length are ``--corpus-size`` and ``--corpus-length``; the median future window per
context and per readout is recorded in the artifact. The JVP method is the flag when given and
otherwise the one this model's preflight established, and a run with neither fails closed.
Residual capture takes the registry's ``probes.capture_dtype`` (R18b); the float32 cast
boundary is inside ``ArchitectureView``, and the preflight's ``fp32_manual_vs_native`` block is
copied into every artifact (R18a).

Comparability (R35). Two J-lens tables are comparable only where policy, derivative method,
layer selection (as fractions and kinds), generator version, prompt rendering and estimator
variant agree, or every difference is named in both artifacts. Base-versus-adapter is a named
difference, not a comparison. The artifact's ``comparability`` block carries all of them.

No model runs from this docstring: the CLI loads weights, and it runs only under a Director
lift.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_llm_lab.pipeline.jlens import (
    DEFAULT_CORPUS,
    DEFAULT_CORPUS_LENGTH,
    DEFAULT_SOURCE_POSITIONS,
    JVP_METHODS,
    PRIMARY_READOUT,
    READOUTS,
    EmptyFutureWindowError,
    JvpMethodUnresolved,
    SourcePosition,
    build_corpus,
    comparability_block,
    conformance_block,
    distribution,
    encode,
    hybrid_period,
    jlens_readouts,
    kind_matched_layer_family,
    probe_layer_kind,
    residual_at,
    resolve_jvp_method,
    resolve_source_positions,
)

__all__ = [
    "MIN_CASES",
    "PROBE_STEP",
    "TASK_SPLIT",
    "holm_adjust",
    "main",
    "note_prefix",
    "render_markdown",
    "run_sweep",
    "select_cases",
    "sign_test",
    "strip_pending",
    "suffix_of",
]

#: The first read whose directory listing has fallen out of the two-observation window.
PROBE_STEP = 3
#: Below this the sign test has no power worth reporting; the run refuses rather than pretend.
MIN_CASES = 4
TASK_SPLIT = "jsweep"
_PENDING = re.compile(r"; pending: [^\n]*")


def note_prefix(thought: str, short_name: str) -> str | None:
    """The expert note truncated to just before the filename's random suffix."""
    stem = short_name.rsplit("-", 1)[0] + "-"  # "invoice-2-537.txt" -> "invoice-2-"
    if short_name not in thought:
        return None
    return thought[: thought.index(short_name)] + stem


def strip_pending(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the ``pending:`` lists from assistant notes, reproducing the run-A condition."""
    out = []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            content = _PENDING.sub(".", content)
        out.append({**message, "content": content})
    return out


def suffix_of(path: str) -> str:
    """``lab/x/ledger/invoice-2-537.txt`` -> ``537``."""
    return path.rsplit("-", 1)[-1].split(".")[0]


def select_cases(
    tasks: Sequence[Any],
    tokenizer: Any,
    *,
    spec: Any,
    probe_step: int = PROBE_STEP,
) -> list[dict[str, Any]]:
    """Build the probe points: one forced-prefix context and one candidate pair per task.

    The selection rule is the recorded sweep's, unchanged (EXP-001 §3.1): step ``probe_step``,
    which must be a ``read_file``; ``pending:`` lists stripped so the filename is genuinely
    unavailable; the context forced to end mid-note at the target's stem; a leakage guard that
    drops any case whose true suffix is visible anywhere in the rendered context, since that is
    no longer a memory test; and pairs whose first suffix tokens coincide are dropped because
    the comparison would be between a token and itself.

    What changed is the rendering: the context goes through ``protocol.build_prompt`` with the
    model's own ``template_kwargs`` (B2). The rows are already windowed by ``build_rows``, so
    ``keep_last`` is set to the row's own tool count -- re-windowing an already-stubbed
    observation would rewrite its line count and corrupt the context.
    """
    from local_llm_lab.pipeline.data import build_rows
    from local_llm_lab.pipeline.protocol import build_prompt

    cases: list[dict[str, Any]] = []
    for task in tasks:
        rows = build_rows(task)
        if len(rows) <= probe_step or probe_step >= len(task.steps):
            continue
        step = task.steps[probe_step]
        if step.action.name != "read_file":
            continue
        target_path = str(step.action.arguments["path"])
        short = target_path.rsplit("/", 1)[-1]
        prefix = note_prefix(step.thought, short)
        if prefix is None:
            continue
        earlier = [
            str(s.action.arguments["path"])
            for s in task.steps[:probe_step]
            if s.action.name == "read_file"
        ]
        if not earlier:
            continue
        true_suffix = suffix_of(target_path)
        false_suffix = suffix_of(earlier[-1])
        if true_suffix == false_suffix:
            continue
        messages = strip_pending(rows[probe_step]["messages"][:-1])
        tool_messages = sum(1 for message in messages if message["role"] == "tool")
        prompt = (
            build_prompt(tokenizer, messages, keep_last=tool_messages, spec=spec) + prefix
        )
        if true_suffix in prompt:
            continue  # the answer leaked into the context; not a memory test
        first_true = encode(tokenizer, true_suffix)
        first_false = encode(tokenizer, false_suffix)
        if not first_true or not first_false or first_true[0] == first_false[0]:
            continue
        cases.append(
            {
                "task_id": task.task_id,
                "difficulty": getattr(task, "difficulty", None),
                "prompt": prompt,
                "prefix": prefix,
                "true": true_suffix,
                "false": false_suffix,
                "true_token": first_true[0],
                "false_token": first_false[0],
            }
        )
    return cases


def sign_test(wins: int, total: int) -> float:
    """Two-sided binomial p-value against a fair coin, computed exactly."""
    from math import comb

    if total == 0:
        return 1.0
    tail = [comb(total, i) for i in range(total + 1)]
    observed = tail[wins]
    return min(1.0, sum(t for t in tail if t <= observed) / 2**total)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment across one family of tests (EXP-001 §2, multiplicity).

    The family is *every layer the sweep scored under one readout*: the in-band fractions,
    their kind-matched partners and the two reported fractions alike, one family per readout.
    EXP-001 §2 pre-registers Holm across it, as the P2 tables already do, because the World A
    row would otherwise count a dozen uncorrected nulls as evidence -- and because a partner
    layer is a test that was run, so excluding it would understate the multiplicity.
    """
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return {key: adjusted[key] for key in pvalues}


def _probabilities(
    view: Any,
    tokenizer: Any,
    prompt: str,
    candidates: dict[str, int],
    *,
    layers: Sequence[int],
    corpus_ids: Sequence[Any],
    source_positions: Sequence[SourcePosition],
    method: str,
    capture_dtype: str,
) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, Any]]]:
    """Per-layer, per-readout probability of each candidate, plus the model's own output.

    The model's own next-token distribution is the decisive measurement (EXP-001 §2); the lens
    readouts corroborate it. Both are returned from the same context so the pair is exact.
    """
    ids = encode(tokenizer, prompt)
    result: dict[str, dict[str, float]] = {}
    stats_by_layer: dict[int, dict[str, Any]] = {}
    for layer in layers:
        residual = residual_at(view, ids, layer, capture_dtype=capture_dtype)
        probe = residual[0, -1]
        mapped, stats = jlens_readouts(
            view,
            layer,
            probe,
            list(corpus_ids),
            source_positions=source_positions,
            readouts=READOUTS,
            method=method,
            capture_dtype=capture_dtype,
        )
        stats_by_layer[layer] = stats
        for name in READOUTS:
            probs = distribution(view, mapped[name]).tolist()
            result[f"jlens_L{layer}_{name}"] = {
                label: float(probs[token]) for label, token in candidates.items()
            }
        logit = distribution(view, probe).tolist()
        result[f"logit_lens_L{layer}"] = {
            label: float(logit[token]) for label, token in candidates.items()
        }
    final = residual_at(view, ids, view.num_layers, capture_dtype=capture_dtype)[0, -1]
    output = distribution(view, final).tolist()
    result["model_output"] = {
        label: float(output[token]) for label, token in candidates.items()
    }
    return result, stats_by_layer


def run_sweep(
    view: Any,
    tokenizer: Any,
    cases: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    corpus_ids: Sequence[Any],
    source_positions: Sequence[SourcePosition],
    method: str,
    capture_dtype: str,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Score every probe point in matched and mismatched contexts and aggregate the sign tests.

    ``progress`` is called once per probe point -- the outer unit of work here, since layers
    share the expensive prefill inside it -- so the run is never silent for longer than one
    point (R26 g).
    """
    import mlx.core as mx

    matched: dict[str, list[bool]] = {}
    mismatched: dict[str, list[bool]] = {}
    per_case: list[dict[str, Any]] = []
    stats_by_layer: dict[int, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        candidates = {"true": case["true_token"], "false": case["false_token"]}
        own, stats = _probabilities(
            view,
            tokenizer,
            case["prompt"],
            candidates,
            layers=layers,
            corpus_ids=corpus_ids,
            source_positions=source_positions,
            method=method,
            capture_dtype=capture_dtype,
        )
        stats_by_layer.update(stats)
        other = cases[(index + 1) % len(cases)]
        alien, _ = _probabilities(
            view,
            tokenizer,
            other["prompt"],
            candidates,
            layers=layers,
            corpus_ids=corpus_ids,
            source_positions=source_positions,
            method=method,
            capture_dtype=capture_dtype,
        )
        for key, values in own.items():
            matched.setdefault(key, []).append(values["true"] > values["false"])
        for key, values in alien.items():
            mismatched.setdefault(key, []).append(values["true"] > values["false"])
        per_case.append(
            {
                "task_id": case["task_id"],
                "difficulty": case["difficulty"],
                "true": case["true"],
                "false": case["false"],
                "mismatched_task_id": other["task_id"],
                "matched": own,
                "mismatched": alien,
            }
        )
        mx.clear_cache()
        if progress is not None:
            progress(index + 1, len(cases), "probe point", task_id=case["task_id"])

    results: dict[str, Any] = {}
    for key, wins in matched.items():
        null = mismatched[key]
        results[key] = {
            "matched_wins": sum(wins),
            "matched_n": len(wins),
            "matched_rate": sum(wins) / len(wins) if wins else 0.0,
            "mismatched_wins": sum(null),
            "mismatched_n": len(null),
            "mismatched_rate": sum(null) / len(null) if null else 0.0,
            "matched_p": sign_test(sum(wins), len(wins)),
            "mismatched_p": sign_test(sum(null), len(null)),
        }
    # Holm across the layer family, per readout: the family is every layer scored under one
    # readout -- in-band fractions, their kind-matched partners and the reported fractions
    # alike, since ``layers`` is the derived family -- exactly as EXP-001 §2 pre-registers it.
    for suffix in (*READOUTS, None):
        family = {
            key: value["matched_p"]
            for key, value in results.items()
            if key.startswith("jlens_L") and (suffix is None or key.endswith(f"_{suffix}"))
        }
        if suffix is None or not family:
            continue
        for key, adjusted in holm_adjust(family).items():
            results[key]["matched_p_holm"] = adjusted
    return {
        "cases": len(cases),
        "layers": list(layers),
        "results": results,
        "per_case": per_case,
        "layer_stats": {str(layer): stats for layer, stats in stats_by_layer.items()},
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """The table in the form of ``research/jspace_probe.md``, plus the conformance statement."""
    conformance = payload.get("conformance", {})
    family = conformance.get("layer_family") or {}
    lines = [
        "# J-space sweep",
        "",
        f"model: `{payload.get('model')}`  policy: `{payload.get('policy')}`  "
        f"cases: {payload.get('cases')}",
        "",
        "## Conformance (R34)",
        "",
        f"- layer index convention: {conformance.get('layer_index_convention')}",
        f"- layers: {conformance.get('layers')} kinds: {conformance.get('layer_kinds')}",
        f"- roles: {family.get('roles')} (primary = in-band, EXP-001 §2)",
        f"- hybrid period: {family.get('hybrid_period')} "
        f"(from {family.get('hybrid_period_source')})",
        f"- kind-matched pairs: {family.get('pairs')} partners: {family.get('partners')}",
        f"- family derived: {family.get('derived')} -- {family.get('reason')}",
        f"- source positions: {conformance.get('source_positions')}",
        f"- output positions read: {conformance.get('output_positions_read')}",
        f"- corpus: {conformance.get('corpus_size')} contexts, {conformance.get('corpus_length')}",
        f"- window: {conformance.get('window')}",
        f"- JVP: {conformance.get('jvp_method')} (from {conformance.get('jvp_method_source')})",
        f"- capture dtype: {conformance.get('capture_dtype')}",
        f"- self-only limiting case: {conformance.get('self_only_limiting_case')}",
        "",
        "## How often is the TRUE suffix more probable than a wrong, previously-seen one?",
        "",
        "| readout | matched | mismatched (null) | matched p | Holm |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, value in payload.get("results", {}).items():
        holm = value.get("matched_p_holm")
        lines.append(
            f"| {key} | {value['matched_wins']}/{value['matched_n']} "
            f"({value['matched_rate']:.0%}) | {value['mismatched_wins']}/"
            f"{value['mismatched_n']} ({value['mismatched_rate']:.0%}) | "
            f"{value['matched_p']:.3f} | {'-' if holm is None else f'{holm:.3f}'} |"
        )
    lines += [
        "",
        "If a readout tracks the task, its matched rate is well above both 50% and its own",
        "mismatched rate. If it only tracks digit frequency, the two columns agree.",
        "",
        f"Comparability (R35): {json.dumps(payload.get('comparability', {}), sort_keys=True)}",
    ]
    return "\n".join(lines)


def main() -> None:  # noqa: C901 - probe CLI orchestration
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline import preflight as preflight_module
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.pipeline.tasks import (
        GENERATOR_VERSION,
        JSPACE_SPLIT_LIMIT,
        JSPACE_SPLIT_NAMES,
        make_jspace_tasks,
        make_tasks,
    )
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu
    from local_llm_lab.probes.policies import (
        resolve_layers,
        resolve_policy,
        validate_layer_syntax,
    )
    from local_llm_lab.probes.state_probe import preflight_precision_block, spec_capture_dtype
    from local_llm_lab.provenance import write_provenance
    from local_llm_lab.runlog import RunLog, git_commit

    parser = argparse.ArgumentParser(
        description=(
            "J-space sweep: paired sign test of the J-lens and the model's own output on "
            "whether a hidden filename survives the observation window."
        )
    )
    parser.add_argument("--model", default="qwen25-coder-3b")
    parser.add_argument(
        "--policy", default="base", help="a policy named by the selected model; default base"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Deprecated explicit adapter-directory alias; omit for the untouched base.",
    )
    parser.add_argument(
        "--layers",
        help=(
            "comma-separated layer indices or fractions, honoured verbatim; omit for the "
            "registry fractions plus their kind-matched partners (EXP-001 §3.5)"
        ),
    )
    parser.add_argument("--corpus-size", type=int, default=16)
    parser.add_argument("--corpus-length", type=int, default=DEFAULT_CORPUS_LENGTH)
    parser.add_argument("--source-positions", default=DEFAULT_SOURCE_POSITIONS)
    parser.add_argument("--count", type=int, default=JSPACE_SPLIT_LIMIT)
    parser.add_argument("--split", default=TASK_SPLIT)
    parser.add_argument("--probe-step", type=int, default=PROBE_STEP)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument(
        "--generator-version",
        type=int,
        default=None,
        help="Generator version the probe points are bound to; defaults to HEAD, recorded.",
    )
    parser.add_argument("--jvp-method", choices=JVP_METHODS, default=None)
    parser.add_argument("--skip-preflight-check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    add_gpu_arguments(parser)
    args = parser.parse_args()

    if args.policy != "base" and args.adapter is not None:
        parser.error("--policy and --adapter cannot be used together")
    spec = load_model_spec(args.model)
    try:
        validate_layer_syntax(args.layers)
        source_positions = resolve_source_positions(args.source_positions)
        adapter = resolve_policy(
            str(args.adapter) if args.adapter is not None else args.policy, spec
        )
    except ValueError as error:
        parser.error(str(error))
    for name, value in (
        ("--corpus-size", args.corpus_size),
        ("--corpus-length", args.corpus_length),
        ("--count", args.count),
    ):
        if value < 1:
            parser.error(f"{name} must be positive")
    if args.probe_step < 0:
        parser.error("--probe-step must be non-negative")
    try:
        jvp_method, jvp_method_source = resolve_jvp_method(args.jvp_method, spec)
    except (JvpMethodUnresolved, ValueError) as error:
        parser.error(str(error))

    # The seed is always recorded, whether it came from the flag or from the generator's own
    # default: an artifact that does not name its seed cannot be replayed (N3, R12/R23).
    data_seed_source = "cli" if args.data_seed is not None else "generator-default"
    data_seed = args.data_seed if args.data_seed is not None else _default_seed()
    generator_version = (
        GENERATOR_VERSION if args.generator_version is None else args.generator_version
    )
    if generator_version != GENERATOR_VERSION:
        parser.error(
            f"--generator-version {generator_version} does not match HEAD's "
            f"{GENERATOR_VERSION}; replaying an older generator is an R12/R23 tool's job, not "
            "this probe's"
        )
    capture_dtype = spec_capture_dtype(spec)

    # R26(e): the log opens once the identity is known, before the preflight gate, the GPU
    # guard and the model load, so run.log alone says what ran and covers the whole run.
    with RunLog.open(
        args.output,
        name="jspace-sweep",
        command=sys.argv,
        identity={
            "model": spec.name,
            "hf_id": spec.hf_id,
            "policy": args.policy if args.adapter is None else None,
            "adapter": str(adapter) if adapter is not None else None,
            "layers": args.layers,
            "split": args.split,
            "count": args.count,
            "corpus_size": args.corpus_size,
            "corpus_length": args.corpus_length,
            "source_positions": args.source_positions,
            "data_seed": data_seed,
            "data_seed_source": data_seed_source,
            "generator_version": generator_version,
            "jvp_method": jvp_method,
            "jvp_method_source": jvp_method_source,
            "capture_dtype": capture_dtype,
            "git_commit": git_commit(),
        },
    ) as log:
        preflight_module.require_preflight(spec, skip=args.skip_preflight_check)
        if args.split in JSPACE_SPLIT_NAMES:
            tasks = make_jspace_tasks(args.split, args.count, data_seed)
        else:
            tasks = make_tasks(args.split, args.count, data_seed)
        ledger = [task for task in tasks if task.family == "ledger_reconcile"]
        log.info("tasks", split=args.split, generated=len(tasks), ledger=len(ledger))

        require_idle_gpu(parser, args, "loading the J-space sweep model")
        log.info("loading policy", model=args.model, policy=args.policy)
        model, tokenizer, view, resolved = load_policy(spec, adapter)
        del model
        try:
            selection = resolve_layers(args.layers, spec, view.num_layers)
        except ValueError as error:
            parser.error(str(error))
        # EXP-001 §3.5: the default sweep is the registry fractions *plus* the kind-matched
        # partners the config's hybrid period implies, so §2's per-kind contrast at
        # comparable depth is guaranteed to be in the run. An explicit --layers is verbatim.
        period, period_source = hybrid_period(view)
        family = kind_matched_layer_family(
            selection.indices,
            num_layers=view.num_layers,
            period=period,
            period_source=period_source,
            kind_of=lambda layer: probe_layer_kind(view, layer),
            derive=selection.source != "cli",
        )
        layers = list(family.layers)
        layer_kinds = dict(family.kinds)
        log.info(
            "layers",
            selected=layers,
            kinds=[layer_kinds[layer] for layer in layers],
            roles=[family.roles[layer] for layer in layers],
            hybrid_period=family.period,
        )

        cases = select_cases(ledger, tokenizer, spec=spec, probe_step=args.probe_step)
        log.info("probe points", usable=len(cases))
        if len(cases) < MIN_CASES:
            parser.error(
                f"only {len(cases)} usable probe point(s); the sign test needs at least "
                f"{MIN_CASES}"
            )

        corpus_ids = build_corpus(
            tokenizer, DEFAULT_CORPUS, size=args.corpus_size, length=args.corpus_length
        )
        lengths = [len(ids) for ids in corpus_ids]
        log.info(
            "corpus",
            contexts=len(corpus_ids),
            median_tokens=statistics.median(lengths) if lengths else 0,
            min_tokens=min(lengths) if lengths else 0,
        )

        try:
            payload = run_sweep(
                view,
                tokenizer,
                cases,
                layers=layers,
                corpus_ids=corpus_ids,
                source_positions=source_positions,
                method=jvp_method,
                capture_dtype=capture_dtype,
                progress=log.progress,
            )
        except EmptyFutureWindowError as error:
            log.error("empty future window", detail=str(error))
            parser.error(str(error))

        first_stats = next(iter(payload["layer_stats"].values()), {})
        conformance = conformance_block(
            layers=layers,
            layer_kinds=layer_kinds,
            layer_selection=selection.as_dict(),
            layer_family=family.as_dict(),
            source_positions=source_positions,
            readouts=READOUTS,
            corpus_size=len(corpus_ids),
            corpus_length={
                "requested_minimum": args.corpus_length,
                "median_tokens": statistics.median(lengths) if lengths else 0,
                "min_tokens": min(lengths) if lengths else 0,
            },
            window=first_stats.get("window", {}),
            jvp_method=jvp_method,
            jvp_method_source=jvp_method_source,
            capture_dtype=_capture_dtype_report(view, capture_dtype),
        )
        comparability = comparability_block(
            model=spec.name,
            hf_id=spec.hf_id,
            policy=args.policy if args.adapter is None else None,
            adapter=None if adapter is None else str(adapter),
            jvp_method=jvp_method,
            template_kwargs=dict(spec.chat.template_kwargs),
            keep_last=None,
            estimator_variant=PRIMARY_READOUT,
            layer_selection=selection.as_dict(),
            layer_kinds=layer_kinds,
            layer_family=family.as_dict(),
            generator_version=generator_version,
            data_seed=data_seed,
            fp32_manual_vs_native=preflight_precision_block(spec),
        )
        payload.update(
            {
                "model": spec.name,
                "hf_id": spec.hf_id,
                "policy": args.policy if args.adapter is None else None,
                "adapter": None if adapter is None else str(adapter),
                "split": args.split,
                "count": args.count,
                "probe_step": args.probe_step,
                "data_seed": data_seed,
                "data_seed_source": data_seed_source,
                "generator_version": generator_version,
                "layer_selection": selection.as_dict(),
                "layer_family": family.as_dict(),
                "layer_kinds": {str(layer): kind for layer, kind in layer_kinds.items()},
                "layer_roles": {str(layer): role for layer, role in family.roles.items()},
                "jvp_method": jvp_method,
                "jvp_method_source": jvp_method_source,
                "capture_dtype": conformance["capture_dtype"],
                "conformance": conformance,
                "comparability": comparability,
            }
        )

        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "sweep.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (args.output / "sweep.md").write_text(
            render_markdown(payload) + "\n", encoding="utf-8"
        )
        log.info("wrote artifact", path=str(args.output / "sweep.json"))
        write_provenance(
            args.output,
            resolved=resolved,
            spec=spec,
            extra={
                "stage": "jspace-sweep",
                "conformance": conformance,
                "comparability": comparability,
            },
        )


def _default_seed() -> int:
    """The generator's own default data seed; never a literal in this module."""
    import inspect

    from local_llm_lab.pipeline.tasks import make_tasks

    return inspect.signature(make_tasks).parameters["seed"].default


def _capture_dtype_report(view: Any, requested: str) -> dict[str, Any]:
    from local_llm_lab.pipeline.jlens import resolve_capture_dtype

    return resolve_capture_dtype(view, requested)


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
