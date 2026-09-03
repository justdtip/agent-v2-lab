"""Jacobian-lens ("J-lens") interpretability probe.

Implements the readout described in Gurnee et al. 2026, "Verbalizable Representations Form a
Global Workspace in Language Models". The standard logit lens reads a mid-network residual
stream ``h`` by pretending it is already the final-layer residual: ``softmax(W_U * norm(h))``.
That is a poor approximation for early/mid layers because the network still has many more
layers of transformation to apply to ``h`` before it reaches the unembedding.

The J-lens instead approximates the *linearised* map the rest of the network applies. For a
layer ``L``, define

    J_L = E_corpus[ d h_final / d h_L ]

the Jacobian of the final residual stream with respect to the layer-``L`` residual stream,
evaluated at a token position and averaged over token positions and a corpus of contexts (the
network is highly non-linear, so this local linearisation is corpus- and position-dependent;
averaging over many contexts gives a single direction-independent summary of "what layer L's
residual space typically maps to"). The lens readout for an activation ``h`` is then

    lens(h) = softmax( W_U . norm( J_L . h ) )

i.e. push ``h`` through the *averaged local linearisation* of the remaining layers, apply the
model's final norm, and unembed. Sorting that distribution gives the tokens the model is
"prepared to verbalise" from that activation once its remaining transformation is accounted
for. The subspace spanned by these readout vectors is called J-space.

Efficiency (the reason this is tractable at all): we never materialise the full ``d x d``
Jacobian ``J_c`` for a context ``c``, let alone the average ``J_L``. We only ever need
``J_L . h`` for one probe vector ``h``, and because expectation and the (locally) linear map
commute,

    E_c[J_c] . h = E_c[J_c . h]

each ``J_c . h`` is exactly a forward-mode Jacobian-vector product (JVP) of the tail of the
network (from layer ``L`` to the final norm) evaluated at context ``c``, with tangent ``h``
placed at the probed token position and zero elsewhere. So one J-lens readout costs one JVP per
corpus context (``N`` forward-mode passes through the tail of the network), not one pass per
dimension of ``h`` as a naive finite-difference or reverse-mode Jacobian materialisation would.
MLX's ``mx.jvp(fn, primals, tangents)`` computes exactly this, in a single forward sweep, for
both arguments and both results as lists.

Numerics: the tail-network JVP is run and accumulated in float32. On the real (4-bit quantized)
checkpoint the tangent has been observed to overflow to infinity in float16, so float32
accumulation throughout (primal, tangent, and the corpus average) is mandatory, not a nicety.

``DEFAULT_CORPUS`` below is a small (24-snippet), hand-written stand-in for the pretraining
distribution the paper averages over. It is not a sample from any real pretraining corpus, so
J-lens estimates computed from it are approximate and corpus-dependent; treat differences
between the J-lens and the logit-lens baseline as suggestive, not as a calibrated probability
of "the model can verbalise this", and prefer changes in relative ranking over absolute
probabilities.

Attention masking: layers are invoked with the same causal mask the model builds itself, via
``create_attention_mask``. Calling transformer blocks directly bypasses
``Qwen2Model.__call__``, which is where that mask is normally constructed; passing ``None``
would give every multi-token prompt bidirectional self-attention, so the residual stream would
encode something the model never computes at inference and every readout taken from it would be
meaningless. ``residual_at`` composed with ``jacobian_vector_product``'s tail is verified to
reproduce the model's own forward pass exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from local_llm_lab.arch import ArchitectureView

__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_MODEL",
    "distribution",
    "encode",
    "jacobian_vector_product",
    "jlens_map",
    "logit_lens",
    "main",
    "probe_layers",
    "readout",
    "residual_at",
    "token_evidence",
]

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"

# A small, hand-written stand-in for a pretraining distribution: plain prose, code, structured
# key=value lines, file paths, and a little dialogue. This is NOT a sample of the model's real
# pretraining data, so J-lens averages over it are an approximation; keep it varied rather than
# representative of any one domain.
DEFAULT_CORPUS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "def add(a, b):\n    return a + b",
    "status=ready\nretries=3\nowner=platform-team",
    '"Are you coming?" she asked, glancing at the clock above the door.',
    "/var/log/app/error-2024-11-03.log",
    "In 1969, astronauts first walked on the surface of the Moon.",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name;",
    "invoice-2-537.txt: total due $412.50, payment terms net 30.",
    "The committee will reconvene next Tuesday to finalise the budget.",
    "import numpy as np\narr = np.zeros((3, 3))",
    "host=10.0.0.14 port=8080 protocol=https timeout=30s",
    "He whispered, 'Not now, someone might hear us,' and stepped back.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "class Node:\n    def __init__(self, value):\n        self.value = value",
    "reports/quarterly/q3-2025-summary.csv",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "git commit -m 'fix off-by-one error in the retry loop'",
    "name=Ada Lovelace\nrole=mathematician\nborn=1815",
    "Despite the storm, the ferry departed only twenty minutes late.",
    "for i in range(10):\n    print(i * i)",
    "config/production.yaml",
    '"I disagree," he said flatly, folding his arms across his chest.',
    "Mount Everest is the tallest mountain above sea level on Earth.",
    "user_id=48213 action=login result=success timestamp=2026-01-04T09:12:00Z",
)


def encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids for ``text`` with no added special tokens, where the tokenizer supports that."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _view(value: Any) -> Any:
    """Return a view, temporarily adapting legacy model callers until Task 6."""
    if callable(getattr(value, "residuals", None)) and callable(getattr(value, "tail", None)):
        return value
    return ArchitectureView.from_model(value)


def residual_at(view: ArchitectureView, token_ids: Any, layer: int) -> Any:
    """Residual stream entering layer ``layer`` for a single context.

    Runs the view's embedding plus the blocks before ``layer`` (an empty prefix for
    ``layer == 0`` returns the raw embedding), each layer called with the model's own causal
    mask so the result matches its true forward pass. Returns an
    ``mx.array`` of shape ``(1, T, d)`` in float32.
    """
    return _view(view).residuals(token_ids, [layer])[layer]


def jacobian_vector_product(
    view: ArchitectureView,
    layer: int,
    primal: Any,
    tangent: Any,
    *,
    method: Literal["forward", "finite_difference"] = "forward",
) -> Any:
    """One forward-mode JVP of the tail of the network: layers ``[layer:]`` then the final norm.

    ``primal`` and ``tangent`` are both ``(1, T, d)``. This is ``J_c . tangent`` for the single
    context that produced ``primal`` (``J_c`` being the local Jacobian of
    ``norm(layers[layer:](.))`` at ``primal``), computed without ever materialising ``J_c``.
    Both primal and tangent are cast to float32 before the JVP (mandatory: the tangent has been
    observed to overflow to infinity in float16 on the real quantized checkpoint), and the
    result is returned in float32.
    """
    import mlx.core as mx

    primal32 = primal.astype(mx.float32)
    tangent32 = tangent.astype(mx.float32)

    fn = _view(view).tail(layer)
    if method == "forward":
        _, (tangent_out,) = mx.jvp(fn, [primal32], [tangent32])
        return tangent_out.astype(mx.float32)
    if method != "finite_difference":
        raise ValueError(f"unknown JVP method {method!r}")
    tangent_norm = mx.sqrt(mx.sum(tangent32 * tangent32))
    if float(tangent_norm.item()) == 0.0:
        return mx.zeros_like(primal32)
    primal_norm = mx.sqrt(mx.sum(primal32 * primal32))
    eps = 1e-2 * primal_norm / tangent_norm
    return ((fn(primal32 + eps * tangent32) - fn(primal32 - eps * tangent32)) / (2.0 * eps)).astype(mx.float32)


def jlens_map(
    view: ArchitectureView,
    layer: int,
    probe: Any,
    corpus_ids: list[Any],
    *,
    position: int = -1,
    method: Literal["forward", "finite_difference"] = "forward",
) -> tuple[Any, dict[str, Any]]:
    """Estimate ``J_layer . probe`` (shape ``(d,)``), averaged over ``corpus_ids``.

    For each context: builds ``primal = residual_at(view, ids, layer)``, a tangent that is
    zero everywhere except ``probe`` placed at token ``position``, and reads off the output
    tangent at that same position via :func:`jacobian_vector_product`. Because
    ``E_c[J_c] . probe == E_c[J_c . probe]``, the mean of these per-context vectors *is*
    ``J_layer . probe`` for the corpus-averaged Jacobian; each JVP costs one tail-network
    forward-mode pass rather than one pass per dimension of ``probe``.

    Any context whose output tangent is not finite (or whose ``position`` does not exist in
    that context) is skipped rather than allowed to poison the mean. The returned statistics
    record ``used``, ``skipped``, and the requested derivative ``method``.

    Raises ``ValueError`` if every context was skipped.
    """
    import mlx.core as mx

    view_call = callable(getattr(view, "residuals", None)) and callable(getattr(view, "tail", None))
    architecture = _view(view)
    probe32 = probe.astype(mx.float32)
    total = mx.zeros_like(probe32)
    used = 0
    skipped = 0
    for ids in corpus_ids:
        primal = residual_at(architecture, token_ids=ids, layer=layer)
        length = primal.shape[1]
        pos = position if position >= 0 else length + position
        if not (0 <= pos < length):
            skipped += 1
            continue
        tangent = mx.zeros_like(primal)
        tangent[0, pos] = probe32
        tangent_out = jacobian_vector_product(architecture, layer, primal, tangent, method=method)
        vector = tangent_out[0, pos]
        if not bool(mx.all(mx.isfinite(vector)).item()):
            skipped += 1
            continue
        total = total + vector
        used += 1
    stats: dict[str, Any] = {"used": used, "skipped": skipped, "method": method}
    if used == 0:
        raise ValueError("jlens_map: every corpus context was skipped (empty or non-finite)")
    mapped = (total / used).astype(mx.float32)
    # Task 6 removes raw-model callers.  The temporary coercion branch keeps their historical
    # mapped-array result while ArchitectureView callers receive the binding tuple contract.
    if not view_call:
        return mapped  # type: ignore[return-value]
    return mapped, stats


def distribution(view: ArchitectureView, vector: Any) -> Any:
    """``softmax(W_U . norm(vector))`` over the vocabulary, in float32; ``vector`` is ``(d,)``."""
    import mlx.core as mx

    v = vector.astype(mx.float32)
    architecture = _view(view)
    return mx.softmax(architecture.unembed(architecture.final_norm(v)).astype(mx.float32), axis=-1)


def _top_k(distribution: Any, tokenizer: Any, k: int) -> list[tuple[str, float, int]]:
    import mlx.core as mx

    order = mx.argsort(-distribution)[:k]
    ids = [int(i) for i in order.tolist()]
    probs = distribution.tolist()
    return [(tokenizer.decode([token_id]), float(probs[token_id]), token_id) for token_id in ids]


def readout(view: ArchitectureView, vector: Any, tokenizer: Any, k: int = 20) -> list[tuple[str, float, int]]:
    """The J-lens readout of ``vector``: ``top_k(softmax(W_U . norm(vector)))``.

    ``vector`` is expected to already be a Jacobian-mapped activation (e.g. the output of
    :func:`jlens_map`); this function itself applies no Jacobian, only the final norm and
    unembedding, so it doubles as the shared machinery behind :func:`logit_lens`. The
    ArchitectureView selects tied or untied unembedding. Returns up to ``k``
    ``(token_string, probability, token_id)`` triples sorted by descending probability.
    """
    return _top_k(distribution(view, vector), tokenizer, k)


def logit_lens(
    view: ArchitectureView, vector: Any, tokenizer: Any, k: int = 20
) -> list[tuple[str, float, int]]:
    """The standard logit-lens baseline: identical machinery to :func:`readout`, but intended
    to be called on a raw (non-Jacobian-mapped) activation. Comparing this against
    :func:`readout` on the same activation, run through :func:`jlens_map` first, is the control
    that shows whether the J-lens adds anything over just norming and unembedding directly.
    """
    return _top_k(distribution(view, vector), tokenizer, k)


def token_evidence(
    distribution: Any, tokenizer: Any, candidates: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """How strongly a full vocabulary ``distribution`` supports each candidate string.

    ``candidates`` maps a label (e.g. ``"target"``, ``"already_read"``) to a literal string.

    Candidates in this experiment are sibling file paths that share almost every token
    (``lab/test/0007/ledger/invoice-2-537.txt`` versus ``...invoice-1-924.txt``), so scoring
    over all tokens measures the shared prefix and returns an identical answer for every
    candidate, which is useless. The distinctive metrics below therefore restrict attention to
    the tokens unique to one candidate, which is what actually discriminates them. Both are
    reported so the difference stays visible. Returns, per label::

        {
            "tokens": [...], "max_probability": ..., "best_rank": ..., "total_probability": ...,
            "distinctive_tokens": [...],        # tokens no other candidate contains
            "distinctive_max_probability": ..., # the metric to read
            "distinctive_best_rank": ...,
            "distinctive_total_probability": ...,
        }
    """
    import mlx.core as mx

    order = mx.argsort(-distribution)
    order_list = [int(i) for i in order.tolist()]
    rank_of = {token_id: rank + 1 for rank, token_id in enumerate(order_list)}
    probs = distribution.tolist()
    encoded = {label: encode(tokenizer, text) for label, text in candidates.items()}
    evidence: dict[str, dict[str, Any]] = {}
    for label, ids in encoded.items():
        shared = {i for other, other_ids in encoded.items() if other != label for i in other_ids}
        distinctive = [token_id for token_id in ids if token_id not in shared]

        def summarize(token_ids: list[int]) -> tuple[float, int | None, float]:
            if not token_ids:
                return 0.0, None, 0.0
            piece_probs = [float(probs[token_id]) for token_id in token_ids]
            return (
                max(piece_probs),
                min(rank_of[token_id] for token_id in token_ids),
                sum(piece_probs),
            )

        all_max, all_rank, all_total = summarize(list(ids))
        dist_max, dist_rank, dist_total = summarize(distinctive)
        evidence[label] = {
            "tokens": [tokenizer.decode([token_id]) for token_id in ids],
            "max_probability": all_max,
            "best_rank": all_rank,
            "total_probability": all_total,
            "distinctive_tokens": [tokenizer.decode([token_id]) for token_id in distinctive],
            "distinctive_max_probability": dist_max,
            "distinctive_best_rank": dist_rank,
            "distinctive_total_probability": dist_total,
        }
    return evidence


def probe_layers(
    view: ArchitectureView,
    tokenizer: Any,
    token_ids: Any,
    layers: list[int],
    corpus_ids: list[Any],
    candidates: dict[str, str],
    position: int = -1,
    *,
    k: int = 20,
) -> list[dict[str, Any]]:
    """Run the full J-lens vs. logit-lens comparison at each layer in ``layers``.

    For every layer: takes the layer's residual stream at ``token_ids`` (position
    ``position``) as the probe, estimates ``J_layer . probe`` with :func:`jlens_map` over
    ``corpus_ids``, and computes both the J-lens distribution (from that estimate) and the
    logit-lens distribution (from the raw probe). Returns one record per layer, in the order
    ``layers`` was given, with the top-``k`` tokens and :func:`token_evidence` for both lenses.
    """
    architecture = _view(view)
    records = []
    for layer in layers:
        residual = residual_at(architecture, token_ids, layer)
        length = residual.shape[1]
        pos = position if position >= 0 else length + position
        probe = residual[0, pos]
        jmap, stats = jlens_map(architecture, layer, probe, corpus_ids, position=position)
        jlens_distribution = distribution(architecture, jmap)
        logit_distribution = distribution(architecture, probe)
        records.append(
            {
                "layer": layer,
                "corpus_used": stats.get("used", 0),
                "corpus_skipped": stats.get("skipped", 0),
                "jlens_top_k": _top_k(jlens_distribution, tokenizer, k),
                "logit_lens_top_k": _top_k(logit_distribution, tokenizer, k),
                "jlens_evidence": token_evidence(jlens_distribution, tokenizer, candidates),
                "logit_lens_evidence": token_evidence(logit_distribution, tokenizer, candidates),
            }
        )
    return records


# --------------------------------------------------------------------------- CLI


def _replay_to_step(task: Any, step: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Replay ``task``'s EXPERT trajectory through a fresh :class:`Simulator` up to (not
    including) ``step``, returning the exact message list ``build_prompt`` would see just
    before the policy generates that step, and the paths read by earlier ``read_file`` steps.

    Deliberately replays the recorded expert actions rather than generating: this makes the
    probe's context, and therefore its result, independent of any policy's actual behaviour.
    """
    from local_llm_lab.pipeline.env import Simulator
    from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT, assistant_message, tool_message

    if not 0 <= step < len(task.steps):
        raise ValueError(f"--step {step} out of range: task horizon is {len(task.steps)}")
    simulator = Simulator.for_task(task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    read_paths: list[str] = []
    for index, expert_step in enumerate(task.steps):
        if index == step:
            break
        observation = simulator.execute(expert_step.action)
        if expert_step.action.name == "read_file":
            read_paths.append(str(expert_step.action.arguments["path"]))
        messages.append(assistant_message(expert_step.thought, expert_step.action))
        messages.append(tool_message(expert_step.action.name, observation))
    return messages, read_paths


def _unseen_path(split: str, seed: int, task_index: int, exclude: dict[str, str]) -> str:
    """A file path drawn from a different task, guaranteed absent from ``exclude``'s files."""
    from local_llm_lab.pipeline.tasks import make_tasks

    pool = make_tasks(split, max(task_index + 8, 8), seed)
    for index, other in enumerate(pool):
        if index == task_index:
            continue
        for path in other.files:
            if path not in exclude:
                return path
    raise RuntimeError("could not find a path from another task that is unseen by this one")


def _print_table(records: list[dict[str, Any]], candidates: dict[str, str]) -> None:
    for record in records:
        print(f"\n== layer {record['layer']} " + "=" * 40)
        print(f"corpus contexts used={record['corpus_used']} skipped={record['corpus_skipped']}")
        for lens_name, top_key, evidence_key in (
            ("logit-lens", "logit_lens_top_k", "logit_lens_evidence"),
            ("j-lens", "jlens_top_k", "jlens_evidence"),
        ):
            top = ", ".join(f"{tok!r}:{prob:.3f}" for tok, prob, _ in record[top_key][:8])
            print(f"  {lens_name} top tokens: {top}")
            for label in candidates:
                evidence = record[evidence_key][label]
                print(
                    f"    {label:12s} distinctive={evidence['distinctive_tokens']} "
                    f"rank={evidence['distinctive_best_rank']} "
                    f"max_p={evidence['distinctive_max_probability']:.6f} "
                    f"(all-token rank={evidence['best_rank']})"
                )


def main() -> None:
    from local_llm_lab.pipeline.protocol import build_prompt
    from local_llm_lab.pipeline.tasks import make_tasks
    from local_llm_lab.probes.guard import add_gpu_arguments, require_idle_gpu
    from local_llm_lab.project import configure_local_cache

    parser = argparse.ArgumentParser(
        description=(
            "J-lens interpretability probe: compare the Jacobian-lens and logit-lens readouts "
            "of a model's residual stream at a chosen step of an expert agent trajectory."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter", type=Path, help="Adapter directory; omit for the untouched base."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Which decision point in the expert trajectory to probe.",
    )
    parser.add_argument(
        "--force-prefix",
        help=(
            "Text appended to the prompt before probing, e.g. the note plus a partial tool "
            "call. Use it to probe the position where the model is actually due to emit the "
            "candidate, rather than the start of its note."
        ),
    )
    parser.add_argument("--layers", default="6,12,18,24,30")
    parser.add_argument("--corpus-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    add_gpu_arguments(parser)
    args = parser.parse_args()

    try:
        layers = [int(part) for part in args.layers.split(",") if part.strip()]
    except ValueError:
        parser.error("--layers must be a comma-separated list of integers")
    if not layers:
        parser.error("--layers must list at least one layer index")
    if args.corpus_size < 1:
        parser.error("--corpus-size must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    seed = 20260902
    tasks = make_tasks(args.split, max(args.task_index + 1, 1), seed)
    if args.task_index >= len(tasks):
        parser.error(f"--task-index {args.task_index} out of range for split {args.split!r}")
    task = tasks[args.task_index]
    try:
        messages, read_paths = _replay_to_step(task, args.step)
    except ValueError as error:
        parser.error(str(error))
        return

    target_step = task.steps[args.step]
    if target_step.action.name != "read_file":
        parser.error(
            f"--step {args.step} is a {target_step.action.name!r} call, not read_file; "
            "choose a step whose expert action reads a file"
        )
    target_path = str(target_step.action.arguments["path"])
    already_read = next((path for path in reversed(read_paths) if path != target_path), None)
    if already_read is None:
        already_read = next(
            (path for path in sorted(task.files) if path != target_path), target_path
        )
    unseen_path = _unseen_path(args.split, seed, args.task_index, task.files)
    candidates = {"target": target_path, "already_read": already_read, "unseen": unseen_path}

    require_idle_gpu(parser, args, "loading the J-lens model")
    configure_local_cache()
    from mlx_lm import load

    model, tokenizer = load(
        args.model, adapter_path=None if args.adapter is None else str(args.adapter.resolve())
    )

    prompt = build_prompt(tokenizer, messages)
    if args.force_prefix:
        prompt = prompt + args.force_prefix
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not prompt.startswith(bos)
    try:
        token_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    except TypeError:
        token_ids = tokenizer.encode(prompt)

    view = ArchitectureView.from_model(model)
    corpus = DEFAULT_CORPUS[: args.corpus_size]
    corpus_ids = [encode(tokenizer, text) for text in corpus]

    records = probe_layers(
        view, tokenizer, token_ids, layers, corpus_ids, candidates, k=args.top_k
    )

    print(f"task={task.task_id} step={args.step} layers={layers} corpus_size={len(corpus_ids)}")
    print(f"candidates={candidates}")
    _print_table(records, candidates)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "adapter": None if args.adapter is None else str(args.adapter.resolve()),
            "split": args.split,
            "task_id": task.task_id,
            "step": args.step,
            "layers": layers,
            "corpus_size": len(corpus_ids),
            "candidates": candidates,
            "records": records,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\nWrote {args.output}")
