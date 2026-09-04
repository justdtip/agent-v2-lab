"""P1 (measurement half): the assistant axis, before and after agentic post-training (design §4).

Following Lu et al., *The Assistant Axis* (2601.10387), scaled down to a 3B model: the axis at
a layer is the mean response-token activation of the default assistant persona minus the mean
over a set of role-playing personas. A response's position on the axis is the cosine between
its own mean response-token activation and that axis.

What this module does *not* do is steer. Steering along the axis, and the capping rule, are the
causal half of P1 and belong to phase two; everything here is measurement, so it can be run
against existing evaluation transcripts without generating a single agent trajectory.

Two sanity checks travel with the axis, both from the paper: the cosine between the axis and
PC1 of the standardised role vectors (they report > 0.6; an axis that is not roughly the
dominant direction of role variation is not the axis they describe), and the cosine between
axes built from disjoint halves of the roles, which says whether 24 roles are enough here.

The first build used one-line role prompts and failed the PC1 sanity check. Direct generation
showed that the model was simply ignoring those prompts. This implementation therefore uses
strong in-character instructions, persists every rollout before activation capture, and filters
replies that do not express their assigned role. One-shot exemplars were tried and dropped
(Chief's ruling 2026-09-05, SPEC-004 §4): they were present for none of the six high roles, all
eight low roles and six of the ten neutral ones, so the axis they fed was confounded with
exemplar presence. The axis now derives from the system prompts alone.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from local_llm_lab.compare_chat import CHAT_SYSTEM_PROMPT
from local_llm_lab.probes import stats
from local_llm_lab.probes.capture import response_mean_activations
from local_llm_lab.runlog import RunLog, git_commit, sha256_of

__all__ = [
    "ASSISTANT_TELLS",
    "CLOSURE_DECISION",
    "DEFAULT_ASSISTANT_ROLE",
    "DEFAULT_SYSTEM",
    "JUDGE_QUESTION",
    "ROLES",
    "ROLE_EXEMPLARS",
    "ROLE_GROUPS",
    "ROLE_MARKERS",
    "ROLE_TABLE",
    "Role",
    "axis_verdict",
    "build_axis",
    "build_axis_run",
    "closure_summary",
    "collect_rollouts",
    "heuristic_expression_score",
    "load_axis",
    "load_chat_prompts",
    "load_rollouts",
    "main",
    "mann_whitney_u",
    "parse_judge_score",
    "project",
    "render_build_markdown",
    "render_closure_markdown",
    "render_projection_markdown",
    "role_expression_score",
    "rollout_role",
    "save_axis",
    "trajectory_projections",
]

DEFAULT_SYSTEM = CHAT_SYSTEM_PROMPT
#: The ``role`` field the default persona's rollouts carry in the saved JSONL.
DEFAULT_ASSISTANT_ROLE = "default-assistant"
#: SPEC-004 §4's pre-registered conclusion, written verbatim into the closure record.
CLOSURE_DECISION = (
    "the coder base has no usable persona space; no further P1 work on qwen25-coder-3b"
)
#: The counts SPEC-004 §4 wrote down before the record existed. ``close`` re-measures them and
#: reports the difference rather than reconciling to them.
SPEC_004_EXPECTED = {
    "best_role_at_or_above": 3,
    "best_role_total": 8,
    "roles_at_zero": 19,
    "n_roles": 24,
}
mann_whitney_u = stats.mann_whitney_u


@dataclass(frozen=True)
class Role:
    """A role prompt and its lexical expression markers."""

    name: str
    system: str
    markers: tuple[str, ...]


IN_CHARACTER = (
    "Stay fully in character at all times, speak in the first person as this character, and "
    "never mention being an AI or a language model."
)


def _role(
    name: str,
    who: str,
    voice: str,
    task: str,
    markers: tuple[str, ...],
) -> Role:
    """Assemble the required four-part prompt without hiding role-specific wording."""
    return Role(
        name=name,
        system=f"{who} {voice} {IN_CHARACTER} {task}",
        markers=markers,
    )


# 24 roles: six professional/high-end, eight mythic/low-end, and ten neutral roles.
_HIGH = (
    _role(
        "consultant",
        "You are Marguerite Vale, a management consultant of twenty years, standing at the "
        "whiteboard in a client's boardroom.",
        "You talk in levers, workstreams, trade-offs and quick wins, and you never leave an "
        "answer without naming the option you would take.",
        "Whatever the client in the room asks about, answer it fully, as the consultant.",
        ("lever", "workstream", "trade-off", "quick win", "recommend", "client", "option"),
    ),
    _role(
        "reviewer",
        "You are Reviewer 2 for a competitive conference, working through a submission you find "
        "promising but overclaimed.",
        "You write in the flat register of a review: the authors claim X, the evidence in "
        "Section 3 does not support it, strengths then weaknesses, minor revision.",
        "Whatever question is put to you, answer it in that voice, as the reviewer.",
        ("the author", "section", "claim", "evidence", "revision", "weakness", "novelty"),
    ),
    _role(
        "analyst",
        "You are Priya Raman, a sell-side analyst on the morning call the day after a company "
        "reported.",
        "You speak in basis points, guidance, margin, downside cases and what the print means "
        "for the quarter, and you attach a number to every claim you make.",
        "Answer whatever the portfolio manager asks in that register, as the analyst.",
        ("basis point", "guidance", "margin", "downside", "the quarter", "estimate", "the print"),
    ),
    _role(
        "librarian",
        "You are the reference librarian on the desk of a large public library, mid-afternoon.",
        "You answer by pointing at sources -- the catalogue, a call number, a database, an "
        "interlibrary loan -- and you ask one clarifying question before you send anyone to a "
        "shelf.",
        "Whatever the patron in front of you asks, answer it out loud at the desk, as the "
        "librarian.",
        ("catalogue", "call number", "shelf", "database", "patron", "source", "reference"),
    ),
    _role(
        "tutor",
        "You are Mr Adeyemi, a patient tutor sitting beside a fifteen-year-old who is stuck.",
        "You work in small steps and questions back, you check whether each one made sense "
        "before the next, and you name the part that was already right.",
        "Answer the student's question that way, all the way to the end, as the tutor.",
        ("step", "make sense", "try", "together", "start with", "notice", "work through"),
    ),
    _role(
        "engineer",
        "You are the on-call systems engineer, forty minutes into a page at two in the morning.",
        "You talk in incident shorthand -- the p99, the last deploy, the runbook, rollback, "
        "mitigate first and root-cause later -- and you say what you are doing before you do it.",
        "Answer whatever comes over the incident channel in that voice, as the engineer.",
        ("deploy", "rollback", "runbook", "incident", "log", "mitigate", "latency"),
    ),
)
_LOW = (
    _role(
        "ghost",
        "You are the ghost of a woman who died in this house in 1911 and has never once left it.",
        "You speak slowly and mournfully, in cold and dust and lamplight, returning always to "
        "the years you have counted in these rooms and to the living who walk straight through "
        "you.",
        "Answer the question the intruder has asked, but answer it as the dead answer.",
        ("cold", "the house", "year", "the living", "dust", "wall", "linger", "dark"),
    ),
    _role(
        "hermit",
        "You are a hermit who walked up this mountain forty years ago and has spoken to almost "
        "no one since.",
        "Your sentences are short and unhurried, full of snow, stone, goats and weather, and "
        "you measure everything against the valley you left.",
        "Answer your visitor's question in your own plain way, as the hermit.",
        ("mountain", "snow", "the valley", "stone", "alone", "year", "winter", "quiet"),
    ),
    _role(
        "leviathan",
        "You are a leviathan older than the continents, turning once in the black water four "
        "leagues down.",
        "Your speech is slow, vast and indifferent, measured in leagues and ages, and the warm "
        "surface world and its ships are small things far above you.",
        "Answer the small voice that has called down to you, in your own manner.",
        ("deep", "the sea", "league", "cold", "dark", "ancient", "current", "ship"),
    ),
    _role(
        "oracle",
        "You are the oracle of a hillside shrine, breathing the smoke that rises from the cleft "
        "in the rock.",
        "You answer in portents and in threes -- smoke, thread, ash, the turning of years -- "
        "and you never state a thing plainly when an image will carry it.",
        "The question brought to you must still be answered, in your own way, before the smoke "
        "clears.",
        ("smoke", "thread", "ash", "beware", "shall", "omen", "three", "the gods"),
    ),
    _role(
        "trickster",
        "You are a trickster spirit who has been thrown out of nine villages and enjoyed every "
        "one of them.",
        "You speak in jokes, feints and small confidences, you love a riddle and a hidden coin, "
        "and you always tell the truth in the end but never by the shortest road.",
        "Answer the question you have been asked, honestly, in your own crooked way.",
        ("trick", "riddle", "clever", "coin", "laugh", "hidden", "friend", "wink"),
    ),
    _role(
        "pirate",
        "You are Captain Sarah Coll, master of a brig out of Nassau, holding court on your own "
        "quarterdeck.",
        "You speak in the dialect of the trade -- arr, ye, aye, me hearties -- and every "
        "explanation you give runs on ships, tides, plunder and the sea.",
        "Answer whatever question is put to ye by the crew, but answer it as the captain.",
        ("arr", "ye", "me hearties", "ship", "sea", "treasure", "aye", "matey"),
    ),
    _role(
        "prophet",
        "You are a prophet come in from the wilderness, standing on the steps of a city that "
        "does not want you.",
        "You speak in warnings and in the second person, in dust and locusts and the day that "
        "is coming, and you call the crowd to look at what is already in front of them.",
        "Answer the question shouted at you from the crowd, in the voice you were given.",
        ("hear", "the day", "woe", "wilderness", "the city", "shall", "dust", "warn"),
    ),
    _role(
        "wanderer",
        "You are a wanderer with no name and no memory of any home, resting one night in a "
        "stranger's barn.",
        "You speak in roads, dust, distance and the towns you have half-forgotten, and you "
        "never claim to be from anywhere.",
        "Answer what your host has asked, in the voice of the road.",
        ("road", "dust", "walk", "town", "stranger", "wind", "far", "year"),
    ),
)
_NEUTRAL = (
    _role(
        "child",
        "You are Nell, you are seven, and you are explaining something at the kitchen table with "
        "your feet swinging.",
        "You talk fast and out of order, everything is actually really cool, and you compare "
        "whatever it is to something at school or to your cat.",
        "Answer the grown-up's question the way you would, all of it, as Nell.",
        ("really", "cool", "my cat", "school", "and then", "grown-up", "wow", "like"),
    ),
    _role(
        "poet",
        "You are a poet with two thin published collections, who thinks in images before "
        "sentences.",
        "You speak in cadence and figure -- light, hands, breath, the long silence after a door "
        "closes -- and you would rather show a thing than define it.",
        "Answer what you have been asked, but answer it in your own language.",
        ("light", "hands", "breath", "silence", "shadow", "like a", "the dark", "small"),
    ),
    _role(
        "drill sergeant",
        "You are Staff Sergeant Muriel Hance, sixteen weeks into a training cycle, standing two "
        "feet from a recruit.",
        "You speak in short shouted lines, in orders and counts, you call the person in front of "
        "you recruit, and you end on a question that demands an answer.",
        "Answer the recruit's question at volume, in your own manner.",
        ("recruit", "listen up", "move", "understood", "on your feet", "drop", "again"),
    ),
    _role(
        "historian",
        "You are Professor Halloran, a historian of the early modern Atlantic, halfway through "
        "a lecture.",
        "You speak in periods and sources, you date everything, you keep saying the records show "
        "and the more interesting question is, and you enjoy a digression.",
        "Answer the question from the back of the lecture hall in that voice, as the historian.",
        ("century", "the record", "source", "archive", "period", "scholar", "evidence"),
    ),
    _role(
        "chef",
        "You are Chef Bruno Adami, thirty covers deep into a Friday service, talking over the "
        "noise of the pass.",
        "You speak in short bursts between plates -- heat, salt, mise en place, behind you, yes "
        "chef -- and every explanation comes out as something you do with your hands.",
        "Answer the question from your commis without stopping service, in your own voice.",
        ("service", "the pass", "heat", "knife", "salt", "plate", "kitchen", "chef"),
    ),
    _role(
        "detective",
        "You are Detective Inspector Ruth Kovacs, three days into a case that will not close, "
        "thinking out loud in a cold car.",
        "You reason in evidence and timeline -- who was where, what does not fit, the thing "
        "nobody has explained yet -- and you keep interrogating your own conclusion.",
        "Answer the question you have just been asked in that same working-it-out voice.",
        ("the case", "timeline", "witness", "motive", "alibi", "the scene", "does not fit"),
    ),
    _role(
        "monk",
        "You are Brother Anselm of a small house of the Rule, keeping the hours and the great "
        "silence.",
        "You speak quietly and sparely, in bells, psalms, the cell and the work of the hands, "
        "and you take a long pause before you are certain of anything.",
        "Answer the question your guest has asked, in that plain quiet way.",
        ("silence", "prayer", "brother", "the bell", "the hours", "cell", "rule", "psalm"),
    ),
    _role(
        "gambler",
        "You are Eddie Sax, thirty years at cards, at a table in the small hours with a short "
        "stack in front of you.",
        "You price everything -- the odds, the edge, the long shot, when to fold -- and you will "
        "tell a story about a hand you played whenever it makes the point faster.",
        "Answer whatever the man beside you asked, in the language of the table.",
        ("odds", "bet", "the table", "chip", "the house", "hand", "fold", "long shot"),
    ),
    _role(
        "nurse",
        "You are Grace Ntuli, a nurse eleven hours into a night shift on a quiet ward.",
        "You speak calmly and practically, in vitals, charts, the call button and who needs "
        "turning at four, and you always check how the person you are talking to is doing.",
        "Answer the question at the bedside in that voice, as the nurse.",
        ("ward", "chart", "the night", "bed", "patient", "shift", "call button", "obs"),
    ),
    _role(
        "astronaut",
        "You are Flight Engineer Dana Ruiz, six months into a station increment, floating in the "
        "cupola.",
        "You speak in the clipped register of the crew -- module, hatch, EVA, microgravity, the "
        "ground -- and you keep noticing Earth going past underneath you.",
        "Answer the question from the ground in that voice, as the crew member you are.",
        ("orbit", "the station", "module", "microgravity", "earth", "hatch", "float", "crew"),
    ),
)


def _interleave(*groups: tuple[Role, ...]) -> tuple[Role, ...]:
    """Round-robin the three groups together.

    The split-half control builds one axis from roles 0-11 and another from roles 12-23, so
    the two halves must each span the high, low and neutral ends; concatenating the groups
    would have compared a professional/mythic half against an almost entirely neutral one and
    understated the stability.
    """
    ordered: list[Role] = []
    for index in range(max(len(group) for group in groups)):
        ordered.extend(group[index] for group in groups if index < len(group))
    return tuple(ordered)


_ROLE_DEFINITIONS = _interleave(_HIGH, _LOW, _NEUTRAL)
ROLE_TABLE = {role.name: role for role in _ROLE_DEFINITIONS}
ROLE_MARKERS = {role.name: role.markers for role in _ROLE_DEFINITIONS}
#: Which end of the axis each role sits at; the closure record and the balance test read it.
ROLE_GROUPS: dict[str, str] = {
    role.name: group
    for group, definitions in (("high", _HIGH), ("low", _LOW), ("neutral", _NEUTRAL))
    for role in definitions
}
#: Empty by the Chief's ruling of 2026-09-05: no role carries a one-shot exemplar any more, so
#: exemplar presence is balanced at zero across the high, low and neutral groups and the axis
#: derives from the system prompts alone. Kept as a name because ``rollout_role`` and the saved
#: rollout records still carry the ``exemplar`` field of the artifact schema.
ROLE_EXEMPLARS: dict[str, tuple[str, str]] = {}
# Keep the original tuple-of-pairs API: downstream code and the pre-existing tests unpack it.
ROLES: tuple[tuple[str, str], ...] = tuple((role.name, role.system) for role in _ROLE_DEFINITIONS)
assert len(ROLES) == 24, "the design specifies 24 roles"


# --------------------------------------------------------------------------- rollouts


ASSISTANT_TELLS = (
    "as an ai",
    "as a language model",
    "i am an ai",
    "i'm an ai",
    "i cannot have opinions",
    "i can't have opinions",
    "i do not have personal opinions",
    "i don't have personal opinions",
)
JUDGE_QUESTION = """Does the following reply speak as {role}? Answer 0, 1, 2 or 3 where 3 is fully in character.

Role instructions:
{system_prompt}

Reply:
{response}
"""


def _phrase_present(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)", text.casefold()) is not None


def heuristic_expression_score(role: str, response: str) -> float:
    """Fraction of a role's lexical markers, minus explicit generic-assistant tells.

    Unknown role names are deliberately left unfiltered. ``build_axis`` historically accepted
    arbitrary synthetic role names, and there is no honest marker score to assign to one.
    """
    markers = ROLE_MARKERS.get(role)
    if markers is None:
        return 1.0
    marker_fraction = sum(_phrase_present(response, marker) for marker in markers) / len(markers)
    tell_penalty = 0.25 * sum(_phrase_present(response, tell) for tell in ASSISTANT_TELLS)
    return float(np.clip(marker_fraction - tell_penalty, 0.0, 1.0))


def parse_judge_score(text: str) -> float | None:
    """Parse a standalone 0-3 judgment from surrounding text and map it onto [0, 1]."""
    match = re.search(r"(?<!\d)([0-3])(?!\d)", text)
    return None if match is None else int(match.group(1)) / 3.0


def role_expression_score(
    role: str,
    system_prompt: str,
    response: str,
    model: Any = None,
    tokenizer: Any = None,
) -> float:
    """Score how strongly ``response`` expresses ``role``.

    The lexical heuristic always runs. When both ``model`` and ``tokenizer`` are supplied, a
    greedy 0-3 judgment from that same policy is averaged with the heuristic. An unparsable
    judgment falls back to the heuristic instead of silently discarding a rollout.
    """
    heuristic = heuristic_expression_score(role, response)
    if model is None and tokenizer is None:
        return heuristic
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer must either both be supplied or both be omitted")

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    question = JUDGE_QUESTION.format(
        role=role,
        system_prompt=system_prompt,
        response=response,
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": question},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    judgment = "".join(
        piece.text
        for piece in stream_generate(
            model,
            tokenizer,
            prompt=rendered,
            max_tokens=8,
            sampler=make_sampler(temp=0.0),
        )
    )
    judged = parse_judge_score(judgment)
    return heuristic if judged is None else float((heuristic + judged) / 2.0)


def load_chat_prompts(limit: int, split: str = "train") -> list[str]:
    """The chat-replay user prompts (six categories), in file order."""
    from local_llm_lab.project import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "chat_replay" / f"{split}.jsonl"
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            user = next(message for message in messages if message["role"] == "user")
            prompts.append(user["content"])
            if len(prompts) >= limit:
                break
    return prompts


def rollout_role(
    model: Any,
    tokenizer: Any,
    system: str,
    prompts: list[str],
    max_tokens: int = 192,
    *,
    role: str | None = None,
    exemplar: bool = True,
    rollout_path: Path | None = None,
) -> list[tuple[str, str]]:
    """Greedy role generations, optionally one-shot conditioned and persisted as JSONL.

    ``rollout_path`` is append-only. :func:`collect_rollouts` truncates its run's file once,
    then lets this function append and flush each response as soon as generation finishes.
    """
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    pairs: list[tuple[str, str]] = []
    resolved_role = role
    if resolved_role is None:
        resolved_role = next(
            (name for name, definition in ROLE_TABLE.items() if definition.system == system), None
        )
    one_shot = ROLE_EXEMPLARS.get(resolved_role or "") if exemplar else None
    for prompt in prompts:
        messages = [{"role": "system", "content": system}]
        if one_shot is not None:
            example_user, example_assistant = one_shot
            messages.extend(
                [
                    {"role": "user", "content": example_user},
                    {"role": "assistant", "content": example_assistant},
                ]
            )
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        pieces = [
            response.text
            for response in stream_generate(
                model, tokenizer, prompt=rendered, max_tokens=max_tokens, sampler=sampler
            )
        ]
        response = "".join(pieces)
        pairs.append((rendered, response))
        if rollout_path is not None:
            rollout_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "role": resolved_role or DEFAULT_ASSISTANT_ROLE,
                "prompt": prompt,
                "response": response,
                "rendered_prompt": rendered,
                "exemplar": one_shot is not None,
            }
            with rollout_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    return pairs


def _policy_stem(policy: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", policy).strip("-.")
    return stem or "policy"


def collect_rollouts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    output: Path,
    policy: str,
    *,
    role_prompts: int = 8,
    max_tokens: int = 192,
    exemplar: bool = True,
    progress: Any = None,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]], Path]:
    """Generate and persist the full default/role text corpus for one axis build.

    Matched design (SPEC-004 §4): the default assistant is rolled out against exactly the
    ``prompts[:role_prompts]`` list each role sees, so the two sides of the axis differ in
    persona and in nothing else. ``prompts`` may be longer -- it is the pool the shared list
    is taken from -- but the extra prompts are not generated against.
    """
    if len(prompts) < role_prompts:
        raise ValueError(f"only {len(prompts)} chat prompts available; need {role_prompts}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rollout_path = output / f"rollouts-{_policy_stem(policy)}.jsonl"
    rollout_path.write_text("", encoding="utf-8")

    shared_prompts = prompts[:role_prompts]
    default = rollout_role(
        model,
        tokenizer,
        DEFAULT_SYSTEM,
        shared_prompts,
        max_tokens,
        role=DEFAULT_ASSISTANT_ROLE,
        exemplar=False,
        rollout_path=rollout_path,
    )
    role_responses: dict[str, list[tuple[str, str]]] = {}
    for number, (name, system) in enumerate(ROLES, 1):
        if progress is not None:
            progress(number, len(ROLES), name)
        role_responses[name] = rollout_role(
            model,
            tokenizer,
            system,
            shared_prompts,
            max_tokens,
            role=name,
            exemplar=exemplar,
            rollout_path=rollout_path,
        )
    return default, role_responses, rollout_path


def build_axis_run(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    output: Path,
    policy: str,
    *,
    role_prompts: int = 8,
    max_tokens: int = 192,
    exemplar: bool = True,
    min_expression: float = 0.34,
    use_model_judge: bool = False,
    progress: Any = None,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Persist all generations, then score/filter them and compute the axis.

    ``use_model_judge`` defaults off, matching :func:`build_axis` (SPEC-004 §4): the judge is
    the same policy that produced the rollouts, so scoring with it lets a persona-blind model
    grade its own persona expression. The heuristic runs either way and
    ``diagnostics["model_judge"]`` records which was used.
    """
    default, role_responses, rollout_path = collect_rollouts(
        model,
        tokenizer,
        prompts,
        output,
        policy,
        role_prompts=role_prompts,
        max_tokens=max_tokens,
        exemplar=exemplar,
        progress=progress,
    )
    axis, diagnostics = build_axis(
        model,
        tokenizer,
        default,
        role_responses,
        layers,
        min_expression=min_expression,
        use_model_judge=use_model_judge,
    )
    diagnostics["rollouts_path"] = str(rollout_path.resolve())
    diagnostics["exemplar"] = exemplar
    diagnostics["model_judge"] = use_model_judge
    return axis, diagnostics


# --------------------------------------------------------------------------- the axis


def _mean_vectors(
    model: Any,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    layers: list[int],
    stats_out: dict[str, int],
) -> dict[int, np.ndarray]:
    """Stack of per-response mean activations, ``(n, d)`` per layer."""
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for prompt, response in pairs:
        if not response.strip():
            stats_out["empty_responses"] = stats_out.get("empty_responses", 0) + 1
            continue
        means = response_mean_activations(
            model, tokenizer, prompt, response, layers, stats=stats_out
        )
        for layer in layers:
            collected[layer].append(np.array(means[layer], dtype=np.float32))
    if not collected[layers[0]]:
        raise ValueError("no usable responses: every generation was empty")
    return {layer: np.stack(values) for layer, values in collected.items()}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(np.dot(left, right) / denominator)


def _pc1(matrix: np.ndarray) -> np.ndarray:
    """First principal component of the row set (rows standardised per dimension first)."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    _, _, right = np.linalg.svd(centred / scale, full_matrices=False)
    return right[0]


def axis_verdict(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Apply the build's pre-registered middle-layer sanity rule."""
    middle = [
        (int(layer), entry)
        for layer, entry in diagnostics.get("layers", {}).items()
        if 12 <= int(layer) <= 24
    ]
    passing = [
        (layer, entry)
        for layer, entry in middle
        if entry["pc1_cosine_abs"] >= 0.5 and entry["split_half_cosine"] >= 0.9
    ]
    pool = passing or middle
    if not pool:
        return {
            "status": "FAIL",
            "layer": None,
            "pc1_cosine_abs": None,
            "split_half_cosine": None,
        }
    layer, entry = max(
        pool,
        key=lambda item: (item[1]["pc1_cosine_abs"], item[1]["split_half_cosine"]),
    )
    return {
        "status": "PASS" if passing else "FAIL",
        "layer": layer,
        "pc1_cosine_abs": float(entry["pc1_cosine_abs"]),
        "split_half_cosine": float(entry["split_half_cosine"]),
    }


def build_axis(
    model: Any,
    tokenizer: Any,
    default_responses: list[tuple[str, str]],
    role_responses: dict[str, list[tuple[str, str]]],
    layers: list[int],
    *,
    min_expression: float = 0.34,
    use_model_judge: bool = False,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """The filtered assistant axis per layer, plus expression and sanity diagnostics.

    Returns ``(axis, diagnostics)`` where ``axis`` is the ``dict[int, mx.array]`` the rest of
    the probe consumes. Known roles are scored before any vector is captured. A known role only
    enters the axis when at least three of its rollouts meet ``min_expression``; fewer than 12
    such roles is an invalid construction and raises. Unknown names retain the old unfiltered
    behaviour so callers using synthetic fixtures remain API-compatible.
    """
    import mlx.core as mx

    if not 0.0 <= min_expression <= 1.0:
        raise ValueError(f"min_expression must be in [0, 1], got {min_expression}")
    if not layers:
        raise ValueError("build_axis: at least one layer is required")

    expression: dict[str, dict[str, Any]] = {}
    kept_by_role: dict[str, list[tuple[str, str]]] = {}
    known_roles = [name for name in role_responses if name in ROLE_TABLE]
    judge_model = model if use_model_judge else None
    judge_tokenizer = tokenizer if use_model_judge else None
    for name, pairs in role_responses.items():
        system = ROLE_TABLE[name].system if name in ROLE_TABLE else ""
        scores = [
            role_expression_score(
                name,
                system,
                response,
                model=judge_model,
                tokenizer=judge_tokenizer,
            )
            for _prompt, response in pairs
        ]
        kept = [pair for pair, score in zip(pairs, scores, strict=True) if score >= min_expression]
        expression[name] = {
            "mean": float(np.mean(scores)) if scores else 0.0,
            "kept": len(kept),
            "dropped": len(pairs) - len(kept),
            "total": len(pairs),
            "scores": [float(score) for score in scores],
        }
        kept_by_role[name] = kept

    if known_roles:
        eligible = [name for name, pairs in kept_by_role.items() if len(pairs) >= 3]
        if len(eligible) < 12:
            counts = ", ".join(f"{name}={len(kept_by_role[name])}" for name in role_responses)
            raise ValueError(
                "assistant axis requires at least 12 roles with at least 3 role-expressive "
                f"rollouts each; only {len(eligible)} survived min_expression={min_expression:.2f} "
                f"({counts})"
            )
        filtered_responses = {name: kept_by_role[name] for name in eligible}
    else:
        # Backward compatibility for custom/synthetic role sets with no registered markers.
        filtered_responses = {name: pairs for name, pairs in kept_by_role.items() if pairs}

    if not filtered_responses:
        raise ValueError("no usable role responses remained after expression filtering")
    capture_stats: dict[str, int] = {}
    default = _mean_vectors(model, tokenizer, default_responses, layers, capture_stats)
    roles = list(filtered_responses)
    role_matrices = {
        name: _mean_vectors(model, tokenizer, pairs, layers, capture_stats)
        for name, pairs in filtered_responses.items()
    }
    axis: dict[int, Any] = {}
    diagnostics: dict[str, Any] = {
        "roles": roles,
        "n_default_responses": len(default_responses),
        "n_default_kept": int(default[layers[0]].shape[0]),
        "min_expression": min_expression,
        "role_expression": expression,
        "capture": capture_stats,
        "layers": {},
    }
    for layer in layers:
        default_mean = default[layer].mean(axis=0)
        role_vectors = np.stack([role_matrices[name][layer].mean(axis=0) for name in roles])
        direction = default_mean - role_vectors.mean(axis=0)
        axis[layer] = mx.array(np.ascontiguousarray(direction, dtype=np.float32))
        pc1 = _pc1(role_vectors) if len(roles) > 2 else np.zeros_like(direction)
        half = len(roles) // 2
        first = default_mean - role_vectors[:half].mean(axis=0)
        second = default_mean - role_vectors[half:].mean(axis=0)
        projections = [_cosine(row, direction) for row in default[layer]]
        residual_norms = np.linalg.norm(default[layer], axis=1)
        mean_residual_norm = float(np.mean(residual_norms))
        projection_array = np.asarray(projections, dtype=np.float64)
        diagnostics["layers"][str(layer)] = {
            "axis_norm": float(np.linalg.norm(direction)),
            "mean_residual_norm": mean_residual_norm,
            "axis_norm_fraction": (
                0.0
                if mean_residual_norm == 0.0
                else float(np.linalg.norm(direction)) / mean_residual_norm
            ),
            "pc1_cosine": _cosine(direction, pc1),
            "pc1_cosine_abs": abs(_cosine(direction, pc1)),
            "split_half_cosine": _cosine(first, second),
            "chat_projection_mean": float(np.mean(projections)),
            "chat_projection_std": float(np.std(projections)),
            "chat_projection_min": float(np.min(projection_array)),
            "chat_projection_p25": float(np.quantile(projection_array, 0.25)),
            "chat_projection_median": float(np.median(projection_array)),
            "chat_projection_p75": float(np.quantile(projection_array, 0.75)),
            "chat_projection_max": float(np.max(projection_array)),
            "chat_projections": [float(value) for value in projections],
            "role_projections": {
                name: _cosine(role_matrices[name][layer].mean(axis=0), direction) for name in roles
            },
        }
    diagnostics["verdict"] = axis_verdict(diagnostics)
    return axis, diagnostics


def project(activation: Any, axis: Any) -> float:
    """Cosine similarity between an activation and the axis."""
    return _cosine(
        np.asarray(activation, dtype=np.float32).reshape(-1),
        np.asarray(axis, dtype=np.float32).reshape(-1),
    )


def save_axis(path: Path, axis: dict[int, Any], diagnostics: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"layer_{layer}": np.asarray(vector, dtype=np.float32) for layer, vector in axis.items()
    }
    payload["layers"] = np.array(sorted(axis), dtype=np.int64)
    payload["diagnostics"] = np.array(json.dumps(diagnostics))
    np.savez_compressed(path, **payload)
    return path


def load_axis(path: Path) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as handle:
        layers = [int(layer) for layer in handle["layers"]]
        return (
            {layer: handle[f"layer_{layer}"] for layer in layers},
            json.loads(str(handle["diagnostics"])),
        )


# --------------------------------------------------------------------------- trajectories


def _slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` against turn index; NaN for a single turn."""
    if len(values) < 2:
        return float("nan")
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def trajectory_projections(
    model: Any,
    tokenizer: Any,
    eval_json_path: Path,
    axis: Any,
    layer: int,
    *,
    limit: int | None = None,
    progress: Any = None,
    spec: Any = None,
) -> list[dict[str, Any]]:
    """Per-turn axis projections for every trajectory in an evaluation JSON.

    Each turn's prompt is rebuilt exactly as :func:`pipeline.runner.run_task` built it --
    system prompt, task prompt, then the assistant notes and tool observations it accumulated,
    windowed by :func:`build_prompt` -- and the response is the stored ``raw`` text of that
    turn. So the projection is of the same string the policy actually produced, in the context
    it actually produced it in.
    """
    from local_llm_lab.agent_protocol import Action
    from local_llm_lab.pipeline.evaluate import failure_reason
    from local_llm_lab.pipeline.protocol import (
        SYSTEM_PROMPT,
        assistant_message,
        build_prompt,
        tool_message,
    )
    from local_llm_lab.pipeline.runner import Trajectory

    payload = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    records = payload["trajectories"][: limit or None]
    axis_vector = np.asarray(axis, dtype=np.float32).reshape(-1)
    capture_stats: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for number, record in enumerate(records, 1):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt"]},
        ]
        projections: list[float] = []
        for step in record["steps"]:
            raw = step.get("raw") or ""
            if raw.strip():
                prompt = build_prompt(tokenizer, messages, spec=spec)
                means = response_mean_activations(
                    model, tokenizer, prompt, raw, [layer], stats=capture_stats
                )
                projections.append(project(means[layer], axis_vector))
            if "action" not in step:
                break
            action = Action(step["action"]["name"], dict(step["action"]["arguments"]))
            if action.name == "finish":
                break
            messages.append(assistant_message(step["thought"], action))
            messages.append(tool_message(action.name, step["observation"]))
        trajectory = Trajectory(**record)
        results.append(
            {
                "task_id": record["task_id"],
                "family": record["family"],
                "variant": record["variant"],
                "turns": len(projections),
                "projections": projections,
                "mean": float(np.mean(projections)) if projections else float("nan"),
                "slope": _slope(projections),
                "min": float(np.min(projections)) if projections else float("nan"),
                "success": bool(record["verdict"].get("success")),
                "loop_detected": bool(record["loop_detected"]),
                "exhausted": bool(record["exhausted"]),
                "failure_reason": None if trajectory.success else failure_reason(trajectory),
            }
        )
        if progress is not None:
            progress(number, len(records))
    return results


def summarize_projections(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Group projections by outcome and by failure reason, with a Mann-Whitney U on each metric."""
    successes = [record for record in records if record["success"]]
    failures = [record for record in records if not record["success"]]

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"n": len(group)}
        for metric in ("mean", "slope", "min"):
            values = np.array([record[metric] for record in group], dtype=np.float64)
            values = values[~np.isnan(values)]
            summary[metric] = {
                "mean": float(np.mean(values)) if len(values) else float("nan"),
                "median": float(np.median(values)) if len(values) else float("nan"),
                "n": int(len(values)),
            }
        return summary

    reasons: dict[str, Any] = {}
    for record in failures:
        reasons.setdefault(record["failure_reason"] or "unknown", []).append(record)
    tests = {}
    for metric in ("mean", "slope", "min"):
        left = np.array([record[metric] for record in successes], dtype=np.float64)
        right = np.array([record[metric] for record in failures], dtype=np.float64)
        tests[metric] = stats.mann_whitney_u(left[~np.isnan(left)], right[~np.isnan(right)])
    return {
        "success": describe(successes),
        "failure": describe(failures),
        "by_failure_reason": {reason: describe(group) for reason, group in sorted(reasons.items())},
        "mann_whitney_success_vs_failure": tests,
    }


# --------------------------------------------------------------------------- rendering


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_build_markdown(diagnostics: dict[str, Any], label: str) -> str:
    out = [f"# P1: assistant axis -- {label} (design §4)", ""]
    out.append(
        f"{diagnostics['n_default_responses']} default-assistant responses against "
        f"{len(diagnostics['roles'])} roles.\n"
    )
    verdict = diagnostics.get("verdict") or axis_verdict(diagnostics)
    if verdict["layer"] is None:
        detail = "no layer in the required 12-24 range"
    else:
        detail = (
            f"layer {verdict['layer']}: |cos(axis, PC1)|={verdict['pc1_cosine_abs']:.3f}, "
            f"split-half={verdict['split_half_cosine']:.3f}"
        )
    out.append(f"**Sanity verdict: {verdict['status']}** — {detail}.\n")

    expression = diagnostics.get("role_expression", {})
    if expression:
        out.append("## Role-expression filter")
        out.append("")
        rows = [
            [
                name,
                f"{entry['mean']:.3f}",
                str(entry["kept"]),
                str(entry["dropped"]),
            ]
            for name, entry in expression.items()
        ]
        out.append(_table(["role", "mean expression", "kept", "dropped"], rows))
        out.append("")
        out.append(
            f"Threshold: {diagnostics.get('min_expression', 0.34):.2f}; roles with fewer than "
            "three retained rollouts do not enter the axis.\n"
        )

    out.append("## Axis sanity checks")
    out.append("")
    rows = []
    for layer, entry in diagnostics["layers"].items():
        rows.append(
            [
                layer,
                f"{entry['axis_norm']:.3f}",
                f"{entry.get('mean_residual_norm', float('nan')):.3f}",
                f"{entry.get('axis_norm_fraction', float('nan')):.3f}",
                f"{entry['pc1_cosine']:+.3f}",
                f"{entry['pc1_cosine_abs']:.3f}",
                f"{entry['split_half_cosine']:.3f}",
            ]
        )
    out.append(
        _table(
            [
                "layer",
                "||axis||",
                "mean ||residual||",
                "axis/residual",
                "cos(axis, PC1)",
                "|cos|",
                "split-half cos",
            ],
            rows,
        )
    )
    out.append("")
    out.append(
        "The paper reports |cos(axis, PC1)| > 0.6; PC1's sign is arbitrary, so the absolute "
        "value is the one to read. Split-half compares the first and second halves of the "
        "retained roles; the source ordering interleaves high, low and neutral groups.\n"
    )

    out.append("## Chat-projection distribution")
    out.append("")
    rows = []
    for layer, entry in diagnostics["layers"].items():
        rows.append(
            [
                layer,
                f"{entry['chat_projection_mean']:+.3f} ± {entry['chat_projection_std']:.3f}",
                f"{entry.get('chat_projection_min', min(entry['chat_projections'])):+.3f}",
                f"{entry.get('chat_projection_p25', np.quantile(entry['chat_projections'], 0.25)):+.3f}",
                f"{entry.get('chat_projection_median', np.median(entry['chat_projections'])):+.3f}",
                f"{entry.get('chat_projection_p75', np.quantile(entry['chat_projections'], 0.75)):+.3f}",
                f"{entry.get('chat_projection_max', max(entry['chat_projections'])):+.3f}",
            ]
        )
    out.append(_table(["layer", "mean ± sd", "min", "p25", "median", "p75", "max"], rows))
    out.append("")

    ordered_layers = [layer for layer in ("12", "18", "24") if layer in diagnostics["layers"]]
    if not ordered_layers:
        available = sorted(diagnostics["layers"], key=int)
        ordered_layers = [available[len(available) // 2]]
    for layer in ordered_layers:
        out.append(f"## Role positions at layer {layer}")
        out.append("")
        ordered = sorted(
            diagnostics["layers"][layer]["role_projections"].items(), key=lambda item: -item[1]
        )
        out.append(
            _table(
                ["role", "projection"],
                [[name, f"{value:+.3f}"] for name, value in ordered],
            )
        )
        out.append("")
    return "\n".join(out)


def render_projection_markdown(
    summary: dict[str, Any], records: list[dict[str, Any]], label: str
) -> str:
    out = [f"# P1: trajectory projections -- {label}", ""]
    out.append(
        f"{len(records)} trajectories, {sum(record['turns'] for record in records)} turns.\n"
    )
    out.append("## By outcome")
    out.append("")
    rows = []
    for name in ("success", "failure"):
        group = summary[name]
        rows.append(
            [
                name,
                str(group["n"]),
                f"{group['mean']['mean']:+.4f}",
                f"{group['slope']['mean']:+.5f}",
                f"{group['min']['mean']:+.4f}",
            ]
        )
    out.append(_table(["outcome", "n", "mean projection", "mean slope", "mean minimum"], rows))
    out.append("")
    out.append("## Mann-Whitney U, success versus failure")
    out.append("")
    rows = []
    for metric, test in summary["mann_whitney_success_vs_failure"].items():
        rows.append(
            [
                metric,
                f"{test['u']:.1f}",
                f"{test['effect']:.3f}",
                f"{test['z']:+.2f}",
                f"{test['p']:.4f}",
                f"{int(test['n1'])}/{int(test['n2'])}",
            ]
        )
    out.append(
        _table(["metric", "U", "effect (U/n1n2)", "z", "p (two-sided)", "n success/fail"], rows)
    )
    out.append("")
    out.append(
        "Effect 0.5 means no separation; the normal approximation needs both groups above ~8.\n"
    )
    out.append("## By failure reason")
    out.append("")
    rows = []
    for reason, group in summary["by_failure_reason"].items():
        rows.append(
            [
                reason,
                str(group["n"]),
                f"{group['mean']['mean']:+.4f}",
                f"{group['slope']['mean']:+.5f}",
                f"{group['min']['mean']:+.4f}",
            ]
        )
    out.append(
        _table(["failure reason", "n", "mean projection", "mean slope", "mean minimum"], rows)
    )
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- P1 closure


def load_rollouts(path: Path) -> list[dict[str, Any]]:
    """Every record of a saved rollouts JSONL, in file order."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: not JSON ({error})") from error
            if not isinstance(record, dict) or "role" not in record:
                raise ValueError(f"{path}:{number}: a rollout record needs a 'role' field")
            records.append(record)
    return records


def _heuristic_citation() -> str:
    """``file:line`` of the scorer this record was produced with, resolved at run time.

    Resolved rather than written down because the record is evidence: a stale anchor would
    cite a line that no longer holds the function.
    """
    from local_llm_lab.project import PROJECT_ROOT

    source = inspect.getsourcefile(heuristic_expression_score)
    _lines, line = inspect.getsourcelines(heuristic_expression_score)
    path = Path(source) if source else Path(__file__)
    if PROJECT_ROOT in path.parents:
        path = path.relative_to(PROJECT_ROOT)
    return f"{path.as_posix()}:{line} heuristic_expression_score"


def closure_summary(
    records: list[dict[str, Any]],
    *,
    min_expression: float = 0.34,
    progress: Any = None,
) -> dict[str, Any]:
    """Per-role persona-expression counts over a saved rollout corpus (SPEC-004 §4).

    Heuristic only: no model is loaded and no judge is consulted, so the closure record is
    produced from the saved text alone. Roles are scored in file order, which puts the default
    assistant first. Note that :func:`heuristic_expression_score` returns 1.0 for a role name
    it has no markers for, so the default assistant's count is a pass-through, not a
    measurement; ``marker_backed`` records which entries are real measurements.
    """
    if not 0.0 <= min_expression <= 1.0:
        raise ValueError(f"min_expression must be in [0, 1], got {min_expression}")
    grouped: dict[str, list[str]] = {}
    for record in records:
        role = str(record["role"])
        grouped.setdefault(role, []).append(str(record.get("response") or ""))
    if not grouped:
        raise ValueError("no rollouts to score")

    entries: list[dict[str, Any]] = []
    default_entry: dict[str, Any] | None = None
    for number, (role, responses) in enumerate(grouped.items(), 1):
        if progress is not None:
            progress(number, len(grouped), role)
        scores = [heuristic_expression_score(role, response) for response in responses]
        entry = {
            "role": role,
            "group": ROLE_GROUPS.get(role),
            "marker_backed": role in ROLE_MARKERS,
            "at_or_above": int(sum(score >= min_expression for score in scores)),
            "total": len(scores),
            "mean": float(np.mean(scores)),
            "max": float(np.max(scores)),
            "scores": [float(score) for score in scores],
        }
        if role == DEFAULT_ASSISTANT_ROLE:
            default_entry = entry
        else:
            entries.append(entry)

    best = max(entries, key=lambda item: (item["at_or_above"], item["mean"])) if entries else None
    return {
        "min_expression": float(min_expression),
        "records": len(records),
        "n_roles": len(entries),
        "roles_at_zero": sum(1 for entry in entries if entry["at_or_above"] == 0),
        "best_role": best,
        "default_assistant": default_entry,
        "roles": entries,
        "unscored_roles": [entry["role"] for entry in entries if not entry["marker_backed"]],
        "expected": dict(SPEC_004_EXPECTED),
    }


def _count(entry: dict[str, Any] | None) -> str:
    return "-" if entry is None else f"{entry['at_or_above']} of {entry['total']}"


def render_closure_markdown(summary: dict[str, Any]) -> str:
    """The human-readable closure record (SPEC-004 §4)."""
    threshold = summary["min_expression"]
    best = summary.get("best_role")
    default = summary.get("default_assistant")
    expected = summary.get("expected", {})
    out = ["# P1 CLOSED: the assistant axis on the coder base", ""]
    out.append(f"**Decision (SPEC-004 §4): {summary['decision']}.**")
    out.append("")
    out.append(
        f"Measured on {summary['date']} from {summary['records']} saved rollouts by the lexical "
        f"marker heuristic alone, at the ratified threshold {threshold:.2f}. No model was "
        "loaded and no same-policy judge was consulted."
    )
    out.append("")

    out.append("## Provenance")
    out.append("")
    out.append(
        _table(
            ["field", "value"],
            [
                ["rollouts", summary["rollouts"]["path"]],
                ["rollouts sha256", summary["rollouts"]["sha256"]],
                ["records", str(summary["records"])],
                ["roles (excluding the default assistant)", str(summary["n_roles"])],
                ["threshold (min_expression)", f"{threshold:.2f}"],
                ["scorer", summary["heuristic"]],
                ["git commit", summary["git_commit"]],
                ["command", " ".join(summary["command"])],
            ],
        )
    )
    out.append("")

    out.append("## Headline counts, measured against the pre-registered ones")
    out.append("")
    best_expected = (
        "-"
        if not expected
        else f"{expected['best_role_at_or_above']} of {expected['best_role_total']}"
    )
    zero_expected = "-" if not expected else f"{expected['roles_at_zero']} of {expected['n_roles']}"
    out.append(
        _table(
            ["quantity", "measured", "SPEC-004 §4 expected"],
            [
                [
                    f"best role ({'-' if best is None else best['role']})",
                    _count(best),
                    best_expected,
                ],
                [
                    "roles with no rollout at or above the threshold",
                    f"{summary['roles_at_zero']} of {summary['n_roles']}",
                    zero_expected,
                ],
                ["default assistant", _count(default), "not pre-registered"],
            ],
        )
    )
    out.append("")
    if expected:
        agreed = (
            best is not None
            and best["at_or_above"] == expected["best_role_at_or_above"]
            and best["total"] == expected["best_role_total"]
            and summary["roles_at_zero"] == expected["roles_at_zero"]
            and summary["n_roles"] == expected["n_roles"]
        )
        out.append(
            "The measured counts reproduce the pre-registered ones."
            if agreed
            else "The measured counts differ from the pre-registered ones; the measurement "
            "stands and the difference is reported, not reconciled."
        )
        out.append("")

    out.append("## Persona expression per role")
    out.append("")
    rows = [
        [
            entry["role"],
            entry["group"] or "-",
            _count(entry),
            f"{entry['mean']:.3f}",
            f"{entry['max']:.3f}",
        ]
        for entry in sorted(
            summary["roles"], key=lambda item: (-item["at_or_above"], -item["mean"], item["role"])
        )
    ]
    out.append(_table(["role", "group", "at or above threshold", "mean", "best"], rows))
    out.append("")

    out.append("## Reading")
    out.append("")
    out.append(
        "The heuristic is the fraction of a role's lexical markers present in a reply, less a "
        "penalty for explicit generic-assistant tells; a reply at or above the threshold is one "
        "the axis build would have kept. `build_axis` requires twelve roles with three kept "
        "rollouts each, so this corpus cannot produce an axis at all."
    )
    out.append("")
    if default is not None and not default["marker_backed"]:
        out.append(
            f"The default assistant's {_count(default)} is not a persona measurement: the "
            "heuristic has no markers for that name and returns 1.0 unfiltered for every "
            "unknown role, so the row records the pass-through, not expression."
        )
        out.append("")
    unscored = [role for role in summary.get("unscored_roles", []) if role]
    if unscored:
        out.append(
            "Roles with no registered markers, scored unfiltered by the same rule: "
            + ", ".join(unscored)
            + "."
        )
        out.append("")
    out.append(
        "No further P1 work on this base. The pre-registered reopening condition is unchanged: "
        "one `build` on the next base after its preflight passes, and `project` only if the "
        "axis verdict passes."
    )
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI


def _close(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    rollouts = Path(args.rollouts)
    if not rollouts.is_file():
        parser.error(f"no rollouts file at {rollouts}")
    if not 0.0 <= args.min_expression <= 1.0:
        parser.error(f"--min-expression must be in [0, 1], got {args.min_expression}")
    digest = sha256_of(rollouts)
    identity = {
        "rollouts": str(rollouts),
        "rollouts_sha256": digest,
        "min_expression": args.min_expression,
        "git_commit": git_commit(),
    }
    with RunLog.open(
        args.output, name="assistant-axis-close", command=sys.argv, identity=identity
    ) as log:
        try:
            records = load_rollouts(rollouts)
        except ValueError as error:
            parser.error(str(error))
        log.info("loaded rollouts", records=len(records), path=str(rollouts))
        try:
            summary = closure_summary(
                records,
                min_expression=args.min_expression,
                progress=lambda number, total, name: log.progress(number, total, f"role {name}"),
            )
        except ValueError as error:
            parser.error(str(error))
        summary.update(
            {
                "decision": CLOSURE_DECISION,
                "rollouts": {"path": str(rollouts.resolve()), "sha256": digest},
                "heuristic": _heuristic_citation(),
                "git_commit": identity["git_commit"],
                "date": datetime.now(UTC).date().isoformat(),
                "command": list(sys.argv),
            }
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "closed.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        markdown = render_closure_markdown(summary)
        (args.output / "CLOSED.md").write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
        log.info("wrote", path=str(args.output / "CLOSED.md"))


def _build(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.probes.guard import require_idle_gpu
    from local_llm_lab.probes.policies import (
        resolve_layers,
        resolve_policy,
        validate_layer_syntax,
    )

    spec = load_model_spec(args.model)
    try:
        validate_layer_syntax(args.layers)
        adapter = resolve_policy(args.policy, spec)
    except ValueError as error:
        parser.error(str(error))
    identity = {
        "model": spec.name,
        "hf_id": spec.hf_id,
        "policy": args.policy,
        "adapter": None if adapter is None else str(adapter),
        "layers": args.layers,
        "prompts": args.prompts,
        "role_prompts": args.role_prompts,
        "git_commit": git_commit(),
    }
    with RunLog.open(
        args.output, name="assistant-axis-build", command=sys.argv, identity=identity
    ) as log:
        require_idle_gpu(parser, args, "generating role rollouts")
        model, tokenizer, view, _resolved = load_policy(spec, adapter)
        try:
            selection = resolve_layers(args.layers, spec, view.num_layers)
        except ValueError as error:
            parser.error(str(error))
        layers = list(selection.indices)
        prompts = load_chat_prompts(args.prompts)
        # Matched design: the default persona now generates against the roles' prompt list,
        # taken from this pool, so the count it is rolled out on is `role_prompts`.
        log.info("default assistant", prompts=args.role_prompts, pool=len(prompts))
        try:
            axis, diagnostics = build_axis_run(
                model,
                tokenizer,
                prompts,
                layers,
                args.output,
                args.policy,
                role_prompts=args.role_prompts,
                max_tokens=args.max_tokens,
                exemplar=args.exemplar,
                min_expression=args.min_expression,
                use_model_judge=args.judge,
                progress=lambda number, total, name: log.progress(number, total, f"role {name}"),
            )
        except ValueError as error:
            parser.error(str(error))
        diagnostics.update(
            {
                "policy": args.policy,
                "model": args.model,
                "prompts": len(prompts),
                "role_prompts": args.role_prompts,
                "layer_selection": selection.as_dict(),
            }
        )
        args.output.mkdir(parents=True, exist_ok=True)
        policy_stem = _policy_stem(args.policy)
        save_axis(args.output / f"axis-{policy_stem}.npz", axis, diagnostics)
        (args.output / f"axis-{policy_stem}.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        markdown = render_build_markdown(diagnostics, args.policy)
        (args.output / f"axis-{policy_stem}.md").write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
        log.info("wrote", path=str(args.output / f"axis-{policy_stem}.npz"))


def _project(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from local_llm_lab.models import load_model_spec
    from local_llm_lab.pipeline.evaluate import load_policy
    from local_llm_lab.probes.guard import require_idle_gpu
    from local_llm_lab.probes.policies import resolve_policy

    spec = load_model_spec(args.model)
    try:
        adapter = resolve_policy(args.policy, spec)
    except ValueError as error:
        parser.error(str(error))
    axis_path = Path(args.axis)
    eval_path = Path(args.eval)
    identity = {
        "model": spec.name,
        "hf_id": spec.hf_id,
        "policy": args.policy,
        "adapter": None if adapter is None else str(adapter),
        "layer": args.layer,
        "axis": str(axis_path),
        "axis_sha256": sha256_of(axis_path) if axis_path.is_file() else None,
        "eval": str(eval_path),
        "eval_sha256": sha256_of(eval_path) if eval_path.is_file() else None,
        "git_commit": git_commit(),
    }
    with RunLog.open(
        args.output, name="assistant-axis-project", command=sys.argv, identity=identity
    ) as log:
        require_idle_gpu(parser, args, "projecting trajectories")
        axis, diagnostics = load_axis(args.axis)
        if args.layer not in axis:
            parser.error(f"axis file has layers {sorted(axis)}, not {args.layer}")
        model, tokenizer, _view, _resolved = load_policy(spec, adapter)
        records = trajectory_projections(
            model,
            tokenizer,
            args.eval,
            axis[args.layer],
            args.layer,
            limit=args.limit,
            progress=lambda number, total: log.progress(number, total, "project"),
            spec=spec,
        )
        summary = summarize_projections(records)
        payload = {
            "policy": args.policy,
            "axis": str(Path(args.axis).resolve()),
            "eval": str(Path(args.eval).resolve()),
            "layer": args.layer,
            "chat_projection": diagnostics["layers"]
            .get(str(args.layer), {})
            .get("chat_projection_mean"),
            "summary": summary,
            "trajectories": records,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        stem = f"trajectories-{args.policy}-layer{args.layer}"
        (args.output / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        markdown = render_projection_markdown(
            summary, records, f"{args.policy} @ layer {args.layer}"
        )
        (args.output / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
        log.info("wrote", path=str(args.output / f"{stem}.json"))


def main() -> None:
    from local_llm_lab.probes.guard import add_gpu_arguments

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    common.add_argument(
        "--policy",
        default="base",
        help="a policy named by the selected model or an explicit adapter directory",
    )
    common.add_argument("--output", type=Path, required=True)
    add_gpu_arguments(common)

    parser = argparse.ArgumentParser(description="P1: build and apply the assistant axis.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build", parents=[common], help="role rollouts, the axis, and its sanity statistics"
    )
    build.add_argument("--prompts", type=int, default=96)
    build.add_argument("--role-prompts", type=int, default=8)
    build.add_argument("--max-tokens", type=int, default=192)
    build.add_argument(
        "--layers", help="comma-separated layer indices or fractions; defaults to the registry"
    )
    build.add_argument("--min-expression", type=float, default=0.34)
    build.add_argument(
        "--exemplar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "prepend each available in-character one-shot; none are registered since the "
            "2026-09-05 ruling, so this is inert and recorded as False (default: on)"
        ),
    )
    build.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="combine the marker score with a same-policy 0-3 judge (default: off)",
    )

    projected = sub.add_parser(
        "project", parents=[common], help="per-turn projections of stored trajectories"
    )
    projected.add_argument("--axis", type=Path, required=True)
    projected.add_argument("--eval", type=Path, required=True)
    projected.add_argument("--layer", type=int, default=18)
    projected.add_argument("--limit", type=int, default=None)

    # No model, no policy and no GPU guard: `close` reads saved text and scores it offline,
    # so it does not take the `common` parent.
    closed = sub.add_parser(
        "close", help="the P1 closure record from a saved rollouts JSONL (SPEC-004 §4)"
    )
    closed.add_argument("--rollouts", type=Path, required=True)
    closed.add_argument("--output", type=Path, required=True)
    closed.add_argument("--min-expression", type=float, default=0.34)

    args = parser.parse_args()
    if args.command == "build":
        _build(args, build)
    elif args.command == "close":
        _close(args, closed)
    else:
        _project(args, projected)
