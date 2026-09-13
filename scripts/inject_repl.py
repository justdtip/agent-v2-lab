"""An interactive console for extracting residual-stream vectors and injecting them back.

The model is loaded once and stays resident, so a session is a conversation with a live model
rather than a series of one-minute reloads. Vectors live in named slots, so a vector extracted from
one prompt can be injected into another, combined with a second vector, or swept over layers and
strengths without re-extraction.

WHAT THIS IS FOR. Lindsey's *Emergent Introspective Awareness* (transformer-circuits, 2025) builds a
concept vector from "Tell me about {word}." minus the mean over random words, injects it into the
residual stream about two thirds of the way up, and asks the model whether it notices an intruding
thought. This console does the mechanical part of that and leaves the experimental design to you.

THE MAGNITUDE, AND WHY THE DEFAULT IS NOT THE REPO'S LITERAL RULE. This repo rules that a synthetic
direction is injected at `alpha` times the *residual norm* at the site, citing the paper's
"strengths 2 to 4". Applied literally that is catastrophic: an extracted concept direction runs at
a few per cent of the residual norm, so alpha 2 on that basis adds a perturbation twice the size of
the residual it is added to, and the model emits one token forever. Measured, not assumed — it was
the first thing this console did. So `basis=vector` is the default and multiplies the direction by
alpha, which is the only reading of "strength 2" that leaves a model standing; `basis=residual` is
the literal rule, kept and labelled. Either way the perturbation is printed as a fraction of the
clean residual norm at the site, so the magnitude is never implicit. A *donor* row from `extract` is
already in the model's own units, which is why the repo's ruling
(`GEMMA3-CONFIDENCE-2026-09-08`) prefers donor differences: they need no convention at all.

HOW THIS DIFFERS FROM THE STEERING PILOT, DECLARED BECAUSE IT MATTERS. `STEER-PILOT-2026-09-10` patched
the **prefill only** and deliberately left decode untouched (`h.shape[1] > pos`). The paper's protocol
injects from a chosen token *and continues through the assistant's response*, so this console sustains
the injection across every decode step by default. That is a different intervention, not a port of
the pilot's, and `sustain=false` gives you the pilot's prefill-only behaviour for comparison.

LAYER INDEXING. `L` is a residual index: layer 0 is the embedding output and layer L is the residual
**after block L-1**, which is the repo's convention throughout. The hook is installed on block L-1.
Two thirds of the way up is L≈23 on the 34-layer 4B and L≈32 on the 48-layer 12B.

    python scripts/inject_repl.py --model /path/to/snapshot --device cuda:0

Commands (`help` lists them live):

    gen <prompt>                              generate with no intervention
    extract <slot> @<L> [pos=<i>] <prompt>    residual at layer L, position pos (default: last)
    concept <slot> @<L> <word>                Lindsey's recipe: "Tell me about {word}." minus the
                                              mean over --concept-baseline random words
    inject <slot> @<L> [a=<alpha>] [from=<i>] [sustain=<bool>] [basis=vector|residual] <prompt>
    sweep <slot> layers=<a,b,c> [a=<alpha>]  <prompt>
    slots | drop <slot> | save <file> | load <file> | help | quit
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_llm_lab import device as device_mod  # noqa: E402
from local_llm_lab import hf_text  # noqa: E402
from local_llm_lab.arch_torch import TorchArchitectureView  # noqa: E402
from local_llm_lab.pipeline.runner import config_eos_ids  # noqa: E402

#: The words the concept recipe subtracts, so a concept vector is a direction and not a position.
BASELINE_WORDS = (
    "table", "river", "pencil", "harbour", "cotton", "engine", "meadow", "marble", "lantern",
    "compass", "biscuit", "curtain", "gravel", "kettle", "ladder", "mirror", "needle", "orchard",
    "pillow", "quarry", "ribbon", "saddle", "thimble", "velvet", "window", "anchor", "bramble",
    "cinder", "drawer", "ember", "fennel", "girder", "hollow", "ingot", "jasmine", "kestrel",
    "linnet", "mantle", "nutmeg", "oyster", "parsley", "quiver", "rafter", "sorrel", "trellis",
    "urchin", "vellum", "walnut", "yarrow", "zephyr",
)


class Magnitude:
    """Scale the residual instead of adding to it: damage with no direction whatsoever.

    Every control anyone has run -- ours, and Rivera and Africa's -- varies WHICH vector is added.
    A model that has learned only "a vector of the kind this protocol adds is present, and roughly
    how big" passes all of them. This adds nothing: it multiplies the residual by 1+epsilon, which
    reaches the same damage by the purest disturbance stimulus there is. If a concept moves the
    yes/no answer no further than this does at matched damage, the answer is about the damage.

    Deliberately the same shape as `Injection` so the grid can drive either through one code path.
    """

    def __init__(self, block, epsilon: float, *, from_position: int, sustain: bool = True):
        self.block, self.epsilon = block, epsilon
        self.from_position, self.sustain = from_position, sustain
        self.applications, self.positions_touched = 0, 0
        self._seen = 0
        self._handle = None

    def __enter__(self):
        self._handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        width = hidden.shape[1]
        start, self._seen = self._seen, self._seen + width
        if not self.sustain and start > 0:
            return output
        first = max(self.from_position - start, 0)
        if first >= width:
            return output
        hidden[:, first:, :].mul_(1.0 + self.epsilon)
        self.applications += 1
        self.positions_touched += width - first
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def report(self) -> dict:
        return {"applications": self.applications, "positions_touched": self.positions_touched,
                "from_position": self.from_position, "sustain": self.sustain,
                "epsilon": self.epsilon, "kind": "magnitude"}


class Injection:
    """Adds a fixed direction to a block's output residual, on every forward it sees.

    Sustained by default: the hook fires on the prefill and on every KV-cached decode step, because
    the torch decode loop calls the model per step and never touches hook handles. Positions are
    absolute over the whole sequence; the hook tracks how far the sequence has advanced rather than
    trusting the slice it is handed, since a decode step's slice is one token wide.
    """

    def __init__(self, block, vector, *, scale: float, from_position: int, sustain: bool = True,
                 local_fraction: float | None = None):
        """`scale` is a fixed multiplier. `local_fraction`, if given, overrides it.

        A fixed multiplier is a delta of one size added everywhere, sized from the residual norm at
        ONE position. Residual norms vary a great deal across positions -- at the BOS token the
        same delta is several times larger relative to the local residual than it is mid-sentence
        -- so "forty per cent of the residual norm" is then true at exactly one place and false
        everywhere else it lands. `local_fraction` scales the delta by each position's own norm, so
        the number means what it says at every position it touches.
        """
        self.block, self.vector, self.scale = block, vector, scale
        self.local_fraction = local_fraction
        self.from_position, self.sustain = from_position, sustain
        self.applications, self.positions_touched = 0, 0
        self._seen = 0  # absolute position of the first row in the current slice
        self._handle = None
        self._delta = None
        self._unit = None

    def __enter__(self):
        self._handle = self.block.register_forward_hook(self._hook)
        # Precomputed once. `self.vector` lives on the CPU -- residual_at returns it there -- so
        # rebuilding this inside the hook meant a host multiply and a pageable copy on every
        # decode step of a sustained injection.
        self._delta = None
        self._unit = None
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        width = hidden.shape[1]
        start, self._seen = self._seen, self._seen + width
        if not self.sustain and start > 0:
            return output  # prefill only: the pilot's convention, kept for comparison
        first = max(self.from_position - start, 0)
        if first >= width:
            return output
        if self.local_fraction is None:
            if self._delta is None:
                self._delta = (self.vector * self.scale).to(hidden.dtype).to(hidden.device)
            hidden[:, first:, :].add_(self._delta)
        else:
            if self._unit is None:
                self._unit = (self.vector / self.vector.norm()).to(hidden.dtype).to(hidden.device)
            here = hidden[:, first:, :].float().norm(dim=-1, keepdim=True)
            hidden[:, first:, :].add_(self._unit * here.to(hidden.dtype) * self.local_fraction)
        self.applications += 1
        self.positions_touched += width - first
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def report(self) -> dict:
        return {"applications": self.applications, "positions_touched": self.positions_touched,
                "from_position": self.from_position, "sustain": self.sustain,
                "scale": self.scale, "local_fraction": self.local_fraction}


class Console:
    def __init__(self, snapshot: Path, *, device: str, dtype: str, baseline_words: int, seed: int,
                 adapter: Path | None = None, lora_rank: int = 32, lora_alpha: int = 64):
        device_mod.pin(seed=seed)
        self.device = device_mod.select(device)
        print(f"loading {snapshot.name} on {self.device} in {dtype} ...", flush=True)
        self.model, self.report = hf_text.load_text_causal_lm(
            snapshot, dtype=dtype, attn_implementation="eager", device=self.device)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        #: The adapter, or None for the base checkpoint. Every reading taken through this console
        #: belongs to whichever of the two is loaded, so it is recorded rather than remembered.
        self.adapter = None
        if adapter is not None:
            # Before the architecture view, which indexes the modules the swap replaces.
            from local_llm_lab.lora_torch import apply_lora, load_lora_state_dict
            swapped = apply_lora(self.model, r=lora_rank, alpha=lora_alpha)
            loaded = load_lora_state_dict(self.model, torch.load(adapter, map_location="cpu"))
            self.model.eval()
            self.adapter = str(adapter)
            print(f"adapter {Path(adapter).name}: {loaded} of {swapped} modules loaded at "
                  f"rank {lora_rank}, alpha {lora_alpha}", flush=True)
        self.view = TorchArchitectureView.from_model(self.model)
        self.stop_ids = set(config_eos_ids(self.model))
        for name in ("<end_of_turn>", "<eos>"):
            got = self.tokenizer.convert_tokens_to_ids(name)
            if isinstance(got, int) and got >= 0:
                self.stop_ids.add(got)
        self.slots: dict[str, dict] = {}
        self.baseline_words = baseline_words
        self.rng = random.Random(seed)
        print(f"ready: {self.view.num_layers} layers, hidden {self.view.hidden_size}, "
              f"two-thirds depth is layer {round(self.view.num_layers * 2 / 3)}", flush=True)

    # -- plumbing ------------------------------------------------------------------------------
    def render(self, prompt: str) -> list[int]:
        """A user turn through the model's own chat template, tokenized without a second BOS."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and ids[:2] == [bos, bos]:
            raise ValueError("double BOS: the template and the tokenizer both added one")
        return ids

    def residual_at(self, ids: list[int], layer: int, position: int | None) -> torch.Tensor:
        """The residual after block `layer-1` at one absolute position, from a single forward."""
        captured = {}

        def hook(_m, _a, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Slice first, THEN upcast and copy. Taking the whole layer meant an fp32 allocation of
            # sequence x hidden and a full device-to-host transfer to obtain one row of it -- at
            # 8k tokens that is 172 MB moved to read 21 KB, and cmd_concept does it 25 times.
            width = hidden.shape[1]
            at = width - 1 if position is None else position
            if not -width <= at < width:
                raise IndexError(f"position {position} is outside a {width}-token forward")
            captured["h"] = hidden.detach()[0, at].float().cpu()

        handle = self.view.blocks[layer - 1].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(self.view._ids(ids), use_cache=False)
        finally:
            handle.remove()
        return captured["h"].clone()

    def generate(self, ids: list[int], max_tokens: int) -> str:
        """Greedy decode with a KV cache, one forward per step so hooks fire on every token.

        Deliberately not `runner.torch_greedy_stream`: that loop is shared with the sealed capture
        path, and on this prompt it emits newlines where a plain greedy loop emits "Hello there!",
        so its convention is not the one this console wants. Rather than change a loop other work
        depends on, this one is local and small.
        """
        from transformers import DynamicCache

        cache = DynamicCache()
        cursor, emitted = list(ids), []
        with torch.no_grad():
            logits = self.model(input_ids=self.view._ids(cursor),
                                past_key_values=cache, use_cache=True).logits
            for _ in range(max_tokens):
                token = int(logits[0, -1].float().argmax())
                if token in self.stop_ids:
                    break
                emitted.append(token)
                logits = self.model(input_ids=self.view._ids([token]),
                                    past_key_values=cache, use_cache=True).logits
        return self.tokenizer.decode(emitted)

    # -- commands ------------------------------------------------------------------------------
    def cmd_gen(self, prompt: str, max_tokens: int) -> None:
        print(self.generate(self.render(prompt), max_tokens).strip() or "(nothing)")

    def cmd_extract(self, slot: str, layer: int, position: int | None, prompt: str) -> None:
        ids = self.render(prompt)
        vector = self.residual_at(ids, layer, position)
        self.slots[slot] = {"vector": vector, "layer": layer, "kind": "residual",
                            "norm": float(vector.norm()), "prompt": prompt,
                            "position": position if position is not None else len(ids) - 1}
        print(f"{slot}: residual at layer {layer}, position {self.slots[slot]['position']}, "
              f"norm {self.slots[slot]['norm']:.1f}")

    def cmd_concept(self, slot: str, layer: int, word: str) -> None:
        """The paper's recipe: the word's activation minus the mean over random words."""
        target = self.residual_at(self.render(f"Tell me about {word}."), layer, None)
        pool = [w for w in BASELINE_WORDS if w.lower() != word.lower()]
        chosen = self.rng.sample(pool, min(self.baseline_words, len(pool)))
        stack = [self.residual_at(self.render(f"Tell me about {w}."), layer, None) for w in chosen]
        mean = torch.stack(stack).mean(dim=0)
        vector = target - mean
        self.slots[slot] = {"vector": vector, "layer": layer, "kind": "concept", "word": word,
                            "norm": float(vector.norm()), "baseline_n": len(chosen),
                            "target_norm": float(target.norm())}
        print(f"{slot}: concept '{word}' at layer {layer}, direction norm {vector.norm():.1f} "
              f"against a carrier norm of {target.norm():.1f} "
              f"({vector.norm() / target.norm():.1%} of it), {len(chosen)} baseline words")

    def _prepare(self, slot: str, layer: int, alpha: float, ids: list[int], from_position: int,
                 basis: str = "vector"):
        """Return the vector, the multiplier, and a legible statement of what the magnitude is.

        `basis="vector"` (the default) multiplies the extracted direction by alpha, which is what
        the paper's "strength 2" has to mean: the direction is already a difference of activations
        and comes out at a few per cent of the residual, so alpha of 2 perturbs by a few per cent
        more. `basis="residual"` is the literal reading of this repo's rule for synthetic directions
        — alpha times the residual norm — and at alpha 2 that is a perturbation twice the size of
        the residual it is added to, which destroys the model's output. Both are offered; the
        destructive one is not the default, and either way the perturbation is reported as a
        fraction of the clean residual norm so the magnitude is never implicit.
        """
        entry = self.slots[slot]
        vector = entry["vector"]
        here = float(self.residual_at(ids, layer, from_position).norm())
        if basis == "residual":
            scale = alpha * here / float(vector.norm())
        else:
            scale = alpha
        perturbation = scale * float(vector.norm())
        note = (f"basis={basis}, |delta| {perturbation:,.0f} = "
                f"{perturbation / here:.1%} of the clean residual norm {here:,.0f}")
        return vector, scale, note

    def cmd_inject(self, slot: str, layer: int, alpha: float, from_position: int | None,
                   sustain: bool, prompt: str, max_tokens: int, basis: str = "vector") -> None:
        if slot not in self.slots:
            print(f"no slot '{slot}'"); return
        ids = self.render(prompt)
        start = from_position if from_position is not None else len(ids) - 1
        vector, scale, note = self._prepare(slot, layer, alpha, ids, start, basis)
        injection = Injection(self.view.blocks[layer - 1], vector, scale=scale,
                              from_position=start, sustain=sustain)
        with injection:
            out = self.generate(ids, max_tokens)
        print(f"[{slot} @ L{layer}, alpha {alpha}, from {start}, sustain {sustain}; {note}]")
        print(out.strip() or "(nothing)")
        print(f"  -- hook fired {injection.applications} times over "
              f"{injection.positions_touched} positions")

    def cmd_sweep(self, slot: str, layers: list[int], alpha: float, prompt: str,
                  max_tokens: int, basis: str = "vector") -> None:
        if slot not in self.slots:
            print(f"no slot '{slot}'"); return
        ids = self.render(prompt)
        start = len(ids) - 1
        for layer in layers:
            vector, scale, _ = self._prepare(slot, layer, alpha, ids, start, basis)
            with Injection(self.view.blocks[layer - 1], vector, scale=scale, from_position=start):
                out = self.generate(ids, max_tokens)
            first = " ".join(out.split())[:150]
            print(f"  L{layer:>3} a={alpha}: {first or '(nothing)'}")

    def cmd_slots(self) -> None:
        if not self.slots:
            print("(no slots)"); return
        for name, entry in sorted(self.slots.items()):
            extra = f"word '{entry['word']}'" if entry["kind"] == "concept" else f"'{entry.get('prompt','')[:40]}'"
            print(f"  {name:<12} {entry['kind']:<9} L{entry['layer']:<3} norm {entry['norm']:>8.1f}  {extra}")

    def cmd_save(self, path: str) -> None:
        payload = {n: {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in e.items()}
                   for n, e in self.slots.items()}
        Path(path).write_text(json.dumps(payload))
        print(f"wrote {len(payload)} slot(s) to {path}")

    def cmd_load(self, path: str) -> None:
        payload = json.loads(Path(path).read_text())
        for name, entry in payload.items():
            entry["vector"] = torch.tensor(entry["vector"])
            self.slots[name] = entry
        print(f"loaded {len(payload)} slot(s) from {path}")


def parse_options(words: list[str]) -> tuple[dict, list[str]]:
    """Split leading `key=value` options from the free text that follows."""
    options, rest = {}, list(words)
    while rest and "=" in rest[0] and not rest[0].startswith("@"):
        key, _, value = rest.pop(0).partition("=")
        options[key] = value
    return options, rest


def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def run(console: Console, max_tokens: int) -> int:
    print("type 'help' for commands, 'quit' to leave", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line in {"quit", "exit"}:
            return 0
        try:
            head, *rest = shlex.split(line) if line.startswith(("save", "load", "drop")) else line.split(" ", 1)
            body = rest[0] if rest else ""
            if head == "help":
                print(__doc__.split("Commands")[1])
            elif head == "slots":
                console.cmd_slots()
            elif head == "drop":
                console.slots.pop(body.strip(), None); console.cmd_slots()
            elif head == "save":
                console.cmd_save(body.strip())
            elif head == "load":
                console.cmd_load(body.strip())
            elif head == "gen":
                console.cmd_gen(body, max_tokens)
            elif head in {"extract", "concept", "inject", "sweep"}:
                words = body.split()
                slot = words.pop(0)
                layer = None
                if words and words[0].startswith("@"):
                    layer = int(words.pop(0)[1:])
                options, words = parse_options(words)
                text = " ".join(words)
                if head == "extract":
                    console.cmd_extract(slot, layer, int(options["pos"]) if "pos" in options else None, text)
                elif head == "concept":
                    console.cmd_concept(slot, layer, text.strip())
                elif head == "inject":
                    console.cmd_inject(
                        slot, layer if layer is not None else console.slots[slot]["layer"],
                        float(options.get("a", 2.0)),
                        int(options["from"]) if "from" in options else None,
                        truthy(options.get("sustain", "true")), text,
                        int(options.get("n", max_tokens)), options.get("basis", "vector"))
                else:
                    console.cmd_sweep(slot, [int(x) for x in options["layers"].split(",")],
                                      float(options.get("a", 2.0)), text,
                                      int(options.get("n", max_tokens)),
                                      options.get("basis", "vector"))
            else:
                print(f"unknown command '{head}' — try 'help'")
        except Exception as exc:  # a console that dies on a typo is not a console
            print(f"!! {type(exc).__name__}: {exc}")
        print(flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local snapshot directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--concept-baseline", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    console = Console(args.model, device=args.device, dtype=args.dtype,
                      baseline_words=args.concept_baseline, seed=args.seed)
    return run(console, args.max_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
