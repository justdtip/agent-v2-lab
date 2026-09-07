"""Requirements §14: register an offline plan, then explicitly execute it once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from local_llm_lab.models import load_model_spec  # noqa: E402
from local_llm_lab.pipeline.lens_fitting.prose import (  # noqa: E402
    corpus_tokenizer,
    execute_plan,
    make_plan,
    register_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="qwen35-4b")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--lens", type=Path)
    parser.add_argument("--lens-sha256")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    spec = load_model_spec(args.model)
    if args.action == "plan":
        if not all((args.corpus, args.lens, args.lens_sha256)):
            parser.error("plan requires --corpus, --lens and --lens-sha256")
        plan = make_plan(
            args.corpus,
            spec,
            args.lens,
            lens_sha256=args.lens_sha256,
            tokenizer=corpus_tokenizer(args.corpus),
            revision=args.revision,
        )
        register_plan(args.out, plan)
        print(
            json.dumps(
                {
                    "event": "plan_only",
                    "windows": len(plan["windows"]),
                    "plan_sha256": plan["plan_sha256"],
                }
            )
        )
    else:
        if any((args.corpus, args.lens, args.lens_sha256)) or args.revision != "main":
            parser.error("execute consumes the frozen plan; input overrides are not allowed")
        execute_plan(args.out, spec)


if __name__ == "__main__":
    main()
