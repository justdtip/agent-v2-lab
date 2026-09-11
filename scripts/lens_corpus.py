"""Build immutable fitting corpora without loading checkpoint weights.

Run with PYTHONPATH=src, for example:
  python scripts/lens_corpus.py --corpus agentic --evals eval.json --out corpus.json
  python scripts/lens_corpus.py --corpus prose --download-prose --data-dir data/lens \
      --out prose.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_llm_lab.models import load_model_spec
from local_llm_lab.pipeline.lens_fitting.corpus import (
    PROSE_CHUNK_TOKENS,
    build_agentic_corpus,
    build_prose_corpus,
    download_prose,
    load_corpus_tokenizer,
)

# Requirement §3.1 corpus data budget; this does not describe a model dimension.
CORPUS_MAX_TOKENS = 2048


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("agentic", "prose"), required=True)
    parser.add_argument("--evals", nargs="+", type=Path, default=[])
    parser.add_argument("--files", nargs="+", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True, help="immutable JSON manifest path")
    parser.add_argument("--model", default="qwen35-4b")
    parser.add_argument("--local-files-only", action="store_true", help="use cached tokenizer only")
    parser.add_argument("--download-prose", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--prose-chunk-tokens",
        type=int,
        help=f"prose window length, recorded in the manifest (default: {PROSE_CHUNK_TOKENS})",
    )
    parser.add_argument("--workspace-corpus", type=Path)
    parser.add_argument("--capture-index", type=Path)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument(
        "--max-tokens", type=int, help="workspace context refusal cap; never truncates"
    )
    args = parser.parse_args(argv)
    workspace_inputs = (
        args.workspace_corpus,
        args.capture_index,
        args.positions,
        args.tokenizer_json,
    )
    if any(workspace_inputs):
        if (
            not all(workspace_inputs)
            or args.corpus != "agentic"
            or args.evals
            or args.files
            or args.download_prose
            or args.prose_chunk_tokens
        ):
            parser.error(
                "workspace input requires agentic, corpus/index/positions/tokenizer, "
                "and no other source"
            )
        from local_llm_lab.pipeline.lens_fitting.workspace_corpus import build_workspace_corpus

        manifest = build_workspace_corpus(*workspace_inputs, args.out, max_tokens=args.max_tokens)
        print(
            json.dumps(
                {
                    "manifest": str(args.out.absolute()),
                    "domain": manifest["domain"],
                    "counts": manifest["counts"],
                    "dropped": 0,
                }
            )
        )
        return
    if args.max_tokens is not None:
        parser.error("--max-tokens requires workspace inputs")
    if args.prose_chunk_tokens is not None and (
        args.corpus != "prose" or args.prose_chunk_tokens <= 0
    ):
        parser.error("--prose-chunk-tokens requires prose and a positive integer")
    if args.corpus == "agentic":
        if not args.evals or args.files or args.download_prose:
            parser.error("agentic requires --evals and does not accept prose inputs")
    elif args.evals or not (args.files or args.download_prose):
        parser.error("prose requires --files or --download-prose and does not accept --evals")
    if args.download_prose and args.data_dir is None:
        parser.error("--download-prose requires --data-dir")
    if args.out.exists() or args.out.with_suffix(".jsonl").exists():
        raise FileExistsError("immutable corpus output already exists")
    spec = load_model_spec(args.model)
    tokenizer, files = load_corpus_tokenizer(spec, local_files_only=args.local_files_only)
    if args.corpus == "agentic":
        manifest = build_agentic_corpus(
            args.evals,
            tokenizer,
            spec,
            args.out,
            max_tokens=CORPUS_MAX_TOKENS,
            tokenizer_files=files,
        )
    else:
        descriptor = None
        if args.download_prose:
            downloaded, descriptor = download_prose(args.data_dir, revision=args.dataset_revision)
            args.files.append(downloaded)
        manifest = build_prose_corpus(
            args.files,
            tokenizer,
            spec,
            args.out,
            tokenizer_files=files,
            download_descriptor=descriptor,
            chunk_tokens=(
                PROSE_CHUNK_TOKENS if args.prose_chunk_tokens is None else args.prose_chunk_tokens
            ),
        )
    print(
        json.dumps(
            {
                "manifest": str(args.out.absolute()),
                "domain": manifest["domain"],
                "counts": manifest["counts"],
                "dropped": len(manifest["dropped_rows"]),
            }
        )
    )


if __name__ == "__main__":
    main()
