"""Read-only input audit. Tokenizes text, but never imports a model library or loads weights.

The preview is not a fitted corpus, a prompt-render conformance result, or model behavior.
"""

import argparse
import hashlib
import importlib.abc
import json
import re
import subprocess
import sys
from pathlib import Path


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm", "torch"}:
            raise ImportError("No model libraries in input audit: " + fullname)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def run(primary):
    project = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project / "src"))
    sys.meta_path.insert(0, NoNative())
    from tokenizers import Tokenizer

    from local_llm_lab.pipeline.lens_fitting.corpus import read_corpus
    from local_llm_lab.pipeline.protocol import SYSTEM_PROMPT
    from local_llm_lab.pipeline.tasks import make_tasks

    work_order = primary / "design_specifications/pending/GEMMA3-WORK-ORDERS-2026-09-08.md"
    old = primary / "research/records/LENS-FITTING-IMPLEMENTATION-2026-09-07"
    legacy = {}
    for name in ("agentic", "prose"):
        path = old / f"CORPUS-{name}-01.json"
        rows = read_corpus(path)
        metadata = json.loads(path.read_bytes())
        legacy[name] = {
            "manifest_sha256": sha(path),
            "rows": len(rows),
            "model_hf_id": metadata["model_hf_id"],
            "counts": metadata["counts"],
        }
    prose = json.loads((old / "CORPUS-prose-01.json").read_bytes())
    text = Path(prose["sources"][0]["path"])
    descriptor = json.loads(Path(prose["download_descriptor"]["path"]).read_bytes())
    snapshot = (
        primary
        / ".cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots"
        / "093f9f388b31de276ce2de164bdc2081324b9767"
    )
    tokenizer_file = snapshot / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    encoded = tokenizer.encode(text.read_text(), add_special_tokens=True)
    counts = {}
    for width in (128, 1024, 2048):
        windows, tail = divmod(len(encoded.ids), width)
        counts[str(width)] = {
            "windows": windows,
            "fit": windows - windows // 5,
            "held": windows // 5,
            "discarded_tail_tokens": tail,
        }
    selected = re.findall(r'\("(test-[a-z_]+-[0-9]+-clean)", 2\)', work_order.read_text())
    tasks = {t.task_id: t for t in make_tasks("test", 180, 20260902, difficulty=2)}
    if len(selected) != 12 or len(set(selected)) != 12 or not set(selected) <= tasks.keys():
        raise ValueError("work order no longer has the twelve expected resolvable pilot selections")
    prompts = [("system", SYSTEM_PROMPT)] + [(name, tasks[name].prompt) for name in selected]
    whitespace = [
        {"id": name, "leading": len(s) - len(s.lstrip()), "trailing": len(s) - len(s.rstrip())}
        for name, s in prompts
    ]
    return {
        "status": "input_audit_only_no_model_execution",
        "source_head": git_head(project),
        "primary_head": git_head(primary),
        "work_order_sha256": sha(work_order),
        "legacy_corpora_reader_check": legacy,
        "prose_source": {
            "path": str(text),
            "sha256": sha(text),
            "dataset_revision": descriptor["revision"],
            "dataset_split": descriptor["split"],
        },
        "gemma_tokenizer_preview": {
            "tokenizer_sha256": sha(tokenizer_file),
            "revision": snapshot.name,
            "tokens": len(encoded.ids),
            "add_special_tokens": True,
            "ids_sha256": hashlib.sha256(
                json.dumps(encoded.ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidate_partitions": counts,
            "scope": (
                "plain-text tokenizer.json preview, no registry identity "
                "or chat render certification"
            ),
        },
        "pilot_prompt_whitespace": whitespace,
        "whitespace_scope": (
            "initial system and twelve agentic task prompts only; generated turns, "
            "observations and chat require later audit"
        ),
        "pending_dependencies": {
            "gemma_registry_present": (primary / "configs/models/gemma3-4b.yaml").exists(),
            "note": (
                "architecture port, identity guard, official conversions and conformance "
                "evidence must be checked again before any fit"
            ),
        },
    }


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().primary), indent=2))
