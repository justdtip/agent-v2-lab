"""`lab-device`: the device's step zero, one command with five parts.

``preflight`` checks the environment, the weights and the determinism in one run and writes a
report every number of which says what kind of number it is; ``login`` takes the Hugging Face
token from a prompt and hands it to the hub's own store, never to this code; ``fetch`` says which
checkpoints fit the device under which mode and downloads the ones that do; ``pack-data`` and
``verify-data`` carry a rendered dataset, which is untracked and exists once, across with its
manifest and its digests. Plan §16.12 and the first-hour runbook.

Nothing here loads a model. ``preflight`` reads checkpoint headers through
:func:`local_llm_lab.hf_text.checkpoint_metadata`, which loads no tensor.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import platform
import re
import sys
import tarfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

from local_llm_lab.project import configure_local_cache

__all__ = ["DEFAULT_MODELS", "MODES", "main"]

#: The checkpoints worth asking about on a device, largest first within a family. Gemma 3 has no
#: 9B; its sizes are 1B, 4B, 12B and 27B. Any other id is passed on the command line.
DEFAULT_MODELS = (
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
)
#: Bytes per parameter a mode needs resident, from plan §10 and §16: bf16 weights for inference,
#: weights plus a small adapter and its states for LoRA, fp32 master weights, gradients and two
#: AdamW moments for full fine-tuning. Activations and logits come on top and are measured, not
#: assumed (§10.2); the table is a feasibility verdict, not a peak.
MODES = {"inference": 2.0, "lora": 2.5, "full": 16.0}
#: What a checkpoint download takes: weights, configs and tokenizer, never other formats.
_ALLOW = ("*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "*.py")
GIB = 2**30


def _version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _row(check: str, ok: bool | None, detail: str, *, basis: str = "measured-here") -> dict:
    return {"check": check, "ok": ok, "detail": detail, "basis": basis}


# ------------------------------------------------------------------------------- preflight


def _git(args: list[str]) -> str | None:
    from local_llm_lab import spawn

    try:
        done = spawn.run(["git", *args], capture_output=True, text=True, check=True)
    except Exception:  # noqa: BLE001 - a box without git is reported, not refused
        return None
    return done.stdout.strip()


def _snapshot_dir(repo_id: str) -> Path | None:
    """The cached snapshot for ``repo_id`` under the project's HF cache, or ``None``."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, local_files_only=True, allow_patterns=list(_ALLOW)))
    except Exception:  # noqa: BLE001 - absent is a finding, not an error
        return None


def _torch_models() -> list[tuple[str, str]]:
    """Registry entries the torch path can load: ``(name, hf_id)`` with a hub id, not `models/`."""
    from local_llm_lab.models import load_model_spec, registered_models

    out = []
    for name in registered_models():
        spec = load_model_spec(name)
        hf_id = spec.hf_id
        if hf_id.startswith(("/", "models/")) or "mlx" in hf_id.lower():
            continue  # an MLX conversion is the MLX entry's checkpoint, not the torch path's
        out.append((name, hf_id))
    return out


def preflight(args: argparse.Namespace) -> int:
    from local_llm_lab import device

    cache = configure_local_cache()
    rows: list[dict] = [_row("hf cache", True, str(cache))]
    rows.append(_row("python", True, platform.python_version()))
    torch_version = _version("torch")
    rows.append(
        _row("torch >= 2.14", bool(torch_version and torch_version >= "2.14"), str(torch_version))
    )
    rows.append(
        _row("transformers", _version("transformers") is not None, str(_version("transformers")))
    )
    for extra in ("accelerate", "peft", "safetensors"):
        rows.append(_row(extra, _version(extra) is not None, str(_version(extra))))
    rows.append(
        _row(
            "mlx absent",
            _version("mlx") is None,
            str(_version("mlx")) + " (must not be installed on the device)",
        )
    )

    try:
        from local_llm_lab.upstream_ref import EXPECTED_JLENS_COMMIT, load_upstream

        up = load_upstream()
        rows.append(
            _row(
                "jlens",
                up.commit == EXPECTED_JLENS_COMMIT,
                f"{up.path} @ {up.commit} (expected {EXPECTED_JLENS_COMMIT[:7]})",
            )
        )
    except Exception as error:  # noqa: BLE001 - reported as the failing row
        rows.append(_row("jlens", False, f"unavailable: {error}"))

    try:
        backend = device.backend()
    except Exception as error:  # noqa: BLE001
        backend = None
        rows.append(_row("backend", False, str(error)))
    else:
        rows.append(
            _row("backend is torch", backend == "torch", f"{backend} (set LLL_BACKEND=torch)")
        )

    cuda_ok = None
    if backend == "torch":
        import torch

        available = torch.cuda.is_available()
        if available:
            info = device.device_info("all")
            names = ", ".join(
                f"{i}: {e['name']} {e['memory_size'] / GIB:.1f} GiB" for i, e in info.items()
            )
            rows.append(_row("cuda", True, names))
            cuda_ok = True
        else:
            cuda_ok = bool(args.allow_cpu)
            rows.append(
                _row(
                    "cuda",
                    cuda_ok,
                    "not available" + (" (allowed: --allow-cpu)" if args.allow_cpu else ""),
                )
            )
        initialised_before = torch.cuda.is_initialized()
        reading = device.pin(seed=args.seed, attention="eager")
        pinned = (
            reading["determinism"] == "pinned"
            and reading.get("deterministic_algorithms") is True
            and reading.get("tf32_matmul") is False
            and reading.get(device.CUBLAS_ENV) == device.CUBLAS_DETERMINISTIC
        )
        rows.append(
            _row(
                "determinism pinned before first CUDA use",
                pinned and not initialised_before,
                json.dumps(
                    {
                        k: reading.get(k)
                        for k in (
                            "determinism",
                            "deterministic_algorithms",
                            "tf32_matmul",
                            device.CUBLAS_ENV,
                            "attn_implementation" if "attn_implementation" in reading else "pinned",
                        )
                    }
                ),
            )
        )
        budget = device.budget() if available or args.allow_cpu else None
    else:
        budget = None

    from local_llm_lab.hf_text import checkpoint_metadata

    for name, hf_id in _torch_models():
        if args.model and name not in args.model:
            continue
        snapshot = _snapshot_dir(hf_id)
        if snapshot is None:
            rows.append(
                _row(
                    f"weights {name}",
                    False,
                    f"{hf_id}: not in the HF cache; run `lab-device fetch`",
                )
            )
            continue
        try:
            meta = checkpoint_metadata(snapshot)
        except Exception as error:  # noqa: BLE001
            rows.append(_row(f"weights {name}", False, f"{snapshot}: {error}"))
            continue
        params = meta["text_bytes"] / 2 if meta["storage_dtypes"] in (["BF16"], ["F16"]) else None
        verdict = ""
        if params and budget:
            fits = {mode: params * per <= budget for mode, per in MODES.items()}
            verdict = " | fits under the R47 budget: " + ", ".join(
                f"{m}={'yes' if v else 'no'}" for m, v in fits.items()
            )
        rows.append(
            _row(
                f"weights {name}",
                True,
                f"{snapshot.name} format={meta['safetensors_format']} "
                f"text={meta['text_bytes'] / GIB:.2f} GiB"
                + (f" params~{params / 1e9:.2f}B" if params else "")
                + verdict,
                basis="measured-here" if budget else "laptop-basis",
            )
        )

    from local_llm_lab.pipeline.data import require_dataset_manifest

    for directory in args.data or []:
        path = Path(directory)
        try:
            manifest = require_dataset_manifest(path)
            splits = sorted(p.name for p in path.glob("*.jsonl"))
            rows.append(
                _row(
                    f"data {path.name}", bool(splits), f"manifest {manifest.name}; splits {splits}"
                )
            )
        except Exception as error:  # noqa: BLE001
            rows.append(_row(f"data {path.name}", False, str(error)))

    head = _git(["rev-parse", "--short", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    rows.append(
        _row(
            "git",
            head is not None,
            f"HEAD {head}; tree "
            + ("clean" if not dirty else f"MODIFIED ({len(dirty.splitlines())} paths)"),
        )
    )
    try:
        from local_llm_lab.runlock import default_window_path

        window = Path(default_window_path())
        rows.append(
            _row(
                "box window",
                not window.exists(),
                "free" if not window.exists() else f"held: {window}",
            )
        )
    except Exception as error:  # noqa: BLE001
        rows.append(_row("box window", None, str(error)))

    report = {
        "tool": "lab-device preflight",
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device.describe() if backend == "torch" else {"backend": backend},
        "rows": rows,
        "ok": all(r["ok"] is not False for r in rows),
    }
    width = max(len(r["check"]) for r in rows)
    for r in rows:
        mark = {True: "ok  ", False: "FAIL", None: "?   "}[r["ok"]]
        print(f"{mark} {r['check']:<{width}}  {r['detail']}")
    print("preflight:", "OK" if report["ok"] else "FAILED, see the rows marked FAIL")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"report written to {args.json}")
    return 0 if report["ok"] else 1


# ------------------------------------------------------------------------------------ login


def login(args: argparse.Namespace) -> int:
    """Prompt for the token and hand it to the hub's own store; this code never keeps it."""
    cache = configure_local_cache()
    if not sys.stdin.isatty() and not args.stdin:
        print(
            "lab-device login needs a terminal to prompt on (or --stdin to read one line)",
            file=sys.stderr,
        )
        return 2
    token = (
        sys.stdin.readline().strip()
        if args.stdin
        else getpass.getpass("Paste your Hugging Face token (hidden): ")
    ).strip()
    if not token:
        print("no token given", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    from huggingface_hub import login as hub_login

    hub_login(token=token, add_to_git_credential=False)
    del token
    who = HfApi().whoami()
    print(
        f"logged in as {who.get('name', '?')}; the token is stored by huggingface_hub under {cache}"
    )
    print(
        "do not copy that cache directory to another machine; run `lab-device login` there instead"
    )
    return 0


# ------------------------------------------------------------------------------------ fetch


def _hub_size(repo_id: str) -> tuple[float | None, int | None]:
    """(parameter count, safetensors bytes) from the hub's metadata, without downloading."""
    from huggingface_hub import HfApi, get_safetensors_metadata

    params = None
    try:
        meta = get_safetensors_metadata(repo_id)
        params = float(sum(meta.parameter_count.values()))
    except Exception:  # noqa: BLE001 - some repos carry no index; the size below still answers
        params = None
    total = None
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
    except Exception:  # noqa: BLE001
        total = None
    if params is None and total:
        params = total / 2  # bf16 is the hub's norm for these families
    return params, total


def feasibility(repo_id: str, budget_bytes: float, params: float | None) -> dict[str, bool | None]:
    if params is None:
        return {mode: None for mode in MODES}
    return {mode: params * per <= budget_bytes for mode, per in MODES.items()}


def fetch(args: argparse.Namespace) -> int:
    cache = configure_local_cache()
    budget = args.budget_gib * GIB if args.budget_gib else None
    if budget is None:
        from local_llm_lab import device

        try:
            budget = device.budget()
        except Exception as error:  # noqa: BLE001
            print(
                f"no --budget-gib and the device budget could not be read: {error}", file=sys.stderr
            )
            return 2
    print(
        f"budget {budget / GIB:.1f} GiB (R47 fraction of the device); modes: "
        + ", ".join(f"{m}={p}B/param" for m, p in MODES.items())
    )
    chosen: list[str] = []
    for repo_id in args.ids or list(DEFAULT_MODELS):
        params, total = _hub_size(repo_id)
        fits = feasibility(repo_id, budget, params)
        size = f"{total / GIB:.1f} GiB" if total else "size unknown"
        p = f"{params / 1e9:.1f}B" if params else "params unknown"
        verdict = ", ".join(
            f"{m}={'yes' if v else 'no' if v is False else '?'}" for m, v in fits.items()
        )
        wanted = fits.get(args.mode) is True or args.all
        print(f"{'FETCH' if wanted else 'skip '}  {repo_id:<28} {p:>10} {size:>10}  {verdict}")
        if wanted:
            chosen.append(repo_id)
    if args.dry_run:
        print("dry run; nothing downloaded")
        return 0
    from huggingface_hub import snapshot_download

    from local_llm_lab.hf_text import checkpoint_metadata

    for repo_id in chosen:
        print(f"downloading {repo_id} into {cache} ...")
        path = Path(snapshot_download(repo_id, allow_patterns=list(_ALLOW)))
        meta = checkpoint_metadata(path)
        print(
            f"  {path}: format={meta['safetensors_format']} "
            f"text={meta['text_bytes'] / GIB:.2f} GiB wrapper={meta['wrapper']}"
        )
    return 0


# ------------------------------------------------------------------------- dictionaries


#: The Gemma Scope 2 repositories are laid out as ``<site>/layer_<N>_width_<W>_l0_<S>/`` holding
#: ``config.json`` (the hook string and the sparsity), ``params.safetensors`` (the dictionary) and
#: ``examples.safetensors`` (activation examples, large and not needed for the bridge).
#: ``resid_post`` is the deep-dive subset of a few depths; ``resid_post_all`` is every layer. Read
#: from the hub's file listing on 2026-09-10, not assumed. Two traps the D-CRO measured: a folder
#: pull takes ``examples.safetensors`` (816 MB against 336 MB of parameters), so files are named;
#: and the deep-dive and every-layer files at one hook, width and ``l0`` are identical in size and
#: header while ``l0_big`` means 150 in one suite and 120 in the other, so a file is verified by
#: the hub's own digest for its exact path, never by its length.
DICTIONARY_SITES = (
    "resid_post",
    "resid_post_all",
    "attn_out",
    "attn_out_all",
    "mlp_out",
    "mlp_out_all",
)


def _dictionary_patterns(site: str, layers: list[int], width: str, l0: str, examples: bool):
    names = ("config.json", "params.safetensors") + (("examples.safetensors",) if examples else ())
    return [f"{site}/layer_{n}_width_{width}_l0_{l0}/{name}" for n in layers for name in names]


def _declared_digest(sibling) -> tuple[str, str] | None:
    """(algorithm, hex) the hub declares for a file: LFS objects carry a sha256, and every other
    file its git blob id, which is the sha1 of ``blob <size>\\0`` followed by the content."""
    lfs = getattr(sibling, "lfs", None)
    sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if sha256:
        return "sha256", sha256
    if getattr(sibling, "blob_id", None):
        return "blob", sibling.blob_id
    return None


def _file_digest(path: Path, algorithm: str) -> str:
    if algorithm == "blob":
        h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    else:
        h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_dictionary(args: argparse.Namespace) -> int:
    """Download sparse dictionary layers by exact filename, saying which files and how many bytes
    first, and verify each against the digest the hub declares for that path."""
    cache = configure_local_cache()
    if not args.layer and not args.all_layers:
        print("name --layer N (repeatable) or --all-layers", file=sys.stderr)
        return 2
    patterns = _dictionary_patterns(args.site, args.layer or [], args.width, args.l0, args.examples)
    from huggingface_hub import HfApi, snapshot_download

    try:
        info = HfApi().model_info(args.repo, files_metadata=True)
    except Exception as error:  # noqa: BLE001 - a gated repository without a login says so here
        print(f"cannot list {args.repo}: {error}; run `lab-device login` first", file=sys.stderr)
        return 2
    siblings = {s.rfilename: s for s in info.siblings}
    if args.all_layers:
        # Every layer the repository holds at this site, width and sparsity, read from its
        # listing: 0-33 for the 4B, 0-47 for the 12B, and nothing assumed about either.
        stem = re.compile(
            rf"^{re.escape(args.site)}/layer_(\d+)_width_{re.escape(args.width)}_l0_{args.l0}/params\.safetensors$"
        )
        found = sorted({int(m.group(1)) for name in siblings if (m := stem.match(name))})
        if not found:
            print(
                f"{args.repo} has no {args.site} layers at width {args.width} l0 {args.l0}",
                file=sys.stderr,
            )
            return 2
        args.layer = found
        patterns = _dictionary_patterns(args.site, found, args.width, args.l0, args.examples)
        print(f"{len(found)} layers listed: {found[0]}-{found[-1]}")
    missing = [pattern for pattern in patterns if pattern not in siblings]
    if missing:
        print(f"{args.repo} has no such files: {missing}", file=sys.stderr)
        return 2
    total = sum((siblings[pattern].size or 0) for pattern in patterns)
    for pattern in patterns:
        print(f"  {(siblings[pattern].size or 0) / 2**20:8.1f} MiB  {pattern}")
    print(f"{len(patterns)} files, {total / GIB:.2f} GiB, into {cache}")
    if args.dry_run:
        return 0
    root = Path(snapshot_download(args.repo, allow_patterns=patterns))
    for pattern in patterns:
        declared = _declared_digest(siblings[pattern])
        if declared is None:
            print(
                f"{pattern}: the hub declares no digest; refusing to call it verified",
                file=sys.stderr,
            )
            return 1
        algorithm, expected = declared
        actual = _file_digest(root / pattern, algorithm)
        if actual != expected:
            print(f"{pattern}: {algorithm} {actual} != declared {expected}", file=sys.stderr)
            return 1
        print(f"  verified {algorithm} {actual[:16]}…  {pattern}")
    for layer in args.layer:
        config = root / args.site / f"layer_{layer}_width_{args.width}_l0_{args.l0}" / "config.json"
        try:
            declared = json.loads(config.read_text())
        except (OSError, ValueError) as error:
            print(f"layer {layer}: config unreadable: {error}", file=sys.stderr)
            return 1
        # The config's own keys, read from a real one: hf_hook_point_in, model_name, width, l0.
        hook = declared.get("hf_hook_point_in", declared.get("hook_name"))
        print(
            f"layer {layer}: model={declared.get('model_name')!r} hook={hook!r} "
            f"l0={declared.get('l0')} width={declared.get('width')}"
        )
    print(f"dictionary layers under {root}")
    return 0


# -------------------------------------------------------------------------- pack / verify data


def _digests(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def pack_data(args: argparse.Namespace) -> int:
    from local_llm_lab.pipeline.data import require_dataset_manifest

    source = Path(args.directory).resolve()
    require_dataset_manifest(source)
    digests = _digests(source)
    out = Path(args.out) if args.out else Path("outputs/transfer") / f"{source.name}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(source, arcname=source.name)
        sums = json.dumps(digests, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo(f"{source.name}/SHA256SUMS.json")
        info.size = len(sums)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(sums))
    print(f"packed {source} ({len(digests)} files) -> {out}")
    return 0


def verify_data(args: argparse.Namespace) -> int:
    from local_llm_lab.pipeline.data import require_dataset_manifest

    archive = Path(args.archive)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
        names = [m.name.split("/", 1)[0] for m in tar.getmembers()]
    root = dest / names[0]
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    (root / "SHA256SUMS.json").unlink()
    actual = _digests(root)
    wrong = sorted(k for k in set(sums) | set(actual) if sums.get(k) != actual.get(k))
    if wrong:
        print(f"digest mismatch in {root}: {wrong}", file=sys.stderr)
        return 1
    require_dataset_manifest(root)
    print(f"verified {root}: {len(actual)} files, digests match, manifest present")
    return 0


# ------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab-device", description="the device's step zero")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight", help="environment, weights and determinism in one run")
    p.add_argument("--json", help="write the report here")
    p.add_argument(
        "--model",
        action="append",
        help="registry name(s) to check; default: every torch-loadable entry",
    )
    p.add_argument("--data", action="append", help="rendered dataset directory to check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="a box without CUDA passes the cuda row (laptop use)",
    )
    p.set_defaults(func=preflight)
    lg = sub.add_parser("login", help="prompt for the Hugging Face token; the hub stores it")
    lg.add_argument(
        "--stdin", action="store_true", help="read one line from stdin instead of prompting"
    )
    lg.set_defaults(func=login)
    f = sub.add_parser("fetch", help="say which checkpoints fit, and download those that do")
    f.add_argument("ids", nargs="*", help=f"hub ids; default: {', '.join(DEFAULT_MODELS)}")
    f.add_argument(
        "--budget-gib", type=float, help="memory to plan under; default: the device's R47 budget"
    )
    f.add_argument("--mode", choices=sorted(MODES), default="inference", help="which mode must fit")
    f.add_argument("--all", action="store_true", help="download every id regardless of the verdict")
    f.add_argument("--dry-run", action="store_true", help="verdicts only")
    f.set_defaults(func=fetch)
    pk = sub.add_parser("pack-data", help="tar a rendered dataset with its manifest and digests")
    pk.add_argument("directory")
    pk.add_argument("--out")
    pk.set_defaults(func=pack_data)
    fd = sub.add_parser(
        "fetch-dictionary", help="download sparse dictionary layers (Gemma Scope 2)"
    )
    fd.add_argument("repo", help="e.g. google/gemma-scope-2-4b-it")
    fd.add_argument("--layer", type=int, action="append", help="block index; repeatable")
    fd.add_argument(
        "--all-layers", action="store_true", help="every layer the repository lists at this site"
    )
    fd.add_argument("--site", default="resid_post_all", choices=DICTIONARY_SITES)
    fd.add_argument("--width", default="16k")
    fd.add_argument("--l0", default="small", choices=("small", "medium", "big"))
    fd.add_argument("--examples", action="store_true", help="also fetch examples.safetensors")
    fd.add_argument("--dry-run", action="store_true")
    fd.set_defaults(func=fetch_dictionary)
    vf = sub.add_parser(
        "verify-data", help="extract a packed dataset and verify digests and manifest"
    )
    vf.add_argument("archive")
    vf.add_argument("--dest", default="data")
    vf.set_defaults(func=verify_data)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
