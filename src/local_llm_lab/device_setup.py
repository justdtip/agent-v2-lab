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


DICTIONARY_FILES = ("config.json", "params.safetensors")


def dictionary_sidecar(cache: Path, repo: str, folder: str) -> Path:
    """Where ``fetch-dictionary`` records the digests it verified, for an offline preflight."""
    return Path(cache) / "dictionaries" / repo / folder / "DIGEST.json"


def verify_cached_dictionary(repo: str, folder: str, *, offline: bool) -> dict:
    """Each of a dictionary folder's files: present in the cache, and its bytes against a digest.

    The expected digest comes from the hub (basis ``hub-declared``) or, offline or when the hub
    cannot be reached, from what ``fetch-dictionary`` recorded (``recorded-at-fetch``). With
    neither the file is reported as **undecided**, never as verified: a check that cannot compare
    must not say the bytes agree. Nothing here downloads.
    """
    from huggingface_hub import HfApi, hf_hub_download

    cache = configure_local_cache()
    sidecar = dictionary_sidecar(cache, repo, folder)
    recorded = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
    declared: dict[str, tuple[str, str]] = {}
    basis, note = None, None
    if not offline:
        try:
            siblings = {
                s.rfilename: s for s in HfApi().model_info(repo, files_metadata=True).siblings
            }
        except Exception as error:  # noqa: BLE001 - offline is a state, reported not raised
            note = f"hub not reachable: {error}"
        else:
            for name in DICTIONARY_FILES:
                sibling = siblings.get(f"{folder}/{name}")
                digest = _declared_digest(sibling) if sibling is not None else None
                if digest is not None:
                    declared[name] = digest
            basis = "hub-declared"
    if not declared and recorded:
        declared = {k: (v["algorithm"], v["digest"]) for k, v in recorded["files"].items()}
        basis = "recorded-at-fetch"
    out: dict = {"repo": repo, "folder": folder, "basis": basis, "note": note, "files": {}}
    for name in DICTIONARY_FILES:
        try:
            path = Path(hf_hub_download(repo, f"{folder}/{name}", local_files_only=True))
        except Exception:  # noqa: BLE001 - absent is the finding
            path = None
        entry: dict = {"present": path is not None, "path": str(path) if path else None}
        if path is not None and name in declared:
            algorithm, expected = declared[name]
            actual = _file_digest(path, algorithm)
            entry.update(
                algorithm=algorithm, expected=expected, actual=actual, ok=actual == expected
            )
        elif path is not None:
            entry.update(ok=None, note="undecided: no hub answer and no digest recorded at fetch")
        out["files"][name] = entry
    return out


def _write_sidecar(cache: Path, repo: str, folder: str, verified: dict, config: dict) -> Path:
    sidecar = dictionary_sidecar(cache, repo, folder)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo": repo,
        "folder": folder,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digest_source": "the hub's declared digest per exact path: LFS sha256, else git blob id",
        "files": verified,
        "config": config,
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def fetch_dictionary(args: argparse.Namespace) -> int:
    """Download sparse dictionary layers by exact filename, saying which files and how many bytes
    first, verify each against the digest the hub declares for that path, and record the digests
    beside the cache so a later offline preflight can check the bytes again."""
    cache = configure_local_cache()
    if args.offline:
        return _fetch_dictionary_offline(args, cache)
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
        head = rf"^{re.escape(args.site)}/layer_(\d+)_width_{re.escape(args.width)}_l0_{args.l0}"
        stem = re.compile(head + r"/params\.safetensors$")
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
    verified: dict[str, dict] = {}
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
        verified[pattern] = {
            "algorithm": algorithm,
            "digest": actual,
            "bytes": (root / pattern).stat().st_size,
        }
    for layer in args.layer:
        folder = f"{args.site}/layer_{layer}_width_{args.width}_l0_{args.l0}"
        config = root / folder / "config.json"
        try:
            declared_config = json.loads(config.read_text())
        except (OSError, ValueError) as error:
            print(f"layer {layer}: config unreadable: {error}", file=sys.stderr)
            return 1
        # The config's own keys, read from a real one: hf_hook_point_in, model_name, width, l0.
        hook = declared_config.get("hf_hook_point_in", declared_config.get("hook_name"))
        print(
            f"layer {layer}: model={declared_config.get('model_name')!r} hook={hook!r} "
            f"l0={declared_config.get('l0')} width={declared_config.get('width')}"
        )
        files = {p.rsplit("/", 1)[1]: v for p, v in verified.items() if p.startswith(folder + "/")}
        sidecar = _write_sidecar(cache, args.repo, folder, files, declared_config)
        print(f"  recorded {sidecar}")
    print(f"dictionary layers under {root}")
    return 0


def _fetch_dictionary_offline(args: argparse.Namespace, cache: Path) -> int:
    """No hub: verify the cached files of the named layers against the recorded digests."""
    if args.all_layers:
        base = Path(cache) / "dictionaries" / args.repo / args.site
        stem = re.compile(rf"^layer_(\d+)_width_{re.escape(args.width)}_l0_{args.l0}$")
        found = sorted(
            int(m.group(1))
            for d in (base.iterdir() if base.is_dir() else ())
            if (m := stem.match(d.name))
        )
        if not found:
            print(f"offline: no recorded layers under {base}", file=sys.stderr)
            return 2
        args.layer = found
    if not args.layer:
        print("name --layer N (repeatable) or --all-layers", file=sys.stderr)
        return 2
    failed = 0
    for layer in args.layer:
        folder = f"{args.site}/layer_{layer}_width_{args.width}_l0_{args.l0}"
        result = verify_cached_dictionary(args.repo, folder, offline=True)
        for name, entry in result["files"].items():
            state = (
                "absent"
                if not entry["present"]
                else "undecided"
                if entry.get("ok") is None
                else "verified"
                if entry["ok"]
                else "MISMATCH"
            )
            failed += state in ("absent", "undecided", "MISMATCH")
            print(f"  {state:>9}  {folder}/{name}  ({result['basis'] or 'no digest'})")
    return 1 if failed else 0


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


# --------------------------------------------------------------------------------- bootstrap


#: The device's default fetch, sized for a 250-300 GB disk (the Director, 2026-09-10): the two
#: Gemma sizes and their every-layer residual dictionaries, about 31 GiB and 33 GiB by the hub's
#: listings. The 27B (51 GiB) and the Qwen pair are named on the command line when wanted.
BOOTSTRAP_MODELS = ("google/gemma-3-4b-it", "google/gemma-3-12b-it")
BOOTSTRAP_DICTIONARIES = ("google/gemma-scope-2-4b-it", "google/gemma-scope-2-12b-it")


def _logged_in() -> str | None:
    """The hub user the stored token belongs to, or ``None`` when there is no usable token."""
    from huggingface_hub import HfApi

    try:
        return str(HfApi().whoami().get("name", "?"))
    except Exception:  # noqa: BLE001 - not logged in is the finding, not an error
        return None


def _dictionary_plan(repo: str, site: str, width: str, l0: str) -> tuple[int, int]:
    """``(layers, bytes)`` an every-layer fetch of ``repo`` would take, from the hub's listing."""
    from huggingface_hub import HfApi

    stem = re.compile(
        rf"^{re.escape(site)}/layer_(\d+)_width_{re.escape(width)}_l0_{l0}/"
        r"(params\.safetensors|config\.json)$"
    )
    layers: set[int] = set()
    total = 0
    for sibling in HfApi().model_info(repo, files_metadata=True).siblings:
        match = stem.match(sibling.rfilename)
        if match:
            layers.add(int(match.group(1)))
            total += sibling.size or 0
    return len(layers), total


def _entries_by_base() -> dict[str, str]:
    """Registry entries the torch path loads, keyed by the base each descends from."""
    from local_llm_lab.models import base_of_artifact, load_model_spec

    return {base_of_artifact(load_model_spec(name).base): name for name, _ in _torch_models()}


def bootstrap(args: argparse.Namespace) -> int:
    """The device's step zero as one command: login, the disk plan, models, dictionaries, data,
    preflight. Each step runs the same code as its subcommand, in that order; the first failure
    stops the sequence with its row, and the report is written either way. The login prompt is
    the only thing that needs a hand."""
    import shutil

    cache = configure_local_cache()
    started = time.time()
    rows: list[dict] = []
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report = Path(args.report or f"outputs/bootstrap-{stamp}.json")

    def step(name: str, ok: bool | None, detail: str) -> bool:
        rows.append(
            {"step": name, "ok": ok, "detail": detail, "at_s": round(time.time() - started, 1)}
        )
        mark = "ok  " if ok else ("FAIL" if ok is False else "note")
        print(f"{mark} {name:<30} {detail}")
        return ok is not False

    def finish(code: int) -> int:
        report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rows": rows,
            "exit": code,
            "cache": str(cache),
            "seconds": round(time.time() - started, 1),
        }
        report.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"bootstrap {'PASSED' if code == 0 else 'FAILED'} in {payload['seconds']:.0f}s; "
            f"report {report}"
        )
        return code

    # 0. login: the one prompt, skipped when a usable token is already stored.
    who = _logged_in()
    if who is not None:
        step("login", True, f"already logged in as {who}")
    elif args.skip_login:
        step("login", False, "no usable token and --skip-login given")
        return finish(2)
    else:
        code = login(argparse.Namespace(stdin=False))
        detail = "token handed to the hub's store" if code == 0 else f"login exited {code}"
        if not step("login", code == 0, detail):
            return finish(2)

    # 1. the disk plan, from the hub's listings, before any byte is downloaded.
    plan: list[tuple[str, int]] = []
    for repo_id in args.models:
        _params, size = _hub_size(repo_id)
        if not size:
            step(
                "plan",
                False,
                f"{repo_id}: the hub reports no size; is the id right and the token accepted?",
            )
            return finish(2)
        plan.append((repo_id, size))
    for repo in args.dictionaries:
        try:
            layers, size = _dictionary_plan(repo, args.site, args.width, args.l0)
        except Exception as error:  # noqa: BLE001 - a gated listing without access says so here
            step("plan", False, f"cannot list {repo}: {error}")
            return finish(2)
        if not layers:
            step("plan", False, f"{repo}: no {args.site} layers at width {args.width} l0 {args.l0}")
            return finish(2)
        plan.append((f"{repo} ({layers} layers)", size))
    total = sum(size for _, size in plan)
    for label, size in plan:
        print(f"  {size / GIB:8.2f} GiB  {label}")
    free = shutil.disk_usage(cache).free
    reserve = args.disk_reserve_gib * GIB
    detail = (
        f"{total / GIB:.1f} GiB to fetch plus {reserve / GIB:.0f} GiB reserve, against "
        f"{free / GIB:.1f} GiB free under {cache}"
    )
    if not step("plan", total + reserve <= free, detail):
        return finish(1)
    if args.dry_run:
        step("dry-run", None, "stopping before any download")
        return finish(0)

    # 2. models, through `fetch`, every named id regardless of the mode verdicts.
    budget_gib = args.budget_gib
    if budget_gib is None:
        from local_llm_lab import device

        try:
            budget_gib = device.budget() / GIB
        except Exception:  # noqa: BLE001 - the verdict columns then read "no"; --all fetches anyway
            budget_gib = 0.0
    code = fetch(
        argparse.Namespace(
            ids=list(args.models), budget_gib=budget_gib, mode="inference", all=True, dry_run=False
        )
    )
    if not step(
        "models", code == 0, ", ".join(args.models) if code == 0 else f"fetch exited {code}"
    ):
        return finish(1)

    # 3. dictionaries, every layer, each file verified by digest and recorded.
    for repo in args.dictionaries:
        code = fetch_dictionary(
            argparse.Namespace(
                repo=repo,
                layer=None,
                all_layers=True,
                site=args.site,
                width=args.width,
                l0=args.l0,
                examples=False,
                dry_run=False,
                offline=False,
            )
        )
        detail = (
            "every layer fetched and verified by digest"
            if code == 0
            else f"fetch-dictionary exited {code}"
        )
        if not step(f"dictionary {repo}", code == 0, detail):
            return finish(1)

    # 4. data archives, verified by digest into the data directory.
    data_dirs: list[Path] = []
    for archive in args.data_archive or []:
        code = verify_data(argparse.Namespace(archive=archive, dest=args.data_dest))
        target = Path(args.data_dest) / Path(archive).name.removesuffix(".tar.gz")
        detail = f"verified into {target}" if code == 0 else f"verify-data exited {code}"
        if not step(f"data {Path(archive).name}", code == 0, detail):
            return finish(1)
        data_dirs.append(target)

    # 5. preflight per entry, each with the dictionary trained on its base, paired by the config
    # the fetch recorded rather than by position on the command line.
    from local_llm_lab.models import base_of_artifact

    by_base = _entries_by_base()
    folder = f"{args.site}/layer_{args.check_layer}_width_{args.width}_l0_{args.l0}"
    dictionary_for: dict[str, str] = {}
    for repo in args.dictionaries:
        sidecar = dictionary_sidecar(cache, repo, folder)
        if not sidecar.is_file():
            step(
                "pairing",
                False,
                f"{sidecar} missing after the fetch; --check-layer {args.check_layer} names a "
                "layer it did not record",
            )
            return finish(1)
        named = json.loads(sidecar.read_text(encoding="utf-8")).get("config", {}).get("model_name")
        entry = by_base.get(base_of_artifact(named)) if named else None
        if entry is None:
            step(
                "pairing",
                False,
                f"{repo} names {named!r}, which no torch registry entry descends from",
            )
            return finish(1)
        dictionary_for[entry] = f"{repo}:{folder}"
    entries = [by_base.get(base_of_artifact(m)) for m in args.models]
    if any(entry is None for entry in entries):
        missing = [m for m, entry in zip(args.models, entries, strict=True) if entry is None]
        step("pairing", False, f"no torch registry entry descends from {missing}")
        return finish(1)
    render_ready = Path(args.render_source).is_dir() and Path(args.render_manifest).is_file()
    if not render_ready:
        step(
            "render",
            None,
            f"skipped: {args.render_source} or {args.render_manifest} absent; pass both data "
            "archives to run the re-render check",
        )
    for entry in entries:
        with_render = render_ready and entry == args.render_model
        code = preflight(
            argparse.Namespace(
                json=str(report.with_name(f"{report.stem}-preflight-{entry}.json")),
                model=[entry],
                data=[str(d) for d in data_dirs] or None,
                seed=0,
                allow_cpu=args.allow_cpu,
                dictionary=[dictionary_for[entry]] if entry in dictionary_for else None,
                offline=True,
                render_source=args.render_source if with_render else None,
                render_manifest=args.render_manifest if with_render else None,
            )
        )
        detail = (
            "every row passed" + (", the re-render included" if with_render else "")
            if code == 0
            else f"preflight exited {code}; see its report"
        )
        if not step(f"preflight {entry}", code == 0, detail):
            return finish(1)
    return finish(0)


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
    bs = sub.add_parser(
        "bootstrap",
        help="step zero as one command: login, plan, models, dictionaries, data, preflight",
    )
    bs.add_argument("--models", nargs="*", default=list(BOOTSTRAP_MODELS), help="hub ids to fetch")
    bs.add_argument("--dictionaries", nargs="*", default=list(BOOTSTRAP_DICTIONARIES))
    bs.add_argument("--site", default="resid_post_all", choices=DICTIONARY_SITES)
    bs.add_argument("--width", default="16k")
    bs.add_argument("--l0", default="small", choices=("small", "medium", "big"))
    bs.add_argument(
        "--check-layer", type=int, default=17, help="the layer preflight's dictionary rows read"
    )
    bs.add_argument(
        "--data-archive",
        action="append",
        help="a pack-data archive to verify into --data-dest; repeatable",
    )
    bs.add_argument("--data-dest", default="data")
    bs.add_argument("--render-source", default="data/agent_v2e")
    bs.add_argument("--render-manifest", default="data/agent_v2e-gemma3-4b/manifest.json")
    bs.add_argument(
        "--render-model",
        default="gemma3-12b-cuda-bf16",
        help="the entry whose re-render must reproduce the laptop digests",
    )
    bs.add_argument(
        "--disk-reserve-gib",
        type=float,
        default=40.0,
        help="free space kept for checkpoints and captures",
    )
    bs.add_argument(
        "--budget-gib", type=float, help="memory to plan under; default: the device's R47 budget"
    )
    bs.add_argument("--allow-cpu", action="store_true")
    bs.add_argument(
        "--skip-login", action="store_true", help="fail rather than prompt when no token is stored"
    )
    bs.add_argument("--dry-run", action="store_true", help="login and the disk plan only")
    bs.add_argument("--report", help="where the report goes; default outputs/bootstrap-<utc>.json")
    bs.set_defaults(func=bootstrap)
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
    fd.add_argument(
        "--offline",
        action="store_true",
        help="no hub: verify the cache against the recorded digests",
    )
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
