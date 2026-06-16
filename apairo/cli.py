"""apairo command-line interface.

A thin wrapper over the library -- no third-party dependencies. Commands mirror
familiar terminal/git verbs:

* ``apairo init``   -- write the ``.apairo`` sidecar(s) by scanning a directory
                       (sequence -> ``channels.yaml``; root -> ``dataset.yaml``).
* ``apairo status`` -- report what a dataset directory contains: sequences,
                       tracked channels, channels detected on disk but not yet
                       registered ("untracked"), event count, and config issues.

``add`` (register an untracked channel) and ``check`` (consistency check) are
planned follow-ups; ``status`` already surfaces what ``add`` will act on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from apairo.core.config import config_exists, read_config, verify_config
from apairo.dataset.kitti.dataset import _detect_loader
from apairo.dataset.raw import RawDataset
from apairo.dataset.raw.dataset import _read_manifest

# Datasets selectable with ``--as``. Generic (profile-free) for now; profiled
# datasets (Tartan, Semantic, ...) will register here as the CLI grows.
DATASETS = {"RawDataset": RawDataset}
_BAR = "─" * 52


# ── helpers ─────────────────────────────────────────────────────────────────

def _count_frames(seq_dir: Path, channel: str) -> int:
    ts = seq_dir / channel / "timestamps.txt"
    if not ts.exists():
        return 0
    with open(ts) as f:
        return sum(1 for line in f if line.strip())


def _untracked_channels(seq_dir: Path) -> list[str]:
    """Channel-like sub-directories present on disk but absent from channels.yaml."""
    tracked = set(read_config(seq_dir).get("channels", {})) if config_exists(seq_dir) else set()
    return [
        d.name
        for d in sorted(seq_dir.iterdir())
        if d.is_dir()
        and not d.name.startswith(".")
        and d.name not in tracked
        and _detect_loader(d) is not None
    ]


def _seq_info(seq_dir: Path) -> dict:
    cfg = read_config(seq_dir).get("channels", {}) if config_exists(seq_dir) else {}
    return {
        "raw": {k: v.get("loader") for k, v in cfg.items() if v.get("kind", "raw") == "raw"},
        "preprocess": {k: v.get("loader") for k, v in cfg.items() if v.get("kind") == "preprocess"},
        "untracked": _untracked_channels(seq_dir),
        "events": sum(_count_frames(seq_dir, ch) for ch in cfg),
        "issues": verify_config(seq_dir) if config_exists(seq_dir)
        else ["not initialized — run `apairo init`"],
    }


def _is_sequence(path: Path) -> bool:
    return config_exists(path) or RawDataset._is_sequence_layout(path)


def _sequence_dirs(root: Path) -> list[Path]:
    return [
        d for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith(".") and _is_sequence(d)
    ]


def _fmt_channels(d: dict) -> str:
    return ", ".join(f"{k} ({v})" for k, v in sorted(d.items())) if d else "—"


# ── status ──────────────────────────────────────────────────────────────────

def _build_status(path: Path) -> Optional[dict]:
    if _is_sequence(path):
        return {"name": path.name, "kind": "sequence", **_seq_info(path)}

    seq_dirs = _sequence_dirs(path)
    if not seq_dirs:
        return None
    per = {d.name: _seq_info(d) for d in seq_dirs}
    raw: dict = {}
    preprocess: dict = {}
    untracked: set[str] = set()
    issues: list[str] = []
    events = 0
    for name, info in per.items():
        raw.update(info["raw"])
        preprocess.update(info["preprocess"])
        untracked.update(f"{name}/{u}" for u in info["untracked"])
        issues += [f"{name}: {i}" for i in info["issues"]]
        events += info["events"]
    manifest = _read_manifest(path)
    return {
        "name": manifest.get("name", path.name),
        "kind": "root",
        "sequences": list(per),
        "raw": raw,
        "preprocess": preprocess,
        "untracked": sorted(untracked),
        "events": events,
        "issues": issues,
    }


def _print_status(s: dict) -> None:
    if s["kind"] == "root":
        head = f"({s['kind']} · {len(s['sequences'])} sequences)"
    else:
        head = f"({s['kind']})"
    print(f"RawDataset — {s['name']}   {head}")
    print(_BAR)
    if s["kind"] == "root":
        print(f"sequences   {', '.join(s['sequences'])}")
    print(f"raw         {_fmt_channels(s['raw'])}")
    print(f"preprocess  {_fmt_channels(s['preprocess'])}")
    if s["untracked"]:
        print(f"untracked   {', '.join(s['untracked'])}   ← run `apairo add`")
    print(f"events      {s['events']}")
    print(f"issues      {'none' if not s['issues'] else ''}")
    for issue in s["issues"]:
        print(f"            - {issue}")


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2
    status = _build_status(path)
    if status is None:
        print(f"'{path}' is not an apairo dataset (no .apairo, no sequences). "
              f"Run `apairo init` to set it up.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        _print_status(status)
    return 0


# ── init ────────────────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2
    try:
        written = RawDataset.init(
            path, merge=not args.force, overwrite=args.force, name=args.name
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"init failed: {exc}", file=sys.stderr)
        return 1
    rel = written.relative_to(path) if written.is_relative_to(path) else written
    print(f"✓ wrote {rel}")
    _print_status(_build_status(path))
    return 0


# ── entry point ───────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", nargs="?", default=".", help="dataset directory (default: .)")
    p.add_argument("--as", dest="as_", metavar="CLASS", choices=list(DATASETS),
                   default="RawDataset", help="interpret with this dataset class")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="apairo", description="Inspect and initialize apairo datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write .apairo sidecars by scanning a directory")
    _add_common(p_init)
    p_init.add_argument("--name", help="dataset name for the root manifest")
    p_init.add_argument("--force", action="store_true",
                        help="rebuild from scratch (default: merge, non-destructive)")

    p_status = sub.add_parser("status", help="show what a dataset directory contains")
    _add_common(p_status)
    p_status.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)
    handler = {"init": cmd_init, "status": cmd_status}[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
