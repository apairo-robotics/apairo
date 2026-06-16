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

import numpy as np

from apairo.core.config import config_exists, read_config, verify_config
from apairo.dataset.kitti.dataset import _detect_loader
from apairo.dataset.raw import RawDataset
from apairo.dataset.raw.dataset import _read_manifest

# Datasets selectable with ``--as``. Generic (profile-free) for now; profiled
# datasets (Tartan, Semantic, ...) will register here as the CLI grows.
DATASETS = {"RawDataset": RawDataset}
_BAR = "─" * 52


# ── helpers ─────────────────────────────────────────────────────────────────

def _read_timestamps(channel_dir: Path):
    ts_path = channel_dir / "timestamps.txt"
    if not ts_path.exists():
        return None
    try:
        return np.atleast_1d(np.loadtxt(ts_path))
    except Exception:
        return None


def _rate_span(ts):
    """Average rate (Hz) and (first, last) timestamps from a timestamp array."""
    if ts is None or len(ts) == 0:
        return None, None
    t0, t1 = float(ts[0]), float(ts[-1])
    rate = (len(ts) - 1) / (t1 - t0) if len(ts) >= 2 and t1 > t0 else None
    return rate, (t0, t1)


def _channel_shape(channel_dir: Path, loader: Optional[str]):
    """Per-frame shape + dtype from a ``.npy`` header (mmap -- no data read)."""
    npys = sorted(channel_dir.glob("*.npy"))
    if not npys:
        return None, None
    try:
        arr = np.load(npys[0], mmap_mode="r")
    except Exception:
        return None, None
    # A stacked ``npy`` file is (N, *frame); a per-frame ``npys`` file is one frame.
    shape = arr.shape[1:] if loader == "npy" else arr.shape
    return list(shape), str(arr.dtype)


def _count_files(channel_dir: Path) -> int:
    return sum(
        1 for p in channel_dir.iterdir() if p.is_file() and p.name != "timestamps.txt"
    )


def _channel_detail(seq_dir: Path, channel: str, meta: Optional[dict]) -> dict:
    """Per-channel facts, all cheap: timestamps give frames/rate/span, the .npy
    header gives shape/dtype (mmap). ``meta=None`` marks an untracked channel."""
    cdir = seq_dir / channel
    ts = _read_timestamps(cdir)
    rate, span = _rate_span(ts)
    loader = meta.get("loader") if meta else _detect_loader(cdir)
    shape, dtype = _channel_shape(cdir, loader)
    detail = {
        "kind": meta.get("kind", "raw") if meta else "untracked",
        "frame": meta.get("frame") if meta else None,
        "loader": loader,
        "frames": len(ts) if ts is not None else _count_files(cdir),
        "rate_hz": rate,
        "span": list(span) if span else None,
        "shape": shape,
        "dtype": dtype,
    }
    if meta and meta.get("timestamps_from"):
        detail["timestamps_from"] = meta["timestamps_from"]
    if meta and meta.get("sources"):
        detail["sources"] = list(meta["sources"])
    return detail


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
    channels = {k: _channel_detail(seq_dir, k, v) for k, v in sorted(cfg.items())}
    untracked = {u: _channel_detail(seq_dir, u, None) for u in _untracked_channels(seq_dir)}
    starts = [c["span"][0] for c in {**channels, **untracked}.values() if c["span"]]
    return {
        "channels": channels,
        "untracked": untracked,
        "start": min(starts) if starts else None,
        "events": sum(c["frames"] for c in channels.values()),
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
        for ch, d in info["channels"].items():
            (raw if d["kind"] == "raw" else preprocess)[ch] = d["loader"]
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


def _fmt_shape(detail: dict) -> str:
    if detail["shape"] is None:
        return "?"
    s = f"({', '.join(map(str, detail['shape']))})"
    return f"{s} {detail['dtype']}" if detail.get("dtype") else s


def _print_channel_table(channels: dict, untracked: dict, t0_ref: Optional[float]) -> None:
    ref = t0_ref or 0.0
    all_ch = list(channels.items()) + list(untracked.items())
    show_frame = any(c.get("frame") for _, c in all_ch)  # only when declared
    headers = ["channel", "kind"] + (["frame"] if show_frame else []) + \
        ["loader", "frames", "rate", "span", "shape", ""]
    rows = []
    for name, c in all_ch:
        rate = f"{c['rate_hz']:.1f} Hz" if c["rate_hz"] else "—"
        span = f"{c['span'][0] - ref:.2f}–{c['span'][1] - ref:.2f}s" if c["span"] else "—"
        if c["kind"] == "untracked":
            note = "← run `apairo add`"
        elif c.get("timestamps_from"):
            note = f"← from {c['timestamps_from']}"
        else:
            note = ""
        row = [name, c["kind"]] + ([c.get("frame") or "—"] if show_frame else []) + [
            c["loader"] or "?", str(c["frames"]), rate, span, _fmt_shape(c), note,
        ]
        rows.append(row)
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = lambda cols: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)).rstrip()
    print(line(headers))
    for r in rows:
        print(line(r))


def _print_status(s: dict) -> None:
    if s["kind"] == "root":
        print(f"RawDataset — {s['name']}   (root · {len(s['sequences'])} sequences)")
        print(_BAR)
        print(f"sequences   {', '.join(s['sequences'])}")
        print(f"raw         {_fmt_channels(s['raw'])}")
        print(f"preprocess  {_fmt_channels(s['preprocess'])}")
        if s["untracked"]:
            print(f"untracked   {', '.join(s['untracked'])}   ← run `apairo add`")
    else:
        print(f"RawDataset — {s['name']}   (sequence)")
        print(_BAR)
        if s.get("start") is not None:
            print(f"start       {s['start']:.2f}s   (span shown relative to this)")
        if s["channels"] or s["untracked"]:
            _print_channel_table(s["channels"], s["untracked"], s.get("start"))
        else:
            print("(no channels)")
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
