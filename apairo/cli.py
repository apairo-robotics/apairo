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

from apairo.core.config import (
    config_exists,
    read_calibration,
    read_config,
    read_manifest,
    verify_config,
)
from apairo.dataset.kitti.dataset import _detect_loader
from apairo.dataset.raw import RawDataset
from apairo.dataset.rellis import Rellis3DDataset
from apairo.dataset.goose import Goose3DDataset
from apairo.dataset.semantic_kitti import SemanticKittiDataset
from apairo.core.profiled_dataset import ProfiledDataset

# Datasets selectable with ``--as``: the profile-free generic loader plus the
# profiled datasets, whose ``init`` maps canonical channel names from a profile.
# (TartanKittiDataset, multi-sequence, will register here later.)
DATASETS = {
    "RawDataset": RawDataset,
    "SemanticKittiDataset": SemanticKittiDataset,
    "Rellis3DDataset": Rellis3DDataset,
    "Goose3DDataset": Goose3DDataset,
}
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
    if not channel_dir.is_dir():
        return 0
    return sum(
        1 for p in channel_dir.iterdir() if p.is_file() and p.name != "timestamps.txt"
    )


def _channel_detail(seq_dir: Path, channel: str, meta: Optional[dict]) -> dict:
    """Per-channel facts for the channel directory ``seq_dir/channel``."""
    return _channel_detail_dir(seq_dir / channel, meta)


def _channel_detail_dir(cdir: Path, meta: Optional[dict]) -> dict:
    """Per-channel facts for an explicit directory, all cheap: timestamps give
    frames/rate/span, the .npy header gives shape/dtype (mmap). ``meta=None``
    marks an untracked channel.  Taking the directory explicitly lets a profiled
    dataset point this at a nested, resolved channel dir (canonical name != dir)."""
    ts = _read_timestamps(cdir)
    rate, span = _rate_span(ts)
    loader = meta.get("loader") if meta else _detect_loader(cdir)
    shape, dtype = _channel_shape(cdir, loader)
    detail = {
        "kind": meta.get("kind", "raw") if meta else "untracked",
        "frame": meta.get("frame") if meta else None,
        "transform": meta.get("transform") if meta else None,
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

def _profiled_status_class(path: Path):
    """The profiled dataset class this directory was initialized as, if any.

    Read from the root manifest (``.apairo/dataset.yaml``) written by
    ``init --as <Class>``.  Returns ``None`` for a generic (profile-free)
    directory, so status falls through to the generic reading."""
    name = read_manifest(path).get("class")
    cls = DATASETS.get(name)
    if cls is not None and issubclass(cls, ProfiledDataset):
        return cls
    return None


def _build_profiled_status(path: Path, cls) -> dict:
    """Status for a profiled dataset, built from the dataset's own layout.

    The dataset resolves canonical channel names to their real (nested,
    per-sequence) directories via :meth:`ProfiledDataset.inventory`; the CLI then
    runs its generic per-directory census on those resolved paths.  Neither side
    re-derives the other's job (mapping vs counting)."""
    inv = cls.inventory(path)
    prefix = Path(*inv["layout"]["fixed"]) if inv["layout"]["fixed"] else Path()
    sequences = inv["sequences"]
    raw_channels = inv["raw"]["channels"]

    raw = {k: c["loader"] for k, c in raw_channels.items() if c["present"]}
    preprocess = {k: v.get("loader", "?") for k, v in inv["preprocess"].items()}

    # Census (frames) over the resolved per-sequence directories.
    def _chan_dir(seq: str, subdir: str) -> Path:
        return path / prefix / seq / subdir

    events = 0
    for seq in sequences:
        for c in raw_channels.values():
            if c["present"] and not c["sequence_file"]:
                events += _count_files(_chan_dir(seq, c["dir"]))
        for key in inv["preprocess"]:
            events += _count_files(_chan_dir(seq, key))

    issues = [
        f"raw channel '{k}' declared in {cls.__name__} profile but not found on disk"
        for k in inv["raw"]["missing"]
        if not raw_channels[k]["optional"]
    ]

    return {
        "name": inv["name"],
        "class": inv["class"],
        "kind": "root",
        "sequences": sequences,
        "raw": raw,
        "preprocess": preprocess,
        "tf": {},
        "untracked": [],
        "calibration": inv["calibration"],
        "events": events,
        "issues": issues,
    }


def _build_profiled_sequence_status(path: Path, cls, seq_id: str) -> Optional[dict]:
    """Per-channel detail for one sequence of a profiled dataset, addressed by id.

    Resolves each canonical channel to its nested per-sequence directory and runs
    the generic per-directory census on it, so the table shows canonical names
    (``lidar``) with real frames/shape/rate -- which a plain
    ``status Rellis-3D/<seq>`` cannot, the profile mapping being unknown there."""
    inv = cls.inventory(path)
    if seq_id not in inv["sequences"]:
        return None
    prefix = Path(*inv["layout"]["fixed"]) if inv["layout"]["fixed"] else Path()
    seq_base = path / prefix / seq_id

    channels: dict = {}
    for key, c in inv["raw"]["channels"].items():
        # Sequence-level files (e.g. poses.txt) are not per-frame dirs -- skip the
        # per-frame table for them.
        if not c["present"] or c["sequence_file"]:
            continue
        channels[key] = _channel_detail_dir(
            seq_base / c["dir"], {"loader": c["loader"], "kind": "raw"}
        )
    for key, meta in inv["preprocess"].items():
        channels[key] = _channel_detail_dir(seq_base / key, {**meta, "kind": "preprocess"})

    starts = [c["span"][0] for c in channels.values() if c["span"]]
    return {
        "name": seq_id,
        "class": inv["class"],
        "kind": "sequence",
        "calibration": inv["calibration"],
        "channels": channels,
        "untracked": {},
        "start": min(starts) if starts else None,
        "events": sum(c["frames"] for c in channels.values()),
        "issues": [],
    }


def _available_sequences(path: Path) -> list[str]:
    """Sequence ids under *path*, profile-aware -- for error hints."""
    profiled = _profiled_status_class(path)
    if profiled is not None:
        return profiled.inventory(path)["sequences"]
    return [d.name for d in _sequence_dirs(path)]


def _build_sequence_status(path: Path, seq_id: str) -> Optional[dict]:
    """Per-sequence status addressed by id from the root (``status -s <id>``).

    Profiled datasets resolve the sequence through the profile; generic datasets
    resolve it to the ``<root>/<id>`` sub-directory."""
    profiled = _profiled_status_class(path)
    if profiled is not None:
        return _build_profiled_sequence_status(path, profiled, seq_id)
    seq_dir = path / seq_id
    if not _is_sequence(seq_dir):
        return None
    return _build_status(seq_dir)


def _build_status(path: Path) -> Optional[dict]:
    profiled = _profiled_status_class(path)
    if profiled is not None:
        return _build_profiled_status(path, profiled)

    if _is_sequence(path):
        return {
            "name": path.name, "class": "RawDataset", "kind": "sequence",
            "calibration": sorted(read_calibration(path)),
            **_seq_info(path),
        }

    seq_dirs = _sequence_dirs(path)
    if not seq_dirs:
        return None
    per = {d.name: _seq_info(d) for d in seq_dirs}
    raw: dict = {}
    preprocess: dict = {}
    tf: dict = {}
    untracked: set[str] = set()
    issues: list[str] = []
    calibration: set[str] = set()
    events = 0
    for name, info in per.items():
        for ch, d in info["channels"].items():
            if d.get("transform"):
                tf[ch] = d["loader"]
            else:
                (raw if d["kind"] == "raw" else preprocess)[ch] = d["loader"]
        untracked.update(f"{name}/{u}" for u in info["untracked"])
        issues += [f"{name}: {i}" for i in info["issues"]]
        events += info["events"]
    for d in seq_dirs:
        calibration.update(read_calibration(d))
    manifest = read_manifest(path)
    return {
        "name": manifest.get("name", path.name),
        "class": "RawDataset",
        "kind": "root",
        "sequences": list(per),
        "raw": raw,
        "preprocess": preprocess,
        "tf": tf,
        "untracked": sorted(untracked),
        "calibration": sorted(calibration),
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
        elif c.get("transform"):
            tf = c["transform"]
            note = f"← tf {tf.get('parent')}→{tf.get('child')}"
            if tf.get("static"):
                note += " (static)"
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


def _print_status(s: dict, show_tf: bool = False) -> None:
    cls = s.get("class", "RawDataset")
    if s["kind"] == "root":
        print(f"{cls} — {s['name']}   (root · {len(s['sequences'])} sequences)")
        print(_BAR)
        print(f"sequences   {', '.join(s['sequences'])}")
        print(f"raw         {_fmt_channels(s['raw'])}")
        print(f"preprocess  {_fmt_channels(s['preprocess'])}")
        if s["untracked"]:
            print(f"untracked   {', '.join(s['untracked'])}   ← run `apairo add`")
        n_tf = len(s.get("tf", {}))
        if show_tf and s.get("tf"):
            print(f"tf          {_fmt_channels(s['tf'])}")
    else:
        print(f"{cls} — {s['name']}   (sequence)")
        print(_BAR)
        if s.get("start") is not None:
            print(f"start       {s['start']:.2f}s   (span shown relative to this)")
        channels = s["channels"] if show_tf else {
            k: v for k, v in s["channels"].items() if not v.get("transform")
        }
        n_tf = sum(1 for v in s["channels"].values() if v.get("transform"))
        if channels or s["untracked"]:
            _print_channel_table(channels, s["untracked"], s.get("start"))
        else:
            print("(no channels)")
    n_cal = len(s.get("calibration") or [])
    if show_tf:
        if s.get("calibration"):
            print(f"calibration {', '.join(s['calibration'])}   (static, in .apairo/calibration.yaml)")
    elif n_tf or n_cal:
        bits = []
        if n_tf:
            bits.append(f"{n_tf} channel{'s' if n_tf > 1 else ''}")
        if n_cal:
            bits.append(f"{n_cal} static")
        print(f"tf          hidden ({', '.join(bits)}) — pass --show-tf to show")
    print(f"events      {s['events']}")
    print(f"issues      {'none' if not s['issues'] else ''}")
    for issue in s["issues"]:
        print(f"            - {issue}")


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2
    if args.sequence:
        status = _build_sequence_status(path, args.sequence)
        if status is None:
            avail = _available_sequences(path)
            hint = f" Available: {', '.join(avail)}." if avail else ""
            print(f"Sequence '{args.sequence}' not found under '{path}'.{hint}",
                  file=sys.stderr)
            return 1
    else:
        status = _build_status(path)
        if status is None:
            print(f"'{path}' is not an apairo dataset (no .apairo, no sequences). "
                  f"Run `apairo init` to set it up.", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        _print_status(status, show_tf=args.show_tf)
    return 0


# ── init ────────────────────────────────────────────────────────────────────

def _hint_unregistered(cls, path: Path) -> None:
    """After init, surface directories that look like preprocessed channels but
    were not registered, so the user can declare the ones they want. Report-only."""
    detect = getattr(cls, "unregistered_channels", None)
    if detect is None:
        return
    try:
        candidates = detect(path)
    except Exception:
        return
    if not candidates:
        return
    print("\nLooks like preprocessed channels, not registered:")
    for name, loader in sorted(candidates.items()):
        print(f"  - {name} ({loader})")
    print(
        f"Declare the ones you want with {cls.__name__}.register_channel("
        f"root, '<name>', '<loader>', timestamps_from=..., sources=[...])."
    )


def _print_registered(path: Path) -> None:
    """List the channels a profiled init just wrote. The generic status is
    profile-unaware (it would look for canonical names as root-level dirs), so we
    report what was registered rather than re-interpreting the config."""
    channels = read_config(path).get("channels", {}) if config_exists(path) else {}
    listing = ", ".join(
        f"{k} ({v.get('loader', '?')})" for k, v in sorted(channels.items())
    )
    print(f"registered: {listing or '—'}")


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2
    cls = DATASETS[args.as_]
    try:
        written = cls.init(
            path, merge=not args.force, overwrite=args.force, name=args.name
        )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"init failed: {exc}", file=sys.stderr)
        return 1
    rel = written.relative_to(path) if written.is_relative_to(path) else written
    print(f"✓ wrote {rel}  (as {cls.__name__})")
    if issubclass(cls, ProfiledDataset):
        _print_registered(path)
        _hint_unregistered(cls, path)
    else:
        _print_status(_build_status(path))
    return 0


# ── entry point ───────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", nargs="?", default=".", help="dataset directory (default: .)")


def _discover_plugins() -> dict:
    """Ecosystem subcommands registered under the ``apairo.cli_plugins`` entry
    point group (e.g. ``apairo extractor`` from ``apairo_extractor``).

    Discovery is by installed metadata only -- apairo never imports or depends
    on its tools; it dispatches to whatever is installed.
    """
    from importlib.metadata import entry_points

    return {ep.name: ep for ep in entry_points(group="apairo.cli_plugins")}


def _build_parser(plugin_names) -> argparse.ArgumentParser:
    epilog = None
    if plugin_names:
        epilog = ("ecosystem commands: " + ", ".join(sorted(plugin_names))
                  + "   (run `apairo <command> --help`)")
    parser = argparse.ArgumentParser(
        prog="apairo", description="Inspect and initialize apairo datasets.",
        epilog=epilog,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write .apairo sidecars by scanning a directory")
    _add_common(p_init)
    p_init.add_argument("--as", dest="as_", metavar="CLASS", choices=list(DATASETS),
                        default="RawDataset",
                        help="initialize with this dataset class (default: RawDataset)")
    p_init.add_argument("--name", help="dataset name for the root manifest")
    p_init.add_argument("--force", action="store_true",
                        help="rebuild from scratch (default: merge, non-destructive)")

    p_status = sub.add_parser("status", help="show what a dataset directory contains")
    _add_common(p_status)
    p_status.add_argument("--json", action="store_true", help="machine-readable output")
    p_status.add_argument("--sequence", "-s", dest="sequence", metavar="ID",
                          help="show the per-channel detail for a single sequence "
                               "(by id), addressed from the dataset root -- avoids "
                               "pointing status at the nested sequence directory")
    p_status.add_argument("--show-tf", dest="show_tf", action="store_true",
                          help="include the transform layer: static calibration and "
                               "dynamic tf channels (hidden by default)")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Ecosystem dispatch: `apairo <plugin> ...` hands the rest to the plugin,
    # which parses its own arguments. Built-ins (init/status) fall through.
    plugins = _discover_plugins()
    if argv and argv[0] in plugins:
        plugin_main = plugins[argv[0]].load()
        result = plugin_main(argv[1:])
        raise SystemExit(result if isinstance(result, int) else 0)

    args = _build_parser(set(plugins)).parse_args(argv)
    handler = {"init": cmd_init, "status": cmd_status}[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
