"""apairo command-line interface.

A thin wrapper over the library -- no third-party dependencies. Commands mirror
familiar terminal/git verbs:

* ``apairo init``   -- write the ``.apairo`` sidecar(s) by scanning a directory
                       (sequence -> ``channels.yaml``; root -> ``dataset.yaml``).
* ``apairo status`` -- report what a dataset directory contains: sequences,
                       tracked channels, channel coverage across sequences
                       (common vs exceptional; ``--missing`` for the per-sequence
                       breakdown), channels detected on disk but not yet
                       registered ("untracked"), event count, and config issues.

* ``apairo check``  -- validate the ``.apairo`` schema (channels, manifest,
                       calibration) and report issues; exit 1 if any.

``add`` (register an untracked channel) is deferred to post-1.0: ``status``
already surfaces the untracked channels it would act on, and they can be
registered today from Python (``register_raw_channel`` / ``register_channel``)
or by re-running ``apairo init`` (which re-scans and merges).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from apairo.core.config import (
    alias_conflict,
    channel_dependents,
    config_exists,
    read_calibration,
    read_config,
    read_manifest,
    remove_channel,
    set_alias,
    verify_calibration,
    verify_config,
    verify_manifest,
)
from apairo.core.profiled_dataset import ProfiledDataset
from apairo.dataset.async_layout.dataset import _detect_loader
from apairo.dataset.goose import Goose3DDataset
from apairo.dataset.raw import RawDataset
from apairo.dataset.rellis import Rellis3DDataset
from apairo.dataset.semantic_kitti import SemanticKittiDataset

# Datasets selectable with ``--as``: the profile-free generic loader plus the
# profiled datasets, whose ``init`` maps canonical channel names from a profile.
# (TartanKittiDataset, multi-sequence, will register here later.)
DATASETS: dict[str, type[Any]] = {
    "RawDataset": RawDataset,
    "SemanticKittiDataset": SemanticKittiDataset,
    "Rellis3DDataset": Rellis3DDataset,
    "Goose3DDataset": Goose3DDataset,
}
_BAR = "-" * 52


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


def _channel_shape(channel_dir: Path, loader: str | None):
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


def _channel_detail(seq_dir: Path, channel: str, meta: dict | None) -> dict:
    """Per-channel facts for the channel directory ``seq_dir/channel``."""
    return _channel_detail_dir(seq_dir / channel, meta)


def _channel_detail_dir(cdir: Path, meta: dict | None) -> dict:
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
        "alias": meta.get("alias") if meta else None,
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
    tracked = (
        set(read_config(seq_dir).get("channels", {}))
        if config_exists(seq_dir)
        else set()
    )
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
    untracked = {
        u: _channel_detail(seq_dir, u, None) for u in _untracked_channels(seq_dir)
    }
    starts = [c["span"][0] for c in {**channels, **untracked}.values() if c["span"]]
    return {
        "channels": channels,
        "untracked": untracked,
        "start": min(starts) if starts else None,
        "events": sum(c["frames"] for c in channels.values()),
        "issues": verify_config(seq_dir)
        if config_exists(seq_dir)
        else ["not initialized -- run `apairo init`"],
    }


def _is_sequence(path: Path) -> bool:
    return config_exists(path) or RawDataset._is_sequence_layout(path)


def _sequence_dirs(root: Path) -> list[Path]:
    return [
        d
        for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith(".") and _is_sequence(d)
    ]


def _fmt_channels(d: dict) -> str:
    return ", ".join(f"{k} ({v})" for k, v in sorted(d.items())) if d else "-"


# ── status ──────────────────────────────────────────────────────────────────


def _profiled_status_class(path: Path):
    """The profiled dataset class this directory was initialized as, if any.

    Read from the root manifest (``.apairo/dataset.yaml``) written by
    ``init --as <Class>``.  Returns ``None`` for a generic (profile-free)
    directory, so status falls through to the generic reading."""
    name = read_manifest(path).get("class")
    cls = DATASETS.get(name) if isinstance(name, str) else None
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


def _build_profiled_sequence_status(path: Path, cls, seq_id: str) -> dict | None:
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
        channels[key] = _channel_detail_dir(
            seq_base / key, {**meta, "kind": "preprocess"}
        )

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


def _build_sequence_status(path: Path, seq_id: str) -> dict | None:
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


def _build_status(path: Path) -> dict | None:
    profiled = _profiled_status_class(path)
    if profiled is not None:
        return _build_profiled_status(path, profiled)

    if _is_sequence(path):
        return {
            "name": path.name,
            "class": "RawDataset",
            "kind": "sequence",
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
    aliases: dict = {}
    present: dict[str, set[str]] = {}  # non-transform channel -> sequences declaring it
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
                present.setdefault(ch, set()).add(name)
            if d.get("alias"):
                aliases[ch] = d["alias"]
        untracked.update(f"{name}/{u}" for u in info["untracked"])
        issues += [f"{name}: {i}" for i in info["issues"]]
        events += info["events"]
    for d in seq_dirs:
        calibration.update(read_calibration(d))

    # Channel coverage across sequences. ``common`` = present in every sequence
    # (the strict intersection the flat root loads); ``exceptional`` = present in
    # a subset; ``missing`` = per sequence, the channels it lacks vs the union.
    # One presence map, read both ways -- by channel (common/exceptional) and by
    # sequence (missing).
    seq_set = set(per)
    n = len(per)
    common = sorted(ch for ch, seqs in present.items() if seqs == seq_set)
    exceptional = {
        ch: {
            "kind": "preprocess" if ch in preprocess else "raw",
            "loader": (preprocess if ch in preprocess else raw)[ch],
            "seqs": sorted(present[ch]),
            "coverage": f"{len(present[ch])}/{n}",
        }
        for ch in sorted(present)
        if present[ch] != seq_set
    }
    missing = {
        name: sorted(ch for ch, seqs in present.items() if name not in seqs)
        for name in per
    }

    manifest = read_manifest(path)
    return {
        "name": manifest.get("name", path.name),
        "class": "RawDataset",
        "kind": "root",
        "sequences": list(per),
        "raw": raw,
        "preprocess": preprocess,
        "tf": tf,
        "aliases": aliases,
        "common": common,
        "exceptional": exceptional,
        "missing": missing,
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


def _print_channel_table(channels: dict, untracked: dict, t0_ref: float | None) -> None:
    ref = t0_ref or 0.0
    all_ch = list(channels.items()) + list(untracked.items())
    show_frame = any(c.get("frame") for _, c in all_ch)  # only when declared
    headers = (
        ["channel", "kind"]
        + (["frame"] if show_frame else [])
        + ["loader", "frames", "rate", "span", "shape", ""]
    )
    rows = []
    for name, c in all_ch:
        rate = f"{c['rate_hz']:.1f} Hz" if c["rate_hz"] else "-"
        span = (
            f"{c['span'][0] - ref:.2f}-{c['span'][1] - ref:.2f}s" if c["span"] else "-"
        )
        if c["kind"] == "untracked":
            note = "<- run `apairo init`"
        elif c.get("transform"):
            tf = c["transform"]
            note = f"<- tf {tf.get('parent')}->{tf.get('child')}"
            if tf.get("static"):
                note += " (static)"
        elif c.get("timestamps_from"):
            note = f"<- from {c['timestamps_from']}"
        else:
            note = ""
        # An aliased channel is shown alias-first (the name you load it by), with
        # its on-disk directory in parentheses.
        display = f"{c['alias']} ({name})" if c.get("alias") else name
        row = (
            [display, c["kind"]]
            + ([c.get("frame") or "-"] if show_frame else [])
            + [
                c["loader"] or "?",
                str(c["frames"]),
                rate,
                span,
                _fmt_shape(c),
                note,
            ]
        )
        rows.append(row)
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]

    def line(cols):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)).rstrip()

    print(line(headers))
    for r in rows:
        print(line(r))


def _print_exceptional(exceptional: dict) -> None:
    """Channels present in only some sequences -- the negative space of `common`.
    Shown by default (when any exist) so a heterogeneous root reads at a glance."""
    if not exceptional:
        return
    items = sorted(exceptional.items())
    w_ch = max(len(f"{ch} ({c['loader']})") for ch, c in items)
    w_cov = max(len(c["coverage"]) for _, c in items)
    for i, (ch, c) in enumerate(items):
        label = "exceptional" if i == 0 else ""
        chan = f"{ch} ({c['loader']})".ljust(w_ch)
        print(
            f"{label:<12}{chan}  {c['coverage'].ljust(w_cov)}  [{', '.join(c['seqs'])}]"
        )


def _print_missing(missing: dict) -> None:
    """Per-sequence breakdown of which channels a sequence lacks vs the union."""
    lacking = sorted((seq, chans) for seq, chans in missing.items() if chans)
    if not lacking:
        print("missing     none -- every sequence has every channel")
        return
    w = max(len(seq) for seq, _ in lacking)
    for i, (seq, chans) in enumerate(lacking):
        label = "missing" if i == 0 else ""
        print(f"{label:<12}{seq.ljust(w)}  {', '.join(chans)}")


def _print_status(s: dict, show_tf: bool = False, show_missing: bool = False) -> None:
    cls = s.get("class", "RawDataset")
    if s["kind"] == "root":
        print(f"{cls} - {s['name']}   (root, {len(s['sequences'])} sequences)")
        print(_BAR)
        print(f"sequences   {', '.join(s['sequences'])}")
        print(f"raw         {_fmt_channels(s['raw'])}")
        print(f"preprocess  {_fmt_channels(s['preprocess'])}")
        _print_exceptional(s.get("exceptional") or {})
        if s.get("aliases"):
            shown = ", ".join(
                f"{real} as {alias}" for real, alias in sorted(s["aliases"].items())
            )
            print(f"aliases     {shown}")
        if s["untracked"]:
            print(
                f"untracked   {', '.join(s['untracked'])}   <- run `apairo init` to register"
            )
        n_tf = len(s.get("tf", {}))
        if show_tf and s.get("tf"):
            print(f"tf          {_fmt_channels(s['tf'])}")
        if show_missing:
            _print_missing(s.get("missing") or {})
    else:
        print(f"{cls} - {s['name']}   (sequence)")
        print(_BAR)
        if s.get("start") is not None:
            print(f"start       {s['start']:.2f}s   (span shown relative to this)")
        channels = (
            s["channels"]
            if show_tf
            else {k: v for k, v in s["channels"].items() if not v.get("transform")}
        )
        n_tf = sum(1 for v in s["channels"].values() if v.get("transform"))
        if channels or s["untracked"]:
            _print_channel_table(channels, s["untracked"], s.get("start"))
        else:
            print("(no channels)")
    n_cal = len(s.get("calibration") or [])
    if show_tf:
        if s.get("calibration"):
            print(
                f"calibration {', '.join(s['calibration'])}   (static, in .apairo/calibration.yaml)"
            )
    elif n_tf or n_cal:
        bits = []
        if n_tf:
            bits.append(f"{n_tf} channel{'s' if n_tf > 1 else ''}")
        if n_cal:
            bits.append(f"{n_cal} static")
        print(f"tf          hidden ({', '.join(bits)}) -- pass --show-tf to show")
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
            print(
                f"Sequence '{args.sequence}' not found under '{path}'.{hint}",
                file=sys.stderr,
            )
            return 1
    else:
        status = _build_status(path)
        if status is None:
            print(
                f"'{path}' is not an apairo dataset (no .apairo, no sequences). "
                f"Run `apairo init` to set it up.",
                file=sys.stderr,
            )
            return 1
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        _print_status(status, show_tf=args.show_tf, show_missing=args.missing)
    return 0


# ── check ─────────────────────────────────────────────────────────────────────


def _check_issues(path: Path) -> list[str] | None:
    """All version-1 schema / consistency issues for a dataset, or ``None`` if
    *path* is not an apairo dataset.

    Reuses the (profile-aware) ``status`` reading to validate channels, then adds
    the optional manifest and calibration files."""
    status = _build_status(path)
    if status is None:
        return None
    issues = list(status.get("issues", []))
    issues += verify_manifest(path)
    issues += verify_calibration(path)
    # Generic roots can carry per-sequence calibration; profiled roots keep it at
    # the root (covered above) and expose no generic sequence dirs here.
    if status.get("kind") == "root":
        for d in _sequence_dirs(path):
            issues += [f"{d.name}: {i}" for i in verify_calibration(d)]
    return issues


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2
    issues = _check_issues(path)
    if issues is None:
        print(
            f"'{path}' is not an apairo dataset (no .apairo, no sequences). "
            f"Run `apairo init` to set it up.",
            file=sys.stderr,
        )
        return 1
    if args.json:
        print(
            json.dumps({"ok": not issues, "issues": issues}, indent=2, sort_keys=True)
        )
    elif not issues:
        print("OK -- no issues")
    else:
        print(f"{len(issues)} issue{'s' if len(issues) != 1 else ''}:")
        for issue in issues:
            print(f"  - {issue}")
    return 1 if issues else 0


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
    print(f"registered: {listing or '-'}")


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
    print(f"wrote {rel}  (as {cls.__name__})")
    if issubclass(cls, ProfiledDataset):
        _print_registered(path)
        _hint_unregistered(cls, path)
    else:
        status = _build_status(path)
        if status is not None:
            _print_status(status)
    return 0


# ── alias ─────────────────────────────────────────────────────────────────────


def _alias_targets(path: Path, channel: str) -> list[Path]:
    """Sequence directories under *path* whose config declares *channel*.

    A single sequence resolves to itself; a root resolves to every sequence
    holding the channel -- so one command aliases the channel everywhere."""
    if config_exists(path) and channel in read_config(path).get("channels", {}):
        return [path]
    return [
        d
        for d in _sequence_dirs(path)
        if config_exists(d) and channel in read_config(d).get("channels", {})
    ]


def cmd_alias(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2
    if not args.remove and not args.alias:
        print(
            "Provide an alias (`apairo alias <channel> <alias>`) or pass --remove.",
            file=sys.stderr,
        )
        return 2

    targets = _alias_targets(path, args.channel)
    if not targets:
        print(
            f"Channel '{args.channel}' is not declared under '{path}'.", file=sys.stderr
        )
        return 1

    new_alias = None if args.remove else args.alias
    # Validate every target before writing any, so a clash leaves nothing half-set.
    # With --force an alias-vs-alias clash is allowed; a directory-name clash is not.
    clashes = [
        (seq, msg)
        for seq in targets
        if (msg := alias_conflict(seq, args.channel, new_alias, force=args.force))
    ]
    if clashes:
        for seq, msg in clashes:
            print(f"cannot alias in '{seq.name}': {msg}", file=sys.stderr)
        print(
            "Pick another name, clear the conflicting alias with "
            "`apairo alias <channel> --remove`, or pass --force to reassign it.",
            file=sys.stderr,
        )
        return 1

    displaced: set[str] = set()
    for seq in targets:
        displaced.update(set_alias(seq, args.channel, new_alias, force=args.force))

    where = "1 sequence" if len(targets) == 1 else f"{len(targets)} sequences"
    if args.remove:
        print(f"removed alias of '{args.channel}' ({where})")
    else:
        print(f"aliased '{args.channel}' as '{args.alias}' ({where})")
    if displaced:
        print(f"  displaced: {', '.join(sorted(displaced))} (now unaliased)")
    return 0


def _channel_targets(path: Path, channel: str) -> list[Path]:
    """Sequence directories under *path* whose config declares *channel*.

    Mirrors :func:`_alias_targets`: a single sequence resolves to itself, a root
    to every sequence holding the channel -- so one command removes it everywhere."""
    if config_exists(path) and channel in read_config(path).get("channels", {}):
        return [path]
    return [
        d
        for d in _sequence_dirs(path)
        if config_exists(d) and channel in read_config(d).get("channels", {})
    ]


def cmd_channel_remove(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return 2

    targets = _channel_targets(path, args.channel)
    if not targets:
        print(
            f"Channel '{args.channel}' is not declared under '{path}'.", file=sys.stderr
        )
        return 1

    # A profiled dataset keeps one root channels.yaml but stores data per
    # sequence; its own remove_channel cascades --purge across the sequences,
    # which the generic root/key deletion cannot reach.
    profiled = _profiled_status_class(path)
    where = (
        "the dataset"
        if profiled is not None
        else "1 sequence"
        if len(targets) == 1
        else f"{len(targets)} sequences"
    )

    # Across every target: is it raw anywhere, and what still depends on it?
    is_raw = False
    dependents: set[str] = set()
    for seq in targets:
        channels = read_config(seq).get("channels", {})
        if channels[args.channel].get("kind", "raw") == "raw":
            is_raw = True
        dependents.update(channel_dependents(channels, args.channel))

    # Removing a raw (source) channel, or deleting data, is hard to undo: warn,
    # then confirm unless --yes. A preprocessed channel without --purge is cheap
    # (regenerable, data untouched) and removed silently.
    if dependents:
        print(
            f"warning: '{args.channel}' is still referenced by "
            f"{', '.join(sorted(dependents))} (timestamps_from/sources).",
            file=sys.stderr,
        )
    if is_raw:
        print(f"warning: '{args.channel}' is a RAW (source) channel.", file=sys.stderr)
    if args.purge:
        print(
            "warning: --purge will delete the channel's data directory on disk.",
            file=sys.stderr,
        )

    if (is_raw or args.purge) and not args.yes:
        verb = "Remove and delete data for" if args.purge else "Remove"
        try:
            resp = input(f"{verb} '{args.channel}' in {where}? [y/N] ")
        except EOFError:
            resp = ""
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    if profiled is not None:
        profiled.remove_channel(path, args.channel, data=args.purge)
    else:
        for seq in targets:
            remove_channel(seq, args.channel, data=args.purge)

    suffix = " (data deleted)" if args.purge else ""
    print(f"removed channel '{args.channel}' from {where}{suffix}")
    return 0


# ── entry point ───────────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "path", nargs="?", default=".", help="dataset directory (default: .)"
    )


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
        epilog = (
            "ecosystem commands: "
            + ", ".join(sorted(plugin_names))
            + "   (run `apairo <command> --help`)"
        )
    parser = argparse.ArgumentParser(
        prog="apairo",
        description="Inspect and initialize apairo datasets.",
        epilog=epilog,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser(
        "init", help="write .apairo sidecars by scanning a directory"
    )
    _add_common(p_init)
    p_init.add_argument(
        "--as",
        dest="as_",
        metavar="CLASS",
        choices=list(DATASETS),
        default="RawDataset",
        help="initialize with this dataset class (default: RawDataset)",
    )
    p_init.add_argument("--name", help="dataset name for the root manifest")
    p_init.add_argument(
        "--force",
        action="store_true",
        help="rebuild from scratch (default: merge, non-destructive)",
    )

    p_status = sub.add_parser("status", help="show what a dataset directory contains")
    _add_common(p_status)
    p_status.add_argument("--json", action="store_true", help="machine-readable output")
    p_status.add_argument(
        "--sequence",
        "-s",
        dest="sequence",
        metavar="ID",
        help="show the per-channel detail for a single sequence "
        "(by id), addressed from the dataset root -- avoids "
        "pointing status at the nested sequence directory",
    )
    p_status.add_argument(
        "--show-tf",
        dest="show_tf",
        action="store_true",
        help="include the transform layer: static calibration and "
        "dynamic tf channels (hidden by default)",
    )
    p_status.add_argument(
        "--missing",
        action="store_true",
        help="(root) per-sequence breakdown of which channels each "
        "sequence lacks vs the union of all channels",
    )

    p_check = sub.add_parser(
        "check",
        help="validate the .apairo schema (channels / manifest / calibration); exit 1 on any issue",
    )
    _add_common(p_check)
    p_check.add_argument("--json", action="store_true", help="machine-readable output")

    p_alias = sub.add_parser(
        "alias",
        help="expose a channel under a clean public name (e.g. ouster_points as lidar)",
    )
    p_alias.add_argument("channel", help="the channel's on-disk directory name")
    p_alias.add_argument(
        "alias", nargs="?", help="public name to expose it under (omit with --remove)"
    )
    p_alias.add_argument(
        "--path", default=".", help="dataset directory (default: .); root-aware"
    )
    p_alias.add_argument(
        "--remove",
        action="store_true",
        help="clear the channel's alias instead of setting it",
    )
    p_alias.add_argument(
        "--force",
        action="store_true",
        help="reassign the alias even if another channel holds it "
        "(that channel is left unaliased)",
    )

    p_channel = sub.add_parser("channel", help="manage channel declarations in .apairo")
    channel_sub = p_channel.add_subparsers(dest="channel_command", required=True)
    p_channel_remove = channel_sub.add_parser(
        "remove", help="drop a channel's declaration (optionally delete its data)"
    )
    p_channel_remove.add_argument(
        "channel", help="the channel's on-disk directory name"
    )
    p_channel_remove.add_argument(
        "--path", default=".", help="dataset directory (default: .); root-aware"
    )
    p_channel_remove.add_argument(
        "--purge",
        action="store_true",
        help="also delete the channel's data directory on disk "
        "(destructive; default keeps the files)",
    )
    p_channel_remove.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="skip the confirmation prompt for raw channels / --purge",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Ecosystem dispatch: `apairo <plugin> ...` hands the rest to the plugin,
    # which parses its own arguments. Built-ins (init/status) fall through.
    plugins = _discover_plugins()
    if argv and argv[0] in plugins:
        plugin_main = plugins[argv[0]].load()
        result = plugin_main(argv[1:])
        raise SystemExit(result if isinstance(result, int) else 0)

    args = _build_parser(set(plugins)).parse_args(argv)
    handler = {
        "init": cmd_init,
        "status": cmd_status,
        "check": cmd_check,
        "alias": cmd_alias,
        "channel": cmd_channel_remove,  # only sub-action is `remove`
    }[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
