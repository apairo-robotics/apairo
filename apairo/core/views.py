"""A synchronization, frozen: named views in ``.apairo``.

``synchronize()`` recomputes its matching every session. Reduced to its state, a
synchronization is not data but an index matrix: for each kept reference tick,
the event each channel contributes (or the bracketing pair an interpolator
blends). Persisting that matrix -- with the reference clock, the method, the
tolerance and a fingerprint of every source channel -- lets a later session,
or a colleague, reload the very same frames as a synchronous dataset without
copying any data, and refuses the reload once a source has changed.

On disk, in a sequence's ``.apairo``::

    views.yaml          the registry: one entry per view, with its parameters
                        and the fingerprint of its sources
    views/<name>.npz    the reference timestamps and one index array per channel
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from apairo.core.config import CONFIG_DIR, SCHEMA_VERSION
from apairo.core.interpolator import Interpolator

if TYPE_CHECKING:
    from apairo.core.synchronized_view import SynchronizedView

VIEWS_FILE = "views.yaml"
VIEWS_DIR = "views"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_VIEW_FIELDS = frozenset(
    {"file", "reference", "method", "tolerance", "frames", "sources", "created"}
)


def _registry_path(seq_dir: Path) -> Path:
    return Path(seq_dir) / CONFIG_DIR / VIEWS_FILE


def read_views(seq_dir: str | Path) -> dict[str, dict]:
    """The views persisted in a sequence, by name; empty when there are none."""
    path = _registry_path(Path(seq_dir))
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    views = data.get("views") if isinstance(data, dict) else None
    return views if isinstance(views, dict) else {}


def _sequence_dir(dataset: Any) -> Path:
    """The directory of the sequence *dataset* reads, where its views live."""
    directory = getattr(dataset, "root_dir", None)
    if directory is None:
        raise TypeError(
            f"{type(dataset).__name__} has no sequence directory to keep views in."
        )
    return Path(directory)


def clock_fingerprint(timestamps: np.ndarray) -> str:
    """A short hash of a channel's clock: equal clocks, equal fingerprints."""
    data = np.ascontiguousarray(timestamps, dtype=np.float64).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _strategy_spec(strategy: Any) -> str:
    """How a channel was matched, as a string the registry can hold."""
    if isinstance(strategy, Interpolator):
        cls = type(strategy)
        return f"interpolate:{cls.__module__}.{cls.__qualname__}"
    if callable(strategy):
        qualname = getattr(strategy, "__qualname__", type(strategy).__qualname__)
        return f"custom:{qualname}"
    return str(strategy)


def persist_view(view: SynchronizedView, name: str, *, overwrite: bool = False) -> Path:
    """Write *view* into its sequence's ``.apairo`` under *name*; the path of
    the index file. See :meth:`SynchronizedView.persist`."""
    from apairo.dataset.async_layout.dataset import AsyncLayoutDataset

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(
            f"A view name is letters, digits, '_', '-' and '.', not starting "
            f"with a dot; got {name!r}."
        )
    parent = view._parent
    if not isinstance(parent, AsyncLayoutDataset):
        raise TypeError(
            f"Only a view synchronized directly on a sequence can be persisted: "
            f"its indices address that sequence's channels. This one was "
            f"synchronized on a {type(parent).__name__}."
        )
    seq_dir = _sequence_dir(parent)
    views = read_views(seq_dir)
    if name in views and not overwrite:
        raise FileExistsError(
            f"A view named '{name}' is already persisted in {seq_dir}. Pass "
            f"overwrite=True to replace it, or load it with load_view('{name}')."
        )

    keys = list(view._keys)
    arrays = {"reference_timestamps": np.asarray(view._ref_timestamps, dtype=float)}
    for i, key in enumerate(keys):
        arrays[f"index_{i}"] = np.asarray(view._index_map[key], dtype=np.int64)
    rel = Path(VIEWS_DIR) / f"{name}.npz"
    target = seq_dir / CONFIG_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)  # type: ignore[arg-type]

    views[name] = {
        "file": rel.as_posix(),
        "reference": view._reference,
        "method": {key: _strategy_spec(view._strategies[key]) for key in keys},
        "tolerance": None if view._tolerance is None else float(view._tolerance),
        "frames": len(view),
        "sources": {
            key: {
                "frames": len(parent.loaders[key]),
                "clock": clock_fingerprint(view._channel_ts[key]),
            }
            for key in keys
        },
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    registry = _registry_path(seq_dir)
    registry.write_text(
        yaml.safe_dump({"version": SCHEMA_VERSION, "views": views}, sort_keys=False)
    )
    return target


def load_view(
    dataset: Any, name: str, *, interpolators: dict[str, Interpolator] | None = None
) -> SynchronizedView:
    """Reload view *name* of a sequence over *dataset* (that sequence, opened
    as the view was made: same declaration). See
    :meth:`~apairo.dataset.async_layout.dataset.AsyncLayoutDataset.load_view`."""
    from apairo.core.synchronized_view import SynchronizedView

    seq_dir = _sequence_dir(dataset)
    views = read_views(seq_dir)
    if name not in views:
        known = ", ".join(sorted(views)) or "none"
        raise KeyError(f"No view named '{name}' in {seq_dir} (persisted: {known}).")
    entry = views[name]
    keys = list(entry["method"])
    missing = [k for k in keys if k not in dataset.keys]
    if missing:
        dataset = dataset._reopened(keys)

    # Stale: a source channel changed since the view was frozen.
    for key, was in entry["sources"].items():
        if key not in dataset.keys:
            raise ValueError(f"View '{name}' is stale: channel '{key}' is gone.")
        frames, clock = len(dataset.loaders[key]), dataset.timestamps[key]
        if frames != was["frames"]:
            raise ValueError(
                f"View '{name}' is stale: channel '{key}' has {frames} frames, "
                f"{was['frames']} when the view was persisted. Recompute it: "
                f"synchronize(...).persist('{name}', overwrite=True)."
            )
        if clock_fingerprint(clock) != was["clock"]:
            raise ValueError(
                f"View '{name}' is stale: the clock of channel '{key}' changed "
                f"since the view was persisted (a timestamp, a key, a latency, or "
                f"a different declaration). Recompute it: "
                f"synchronize(...).persist('{name}', overwrite=True)."
            )

    strategies: dict[str, Any] = {}
    for key, spec in entry["method"].items():
        if spec.startswith("interpolate:"):
            given = (interpolators or {}).get(key)
            wanted = spec.split(":", 1)[1]
            if given is None:
                raise ValueError(
                    f"View '{name}' interpolates channel '{key}' with {wanted}: "
                    f"pass it again, load_view('{name}', interpolators="
                    f"{{'{key}': ...}}) -- an interpolator is code, not state."
                )
            got = f"{type(given).__module__}.{type(given).__qualname__}"
            if got != wanted:
                raise ValueError(
                    f"View '{name}' interpolated channel '{key}' with {wanted}, "
                    f"not {got}."
                )
            strategies[key] = given
        else:
            strategies[key] = spec  # matched: the indices are all it needs

    with np.load(seq_dir / CONFIG_DIR / entry["file"]) as data:
        ref_ts = np.asarray(data["reference_timestamps"], dtype=float)
        index_map = {key: np.asarray(data[f"index_{i}"]) for i, key in enumerate(keys)}
    return SynchronizedView._from_state(
        dataset,
        reference=entry["reference"],
        strategies=strategies,
        tolerance=entry["tolerance"],
        reference_timestamps=ref_ts,
        index_map=index_map,
    )


def verify_views(seq_dir: str | Path) -> list[str]:
    """Issues with a sequence's view registry, for ``check``: its fields and
    its index files. (Staleness needs the channels' clocks: ``load_view``
    refuses a stale view, naming the channel.)"""
    seq_dir = Path(seq_dir)
    path = _registry_path(seq_dir)
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        return [f"{VIEWS_FILE}: cannot parse: {exc}"]
    views = data.get("views") if isinstance(data, dict) else None
    if not isinstance(views, dict):
        return [f"{VIEWS_FILE}: 'views' is missing or not a mapping"]
    issues: list[str] = []
    for name, entry in views.items():
        if not isinstance(entry, dict):
            issues.append(f"{VIEWS_FILE}: view '{name}' is not a mapping")
            continue
        for field in sorted(set(entry) - _VIEW_FIELDS):
            issues.append(f"{VIEWS_FILE}: view '{name}': unknown field '{field}'")
        file = entry.get("file")
        if not isinstance(file, str) or not (seq_dir / CONFIG_DIR / file).is_file():
            issues.append(f"{VIEWS_FILE}: view '{name}': index file {file!r} not found")
        if not isinstance(entry.get("method"), dict) or not entry["method"]:
            issues.append(f"{VIEWS_FILE}: view '{name}': 'method' names no channel")
    return issues
