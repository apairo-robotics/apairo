from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np
import yaml

CONFIG_DIR = ".apairo"
CHANNELS_FILE = "channels.yaml"
CALIBRATION_FILE = "calibration.yaml"
DATASET_FILE = "dataset.yaml"
CONFIG_FILENAME = CONFIG_DIR  # alias kept for external code that checks (path / CONFIG_FILENAME).exists()

# Keep in sync with str_to_loader (apairo/loader/__init__.py) and WRITERS (apairo/writer/__init__.py).
KNOWN_LOADERS: frozenset[str] = frozenset(
    {"npy", "npys", "bin", "img", "zarr"}
)

# ── .apairo schema, version 1 ────────────────────────────────────────────────
# The on-disk contract. Validation is tolerant: an unknown field is reported as a
# warning and otherwise ignored, so a sidecar written by a newer apairo still
# loads on an older one. See docs/datasets/apairo-schema.md.
SCHEMA_VERSION = 1

_CHANNELS_TOP_FIELDS: frozenset[str] = frozenset({"version", "channels"})
_CHANNEL_FIELDS: frozenset[str] = frozenset(
    {"kind", "loader", "timestamps_from", "sources", "frame", "transform", "alias"}
)
_CHANNEL_KINDS: frozenset[str] = frozenset({"raw", "preprocess"})
_TRANSFORM_FIELDS: frozenset[str] = frozenset({"parent", "child", "static", "format"})

# class (profiled root) | name/sequences/channels (generic root roll-up).
_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {"version", "class", "name", "sequences", "channels"}
)

_CALIBRATION_TOP_FIELDS: frozenset[str] = frozenset({"version", "transforms"})
_CALIBRATION_TRANSFORM_FIELDS: frozenset[str] = frozenset({"parent", "child", "matrix"})


def _unknown(present, known: frozenset[str], where: str) -> list[str]:
    """Warnings for keys in *present* not in the version-1 schema (*known*).

    Tolerant by policy -- unknown fields are ignored at load time; this only
    surfaces them (a typo, or a field from a newer apairo)."""
    if not isinstance(present, dict):
        return []
    return [
        f"{where}: unknown field '{k}' (ignored -- not in the version {SCHEMA_VERSION} schema)"
        for k in present
        if k not in known
    ]


def _apairo_dir(root_dir: Path) -> Path:
    return root_dir / CONFIG_DIR


def _channels_path(root_dir: Path) -> Path:
    return _apairo_dir(root_dir) / CHANNELS_FILE


def config_exists(root_dir: Path) -> bool:
    return _channels_path(root_dir).exists()


def read_config(root_dir: Path) -> dict:
    with open(_channels_path(root_dir)) as f:
        return yaml.safe_load(f)


def write_config(root_dir: Path, config: dict) -> None:
    d = _apairo_dir(root_dir)
    d.mkdir(exist_ok=True)
    with open(d / CHANNELS_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)


def read_manifest(root_dir: str | Path) -> dict:
    """Read ``<root>/.apairo/dataset.yaml`` (the root manifest) if present, else ``{}``.

    The manifest records dataset-root identity -- which dataset ``class`` produced
    the layout, an optional ``name``, and (for the generic root) sequence order --
    as opposed to ``channels.yaml`` which is the per-directory channel registry.
    """
    path = _apairo_dir(Path(root_dir)) / DATASET_FILE
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def write_manifest(root_dir: str | Path, manifest: dict) -> Path:
    """Write ``<root>/.apairo/dataset.yaml`` (the root manifest). Returns its path.

    Stamps the schema ``version`` (``1``) if the caller did not set one, so the
    manifest carries the same version contract as ``channels.yaml`` and
    ``calibration.yaml``."""
    d = _apairo_dir(Path(root_dir))
    d.mkdir(exist_ok=True)
    path = d / DATASET_FILE
    payload = {"version": SCHEMA_VERSION, **manifest}
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=True)
    return path


def register_channel(
    root_dir: str | Path,
    key: str,
    loader: str,
    *,
    timestamps_from: Optional[str] = None,
    sources: Optional[list[str]] = None,
    frame: Optional[str] = None,
) -> None:
    """Register a preprocessed channel in ``root_dir/.apairo/channels.yaml``.

    This is the low-level standalone function.  Most users will prefer the
    classmethod :meth:`ConfigurableDataset.register_channel` so that the call
    site names the dataset type explicitly.

    Existing channels (raw or preprocessed) are preserved -- only ``key`` is
    updated.

    Args:
        root_dir: Dataset root directory.
        key: Channel name -- must match its subdirectory name.
        loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, or ``"img"``.
        timestamps_from: Source channel whose timestamps this channel shares
            (provenance only -- the channel always has its own ``timestamps.txt``).
        sources: Provenance -- raw channels this channel was derived from.
        frame: Coordinate frame the channel's data is expressed in (descriptive
            metadata only; apairo does not apply transforms).
    """
    root_dir = Path(root_dir)
    # Read existing config to preserve all other channels (raw + preprocessed).
    config = (
        read_config(root_dir)
        if config_exists(root_dir)
        else {"version": 1, "channels": {}}
    )

    entry: dict = {"kind": "preprocess", "loader": loader}
    if timestamps_from is not None:
        entry["timestamps_from"] = timestamps_from
    if sources:
        entry["sources"] = list(sources)
    if frame is not None:
        entry["frame"] = frame

    config["channels"][key] = entry
    write_config(root_dir, config)


def register_raw_channel(
    root_dir: str | Path,
    key: str,
    loader: str,
    *,
    frame: Optional[str] = None,
    transform: Optional[dict] = None,
    alias: Optional[str] = None,
) -> None:
    """Declare a raw channel in ``root_dir/.apairo/channels.yaml``.

    Use this to record the raw modalities of datasets (e.g. generic KITTI)
    whose channel layout is not defined by a built-in profile, so the dataset
    can be reconstructed without passing ``keys`` and ``dataset_profile``
    every time.

    Existing channels are preserved -- only ``key`` is updated.

    Args:
        root_dir: Dataset root directory (or sequence directory).
        key: Channel name -- must match its subdirectory name.
        loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, ``"img"``, or
            ``"zarr"``.
        frame: Coordinate frame the channel's data is expressed in (descriptive
            metadata only; apairo does not apply transforms).
        transform: For a channel that *is* a coordinate transform (a pose
            stream), the edge it provides, e.g.
            ``{"parent": "odom", "child": "base_link"}`` (optionally
            ``"static": True``, ``"format": "t_xyz_q_xyzw"``). Descriptive only.
        alias: Public name the channel is exposed under when loaded (e.g. expose
            the on-disk ``ouster_points`` directory as ``lidar``). The directory
            name stays the storage key; the alias is what ``keys=[...]`` and
            ``sample.data`` use. See :func:`set_alias`.
    """
    root_dir = Path(root_dir)
    config = (
        read_config(root_dir)
        if config_exists(root_dir)
        else {"version": 1, "channels": {}}
    )

    entry: dict = {"kind": "raw", "loader": loader}
    if frame is not None:
        entry["frame"] = frame
    if transform is not None:
        entry["transform"] = transform
    if alias is not None:
        entry["alias"] = alias
    config["channels"][key] = entry
    write_config(root_dir, config)


def set_alias(
    root_dir: str | Path, channel: str, alias: Optional[str], *, force: bool = False
) -> list[str]:
    """Set (or clear) the public alias of a raw channel in ``channels.yaml``.

    An alias is the name the channel is exposed under at load time: the on-disk
    directory keeps its real name, but ``Dataset(root, keys=[alias])`` loads it
    and ``sample.data[alias]`` returns it. Honoured by both
    :class:`~apairo.dataset.raw.RawDataset` and
    :class:`~apairo.core.profiled_dataset.ProfiledDataset`, so channel names can
    be unified across heterogeneous datasets in one pipeline -- naming lives in
    ``.apairo``, not in the call site.

    Args:
        root_dir: Sequence directory holding ``.apairo/channels.yaml``.
        channel: The channel's on-disk directory name. Must already be declared.
        alias: Public name to expose it under; ``None`` clears any alias.
        force: Reassign *alias* even if another channel already holds it -- the
            previous holder is left **unaliased** (reverts to its directory
            name). Has no effect on a clash with a real directory *name*, which
            can never be reassigned.

    Returns:
        The channels whose alias was cleared to make room (empty unless
        ``force`` displaced a previous holder).

    Raises:
        FileNotFoundError: if no ``channels.yaml`` exists at *root_dir*.
        KeyError: if *channel* is not declared in the config.
        ValueError: if *alias* would collide with another channel's directory
            name (never reassignable), or with another channel's alias and
            ``force`` is not set. Clear the conflicting alias first, pass
            ``force=True``, or pick another name.
    """
    root_dir = Path(root_dir)
    if not config_exists(root_dir):
        raise FileNotFoundError(
            f"No {CONFIG_DIR}/{CHANNELS_FILE} in '{root_dir}'. Run `apairo init` first."
        )
    config = read_config(root_dir)
    channels = config.get("channels", {})
    if channel not in channels:
        raise KeyError(
            f"Channel '{channel}' is not declared in '{root_dir}'. "
            f"Available: {sorted(channels)}."
        )
    displaced: list[str] = []
    if alias:
        clash = _alias_conflict(channels, channel, alias, force=force)
        if clash:
            raise ValueError(f"Cannot alias '{channel}' as '{alias}': {clash}.")
        if force:
            displaced = _alias_holders(channels, channel, alias)
            for other in displaced:
                channels[other].pop("alias", None)
        channels[channel]["alias"] = alias
    else:
        channels[channel].pop("alias", None)
    write_config(root_dir, config)
    return displaced


def remove_channel(root_dir: str | Path, channel: str, *, data: bool = False) -> dict:
    """Remove a channel's declaration from ``root_dir/.apairo/channels.yaml``.

    The inverse of :func:`register_channel` / :func:`register_raw_channel`: drops
    the channel so the dataset stops loading it. By default the on-disk files are
    left untouched, so the removal is reversible (re-run ``init`` or re-register
    the channel); pass ``data=True`` to also delete the channel's directory from
    disk -- destructive and irreversible.

    This is the low-level standalone function. Most users will prefer the
    classmethod :meth:`ConfigurableDataset.remove_channel`, or the CLI
    (``apairo channel remove``), which warns before dropping a *raw* (source)
    channel and before deleting data.

    Args:
        root_dir: Dataset root (or sequence) directory.
        channel: The channel's declared name (its on-disk directory name).
        data: Also delete the channel's directory (``root_dir/channel``) from
            disk. The raw/preprocessed files are gone for good.

    Returns:
        The removed channel's metadata entry -- so a caller can tell whether it
        was ``raw`` or ``preprocess`` (or restore it).

    Raises:
        FileNotFoundError: if no ``channels.yaml`` exists at *root_dir*.
        KeyError: if *channel* is not declared in the config.
    """
    root_dir = Path(root_dir)
    if not config_exists(root_dir):
        raise FileNotFoundError(
            f"No {CONFIG_DIR}/{CHANNELS_FILE} in '{root_dir}'. Run `apairo init` first."
        )
    config = read_config(root_dir)
    channels = config.get("channels", {})
    if channel not in channels:
        raise KeyError(
            f"Channel '{channel}' is not declared in '{root_dir}'. "
            f"Available: {sorted(channels)}."
        )
    entry = channels.pop(channel)
    write_config(root_dir, config)
    if data:
        import shutil

        channel_dir = root_dir / channel
        if channel_dir.is_dir():
            shutil.rmtree(channel_dir)
    return entry


def channel_dependents(channels: dict, channel: str) -> list[str]:
    """Channels that reference *channel* -- borrowing its clock
    (``timestamps_from``) or naming it as a derivation ``source``.

    Removing *channel* leaves these dangling (``verify_config`` would flag them),
    so callers can warn first. Takes the ``channels`` mapping (not a path) so a
    root-aware caller can reuse a single read."""
    out = []
    for name, meta in channels.items():
        if name == channel or not isinstance(meta, dict):
            continue
        if meta.get("timestamps_from") == channel or channel in (meta.get("sources") or []):
            out.append(name)
    return out


def _alias_holders(channels: dict, channel: str, alias: str) -> list[str]:
    """Other channels currently exposing *alias* as their public name."""
    return [
        other for other, meta in channels.items()
        if other != channel and meta.get("alias") == alias
    ]


def _alias_conflict(
    channels: dict, channel: str, alias: str, force: bool = False
) -> Optional[str]:
    """Reason aliasing *channel* as *alias* would clash within *channels*, or None.

    A public name must be unique: it cannot shadow another channel's on-disk
    directory name, nor duplicate another channel's alias -- otherwise two
    channels would claim the same loaded key and the dataset fails to build.
    With *force*, an alias-vs-alias clash is allowed (the holder is displaced);
    a clash with a directory *name* is never reassignable."""
    if alias in channels and alias != channel:
        return f"'{alias}' is already a channel directory name (cannot be reassigned)"
    if not force and _alias_holders(channels, channel, alias):
        holder = _alias_holders(channels, channel, alias)[0]
        return f"'{alias}' is already the alias of '{holder}' (pass force to reassign)"
    return None


def alias_conflict(
    root_dir: str | Path, channel: str, alias: Optional[str], force: bool = False
) -> Optional[str]:
    """Message if aliasing *channel* as *alias* would clash in *root_dir*, else None.

    Read-only counterpart to :func:`set_alias`'s guard -- lets a caller validate
    across several sequences before writing any of them. With *force*, only an
    unreassignable directory-name clash is reported."""
    if not alias:
        return None
    root_dir = Path(root_dir)
    channels = read_config(root_dir).get("channels", {}) if config_exists(root_dir) else {}
    return _alias_conflict(channels, channel, alias, force=force)


def _invert_rigid(T: np.ndarray) -> np.ndarray:
    """Exact inverse of a 4x4 rigid transform (transpose R, re-rotate t)."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


class Calibration(dict):
    """A dataset's static extrinsics: ``{"<parent>_to_<child>": (4,4) float64}``.

    A plain ``dict`` (``cal["lidar_to_base"]`` and iteration work) that can also
    *resolve* the transform between any two connected frames -- the one canonical
    operation a static-transform graph supports. It resolves; applying the matrix
    to data is the caller's job (e.g. ``apairo_transform.ApplyMatrix``), since that
    depends on what the data is (points, poses, normals...).

    Each edge ``"<parent>_to_<child>"`` is ``T_parent_from_child`` (ROS ``/tf``):
    it maps a point in *child* coordinates into *parent*.
    """

    def get_tf(self, source: str, target: str) -> np.ndarray:
        """``T_target_from_source`` -- ``p_target = get_tf(source, target) @ p_source``.

        Walks the undirected transform tree (composing edges and their rigid
        inverses); identity when ``source == target``.

        Raises:
            KeyError: if no path connects them (the message lists the frames
                reachable from *source*).
            ValueError: if a key is not ``"<parent>_to_<child>"``.
        """
        if source == target:
            return np.eye(4)
        adj: dict[str, list[tuple[str, np.ndarray]]] = {}
        for key, matrix in self.items():
            parent, sep, child = key.partition("_to_")
            if not sep:
                raise ValueError(f"Calibration key {key!r} is not '<parent>_to_<child>'.")
            T = np.asarray(matrix, dtype=np.float64)
            adj.setdefault(child, []).append((parent, T))
            adj.setdefault(parent, []).append((child, _invert_rigid(T)))
        seen = {source}
        queue: "deque[tuple[str, np.ndarray]]" = deque([(source, np.eye(4))])
        while queue:
            frame, T_frame_from_source = queue.popleft()
            if frame == target:
                return T_frame_from_source
            for nxt, T_nxt_from_frame in adj.get(frame, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, T_nxt_from_frame @ T_frame_from_source))
        raise KeyError(
            f"No static-transform path from {source!r} to {target!r}. "
            f"Reachable from {source!r}: {sorted(seen)}"
        )


def read_calibration(root_dir: str | Path) -> Calibration:
    """Static extrinsics from ``root_dir/.apairo/calibration.yaml`` as a
    :class:`Calibration` (empty if absent)."""
    path = Path(root_dir) / CONFIG_DIR / CALIBRATION_FILE
    out = Calibration()
    if not path.exists():
        return out
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for key, entry in (data.get("transforms") or {}).items():
        matrix = entry["matrix"] if isinstance(entry, dict) else entry
        out[key] = np.asarray(matrix, dtype=np.float64)
    return out


def register_static_transform(
    root_dir: str | Path, parent: str, child: str, matrix,
) -> None:
    """Record a static transform (extrinsic) in ``.apairo/calibration.yaml``.

    A static transform is time-independent, so it belongs in calibration -- not
    in a per-frame channel. Existing entries are preserved.

    Args:
        root_dir: Dataset root (or sequence) directory.
        parent: Parent frame.
        child: Child frame.
        matrix: 4x4 homogeneous transform (array-like).
    """
    root_dir = Path(root_dir)
    path = root_dir / CONFIG_DIR / CALIBRATION_FILE
    data: dict = {}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    transforms = data.get("transforms") or {}
    transforms[f"{parent}_to_{child}"] = {
        "parent": parent,
        "child": child,
        "matrix": np.asarray(matrix, dtype=float).tolist(),
    }
    (root_dir / CONFIG_DIR).mkdir(exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"version": 1, "transforms": transforms}, f,
                  default_flow_style=False, sort_keys=True)


def verify_config(root_dir: str | Path) -> list[str]:
    """Check ``.apairo/channels.yaml`` for inconsistencies.

    Returns a list of human-readable issue strings.  An empty list means the
    config is coherent with what is present on disk.

    Checks performed:

    * ``channels.yaml`` exists and is parseable YAML.
    * ``version`` is ``1``.
    * Every channel directory is present on disk.
    * Every ``loader`` value is a known loader type.
    * Every ``timestamps_from`` reference names an existing channel.
    * Every ``sources`` entry names an existing channel.

    Args:
        root_dir: Dataset root (or sequence) directory that contains
            ``.apairo/channels.yaml``.

    Returns:
        List of issue strings.  Empty list -> config is consistent.

    Example::

        issues = verify_config("/data/my_dataset/seq_01")
        if issues:
            for issue in issues:
                print("  -", issue)
    """
    root_dir = Path(root_dir)
    issues: list[str] = []

    if not config_exists(root_dir):
        return [".apairo/channels.yaml does not exist"]

    try:
        config = read_config(root_dir)
    except Exception as exc:
        return [f"Cannot parse channels.yaml: {exc}"]

    version = config.get("version")
    if version != SCHEMA_VERSION:
        issues.append(f"Unknown version: {version!r} (expected {SCHEMA_VERSION})")
    issues += _unknown(config, _CHANNELS_TOP_FIELDS, "channels.yaml")

    channels = config.get("channels", {})
    if not isinstance(channels, dict):
        issues.append("'channels' field is not a mapping")
        return issues

    for key, meta in channels.items():
        if not isinstance(meta, dict):
            issues.append(f"Channel '{key}': entry is not a mapping")
            continue

        if not (root_dir / key).is_dir():
            issues.append(
                f"Channel '{key}': directory not found on disk ({root_dir / key})"
            )

        issues += _unknown(meta, _CHANNEL_FIELDS, f"channel '{key}'")

        kind = meta.get("kind")
        if kind is not None and kind not in _CHANNEL_KINDS:
            issues.append(
                f"Channel '{key}': unknown kind '{kind}' (expected one of "
                f"{sorted(_CHANNEL_KINDS)})"
            )

        loader = meta.get("loader")
        if loader and loader not in KNOWN_LOADERS:
            issues.append(f"Channel '{key}': unknown loader '{loader}'")

        tf = meta.get("transform")
        if tf is not None:
            if not isinstance(tf, dict):
                issues.append(f"Channel '{key}': 'transform' is not a mapping")
            else:
                for field in ("parent", "child"):
                    if field not in tf:
                        issues.append(
                            f"Channel '{key}': transform is missing '{field}'"
                        )
                issues += _unknown(
                    tf, _TRANSFORM_FIELDS, f"channel '{key}' transform"
                )

        ts_from = meta.get("timestamps_from")
        if ts_from and ts_from not in channels:
            issues.append(
                f"Channel '{key}': timestamps_from='{ts_from}' "
                f"is not declared in channels"
            )

        for src in meta.get("sources", []):
            if src not in channels:
                issues.append(
                    f"Channel '{key}': source '{src}' is not declared in channels"
                )

    # Aliases must be unique and must not shadow a real channel directory.
    seen_alias: dict[str, str] = {}
    for key, meta in channels.items():
        alias = meta.get("alias")
        if not alias:
            continue
        if alias in channels:
            issues.append(
                f"Channel '{key}': alias '{alias}' collides with an existing channel name"
            )
        if alias in seen_alias:
            issues.append(
                f"Channel '{key}': alias '{alias}' is already used by '{seen_alias[alias]}'"
            )
        seen_alias[alias] = key

    return issues


def verify_manifest(root_dir: str | Path) -> list[str]:
    """Check ``.apairo/dataset.yaml`` (the root manifest) against the version-1
    schema.  The manifest is **optional**: an absent file is not an issue
    (returns ``[]``).  Validates ``version`` and warns on unknown fields.
    """
    path = _apairo_dir(Path(root_dir)) / DATASET_FILE
    if not path.exists():
        return []
    try:
        with open(path) as f:
            manifest = yaml.safe_load(f) or {}
    except Exception as exc:
        return [f"Cannot parse dataset.yaml: {exc}"]
    if not isinstance(manifest, dict):
        return ["dataset.yaml: top level is not a mapping"]

    issues: list[str] = []
    version = manifest.get("version")
    if version is not None and version != SCHEMA_VERSION:
        issues.append(f"dataset.yaml: unknown version {version!r} (expected {SCHEMA_VERSION})")
    issues += _unknown(manifest, _MANIFEST_FIELDS, "dataset.yaml")
    return issues


def verify_calibration(root_dir: str | Path) -> list[str]:
    """Check ``.apairo/calibration.yaml`` against the version-1 schema.

    Calibration is **optional** (many datasets ship already-calibrated data or
    have no extrinsics): an absent file is not an issue (returns ``[]``).  When
    present, validates ``version``, the ``transforms`` mapping, each entry's
    ``parent``/``child``/``matrix`` (a 4x4), and warns on unknown fields.
    """
    path = _apairo_dir(Path(root_dir)) / CALIBRATION_FILE
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        return [f"Cannot parse calibration.yaml: {exc}"]
    if not isinstance(data, dict):
        return ["calibration.yaml: top level is not a mapping"]

    issues: list[str] = []
    version = data.get("version")
    if version is not None and version != SCHEMA_VERSION:
        issues.append(
            f"calibration.yaml: unknown version {version!r} (expected {SCHEMA_VERSION})"
        )
    issues += _unknown(data, _CALIBRATION_TOP_FIELDS, "calibration.yaml")

    transforms = data.get("transforms") or {}
    if not isinstance(transforms, dict):
        issues.append("calibration.yaml: 'transforms' is not a mapping")
        return issues

    for name, entry in transforms.items():
        # A bare 4x4 (no parent/child wrapper) is accepted by read_calibration.
        if not isinstance(entry, dict):
            if not _is_4x4(entry):
                issues.append(f"transform '{name}': not a 4x4 matrix")
            continue
        for field in ("parent", "child", "matrix"):
            if field not in entry:
                issues.append(f"transform '{name}': missing '{field}'")
        if "matrix" in entry and not _is_4x4(entry["matrix"]):
            issues.append(f"transform '{name}': 'matrix' is not 4x4")
        issues += _unknown(entry, _CALIBRATION_TRANSFORM_FIELDS, f"transform '{name}'")
    return issues


def _is_4x4(matrix) -> bool:
    try:
        return np.asarray(matrix, dtype=float).shape == (4, 4)
    except Exception:
        return False
