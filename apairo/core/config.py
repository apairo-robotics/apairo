from __future__ import annotations
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
    """Write ``<root>/.apairo/dataset.yaml`` (the root manifest). Returns its path."""
    d = _apairo_dir(Path(root_dir))
    d.mkdir(exist_ok=True)
    path = d / DATASET_FILE
    with open(path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=True)
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
    directory keeps its real name, but ``RawDataset(root, keys=[alias])`` loads
    it and ``sample.data[alias]`` returns it. This brings the profile-free
    :class:`~apairo.dataset.raw.RawDataset` the canonical-naming ergonomics that
    profiled datasets get from their layout -- naming lives in ``.apairo``, not
    in the call site.

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


def read_calibration(root_dir: str | Path) -> dict[str, np.ndarray]:
    """Static extrinsics from ``root_dir/.apairo/calibration.yaml``.

    Returns ``{"<parent>_to_<child>": 4x4 float64 ndarray}`` (empty if absent).
    apairo only *exposes* these transforms; it never applies them.
    """
    path = Path(root_dir) / CONFIG_DIR / CALIBRATION_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, np.ndarray] = {}
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
    if version != 1:
        issues.append(f"Unknown version: {version!r} (expected 1)")

    channels = config.get("channels", {})
    if not isinstance(channels, dict):
        issues.append("'channels' field is not a mapping")
        return issues

    for key, meta in channels.items():
        if not (root_dir / key).is_dir():
            issues.append(
                f"Channel '{key}': directory not found on disk ({root_dir / key})"
            )

        loader = meta.get("loader")
        if loader and loader not in KNOWN_LOADERS:
            issues.append(f"Channel '{key}': unknown loader '{loader}'")

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
