from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml

CONFIG_DIR = ".apairo"
CHANNELS_FILE = "channels.yaml"
CONFIG_FILENAME = CONFIG_DIR  # alias kept for external code that checks (path / CONFIG_FILENAME).exists()

# Keep in sync with str_to_loader (apairo/loader/__init__.py) and WRITERS (apairo/writer/__init__.py).
KNOWN_LOADERS: frozenset[str] = frozenset(
    {"npy", "npys", "npys_img", "bin", "img", "zarr"}
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

    has_ts = (root_dir / key / "timestamps.txt").exists()
    entry: dict = {"has_timestamps": has_ts, "kind": "preprocess", "loader": loader}
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
    has_timestamps: Optional[bool] = None,
    frame: Optional[str] = None,
    transform: Optional[dict] = None,
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
        loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, or ``"img"``.
        has_timestamps: Whether the channel directory contains a
            ``timestamps.txt``.  Auto-detected from disk when ``None``.
        frame: Coordinate frame the channel's data is expressed in (descriptive
            metadata only; apairo does not apply transforms).
        transform: For a channel that *is* a coordinate transform (a pose
            stream), the edge it provides, e.g.
            ``{"parent": "odom", "child": "base_link"}`` (optionally
            ``"static": True``, ``"format": "t_xyz_q_xyzw"``). Descriptive only.
    """
    root_dir = Path(root_dir)
    config = (
        read_config(root_dir)
        if config_exists(root_dir)
        else {"version": 1, "channels": {}}
    )

    if has_timestamps is None:
        has_timestamps = (root_dir / key / "timestamps.txt").exists()

    entry: dict = {"has_timestamps": has_timestamps, "kind": "raw", "loader": loader}
    if frame is not None:
        entry["frame"] = frame
    if transform is not None:
        entry["transform"] = transform
    config["channels"][key] = entry
    write_config(root_dir, config)


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

    return issues
