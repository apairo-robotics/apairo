from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml

CONFIG_DIR = ".apairo"
CHANNELS_FILE = "channels.yaml"
CONFIG_FILENAME = CONFIG_DIR  # alias kept for external code that checks (path / CONFIG_FILENAME).exists()


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
) -> None:
    """Register a preprocessed channel in ``root_dir/.apairo/channels.yaml``.

    This is the low-level standalone function.  Most users will prefer the
    classmethod :meth:`ConfigurableDataset.register_channel` so that the call
    site names the dataset type explicitly.

    Args:
        root_dir: Dataset root directory.
        key: Channel name — must match its subdirectory name.
        loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, or ``"img"``.
        timestamps_from: Channel whose timestamps to share when this channel
            has no ``timestamps.txt`` of its own.
        sources: Provenance — raw channels this channel was derived from.
    """
    root_dir = Path(root_dir)
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

    config["channels"][key] = entry
    write_config(root_dir, config)
