"""RawDataset -- generic loader for apairo raw datasets (``channels.yaml``-driven).

Loads any dataset laid out as one subdirectory per channel with a
``.apairo/channels.yaml`` describing each channel's loader -- the layout produced
by tools like ``apairo-extractor``.  No structural profile is needed:
``channels.yaml`` is the single source of truth, so multi-rate channels and any
loader format (``npy``, ``npys``, ``bin``, ``img``, ``zarr``) load correctly.

It is the *profile-free* member of the asynchronous layout family: the on-disk
format comes from :class:`~apairo.dataset.kitti.AsyncLayoutDataset`, the
multi-sequence root behaviour from
:class:`~apairo.core.root_sequence.RootSequenceMixin`, and -- unlike
:class:`~apairo.dataset.tartan_kitti.TartanKittiDataset` -- it pins *no* channel
set: the channels are whatever ``channels.yaml`` declares.  It mixes in
:class:`~apairo.core.configurable_dataset.ConfigurableDataset`, so derived
channels can be computed with :meth:`run_preprocess` and registered back.

Like the rest of the family it is **asynchronous** (each channel keeps its own
``timestamps.txt``); call :meth:`synchronize` to obtain synchronous frames.

Layout::

    <root>/                         # dataset root
        .apairo/dataset.yaml        # optional: name + sequence order
        seq_a/                      # a sequence
            .apairo/channels.yaml   # channel -> loader
            lidar/  000000.npy ... timestamps.txt
            imu/    imu.npy         timestamps.txt
        seq_b/ ...

Usage::

    ds = RawDataset("/data/my_dataset")          # whole dataset (all sequences)
    ds = RawDataset("/data/my_dataset/seq_a")    # a single sequence
    ds = RawDataset(root, keys=["lidar", "imu"]) # restrict channels
    ds_sync = ds.synchronize(reference="lidar")  # synchronous frames
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

from apairo.core.config import CONFIG_DIR, config_exists
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.root_sequence import RootSequenceMixin
from apairo.dataset.kitti import AsyncLayoutDataset
from apairo.dataset.kitti.dataset import _detect_loader
from apairo.utils.files import get_files

_MANIFEST_FILE = "dataset.yaml"


def _read_manifest(root: Path) -> dict:
    """Read ``<root>/.apairo/dataset.yaml`` if present, else ``{}``."""
    path = root / CONFIG_DIR / _MANIFEST_FILE
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _is_dataset_root(path: Path) -> bool:
    """A root has a dataset manifest, or subdirectories that are sequences."""
    if (path / CONFIG_DIR / _MANIFEST_FILE).exists():
        return True
    return any(
        config_exists(d)
        for d in path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


class RawDataset(RootSequenceMixin, AsyncLayoutDataset, ConfigurableDataset):
    r"""Generic ``channels.yaml``-driven dataset; single sequence or dataset root.

    Args:
        directory: A sequence directory (has ``.apairo/channels.yaml``) **or** a
            dataset root directory (has ``.apairo/dataset.yaml`` or sequence
            subdirectories) -- auto-detected.
        keys: Channels to load. ``None`` -> every channel declared in
            ``channels.yaml``.

    Example::

        RawDataset.init(seq_dir)                      # once: write .apairo
        ds = RawDataset(root, keys=["lidar", "imu"])  # whole dataset
        ds.run_preprocess(MyLabeler())                # persisted as a new channel
    """

    def __init__(
        self,
        directory: str | Path,
        keys: Optional[List[str]] = None,
    ) -> None:
        path = Path(directory)

        if config_exists(path):
            self._is_root = False
            self._sequence_dir = path
            self._name = path.name
            super().__init__(path, keys=keys)
        elif _is_dataset_root(path):
            self._init_raw_root(path, keys)
        else:
            raise FileNotFoundError(
                f"'{path}' is neither a sequence (no {CONFIG_DIR}/channels.yaml) nor a "
                f"dataset root (no {CONFIG_DIR}/{_MANIFEST_FILE} and no sequence "
                f"subdirectories). Initialize a sequence with RawDataset.init(<seq>)."
            )

    # ------------------------------------------------------------------ root

    def _init_raw_root(self, root: Path, keys: Optional[List[str]]) -> None:
        manifest = _read_manifest(root)
        self._name = manifest.get("name", root.name)

        # Sequence order: manifest order if given, else sorted discovery.
        if manifest.get("sequences"):
            seq_dirs = [
                root / s for s in manifest["sequences"] if config_exists(root / s)
            ]
        else:
            seq_dirs = sorted(
                d
                for d in root.iterdir()
                if d.is_dir() and not d.name.startswith(".") and config_exists(d)
            )
        if not seq_dirs:
            raise FileNotFoundError(f"No sequences found under '{root}'.")

        super()._init_root(
            root, seq_dirs, lambda d: RawDataset(d, keys=keys), build_index=True
        )

    # ------------------------------------------------------------------ hooks

    def _single_available(self) -> frozenset:
        return frozenset(self._profile)

    def _set_single_keys(self, keys) -> None:
        if keys == "all":
            keys = sorted(self._profile)
        # Delegate to the layout base's setter (validates + re-inits loaders).
        AsyncLayoutDataset.keys.fset(self, list(keys))

    # ------------------------------------------------------------------ public

    @property
    def name(self) -> str:
        """Dataset name (manifest ``name``, else the directory name)."""
        return self._name

    # ------------------------------------------------------------------ helpers

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        return self._sequence_dir / key / f"{idx:06d}.{ext}"

    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        """ConfigurableDataset hook: detect raw channels when .apairo is absent."""
        channels: dict = {}
        for key in sorted(get_files(str(sequence_dir))):
            loader = _detect_loader(Path(sequence_dir) / key)
            if loader is None:
                continue
            has_ts = (Path(sequence_dir) / key / "timestamps.txt").exists()
            channels[key] = {"loader": loader, "kind": "raw", "has_timestamps": has_ts}
        return {"version": 1, "channels": channels}
