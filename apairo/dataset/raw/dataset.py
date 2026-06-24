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

from apairo.core.config import (
    CHANNELS_FILE,
    CONFIG_DIR,
    config_exists,
    read_config,
)
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
    """A root has a dataset manifest, or subdirectories that are sequences --
    already initialised (``.apairo``) or a bare channel layout to bootstrap."""
    if (path / CONFIG_DIR / _MANIFEST_FILE).exists():
        return True
    return any(
        config_exists(d) or RawDataset._is_sequence_layout(d)
        for d in path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


class RawDataset(RootSequenceMixin, AsyncLayoutDataset, ConfigurableDataset):
    r"""Generic ``channels.yaml``-driven dataset; single sequence or dataset root.

    Args:
        directory: A sequence directory or a dataset root directory --
            auto-detected. A directory with no ``.apairo`` is bootstrapped on
            load (loaders inferred from file extensions), so raw data works with
            no manual :meth:`init`.
        keys: Channels to load. ``None`` -> every channel declared in
            ``channels.yaml``.

    Example::

        ds = RawDataset(root, keys=["lidar", "imu"])  # whole dataset
        ds.run_preprocess(MyLabeler())                # persisted as a new channel
    """

    def __init__(
        self,
        directory: str | Path,
        keys: Optional[List[str]] = None,
    ) -> None:
        path = Path(directory)

        # A sequence loads directly; a bare channel layout (no .apairo) is
        # bootstrapped on the spot, so raw data needs no manual init().
        if config_exists(path) or self._is_sequence_layout(path):
            self._is_root = False
            self._sequence_dir = path
            self._name = path.name
            if not config_exists(path):
                self._load_or_create_config(path)
            super().__init__(path, keys=keys)
        elif _is_dataset_root(path):
            self._init_raw_root(path, keys)
        else:
            raise FileNotFoundError(
                f"'{path}' is neither a sequence (no recognizable channel "
                f"sub-directories) nor a dataset root (no sequence sub-directories)."
            )

    # ------------------------------------------------------------------ init

    @classmethod
    def init(
        cls,
        directory: str | Path,
        *,
        merge: bool = False,
        overwrite: bool = False,
        name: Optional[str] = None,
    ) -> Path:
        """Write the ``.apairo`` sidecar(s) by scanning *directory*. Root-aware.

        A **sequence** directory (its sub-directories hold data files) gets a
        ``.apairo/channels.yaml`` with loaders inferred per channel. A **root**
        directory (its sub-directories are sequences) gets each sequence
        initialised, then a ``.apairo/dataset.yaml`` manifest (name + sequence
        order + channel union).

        Args:
            directory: Sequence or dataset-root directory (auto-detected).
            merge: Add newly detected channels without touching existing ones.
            overwrite: Discard existing ``.apairo`` and rebuild from scratch.
            name: Dataset name for the root manifest (default: directory name).

        Returns:
            Path of the file written -- ``channels.yaml`` for a sequence, or
            ``dataset.yaml`` for a root.
        """
        path = Path(directory)

        if cls._is_sequence_layout(path):
            AsyncLayoutDataset.init(path, overwrite=overwrite, merge=merge)
            return path / CONFIG_DIR / CHANNELS_FILE

        seq_dirs: list[Path] = []
        for d in sorted(path.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if cls._is_sequence_layout(d):
                try:
                    AsyncLayoutDataset.init(d, overwrite=overwrite, merge=merge)
                except (FileExistsError, ValueError):
                    # Already initialised (no overwrite/merge), or merge found
                    # nothing new -- either way the sequence is ready. Idempotent.
                    pass
                seq_dirs.append(d)
            elif config_exists(d):
                seq_dirs.append(d)

        if not seq_dirs:
            raise FileNotFoundError(
                f"'{path}' has no channels and no sequence sub-directories to "
                f"initialise. Point init at a sequence or a dataset root."
            )
        return cls._write_manifest(path, name=name)

    @staticmethod
    def _is_sequence_layout(path: Path) -> bool:
        """True when *path*'s own sub-directories include a recognizable channel."""
        return any(
            _detect_loader(d) is not None
            for d in path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    @classmethod
    def _write_manifest(cls, root: str | Path, *, name: Optional[str] = None) -> Path:
        """(Re)write ``<root>/.apairo/dataset.yaml`` from the sequences on disk."""
        root = Path(root)
        sequences = sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and config_exists(d)
        )
        channels: dict = {}
        for seq in sequences:
            for key, meta in read_config(root / seq).get("channels", {}).items():
                channels.setdefault(key, {"kind": meta.get("kind", "raw")})

        manifest = {
            "version": 1,
            "name": name or root.name,
            "sequences": sequences,
            "channels": channels,
        }
        apairo_dir = root / CONFIG_DIR
        apairo_dir.mkdir(exist_ok=True)
        path = apairo_dir / _MANIFEST_FILE
        with open(path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=True)
        return path

    # ------------------------------------------------------------------ root

    def _init_raw_root(self, root: Path, keys: Optional[List[str]]) -> None:
        manifest = _read_manifest(root)
        self._name = manifest.get("name", root.name)

        def is_seq(d: Path) -> bool:  # initialised, or a bare layout to bootstrap
            return config_exists(d) or self._is_sequence_layout(d)

        # Sequence order: manifest order if given, else sorted discovery.
        if manifest.get("sequences"):
            seq_dirs = [root / s for s in manifest["sequences"] if is_seq(root / s)]
        else:
            seq_dirs = sorted(
                d
                for d in root.iterdir()
                if d.is_dir() and not d.name.startswith(".") and is_seq(d)
            )
        if not seq_dirs:
            raise FileNotFoundError(f"No sequences found under '{root}'.")

        # type(self) so a profiled subclass (e.g. TartanKittiDataset) builds
        # sequences of its own kind, keeping its channel profile.
        super()._init_root(
            root, seq_dirs, lambda d: type(self)(d, keys=keys), build_index=True
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
            channels[key] = {"loader": loader, "kind": "raw"}
        return {"version": 1, "channels": channels}
