from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from apairo.utils.timestamps import get_end_of_time
from apairo.loader import str_to_loader, loads_timestamps, load_profile
from apairo.utils.files import get_files
from apairo.core import AbstractDataset, AbstractLoader
from apairo.core.sample import Sample
from apairo.core.config import (
    config_exists,
    read_config,
    write_config,
    register_raw_channel as _register_raw_channel,
)


def _detect_loader(channel_dir: Path) -> str | None:
    """Infer loader type from file extensions found in *channel_dir*."""
    data_files = [
        f for f in channel_dir.iterdir()
        if f.is_file() and f.name != "timestamps.txt"
    ]
    if not data_files:
        return None
    exts = {f.suffix.lower() for f in data_files}
    if ".bin" in exts:
        return "bin"
    if exts & {".png", ".jpg", ".jpeg", ".bmp"}:
        return "img"
    npy_files = [f for f in data_files if f.suffix == ".npy"]
    if npy_files:
        # Multiple per-frame files → npys; single file → npy.
        return "npys" if len(npy_files) > 1 else "npy"
    return None


class KittiDataset(AbstractDataset):
    r"""Generic dataset for KITTI-layout directories (one subdirectory per modality).

    Each modality subdirectory must contain a ``timestamps.txt`` file and data
    files in a format known to the loader registry (``npys``, ``npy``, ``bin``,
    ``img``).

    **Usage with an explicit profile (original API)**::

        ds = KittiDataset(seq_dir, keys=["lidar", "cam"], dataset_profile="my.yaml")

    **Usage with** ``.apairo`` **(after** :meth:`init` **has been called)**::

        KittiDataset.init(seq_dir)          # once, auto-detects channels
        ds = KittiDataset(seq_dir)          # keys and loaders come from .apairo
        ds = KittiDataset(seq_dir, keys=["lidar"])  # restrict to a subset

    Args:
        directory: Path to the dataset root / sequence directory.
        keys: Modality names to load.  ``None`` → all channels declared in
            ``.apairo`` (requires ``.apairo`` to exist).
        dataset_profile: YAML profile filename **or** absolute Path mapping keys
            to loader types.  ``None`` → loaders are read from ``.apairo``
            (requires ``.apairo`` to exist).
    """

    synchronous: bool = False

    def __init__(
        self,
        directory: str | Path,
        keys: Optional[List[str]] = None,
        dataset_profile: Optional[str | Path] = None,
    ) -> None:
        directory = Path(directory)

        if dataset_profile is not None:
            self._profile: Dict[str, str] = load_profile(dataset_profile)
        elif config_exists(directory):
            config = read_config(directory)
            channels = config.get("channels", {})
            self._profile = {k: v["loader"] for k, v in channels.items() if "loader" in v}
            if keys is None:
                keys = sorted(channels.keys())
        else:
            raise FileNotFoundError(
                f"No dataset_profile given and no .apairo found in '{directory}'. "
                f"Either pass dataset_profile=..., or initialize with "
                f"{type(self).__name__}.init('{directory}')."
            )

        if keys is None:
            raise ValueError(
                "keys must be specified when dataset_profile is given. "
                "Pass keys=[...] or use .apairo (call init() first)."
            )

        self._files: Dict[str, str] = get_files(str(directory))

        missing = set(keys) - set(self._files)
        if missing:
            raise KeyError(f"Keys not found in dataset directory: {missing}")

        self._keys: List[str] = []
        self._set_keys(keys)
        self._init()

    @classmethod
    def init(
        cls,
        directory: str | Path,
        *,
        raw_keys: Optional[List[str]] = None,
        overwrite: bool = False,
        merge: bool = False,
    ) -> None:
        """Scan a KITTI-layout directory and write ``.apairo/channels.yaml``.

        All detected subdirectories are registered as raw channels.  Loader
        type is inferred from file extensions:

        * ``.bin`` → ``bin``
        * ``.png`` / ``.jpg`` / … → ``img``
        * multiple ``.npy`` files → ``npys``
        * single ``.npy`` file → ``npy``

        For ambiguous cases (e.g. a single-frame ``.npy`` that is actually
        per-frame), call :func:`~apairo.core.config.register_raw_channel`
        afterwards to override the detected loader.

        Args:
            directory: KITTI root / sequence directory to initialize.
            raw_keys: Subdirectory names to include.  ``None`` → all detected
                subdirectories with recognizable file types.
            overwrite: Discard the existing ``.apairo`` and rebuild from
                scratch.  Incompatible with ``merge``.
            merge: Add newly detected raw channels to an existing ``.apairo``
                without touching channels already declared (raw or
                preprocessed).  If ``.apairo`` does not yet exist, behaves
                like a normal init.  Incompatible with ``overwrite``.

        Raises:
            ValueError: If both ``overwrite`` and ``merge`` are ``True``.
            FileExistsError: If ``.apairo`` already exists and both
                ``overwrite`` and ``merge`` are ``False``.
            ValueError: If no new recognizable channels are found.
        """
        if overwrite and merge:
            raise ValueError("overwrite and merge are mutually exclusive.")

        directory = Path(directory)

        if merge and config_exists(directory):
            existing = read_config(directory).get("channels", {})
            added = 0
            for channel_dir in sorted(directory.iterdir()):
                if not channel_dir.is_dir() or channel_dir.name.startswith("."):
                    continue
                if raw_keys is not None and channel_dir.name not in raw_keys:
                    continue
                if channel_dir.name in existing:
                    continue
                loader = _detect_loader(channel_dir)
                if loader is None:
                    continue
                _register_raw_channel(directory, channel_dir.name, loader)
                added += 1
            if added == 0:
                detail = f" (checked: {raw_keys})" if raw_keys else ""
                raise ValueError(
                    f"No new recognizable channels found in '{directory}'{detail}."
                )
            return

        if config_exists(directory) and not overwrite:
            raise FileExistsError(
                f".apairo already exists in '{directory}'. "
                f"Pass overwrite=True to reinitialize, or merge=True to add new channels."
            )

        channels: dict = {}
        for channel_dir in sorted(directory.iterdir()):
            if not channel_dir.is_dir() or channel_dir.name.startswith("."):
                continue
            if raw_keys is not None and channel_dir.name not in raw_keys:
                continue
            loader = _detect_loader(channel_dir)
            if loader is None:
                continue
            channels[channel_dir.name] = {
                "has_timestamps": (channel_dir / "timestamps.txt").exists(),
                "kind": "raw",
                "loader": loader,
            }

        if not channels:
            detail = f" (checked: {raw_keys})" if raw_keys else ""
            raise ValueError(
                f"No recognizable channels found in '{directory}'{detail}. "
                f"Expected subdirectories containing .bin, .npy, or image files."
            )

        write_config(directory, {"version": 1, "channels": channels})

    # ------------------------------------------------------------------ keys

    @property
    def keys(self) -> List[str]:
        return self._keys

    @keys.setter
    def keys(self, keys: List[str]) -> None:
        missing = set(keys) - set(self._files)
        if missing:
            raise KeyError(f"Keys not found in dataset directory: {missing}")
        self._set_keys(list(keys))
        self._init()

    # ----------------------------------------------------------------- shape

    @property
    def shape(self) -> Dict[str, Tuple[int, ...]]:
        return {key: self.loaders[key].shape for key in self.keys}

    # ----------------------------------------------------------------- init

    def _init(self) -> None:
        if not self._keys:
            return
        self._init_loaders()
        self._init_timeline()

    def _init_loaders(self) -> None:
        self.loaders: Dict[str, AbstractLoader] = {
            key: str_to_loader[self._profile[key]](self._files[key])
            for key in self._keys
        }
        self.timestamps: Dict[str, np.ndarray] = loads_timestamps(
            self._keys, self._files
        )
        self.end_of_time: float = get_end_of_time(self.timestamps) + 1.0

    def _init_timeline(self) -> None:
        """Build the interleaved timeline as two parallel numpy arrays."""
        n_keys = len(self._keys)
        current_idxs = np.zeros(n_keys, dtype=np.intp)
        current_ts = np.array([self.timestamps[k][0] for k in self._keys])

        tl_key_idxs: list[int] = []
        tl_frame_idxs: list[int] = []

        while True:
            ki = int(np.argmin(current_ts))
            if current_ts[ki] >= self.end_of_time:
                break

            tl_key_idxs.append(ki)
            tl_frame_idxs.append(int(current_idxs[ki]))

            current_idxs[ki] += 1
            key = self._keys[ki]
            if current_idxs[ki] >= len(self.timestamps[key]):
                current_ts[ki] = self.end_of_time
            else:
                current_ts[ki] = self.timestamps[key][current_idxs[ki]]

        self._tl_key_idxs: np.ndarray = np.array(tl_key_idxs, dtype=np.intp)
        self._tl_frame_idxs: np.ndarray = np.array(tl_frame_idxs, dtype=np.intp)

    # ------------------------------------------------------------ dunder

    def __len__(self) -> int:
        return len(self._tl_key_idxs)

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        key = self._keys[self._tl_key_idxs[idx]]
        frame = int(self._tl_frame_idxs[idx])
        return Sample(
            data={key: self.loaders[key][frame]},
            timestamp=float(self.timestamps[key][frame]),
        )

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample
