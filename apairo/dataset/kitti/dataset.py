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
    """Infer loader type from the contents of *channel_dir*.

    A channel directory that is itself a Zarr array store (it holds a
    ``.zarray`` / ``zarr.json`` metadata file, with ``timestamps.txt`` placed
    beside the chunks) is detected as ``"zarr"``; otherwise the loader is
    inferred from the data-file extensions.
    """
    if (channel_dir / ".zarray").exists() or (channel_dir / "zarr.json").exists():
        return "zarr"
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


class AsyncLayoutDataset(AbstractDataset):
    r"""Abstract *asynchronous layout* loader (one subdirectory per channel).

    This is the format primitive of the asynchronous dataset family -- it is
    **not** the KITTI dataset (no real KITTI dataset uses it; see
    :class:`~apairo.dataset.semantic_kitti.SemanticKittiDataset`, which is a
    synchronous :class:`~apairo.core.profiled_dataset.ProfiledDataset`).

    It describes *how* channels are stored, never *which* channels exist: each
    channel is a subdirectory with its own ``timestamps.txt`` and data files in
    a format known to the loader registry (``npys``, ``npy``, ``bin``, ``img``,
    ``zarr``). The set of channels is per-instance state, read from
    ``.apairo/channels.yaml`` (or an explicit ``dataset_profile``). Datasets
    with a *fixed* channel set layer a profile on top (e.g.
    :class:`~apairo.dataset.tartan_kitti.TartanKittiDataset`); datasets with
    *dynamic* channels (e.g. ``apairo-extractor`` output) use
    :class:`~apairo.dataset.raw.RawDataset`, which reads the channel set from
    ``.apairo`` with no profile.

    **Usage with an explicit profile (original API)**::

        ds = AsyncLayoutDataset(seq_dir, keys=["lidar", "cam"], dataset_profile="my.yaml")

    **Usage with** ``.apairo`` **(after** :meth:`init` **has been called)**::

        AsyncLayoutDataset.init(seq_dir)          # once, auto-detects channels
        ds = AsyncLayoutDataset(seq_dir)          # keys and loaders come from .apairo
        ds = AsyncLayoutDataset(seq_dir, keys=["lidar"])  # restrict to a subset

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

        # Channel metadata from .apairo (empty when a dataset_profile is passed
        # and no sidecar exists). alias_of maps an on-disk directory name to the
        # public name it is exposed under; timestamp_aliases maps a channel to the
        # one it borrows its clock from (its `timestamps_from`). Everything below
        # is keyed by the public name; the directory name only locates files.
        channels = read_config(directory).get("channels", {}) if config_exists(directory) else {}
        self._alias_of: Dict[str, str] = {
            k: v["alias"] for k, v in channels.items() if v.get("alias")
        }
        self._timestamp_aliases: Dict[str, str] = {
            self._public(k): self._resolve_key(v["timestamps_from"])
            for k, v in channels.items()
            if v.get("timestamps_from")
        }

        if dataset_profile is not None:
            self._profile: Dict[str, str] = load_profile(dataset_profile)
        elif channels:
            self._profile = {
                self._public(k): v["loader"]
                for k, v in channels.items()
                if "loader" in v
            }
            if keys is None:
                keys = sorted(self._profile.keys())
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

        # Re-key the on-disk directories by their public name so a request, a
        # loader and a sample all speak the same (aliased) language.
        self._files: Dict[str, str] = {}
        for real, path in get_files(str(directory)).items():
            public = self._public(real)
            if public in self._files:
                raise ValueError(
                    f"Alias collision in '{directory}': the public name '{public}' "
                    f"is claimed by more than one channel. Clear one alias with "
                    f"`apairo alias <channel> --remove` (see `apairo status`)."
                )
            self._files[public] = path

        keys = [self._resolve_key(k) for k in keys]
        missing = set(keys) - set(self._files)
        if missing:
            raise KeyError(f"Keys not found in dataset directory: {missing}")

        self._keys: List[str] = []
        self._set_keys(keys)
        self._init()

    # ------------------------------------------------------------------ alias

    def _public(self, real_name: str) -> str:
        """Public name a directory is exposed under (its alias, else itself)."""
        return self._alias_of.get(real_name, real_name)

    def _resolve_key(self, key: str) -> str:
        """Normalize a requested key (alias *or* real directory name) to its
        public name. Unknown keys pass through unchanged so the usual
        not-found error still fires."""
        if key in self._alias_of:  # a real name that has an alias -> its alias
            return self._alias_of[key]
        return key  # already a public name (an alias, or an unaliased real name)

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
        keys = [self._resolve_key(k) for k in keys]
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
        self.timestamps: Dict[str, np.ndarray] = self._collect_timestamps()
        self.end_of_time: float = get_end_of_time(self.timestamps) + 1.0

    def _collect_timestamps(self) -> Dict[str, np.ndarray]:
        """Timestamps per loaded key: its own ``timestamps.txt`` when present,
        else the channel named by its ``timestamps_from`` (a derived channel
        sharing its source's clock), else the legacy replacement map handled by
        :func:`~apairo.loader.loads_timestamps`."""
        timestamps: Dict[str, np.ndarray] = {}
        fallback: List[str] = []
        for key in self._keys:
            ts_path = Path(self._files[key]) / "timestamps.txt"
            if ts_path.exists():
                timestamps[key] = np.loadtxt(ts_path)
            elif key in self._timestamp_aliases:
                src = self._timestamp_aliases[key]
                if src not in timestamps:
                    src_path = Path(self._files[src]) / "timestamps.txt"
                    if not src_path.exists():
                        raise ValueError(
                            f"'{key}' shares timestamps with '{src}' (timestamps_from), "
                            f"but '{src}' has no timestamps.txt."
                        )
                    timestamps[src] = np.loadtxt(src_path)
                timestamps[key] = timestamps[src]
            else:
                fallback.append(key)
        if fallback:
            timestamps.update(loads_timestamps(fallback, self._files))
        return timestamps

    def _init_timeline(self) -> None:
        """Build the interleaved timeline as two parallel numpy arrays."""
        from apairo.utils.timestamps import merge_timeline
        self._tl_key_idxs, self._tl_frame_idxs = merge_timeline(
            self.timestamps, self._keys
        )

    # ------------------------------------------------------------ dunder

    def __len__(self) -> int:
        return len(self._tl_key_idxs)

    def _load(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        key = self._keys[self._tl_key_idxs[idx]]
        frame = int(self._tl_frame_idxs[idx])
        return Sample(
            data={key: self.loaders[key][frame]},
            timestamp=float(self.timestamps[key][frame]),
        )

