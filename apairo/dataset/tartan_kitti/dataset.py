from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np

from apairo.loader import str_to_loader, loads_timestamps, load_profile
from apairo.utils.files import get_files
from apairo.utils.timestamps import get_end_of_time
from apairo.dataset.kitti import AsyncLayoutDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.root_sequence import RootSequenceMixin
from apairo.core.config import read_config, config_exists

_PROFILE_PATH = Path(__file__).parent / "profile.yaml"


def _is_sequence_dir(path: Path, raw_profile: dict) -> bool:
    """True if *path* looks like a single TartanDrive sequence directory."""
    return config_exists(path) or any((path / k).is_dir() for k in raw_profile)


class TartanKittiDataset(RootSequenceMixin, AsyncLayoutDataset, ConfigurableDataset):
    r"""TartanDrive v2 dataset (asynchronous layout, fixed channel profile).

    A profiled member of the asynchronous layout family: the on-disk format is
    handled by :class:`~apairo.dataset.kitti.AsyncLayoutDataset`, the
    multi-sequence root behaviour by
    :class:`~apairo.core.root_sequence.RootSequenceMixin`, and this class pins
    the *fixed* TartanDrive channel set via ``profile.yaml``.  (Datasets with a
    *dynamic* channel set use :class:`~apairo.dataset.raw.RawDataset` instead.)

    Accepts either a single sequence directory or a root directory that contains
    multiple sequences -- the structure is auto-detected.

    Single sequence::

        ds = TartanKittiDataset(seq_dir, keys=["velodyne_0", "cmd"])

    Root directory (all sequences, flat access)::

        ds = TartanKittiDataset(root_dir, keys=["velodyne_0"])
        len(ds)           # total events across all sequences
        ds.sequences      # list[TartanKittiDataset], one per sequence

    Lazy init -- inspect before loading::

        ds = TartanKittiDataset(root_or_seq_dir)
        ds.available                 # frozenset of available channels
        ds.keys = ["velodyne_0"]     # initialize loaders
        ds.keys = "all"              # or load everything

    Args:
        path: Single sequence directory **or** root directory.
        keys: Channels to load. ``None`` -> lazy (no loaders). ``"all"`` -> all
            channels present in ``.apairo``.
    """

    available_keys: frozenset = frozenset(load_profile(_PROFILE_PATH).keys())

    def __init__(
        self,
        path: str | Path,
        keys: Optional[List[str] | str] = None,
    ) -> None:
        path = Path(path)
        raw_profile = load_profile(_PROFILE_PATH)

        if _is_sequence_dir(path, raw_profile):
            self._is_root = False
            self._init_sequence(path, keys, raw_profile)
        else:
            self._init_root(path, keys, raw_profile)

    # ---------------------------------------------------------------- sequence

    def _init_sequence(self, sequence_dir: Path, keys, raw_profile: dict) -> None:
        config = self._load_or_create_config(sequence_dir)
        channels: dict = config.get("channels", {})

        if not channels:
            raise FileNotFoundError(
                f"No recognized channels found in '{sequence_dir}'. "
                f"Expected subdirectories matching the TartanDrive v2 profile "
                f"(e.g. velodyne_0, image_left, cmd). "
                f"Verify that the path points to a valid sequence directory."
            )

        self._sequence_dir = sequence_dir
        self._available_channels = channels
        self._effective_profile: dict[str, str] = {
            k: v["loader"] for k, v in channels.items()
        }
        self._timestamp_aliases: dict[str, str] = {
            k: v["timestamps_from"]
            for k, v in channels.items()
            if "timestamps_from" in v
        }

        if keys is None:
            # Lazy: store enough state for _init() to run later via keys setter.
            self._keys = []
            self._profile = raw_profile
            self._files = get_files(str(sequence_dir))
        else:
            if keys == "all":
                keys = sorted(channels.keys())
            unknown = set(keys) - set(channels)
            if unknown:
                raise KeyError(
                    f"Keys {unknown} are not declared in .apairo. "
                    f"Register preprocessed channels with "
                    f"{type(self).__name__}.register_channel()."
                )
            missing_dirs = [k for k in keys if not (sequence_dir / k).is_dir()]
            if missing_dirs:
                raise FileNotFoundError(
                    f"Channel directories missing on disk: {missing_dirs}"
                )
            super().__init__(
                directory=sequence_dir, keys=list(keys), dataset_profile=_PROFILE_PATH
            )

    # ---------------------------------------------------------------- root

    def _init_root(self, root_dir: Path, keys, raw_profile: dict) -> None:
        seq_dirs = sorted(
            d
            for d in root_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and _is_sequence_dir(d, raw_profile)
        )
        if not seq_dirs:
            raise FileNotFoundError(
                f"No TartanDrive sequences found in '{root_dir}'. "
                f"Expected subdirectories that are valid sequence directories."
            )
        super()._init_root(
            root_dir,
            seq_dirs,
            lambda d: TartanKittiDataset(d, keys=keys),
            build_index=keys is not None,
        )

    # ---------------------------------------------------------------- hooks

    def _single_available(self) -> frozenset:
        return frozenset(self._available_channels)

    def _set_single_keys(self, keys) -> None:
        if keys == "all":
            keys = sorted(self._available_channels.keys())
        unknown = set(keys) - set(self._available_channels)
        if unknown:
            raise KeyError(
                f"Keys {unknown} are not declared in .apairo. "
                f"Register preprocessed channels with "
                f"{type(self).__name__}.register_channel()."
            )
        missing_dirs = [k for k in keys if not (self._sequence_dir / k).is_dir()]
        if missing_dirs:
            raise FileNotFoundError(
                f"Channel directories missing on disk: {missing_dirs}"
            )
        self._set_keys(list(keys))
        self._init()

    # ---------------------------------------------------------------- preprocessing

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        return self._sequence_dir / key / f"{idx:06d}.{ext}"

    # ---------------------------------------------------------------- dunder

    def __len__(self) -> int:
        if not self._is_root and not self._keys:
            raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
        return super().__len__()

    def _load(self, idx):
        if not self._is_root and not isinstance(idx, tuple) and not self._keys:
            raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
        return super()._load(idx)

    # ---------------------------------------------------------------- bootstrap

    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        raw_profile = load_profile(_PROFILE_PATH)
        available = get_files(str(sequence_dir))
        channels: dict = {}
        for key in sorted(available):
            if key not in raw_profile:
                continue
            has_ts = (sequence_dir / key / "timestamps.txt").exists()
            channels[key] = {"loader": raw_profile[key], "has_timestamps": has_ts}
        return {"version": 1, "channels": channels}

    # ---------------------------------------------------------------- loaders

    def _init_loaders(self) -> None:
        self.loaders = {
            key: str_to_loader[self._effective_profile[key]](self._files[key])
            for key in self._keys
        }

        self.timestamps: dict[str, np.ndarray] = {}
        raw_fallback: list[str] = []

        for key in self._keys:
            ts_path = Path(self._files[key]) / "timestamps.txt"
            if ts_path.exists():
                self.timestamps[key] = np.loadtxt(ts_path)
            elif key in self._timestamp_aliases:
                # Backward compat: derived channel created before own-timestamps convention.
                src = self._timestamp_aliases[key]
                if src not in self.timestamps:
                    src_path = Path(self._files[src]) / "timestamps.txt"
                    if not src_path.exists():
                        raise ValueError(
                            f"'{key}' has no timestamps.txt and its alias source "
                            f"'{src}' has no timestamps.txt either. "
                            f"Re-run run_preprocess() to regenerate."
                        )
                    self.timestamps[src] = np.loadtxt(src_path)
                self.timestamps[key] = self.timestamps[src]
            else:
                raw_fallback.append(key)

        if raw_fallback:
            self.timestamps.update(loads_timestamps(raw_fallback, self._files))

        self.end_of_time: float = get_end_of_time(self.timestamps) + 1.0

    # ---------------------------------------------------------------- describe

    @classmethod
    def describe(cls, path: str | Path) -> dict:
        """Describe available channels -- auto-detects root vs sequence directory.

        Root directory: lists each sequence with its raw and preprocessed channels.
        Sequence directory: shows raw present/missing and preprocessed channels.
        """
        path = Path(path)
        raw_profile = load_profile(_PROFILE_PATH)

        if _is_sequence_dir(path, raw_profile):
            return super().describe(path)

        seq_dirs = sorted(
            d
            for d in path.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and _is_sequence_dir(d, raw_profile)
        )
        if not seq_dirs:
            print(f"No TartanDrive sequences found in '{path}'.")
            return {}

        n = len(seq_dirs)
        print(f"\n{cls.__name__} -- {path.name} ({n} sequence{'s' if n > 1 else ''})")
        print("─" * 50)

        result = {}
        for seq_dir in seq_dirs:
            if config_exists(seq_dir):
                config = read_config(seq_dir)
            else:
                instance = cls.__new__(cls)
                config = instance._bootstrap_config(seq_dir)

            channels = config.get("channels", {})
            raw = sorted(
                k for k, v in channels.items() if v.get("kind", "raw") == "raw"
            )
            preproc = {
                k: v for k, v in channels.items() if v.get("kind") == "preprocess"
            }

            raw_str = ", ".join(raw) if raw else "(none)"
            preproc_str = f" + {len(preproc)} preprocessed" if preproc else ""
            print(f"  {seq_dir.name:<20} {raw_str}{preproc_str}")

            result[seq_dir.name] = {"raw": raw, "preprocess": preproc}

        print()
        return result
