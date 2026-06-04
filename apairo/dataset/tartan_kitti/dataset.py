from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from apairo.core.sequence_view import SequenceView

from apairo.loader import str_to_loader, loads_timestamps, load_profile
from apairo.utils.files import get_files
from apairo.utils.timestamps import get_end_of_time
from apairo.dataset.kitti import KittiDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.config import read_config, config_exists

_PROFILE_PATH = Path(__file__).parent / "profile.yaml"


def _is_sequence_dir(path: Path, raw_profile: dict) -> bool:
    """True if *path* looks like a single TartanDrive sequence directory."""
    return config_exists(path) or any((path / k).is_dir() for k in raw_profile)


class TartanKittiDataset(KittiDataset, ConfigurableDataset):
    r"""TartanDrive v2 dataset (KITTI layout).

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
            self._is_root = True
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
        self._root_dir = root_dir
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
        self._sequences: list[TartanKittiDataset] = [
            TartanKittiDataset(d, keys=keys) for d in seq_dirs
        ]
        if keys is not None:
            self._build_flat_index()

    def _build_flat_index(self) -> None:
        lengths = [len(s) for s in self._sequences]
        self._cumulative_lengths = np.array([0, *np.cumsum(lengths)], dtype=np.intp)

    # ---------------------------------------------------------------- public API

    @property
    def available(self) -> frozenset:
        """Channels available -- intersection across all sequences for root datasets."""
        if self._is_root:
            if not self._sequences:
                return frozenset()
            common = frozenset(self._sequences[0]._available_channels)
            for seq in self._sequences[1:]:
                common &= frozenset(seq._available_channels)
            return common
        return frozenset(self._available_channels)

    @property
    def sequences(self) -> list[TartanKittiDataset]:
        """Per-sequence datasets (root-level datasets only)."""
        if not self._is_root:
            raise AttributeError(
                "'sequences' is only available on root-level datasets."
            )
        return self._sequences

    @property
    def sequence_ids(self) -> list[str]:
        """Sequence directory names, in discovery order."""
        if not self._is_root:
            raise AttributeError(
                "'sequence_ids' is only available on root-level datasets."
            )
        return [seq._sequence_dir.name for seq in self._sequences]

    def sequence(self, seq_id: str) -> "SequenceView":
        """Return a :class:`~apairo.core.sequence_view.SequenceView` for *seq_id*."""
        if not self._is_root:
            raise AttributeError(
                "'sequence()' is only available on root-level datasets."
            )
        from apairo.core.sequence_view import SequenceView

        for seq in self._sequences:
            if seq._sequence_dir.name == seq_id:
                return SequenceView(seq, range(len(seq)), seq_id)
        raise KeyError(f"Sequence '{seq_id}' not found. Available: {self.sequence_ids}")

    # ---------------------------------------------------------------- preprocessing

    @property
    def root_dir(self) -> Path:
        return self._root_dir if self._is_root else self._sequence_dir

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        return self._sequence_dir / key / f"{idx:06d}.{ext}"

    # ---------------------------------------------------------------- keys

    @property
    def keys(self) -> List[str]:
        if self._is_root:
            return self._sequences[0].keys if self._sequences else []
        return self._keys

    @keys.setter
    def keys(self, keys) -> None:
        if self._is_root:
            if keys == "all":
                keys = sorted(self.available)
            for seq in self._sequences:
                seq.keys = list(keys)
            self._build_flat_index()
        else:
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

    # ---------------------------------------------------------------- dunder

    def __len__(self) -> int:
        if self._is_root:
            if not hasattr(self, "_cumulative_lengths"):
                raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
            return int(self._cumulative_lengths[-1])
        if not self._keys:
            raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
        return super().__len__()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            seq_id, local_idx = idx
            return self.sequence(seq_id)[local_idx]
        if self._is_root:
            if not hasattr(self, "_cumulative_lengths"):
                raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
            if not 0 <= idx < len(self):
                raise IndexError(f"Index {idx} out of range [0, {len(self)})")
            seq_idx = int(
                np.searchsorted(self._cumulative_lengths[1:], idx, side="right")
            )
            local_idx = idx - int(self._cumulative_lengths[seq_idx])
            return self._sequences[seq_idx][local_idx]
        if not self._keys:
            raise RuntimeError("No keys loaded. Set ds.keys = [...] first.")
        return super().__getitem__(idx)

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self):
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample

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
