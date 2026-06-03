"""MNTDataset -- apairo wrapper for the AMIAD MNT pipeline output (Zarr format).

Each *mission directory* produced by the MNT pipeline is treated as one
apairo *sequence*.  A dataset root (parent of several missions) is also
accepted; in that case all missions are concatenated into a flat index.

The MNT pipeline is the AMIAD format that normalises heterogeneous source
datasets (Pendragon ROS1 bags, Isaac Sim HDF5, STONE/nuScenes, TartanGround,
TMOD) into a single zarr layout.

Expected mission layout::

    <mission>/
    ├── images.tar                            # key: "image"       (H,W,3) uint8
    ├── points.zarr/                          # key: "points"      (P,5) float32  [x,y,z,ground,trav]
    ├── trajectory.zarr/
    │   ├── positions.zarr/                   # key: "position"    (2,) float32
    │   ├── yaws.zarr/                        # key: "yaw"         scalar float32
    │   ├── timestamps.zarr/                  # key: "timestamp"   scalar float64
    │   ├── positions_past.zarr/              # key: "position_past"  (N_past,2) float32
    │   ├── yaws_past.zarr/                   # key: "yaw_past"       (N_past,) float32
    │   └── timestamps_past.zarr/             # key: "timestamp_past" (N_past,) float64
    ├── traj_time.zarr/
    │   ├── positions.zarr/                   # key: "waypoints_time"       (W,2) float32
    │   └── yaws.zarr/                        # key: "yaw_waypoints_time"   (W,) float32
    ├── traj_dist.zarr/
    │   ├── positions.zarr/                   # key: "waypoints_dist"       (W,2) float32
    │   └── yaws.zarr/                        # key: "yaw_waypoints_dist"   (W,) float32
    └── metadata.yaml

Preprocessed channels (produced by :class:`~apairo.core.preprocessor.FramePreprocessor`)
are stored as per-frame ``.npy`` files under ``<mission>/<key>/`` and
registered in ``<mission>/.apairo``.

Usage::

    # Single mission
    ds = MNTDataset("/data/my_dataset/mission_001", keys=["image", "points"])
    sample = ds[0]
    # sample.data["image"]  -> np.ndarray (H, W, 3) uint8
    # sample.data["points"] -> np.ndarray (N, 3) float32

    # Full dataset (all missions)
    ds = MNTDataset("/data/my_dataset", keys=["position", "yaw"])
    len(ds)              # total frames across missions
    ds.mission_ids       # list of mission directory names
    seq = ds.sequence("mission_001")   # SequenceView for one mission

    # Preprocessing
    class MyPreprocessor(FramePreprocessor):
        output_key    = "trav_label"
        output_loader = "npys"
        input_keys    = ["points"]
        timestamps_from = "points"
        def process(self, sample): ...

    MNTDataset.run_preprocess(MyPreprocessor(), "/data/my_dataset/mission_001")
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from apairo.core.sequence_view import SequenceView

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.configurable_dataset import ConfigurableDataset
from apairo.core.sample import Sample

# Maps channel key -> sub-path components relative to mission dir.
# Empty list signals a special-case loader (e.g. TarImageLoader for "image").
RAW_CHANNEL_PATHS: dict[str, list[str]] = {
    # ── images / lidar ───────────────────────────────────────────────────────
    "image":                 [],  # images.tar -- handled specially
    "points":                ["points.zarr"],
    # ── current odometry ─────────────────────────────────────────────────────
    "position":              ["trajectory.zarr", "positions.zarr"],
    "yaw":                   ["trajectory.zarr", "yaws.zarr"],
    "timestamp":             ["trajectory.zarr", "timestamps.zarr"],
    # ── past odometry window ─────────────────────────────────────────────────
    "position_past":         ["trajectory.zarr", "positions_past.zarr"],
    "yaw_past":              ["trajectory.zarr", "yaws_past.zarr"],
    "timestamp_past":        ["trajectory.zarr", "timestamps_past.zarr"],
    # ── future trajectory (temporal sampling) ────────────────────────────────
    "waypoints_time":        ["traj_time.zarr", "positions.zarr"],
    "yaw_waypoints_time":    ["traj_time.zarr", "yaws.zarr"],
    # ── future trajectory (spatial sampling) ─────────────────────────────────
    "waypoints_dist":        ["traj_dist.zarr", "positions.zarr"],
    "yaw_waypoints_dist":    ["traj_dist.zarr", "yaws.zarr"],
}

_DEFAULT_KEYS = list(RAW_CHANNEL_PATHS.keys())


def _scan_mission_channels(mission_dir: Path) -> dict:
    """Return a channels config dict for all raw channels present in *mission_dir*."""
    channels: dict = {}
    for key in sorted(RAW_CHANNEL_PATHS.keys()):
        if _channel_exists(mission_dir, key):
            loader_tag = "img" if key == "image" else "zarr"
            channels[key] = {"has_timestamps": False, "kind": "raw", "loader": loader_tag}
    return channels


def _is_mission_dir(path: Path) -> bool:
    """True if *path* looks like a single MNT mission directory."""
    return (
        (path / "trajectory.zarr").is_dir()
        or (path / "images.tar").is_file()
        or (path / "points.zarr").is_dir()
    )


def _channel_exists(mission_dir: Path, key: str) -> bool:
    if key == "image":
        return (mission_dir / "images.tar").is_file()
    parts = RAW_CHANNEL_PATHS.get(key, [])
    if not parts:
        return False
    return (mission_dir / Path(*parts)).is_dir()


def _detect_n_frames(mission_dir: Path) -> int:
    """Return frame count from the first available zarr channel."""
    from apairo.loader.zarr_loader import ZarrLoader

    for key in ("position", "yaw", "timestamp", "points", "waypoints_time", "waypoints_dist"):
        parts = RAW_CHANNEL_PATHS.get(key, [])
        if not parts:
            continue
        zarr_path = mission_dir / Path(*parts)
        if zarr_path.is_dir():
            try:
                return len(ZarrLoader(zarr_path))
            except Exception:
                continue
    raise RuntimeError(
        f"Could not detect number of frames in '{mission_dir}'. "
        "Expected at least one of: trajectory.zarr/positions.zarr, points.zarr, …"
    )


def _build_raw_loader(mission_dir: Path, key: str, n_frames: int):
    """Return a loader for a raw channel, or ``None`` if the channel is absent."""
    if key == "image":
        tar_path = mission_dir / "images.tar"
        if not tar_path.is_file():
            return None
        from apairo.loader.tar_loader import TarImageLoader
        return TarImageLoader(tar_path, n_frames)

    parts = RAW_CHANNEL_PATHS[key]
    zarr_path = mission_dir / Path(*parts)
    if not zarr_path.is_dir():
        return None
    from apairo.loader.zarr_loader import ZarrLoader
    return ZarrLoader(zarr_path)


def _build_derived_loader(mission_dir: Path, key: str):
    """Return an npy-per-frame loader for a preprocessed channel."""
    derived_dir = mission_dir / key
    if not derived_dir.is_dir():
        raise FileNotFoundError(
            f"Derived key '{key}': directory '{derived_dir}' not found. "
            f"Run MNTDataset.run_preprocess(...) first."
        )
    files = sorted(derived_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(
            f"Derived key '{key}': no .npy files in '{derived_dir}'."
        )

    class _NpyPerFrameLoader:
        def __init__(self, paths: list[Path]) -> None:
            self._paths = paths

        def __len__(self) -> int:
            return len(self._paths)

        def __getitem__(self, idx: int) -> np.ndarray:
            return np.load(self._paths[idx])

    return _NpyPerFrameLoader(files)


class MNTDataset(SynchronousDataset, ConfigurableDataset):
    """Apairo dataset for MNT pipeline output (Zarr + tar format).

    Accepts either a single mission directory or a dataset root that contains
    multiple mission directories -- the level is auto-detected.

    Attributes:
        available_keys: Raw channels the dataset can provide.
    """

    available_keys = frozenset(RAW_CHANNEL_PATHS.keys())

    def __init__(
        self,
        root: str | Path,
        keys: Optional[List[str]] = None,
    ) -> None:
        root = Path(root)

        if _is_mission_dir(root):
            self._is_root = False
            self._init_mission(root, keys)
        else:
            self._is_root = True
            self._init_root(root, keys)

    # ---------------------------------------------------------------- init

    def _init_mission(self, mission_dir: Path, keys: Optional[List[str]]) -> None:
        self._mission_dir = mission_dir
        self._root = mission_dir
        self._n_frames = _detect_n_frames(mission_dir)

        # Default: all raw channels present on disk
        if keys is None:
            keys = [k for k in _DEFAULT_KEYS if _channel_exists(mission_dir, k)]

        self._loaders: dict = {}

        for key in keys:
            if key in RAW_CHANNEL_PATHS:
                loader = _build_raw_loader(mission_dir, key, self._n_frames)
            else:
                loader = _build_derived_loader(mission_dir, key)
            if loader is not None:
                self._loaders[key] = loader

        # Drop keys for which no loader could be built
        self._set_keys([k for k in keys if k in self._loaders])

        # Build per-sequence group (single mission = one group)
        self._seq_groups: dict[str, list[int]] = {
            mission_dir.name: list(range(self._n_frames))
        }

    def _init_root(self, root_dir: Path, keys: Optional[List[str]]) -> None:
        self._root_dir = root_dir
        self._root = root_dir
        mission_dirs = sorted(
            d
            for d in root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and _is_mission_dir(d)
        )
        if not mission_dirs:
            raise FileNotFoundError(
                f"No MNT mission directories found in '{root_dir}'. "
                "Expected subdirectories with trajectory.zarr/, points.zarr/, or images.tar."
            )
        self._missions: list[MNTDataset] = [
            MNTDataset(d, keys=keys) for d in mission_dirs
        ]
        if keys is not None:
            self._build_flat_index()

    def _build_flat_index(self) -> None:
        lengths = [len(m) for m in self._missions]
        self._cumulative_lengths = np.array([0, *np.cumsum(lengths)], dtype=np.intp)

    # ---------------------------------------------------------------- ConfigurableDataset

    @classmethod
    def init(
        cls,
        mission_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Scan a mission directory and write ``.apairo/channels.yaml``.

        Detects which raw channels (zarr arrays, ``images.tar``) are present
        and registers them.  Call this once before using
        :class:`MNTDataset` with ``.apairo``-based introspection or
        :meth:`~apairo.core.configurable_dataset.ConfigurableDataset.verify`.

        Args:
            mission_dir: Path to a single MNT mission directory.
            overwrite: Replace an existing ``.apairo`` if present.

        Raises:
            FileExistsError: If ``.apairo`` already exists and
                ``overwrite=False``.
            ValueError: If no known channels are detected.
        """
        from apairo.core.config import config_exists, write_config

        mission_dir = Path(mission_dir)
        if config_exists(mission_dir) and not overwrite:
            raise FileExistsError(
                f".apairo already exists in '{mission_dir}'. "
                f"Pass overwrite=True to reinitialize."
            )

        channels = _scan_mission_channels(mission_dir)
        if not channels:
            raise ValueError(
                f"No known MNT channels found in '{mission_dir}'. "
                f"Expected trajectory.zarr/, points.zarr/, or images.tar."
            )
        write_config(mission_dir, {"version": 1, "channels": channels})

    def _bootstrap_config(self, root_dir: Path) -> dict:
        """Auto-discover raw channels in *root_dir* and write the initial .apairo."""
        return {"version": 1, "channels": _scan_mission_channels(root_dir)}

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        """Return the path where a preprocessed frame should be saved.

        Layout: ``<mission_dir>/<key>/<idx:06d>.<ext>``
        """
        return self._mission_dir / key / f"{idx:06d}.{ext}"

    # ---------------------------------------------------------------- SynchronousDataset

    def __len__(self) -> int:
        if self._is_root:
            if not hasattr(self, "_cumulative_lengths"):
                raise RuntimeError("No keys loaded. Pass keys= to __init__.")
            return int(self._cumulative_lengths[-1])
        return self._n_frames

    def __getitem__(self, idx) -> Sample:
        if isinstance(idx, tuple):
            seq_id, local_idx = idx
            return self.sequence(seq_id)[local_idx]

        if self._is_root:
            if not hasattr(self, "_cumulative_lengths"):
                raise RuntimeError("No keys loaded. Pass keys= to __init__.")
            if not 0 <= idx < len(self):
                raise IndexError(f"Index {idx} out of range [0, {len(self)})")
            seq_idx = int(
                np.searchsorted(self._cumulative_lengths[1:], idx, side="right")
            )
            local_idx = idx - int(self._cumulative_lengths[seq_idx])
            return self._missions[seq_idx][local_idx]

        if not 0 <= idx < self._n_frames:
            raise IndexError(f"Index {idx} out of range [0, {self._n_frames})")
        return Sample(data={key: self._loaders[key][idx] for key in self._keys})

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample

    # ---------------------------------------------------------------- sequences / missions

    @property
    def mission_ids(self) -> list[str]:
        """Names of all mission directories, in discovery order."""
        if self._is_root:
            return [m._mission_dir.name for m in self._missions]
        return [self._mission_dir.name]

    # Alias for consistency with other apairo datasets
    @property
    def sequence_ids(self) -> list[str]:
        return self.mission_ids

    def sequences(self) -> list[SequenceView]:
        """Return one :class:`~apairo.core.sequence_view.SequenceView` per mission."""
        from apairo.core.sequence_view import SequenceView

        if self._is_root:
            return [
                SequenceView(m, list(range(len(m))), m._mission_dir.name)
                for m in self._missions
            ]
        return [SequenceView(self, list(range(len(self))), self._mission_dir.name)]

    def sequence(self, seq_id: str) -> SequenceView:
        """Return a :class:`~apairo.core.sequence_view.SequenceView` for *seq_id*."""
        from apairo.core.sequence_view import SequenceView

        if self._is_root:
            for m in self._missions:
                if m._mission_dir.name == seq_id:
                    return SequenceView(m, list(range(len(m))), seq_id)
            raise KeyError(
                f"Mission '{seq_id}' not found. Available: {self.mission_ids}"
            )
        if seq_id != self._mission_dir.name:
            raise KeyError(
                f"This is a single-mission dataset ('{self._mission_dir.name}'). "
                f"Use MNTDataset(root_dir) to access multiple missions."
            )
        return SequenceView(self, list(range(len(self))), seq_id)

    # ---------------------------------------------------------------- keys

    @property
    def keys(self) -> list[str]:
        if self._is_root:
            return self._missions[0].keys if self._missions else []
        return self._keys

    @keys.setter
    def keys(self, keys: list[str]) -> None:
        if self._is_root:
            for m in self._missions:
                m.keys = list(keys)
            self._build_flat_index()
        else:
            self._init_mission(self._mission_dir, list(keys))

    # ---------------------------------------------------------------- describe

    @classmethod
    def describe(cls, path: str | Path) -> dict:
        """Print available raw and preprocessed channels.

        Auto-detects whether *path* is a mission directory or a dataset root.
        """
        from apairo.core.config import config_exists, read_config

        path = Path(path)

        if _is_mission_dir(path):
            raw_present = sorted(k for k in cls.available_keys if _channel_exists(path, k))
            raw_missing = sorted(k for k in cls.available_keys if not _channel_exists(path, k))
            preprocess = {}
            if config_exists(path):
                config = read_config(path)
                preprocess = {
                    k: v
                    for k, v in config.get("channels", {}).items()
                    if v.get("kind") == "preprocess"
                }
            print(f"\n{cls.__name__} -- {path.name}")
            print("─" * 50)
            print("Raw channels")
            if raw_present:
                print("  present  :", ", ".join(raw_present))
            if raw_missing:
                print("  missing  :", ", ".join(raw_missing))
            print("Preprocessed channels")
            if preprocess:
                for key, meta in sorted(preprocess.items()):
                    ts_info = (
                        f"<- timestamps from {meta['timestamps_from']}"
                        if "timestamps_from" in meta
                        else "<- own timestamps"
                    )
                    print(f"  {key:<20} {meta['loader']:<6} {ts_info}")
            else:
                print("  (none)")
            print()
            return {"raw": {"present": raw_present, "missing": raw_missing}, "preprocess": preprocess}

        mission_dirs = sorted(
            d for d in path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and _is_mission_dir(d)
        )
        n = len(mission_dirs)
        print(f"\n{cls.__name__} -- {path.name} ({n} mission{'s' if n != 1 else ''})")
        print("─" * 50)
        result = {}
        for d in mission_dirs:
            raw = sorted(k for k in cls.available_keys if _channel_exists(d, k))
            preproc = {}
            if config_exists(d):
                cfg = read_config(d)
                preproc = {k: v for k, v in cfg.get("channels", {}).items() if v.get("kind") == "preprocess"}
            suffix = f" + {len(preproc)} preprocessed" if preproc else ""
            print(f"  {d.name:<30} {', '.join(raw) if raw else '(none)'}{suffix}")
            result[d.name] = {"raw": raw, "preprocess": preproc}
        print()
        return result
