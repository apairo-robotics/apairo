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
    │   ├── yaws.zarr/                        # key: "yaw_waypoints_dist"   (W,) float32
    │   ├── positions_past.zarr/              # key: "waypoints_dist_past"  (W_past,2) float32
    │   └── yaws_past.zarr/                   # key: "yaw_waypoints_dist_past" (W_past,) float32
    ├── width_curve_traj_static.zarr/         # key: "width_curve_traj_static" (M,2) float32
    │   ...                                   # (6 variants: {traj,traj_time,traj_dist} × {static,dynamic})
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
from apairo.dataset.mnt.layout import MNT_LAYOUT

# Back-compat view of the layout's channel table (legacy grouped layout).
RAW_CHANNEL_PATHS: dict[str, list[str]] = {
    key: list(spec.path) for key, spec in MNT_LAYOUT.channels.items()
}

_DEFAULT_KEYS = list(RAW_CHANNEL_PATHS.keys())


def channel_path(key: str) -> list[str]:
    """Sub-path components of a channel relative to the mission directory."""
    return list(MNT_LAYOUT.spec(key).path)


def _scan_mission_channels(mission_dir: Path) -> dict:
    """Return a channels config dict for all raw channels present in *mission_dir*."""
    return {
        key: {
            "has_timestamps": False,
            "kind": "raw",
            "loader": "img" if key == "image" else "zarr",
        }
        for key in MNT_LAYOUT.scan(mission_dir)
    }


def _is_mission_dir(path: Path) -> bool:
    """True if *path* looks like a single MNT mission directory."""
    return (
        (path / "trajectory.zarr").is_dir()
        or (path / "images.tar").is_file()
        or (path / "points.zarr").is_dir()
    )


def _channel_exists(mission_dir: Path, key: str) -> bool:
    return MNT_LAYOUT.exists(mission_dir, key)


def _detect_n_frames(mission_dir: Path) -> int:
    """Return frame count from the first available zarr channel."""
    for key in ("position", "yaw", "timestamp", "points", "waypoints_time", "waypoints_dist"):
        loader = MNT_LAYOUT.loader(mission_dir, key, 0)
        if loader is not None:
            try:
                return len(loader)
            except Exception:
                continue
    raise RuntimeError(
        f"Could not detect number of frames in '{mission_dir}'. "
        "Expected at least one of: trajectory.zarr/positions.zarr, points.zarr, …"
    )


def _build_raw_loader(mission_dir: Path, key: str, n_frames: int):
    """Return a loader for a raw channel, or ``None`` if the channel is absent."""
    return MNT_LAYOUT.loader(mission_dir, key, n_frames)


def _build_derived_loader(mission_dir: Path, key: str):
    """Return a loader for a preprocessed channel.

    Two layouts, matching the two preprocessor kinds:

    * per-frame files (``000000.npy``, ...) -- FramePreprocessor output;
    * one stacked ``<key>.npy`` of shape ``(N, ...)`` -- SequencePreprocessor
      output, indexed by row (memory-mapped).
    """
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

    stacked = derived_dir / f"{key}.npy"
    if len(files) == 1 and files[0] == stacked:
        class _NpyRowsLoader:
            def __init__(self, path: Path) -> None:
                self._data = np.load(path, mmap_mode="r")

            def __len__(self) -> int:
                return len(self._data)

            def __getitem__(self, idx: int) -> np.ndarray:
                return np.asarray(self._data[idx])

            @property
            def array(self) -> np.ndarray:
                return self._data

        return _NpyRowsLoader(stacked)

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

        # Default: all raw channels present on disk (table + flat convention)
        if keys is None:
            keys = MNT_LAYOUT.scan(mission_dir)

        self._loaders: dict = {}

        for key in keys:
            loader = _build_raw_loader(mission_dir, key, self._n_frames)
            if loader is None and key not in RAW_CHANNEL_PATHS:
                # Not on disk as a raw channel -> preprocessed channel
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

    @property
    def loaders(self) -> dict:
        """Per-channel loaders, indexed by global frame index (mission-level only)."""
        if self._is_root:
            return {}
        return self._loaders

    def channel_array(self, key: str) -> np.ndarray:
        """Full ``(N, ...)`` array for one channel of a single mission.

        Zero-copy for array-backed channels (zarr, stacked npy); per-frame
        channels (tar images, per-frame npy) are stacked on demand.

        Raises:
            RuntimeError: On a multi-mission (root-level) dataset.
            KeyError: If *key* is not among the loaded channels.
        """
        if self._is_root:
            raise RuntimeError(
                "channel_array() is mission-level. Use .sequence(...) / "
                "MNTDataset(mission_dir) for a single mission."
            )
        if key not in self._loaders:
            loader = _build_raw_loader(self._mission_dir, key, self._n_frames)
            if loader is None and key not in RAW_CHANNEL_PATHS:
                loader = _build_derived_loader(self._mission_dir, key)
            if loader is None:
                raise KeyError(
                    f"Channel '{key}' not found in '{self._mission_dir}'."
                )
            self._loaders[key] = loader
        loader = self._loaders[key]
        if hasattr(loader, "array"):
            return loader.array
        return np.stack([loader[i] for i in range(len(loader))])

    # ---------------------------------------------------------------- SynchronousDataset

    def __len__(self) -> int:
        if self._is_root:
            if not hasattr(self, "_cumulative_lengths"):
                raise RuntimeError("No keys loaded. Pass keys= to __init__.")
            return int(self._cumulative_lengths[-1])
        return self._n_frames

    def _load(self, idx) -> Sample:
        if isinstance(idx, tuple):
            seq_id, local_idx = idx
            view = self.sequence(seq_id)
            return self._load(view._indices[local_idx])

        if self._is_root:
            if not hasattr(self, "_cumulative_lengths"):
                raise RuntimeError("No keys loaded. Pass keys= to __init__.")
            if not 0 <= idx < len(self):
                raise IndexError(f"Index {idx} out of range [0, {len(self)})")
            seq_idx = int(
                np.searchsorted(self._cumulative_lengths[1:], idx, side="right")
            )
            local_idx = idx - int(self._cumulative_lengths[seq_idx])
            return self._missions[seq_idx]._load(local_idx)

        if not 0 <= idx < self._n_frames:
            raise IndexError(f"Index {idx} out of range [0, {self._n_frames})")
        return Sample(data={key: self._loaders[key][idx] for key in self._keys})


    # ---------------------------------------------------------------- sequences / missions

    @property
    def is_root(self) -> bool:
        """True when this dataset spans several mission directories."""
        return self._is_root

    @property
    def mission_dirs(self) -> list[Path]:
        """Paths of all mission directories, in discovery order."""
        if self._is_root:
            return [m._mission_dir for m in self._missions]
        return [self._mission_dir]

    @property
    def mission_ids(self) -> list[str]:
        """Names of all mission directories, in discovery order."""
        return [d.name for d in self.mission_dirs]

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
