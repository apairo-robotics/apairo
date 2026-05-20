from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch

from apairo.utils.timestamps import get_end_of_time
from apairo.loader import str_to_loader, loads_timestamps, load_profile
from apairo.utils.files import get_files
from apairo.core import AbstractDataset, AbstractLoader
from apairo.core.sample import Sample


class KittiDataset(AbstractDataset):
    r"""Dataset for KITTI-formatted robotics data (e.g. Tartan Drive).

    Args:
        directory: Path to the dataset root directory.
        keys: Modality names to load (must match subdirectory names).
        dataset_profile: YAML profile filename mapping keys to loader types.
    """
    synchronous: bool = False

    def __init__(
        self,
        directory: str | Path,
        keys: List[str],
        dataset_profile: str = "tartan_kitti.yaml",
    ) -> None:
        self._profile: Dict[str, str] = load_profile(dataset_profile)
        self._files: Dict[str, str] = get_files(str(directory))

        missing = set(keys) - set(self._files)
        if missing:
            raise KeyError(f"Keys not found in dataset directory: {missing}")

        self._keys: List[str] = []
        self._set_keys(keys)
        self._init()

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
        self.timestamps: Dict[str, np.ndarray] = loads_timestamps(self._keys, self._files)
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
