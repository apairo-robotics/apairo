from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from apairo.core.synchronous_dataset import SynchronousDataset
from apairo.core.sample import Sample

_AVAILABLE_KEYS = {"lidar", "labels"}


class Goose3DDataset(SynchronousDataset):
    r"""Goose3D synchronous dataset (LiDAR + semantic labels).

    Args:
        root_dir: Dataset root. LiDAR files are discovered via
            ``**/lidar/**/*.bin`` and labels via ``**/labels/**/*.label``.
        keys: Modalities to load — any subset of ``{"lidar", "labels"}``.
    """

    def __init__(self, root_dir: str | Path, keys: list[str] | None = None) -> None:
        if keys is None:
            keys = ["lidar", "labels"]
        root = Path(root_dir)
        invalid = set(keys) - _AVAILABLE_KEYS
        if invalid:
            raise KeyError(f"Unknown keys {invalid}. Available: {_AVAILABLE_KEYS}")
        self._set_keys(list(keys))
        self._files: dict[str, list[Path]] = {}
        if "lidar" in self._keys:
            self._files["lidar"] = sorted(root.glob("**/lidar/**/*.bin"))
        if "labels" in self._keys:
            self._files["labels"] = sorted(root.glob("**/labels/**/*.label"))
        lengths = {k: len(v) for k, v in self._files.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Mismatched file counts per key: {lengths}")

    def __len__(self) -> int:
        return len(next(iter(self._files.values())))

    def __getitem__(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        data: dict = {}
        for key in self._keys:
            path = self._files[key][idx]
            if key == "lidar":
                arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
                data[key] = torch.from_numpy(arr)
            elif key == "labels":
                arr = np.fromfile(path, dtype=np.int32)
                data[key] = torch.from_numpy(arr).long()
        return Sample(data=data)

    def __iter__(self):
        self._iter_pos = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_pos >= len(self):
            raise StopIteration
        sample = self[self._iter_pos]
        self._iter_pos += 1
        return sample
