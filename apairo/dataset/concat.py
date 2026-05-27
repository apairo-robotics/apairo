from __future__ import annotations
from typing import List
import numpy as np
from torch.utils.data import Dataset

from apairo.core import AbstractDataset
from apairo.core.sample import Sample


class ConcatDataset(AbstractDataset):
    """Concatenates multiple dataset instances into one.

    Takes the intersection of keys across all datasets so every index returns
    the same set of modalities regardless of which underlying dataset is hit.
    Indexing is O(log n) via binary search over cumulative lengths.

    Args:
        datasets: Non-empty list of dataset instances to concatenate.

    Example::

        sequences = [
            SemanticKittiDataset(f"/data/kitti/seq_{i:02d}", keys=["lidar", "labels"])
            for i in range(11)
        ]
        combined = ConcatDataset(sequences)
        sample = combined[0]

    Raises:
        ValueError: If ``datasets`` is empty.
    """

    def __init__(self, datasets: List[AbstractDataset]) -> None:
        if not datasets:
            raise ValueError("datasets must be non-empty")
        self.datasets = datasets
        self._resolve_keys()
        self._lengths = np.array([len(ds) for ds in self.datasets], dtype=np.intp)
        self._cumulative = np.cumsum(self._lengths)

    def _resolve_keys(self) -> None:
        keys = set(self.datasets[0].keys)
        for ds in self.datasets[1:]:
            keys &= set(ds.keys)
        self._keys = sorted(keys)
        for ds in self.datasets:
            ds.keys = self._keys

    @property
    def keys(self) -> List[str]:
        return self._keys

    @keys.setter
    def keys(self, keys) -> None:
        self._set_keys(list(keys))
        for ds in self.datasets:
            ds.keys = self._keys

    @property
    def timestamps(self) -> dict[str, np.ndarray] | None:
        """None for synchronous datasets, concatenated arrays for temporal ones."""
        if self.datasets[0].timestamps is None:
            return None
        result: dict[str, list[np.ndarray]] = {k: [] for k in self._keys}
        for ds in self.datasets:
            for k in self._keys:
                result[k].append(ds.timestamps[k])
        return {k: np.concatenate(v) for k, v in result.items()}

    @property
    def is_synchronous(self) -> bool:
        return self.datasets[0].timestamps is None

    def _dataset_idx_and_offset(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._cumulative[-1]:
            raise IndexError(f"Index {idx} out of range [0, {self._cumulative[-1]})")
        ds_idx = int(np.searchsorted(self._cumulative, idx, side="right"))
        offset = int(self._cumulative[ds_idx - 1]) if ds_idx > 0 else 0
        return ds_idx, offset

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int) -> Sample:
        ds_idx, offset = self._dataset_idx_and_offset(idx)
        return self.datasets[ds_idx][idx - offset]

    def __iter__(self):
        self._iter_ds = 0
        self._iter_inner = iter(self.datasets[0])
        return self

    def __next__(self) -> Sample:
        while True:
            try:
                return next(self._iter_inner)
            except StopIteration:
                self._iter_ds += 1
                if self._iter_ds >= len(self.datasets):
                    raise
                self._iter_inner = iter(self.datasets[self._iter_ds])


class TorchConcatDataset(ConcatDataset, Dataset):
    r"""Map-style PyTorch Dataset wrapping ConcatDataset."""
