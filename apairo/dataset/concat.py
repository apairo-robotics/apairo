from __future__ import annotations

import functools

import numpy as np

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

    def __init__(self, datasets: list[AbstractDataset]) -> None:
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

    @property
    def keys(self) -> list[str]:
        return self._keys

    @keys.setter
    def keys(self, keys) -> None:
        self._set_keys(list(keys))
        self.__dict__.pop("timestamps", None)

    @functools.cached_property
    def timestamps(self) -> dict[str, np.ndarray] | None:  # type: ignore[override]
        """None for synchronous datasets, concatenated arrays for temporal ones."""
        if self.datasets[0].timestamps is None:
            return None
        result: dict[str, list[np.ndarray]] = {k: [] for k in self._keys}
        for ds in self.datasets:
            assert ds.timestamps is not None  # parents share sync-ness
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

    def _load(self, idx: int) -> Sample:
        ds_idx, offset = self._dataset_idx_and_offset(idx)
        sample = self.datasets[ds_idx][idx - offset]
        return Sample(
            data={k: sample.data[k] for k in self._keys if k in sample.data},
            timestamp=sample.timestamp,
        )
