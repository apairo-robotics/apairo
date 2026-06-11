from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np

from apairo.core.abstract_loader import AbstractLoader


class ZarrLoader(AbstractLoader):
    """Loader for a Zarr array directory.

    Opens the array at construction time (memory-mapped) and returns
    ``arr[idx]`` as a numpy array on each access.

    Args:
        zarr_path: Path to the Zarr array directory (e.g. ``trajectory.zarr/positions.zarr``).
    """

    def __init__(self, zarr_path: str | Path) -> None:
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "zarr is required for ZarrLoader. "
                "Install with: pip install zarr"
            )
        self._path = Path(zarr_path)
        self._array = zarr.open_array(str(self._path), mode="r")

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, idx: int) -> np.ndarray:
        data = self._array[int(idx)]
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        return data

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._array.ndim <= 1:
            return (1,)
        return tuple(self._array.shape[1:])

    @property
    def array(self):
        """The underlying Zarr array (lazy, sliceable without a full load)."""
        return self._array
