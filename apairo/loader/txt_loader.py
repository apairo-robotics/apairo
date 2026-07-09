from __future__ import annotations

from pathlib import Path

import numpy as np

from apairo.core import AbstractLoader


class TXTLoader(AbstractLoader):
    """Loader for a sequence-level text file where each row is one frame.

    Accepts a list of paths (one per sequence) and concatenates them into a
    single array, so multi-sequence datasets are handled transparently.

    Args:
        paths: One ``.txt`` file per sequence, each row encoding one frame.
        reshape: Optional shape to apply to each row (e.g. ``[3, 4]`` for a
            3x4 pose matrix).
    """

    def __init__(self, paths: list[Path], reshape: list | None = None) -> None:
        arrays = [np.atleast_2d(np.loadtxt(p)) for p in sorted(paths)]
        self.array: np.ndarray = np.vstack(arrays) if len(arrays) > 1 else arrays[0]
        self._reshape = reshape
        self._shape: tuple[int, ...] = (
            tuple(reshape) if reshape else tuple(self.array.shape[1:])
        )

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> np.ndarray:
        row = self.array[idx]
        return row.reshape(self._reshape) if self._reshape else row

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
