from pathlib import Path

import numpy as np

from apairo.core import AbstractLoader
from apairo.core.utils.exceptions import FileExtensionError


class NPYLoader(AbstractLoader):
    r"""Loader for a directory containing a single `.npy` file."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        npy_files = sorted(directory.glob("*.npy"))
        if not npy_files:
            raise FileExtensionError(f"No .npy file found in {directory}")
        self.array: np.ndarray = np.load(npy_files[0])

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.array[idx]

    @property
    def shape(self) -> tuple[int, ...]:
        if self.array.ndim == 1:
            return (1,)
        return tuple(self.array.shape[1:])
