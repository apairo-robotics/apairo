from pathlib import Path
from typing import Tuple
import numpy as np
import torch

from apairo.core import AbstractLoader
from apairo.core.utils.exceptions import FileExtensionError


class NPYLoader(AbstractLoader):
    r"""Loader for a directory containing a single `.npy` file."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        npy_files = sorted(directory.glob("*.npy"))
        if not npy_files:
            raise FileExtensionError(f"No .npy file found in {directory}")
        self.array: torch.Tensor = torch.from_numpy(np.load(npy_files[0]))

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.array[idx]

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.array.ndim == 1:
            return (1,)
        return tuple(self.array.shape[1:])
