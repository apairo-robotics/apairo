from typing import Tuple, Union
import numpy as np
import torch
import os

from ..core import AbstractLoader
from ..core.utils.exceptions import FileExtensionError


class NPYLoader(AbstractLoader):
    r"""A :class:`Loader` for directory that contains one `npy` file.

    Args:
        directory (str) :
            The directory that contains the `npy` file.
    """
    array: Union[np.ndarray, torch.Tensor] = np.zeros(0)

    def __init__(self, directory: str) -> None:
        for f in os.listdir(directory):
            if f.split(".")[-1] == "npy":
                self.array = np.load(os.path.join(directory, f))
                break
        if self.array is None:
            raise FileExtensionError(f"There is no numpy file in {directory}")
        self.array = torch.from_numpy(self.array)

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.array[idx]

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.array.ndim == 1:
            return (1,)
        return self.array.shape[1:]
