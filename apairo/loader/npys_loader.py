import os
import numpy as np
import torch

from src.utils import npy_analyser
from ..core import AbstractLoader


class NPYSLoader(AbstractLoader):
    """A :class:`Loader` for `npy` files in a directory.

    In a directory, it is possible to have files of different nature.

    Args:
        directory (str) :
            The directory that contains the `npy` files.

    Example:
        livox
        |--- 000000.npy
        |--- 000000_intensity.npy
        |--- ...
    """

    def __init__(self, directory):
        self.directory = directory
        self.npy_formats = npy_analyser(directory)
        self.files = {npy_format: [] for npy_format in self.npy_formats}
        self.key = ""
        for file in filter(lambda f: f[-3:] == "npy", os.listdir(directory)):
            if "_" in file:
                file_ext = file.split("_")[-1].split(".")[0]
                if file_ext not in self.npy_formats: # Fixed boolean check
                    raise ValueError(f"New format {file_ext} found in {directory}")
            else:
                file_ext = ""
            self.files[file_ext].append(file)

        for key in self.files.keys():
            self.files[key].sort()

        self._shape = np.load(os.path.join(self.directory, self.files[''][0])).shape

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return len(self.files[''])

    def set_format(self, key):
        self.key = key

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.from_numpy(np.load(os.path.join(self.directory, self.files[self.key][idx])))
