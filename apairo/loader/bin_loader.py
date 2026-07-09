import os

import numpy as np

from apairo.core import AbstractLoader


class BINLoader(AbstractLoader):
    def __init__(self, directory: str):
        self.files = sorted([f for f in os.listdir(directory) if f.endswith(".bin")])
        self.directory = directory
        self._shape = (4,)  # (x, y, z, intensity)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> np.ndarray:
        path = os.path.join(self.directory, self.files[idx])
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    @property
    def shape(self):
        return self._shape
