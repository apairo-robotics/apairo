import os

import numpy as np

from apairo.core import AbstractLoader


class BINLoader(AbstractLoader):
    def __init__(self, directory: str, files: list[str] | None = None):
        # `files` = frame-ordered names resolved by the dataset (a key/order regex
        # enumeration); otherwise the legacy default -- all `.bin`, sorted.
        if files is not None:
            self.files = list(files)
        else:
            self.files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
        self.directory = directory
        self._shape = (4,)  # (x, y, z, intensity)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> np.ndarray:
        path = os.path.join(self.directory, self.files[idx])
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 4:
            raise ValueError(
                f"Corrupt/truncated point cloud '{path}': {arr.size} float32 value(s) "
                f"is not a multiple of 4 (x, y, z, intensity)."
            )
        return arr.reshape(-1, 4)

    @property
    def shape(self):
        return self._shape
