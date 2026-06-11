from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from apairo.core import AbstractLoader


class NPYSLoader(AbstractLoader):
    """Loader for an ordered list of per-frame ``.npy`` files in a directory.

    The loader only handles npy mechanics (memory layout, lazy per-frame
    loads); it has no naming convention of its own.  The *dataset* imposes
    one by passing ``files``, the file names ordered by frame index.

    When ``files`` is omitted, the legacy default applies — unsuffixed
    ``<index>.npy`` members sorted lexicographically.  Suffixed variants
    (``000000_intensity.npy``) are ignored by that default: a dataset
    exposing them as separate channels resolves one file list per channel
    (e.g. with :func:`apairo.utils.npy_analyser`) and builds one loader
    per channel.

    Args:
        directory (str) :
            The directory that contains the `npy` files.
        files (list[str], optional) :
            Frame-ordered file names resolved by the dataset.
    """

    def __init__(self, directory, files: Optional[List[str]] = None):
        self.directory = directory
        if files is not None:
            self.files = list(files)
        else:
            self.files = sorted(
                f
                for f in os.listdir(directory)
                if f.endswith(".npy") and "_" not in f
            )
        if not self.files:
            raise FileNotFoundError(f"No .npy frames found in {directory}")
        self._shape = np.load(os.path.join(self.directory, self.files[0])).shape

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> np.ndarray:
        return np.load(os.path.join(self.directory, self.files[idx]))
