from pathlib import Path

import numpy as np

from apairo.core import AbstractLoader
from apairo.core.utils.exceptions import FileExtensionError


class NPYLoader(AbstractLoader):
    r"""Loader for a stacked ``.npy`` array (row 0 = frame 0) in a directory.

    By default the directory holds a single ``.npy`` file and it is loaded. When
    a directory colocates more than one stacked array -- the whole-array analogue
    of the per-frame ``_intensity`` idiom, e.g. ``gicp_poses/poses.npy`` beside
    ``gicp_poses/valid_mask.npy`` -- pass ``file`` to name the exact one. The
    filename lives in ``.apairo`` (the layout), never baked into the loader.
    """

    def __init__(self, directory: str | Path, *, file: str | None = None) -> None:
        directory = Path(directory)
        if file is not None:
            target = directory / file
            if not target.is_file():
                raise FileExtensionError(f"No such .npy file: {target}")
            self.array: np.ndarray = np.load(target)
            return
        npy_files = sorted(directory.glob("*.npy"))
        if not npy_files:
            raise FileExtensionError(f"No .npy file found in {directory}")
        self.array = np.load(npy_files[0])

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> np.ndarray:
        # Copy: self.array is a persistent whole-array cache, so a bare view would
        # let an in-place transform corrupt it and alias repeated reads. The
        # per-frame loaders read a fresh array each call -- match that contract.
        return self.array[idx].copy()

    @property
    def shape(self) -> tuple[int, ...]:
        if self.array.ndim == 1:
            return (1,)
        return tuple(self.array.shape[1:])
