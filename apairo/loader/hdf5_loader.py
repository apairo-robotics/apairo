"""Loader for one array of an HDF5 container file: frame ``i`` is row ``i``.

``h5py`` is imported when a channel is first read (``pip install apairo[hdf5]``),
and a file handle is opened lazily, once per process: an ``h5py`` handle does
not survive a ``fork``, so each ``DataLoader`` worker opens its own.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from apairo.core.abstract_loader import AbstractLoader

EXTENSIONS = frozenset({".h5", ".hdf5", ".he5", ".hdf"})


def locate(directory: Path, array_file: str | None) -> Path:
    """The container file of a channel: ``array_file`` (a name or a glob that
    matches one file), else the only HDF5 file in ``directory``."""
    directory = Path(directory)
    if array_file:
        matches = sorted(directory.glob(array_file))
        if len(matches) != 1:
            found = ", ".join(p.name for p in matches) or "none"
            raise FileNotFoundError(
                f"array_file '{array_file}' must match one file in {directory}, found: {found}"
            )
        return matches[0]
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS)
    if len(files) != 1:
        found = ", ".join(p.name for p in files) or "none"
        raise FileNotFoundError(
            f"{directory} must hold one HDF5 file, or the channel must name it with "
            f"'array_file'; found: {found}"
        )
    return files[0]


class _Handle:
    """An ``h5py.File`` opened on first use, again after a fork."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file = None
        self._pid: int | None = None

    def __call__(self):
        if self._file is None or self._pid != os.getpid():
            try:
                import h5py  # heavy: imported when a channel is read, not at lookup
            except ImportError as exc:
                raise ImportError(
                    "an hdf5 channel needs h5py: pip install 'apairo[hdf5]'"
                ) from exc

            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def __getstate__(self):  # pickled into DataLoader workers without the handle
        return {"path": self.path, "_file": None, "_pid": None}


class Hdf5ArrayLoader(AbstractLoader):
    """Frame ``i`` is row ``i`` of the array ``array`` of an HDF5 file."""

    def __init__(
        self,
        directory: str,
        array: str,
        array_file: str | None = None,
        decode: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.path = locate(self.directory, array_file)
        self.array = array
        self._handle = _Handle(self.path)
        self._video = None
        self._shape: tuple[int, ...]
        dataset = self._dataset()
        if decode is not None:
            if decode != "video":
                raise ValueError(f"decode: only 'video' is known, got {decode!r}")
            if dataset.ndim != 0:
                raise ValueError(
                    f"{self.path.name}:{array} is an array of shape {dataset.shape}; "
                    f"decode: video reads one value holding an encoded video."
                )
            from apairo.loader.video_loader import VideoReader

            # A bound method, not a lambda: the loader pickles into workers.
            self._video = VideoReader(self._encoded)
            self._len = len(self._video)
            self._shape = (self._video.height, self._video.width, 3)
            return
        if dataset.ndim == 0:
            raise ValueError(
                f"{self.path.name}:{array} is a scalar, not an array of frames; "
                f"a per-recording value is metadata, not a channel."
            )
        self._len = int(dataset.shape[0])
        self._shape = tuple(int(s) for s in dataset.shape[1:])

    def _dataset(self, path: str | None = None):
        f = self._handle()
        name = path or self.array
        if name not in f:
            raise KeyError(f"{self.path.name} has no array '{name}'")
        return f[name]

    def _encoded(self) -> bytes:
        value = self._dataset()[()]
        return value.tobytes() if hasattr(value, "tobytes") else bytes(value)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> np.ndarray:
        if self._video is not None:
            return self._video[idx]
        if idx < 0:
            idx += self._len
        if not 0 <= idx < self._len:
            raise IndexError(f"frame {idx} out of range for {self._len} frames")
        return np.asarray(self._dataset()[idx])

    def read(self, path: str) -> np.ndarray:
        """A whole array of the same file (a clock, an index)."""
        return np.asarray(self._dataset(path)[()])

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
