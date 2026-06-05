from __future__ import annotations
from abc import abstractmethod
from pathlib import Path

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample


class SynchronousDataset(AbstractDataset):
    """Base class for datasets where index ``i`` returns a complete synchronous frame.

    All modalities at index ``i`` are co-captured -- no timestamps, no interleaving.
    ``sample.timestamp`` is always ``None``.  Random access and standard PyTorch
    ``DataLoader`` shuffling work without any additional wrappers.

    Subclasses must implement ``__len__``, ``__getitem__``, ``__iter__``, ``__next__``.

    For new synchronous datasets, prefer extending
    :class:`~apairo.core.profiled_dataset.ProfiledDataset` with a YAML profile
    rather than subclassing this directly.

    Attributes:
        timestamps: Always ``None`` -- marks this dataset as synchronous.
    """

    timestamps = None

    @property
    def root_dir(self) -> Path:
        return self._root

    def _seq_root(self, path: Path) -> Path:
        """Return the sequence root directory for a native file path.

        Datasets with deeper file structures (e.g. seq/lidar/scan/file.bin)
        should override this to go up the correct number of levels.
        Default: path.parent.parent (one modality directory deep).
        """
        return path.parent.parent

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        ref = next(iter(self._files.values()))[idx]
        return self._seq_root(ref) / key / f"{ref.stem}.{ext}"

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __next__(self) -> Sample: ...
