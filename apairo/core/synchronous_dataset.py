from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import numpy as np

from apairo.core.abstract_dataset import AbstractDataset


class SynchronousDataset(AbstractDataset):
    """Base class for datasets where index ``i`` returns a complete synchronous frame.

    All modalities at index ``i`` are co-captured -- one row is one sample across
    every channel, no interleaving. That co-capture is what *synchronous* means;
    it is independent of whether the frame carries a timestamp. Random access and
    standard PyTorch ``DataLoader`` shuffling work without any additional wrappers.

    A synchronous dataset may still expose a **shared per-frame clock** in
    :attr:`timestamps` (one entry per global frame, so ``sample.timestamp`` is
    that frame's tick); it defaults to ``None`` -- clockless -- and subclasses
    populate it when the frames carry a timestamp.

    Subclasses must implement ``__len__`` and ``_load``.

    For new synchronous datasets, prefer extending
    :class:`~apairo.core.profiled_dataset.ProfiledDataset` with a YAML profile
    rather than subclassing this directly.

    Attributes:
        synchronous: Always ``True`` -- marks the co-captured (structural) family.
        timestamps: The shared per-frame clock array, or ``None`` when clockless.
    """

    synchronous = True
    timestamps: dict | np.ndarray | None = None

    # Provided by the concrete dataset (see ProfiledDataset.__init__).
    _root: Path
    _files: dict[str, list[Path]]

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
