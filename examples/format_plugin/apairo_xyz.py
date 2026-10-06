"""apairo_xyz -- an apairo format plugin for ASCII point clouds.

One ``.xyz`` file per frame, one ``x y z`` point per line: the format some
scanners and photogrammetry tools export. Installed next to apairo, it makes
``loader: xyz`` a channel like any other -- ``init`` detects it, ``status``
describes it, ``declare`` scaffolds its clock, ``synchronize()`` aligns it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apairo.core.abstract_loader import AbstractLoader
from apairo.core.formats import Format


class XYZLoader(AbstractLoader):
    """Frame ``i`` is the ``(N, 3)`` float32 cloud in ``files[i]``."""

    def __init__(self, directory: str, files: list[str] | None = None) -> None:
        self.directory = Path(directory)
        # `files` is what the core resolved (a key/order regex); otherwise list.
        self.files = (
            list(files)
            if files is not None
            else sorted(p.name for p in self.directory.iterdir() if p.suffix == ".xyz")
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.loadtxt(self.directory / self.files[idx], dtype=np.float32, ndmin=2)

    @property
    def shape(self) -> tuple[int, ...]:
        return self[0].shape


class XyzFormat(Format):
    """Per-frame, found by its extension: the defaults of :class:`Format`
    cover the rest -- detection, opening, status facts."""

    name = "xyz"
    loader = XYZLoader
    extensions = frozenset({".xyz"})
