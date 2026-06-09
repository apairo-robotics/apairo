from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from apairo.core.abstract_dataset import AbstractDataset

if TYPE_CHECKING:
    from apairo.core.sample import Sample


class FilteredView(AbstractDataset):
    """A view over a dataset restricted to a subset of indices.

    Created by :meth:`~apairo.core.abstract_dataset.AbstractDataset.filter`.
    Supports full chaining: ``.transform()``, ``.filter()``, etc.

    Args:
        parent: The underlying dataset.
        indices: Global indices in *parent* to include.

    Example::

        # From a predicate
        view = ds.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)

        # Persist and reload — skips the sweep entirely next run
        np.save("valid_indices.npy", view.indices)
        view = ds.filter(np.load("valid_indices.npy"))
    """

    def __init__(self, parent: AbstractDataset, indices) -> None:
        self._parent = parent
        self._indices = np.asarray(indices, dtype=np.int64)

    @property
    def indices(self) -> np.ndarray:
        """Global indices in the parent dataset that this view covers."""
        return self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def _load(self, idx: int) -> "Sample":
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._parent._load(int(self._indices[idx]))

    def __repr__(self) -> str:
        return f"FilteredView(n={len(self)}, parent={self._parent.__class__.__name__})"
