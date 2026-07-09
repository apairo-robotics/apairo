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

    @property
    def is_synchronous(self) -> bool:
        return self._parent.is_synchronous

    @property
    def frame_sequence_ids(self) -> np.ndarray:
        """Sequence ID for every frame in this view (delegated from parent)."""
        return self._parent.frame_sequence_ids[self._indices]

    @property
    def frame_stems(self) -> np.ndarray:
        """Filename stem for every frame in this view (delegated from parent)."""
        return self._parent.frame_stems[self._indices]

    def frame_info(self, idx: int):
        """Provenance of a view frame -- the parent's, at the remapped index."""
        return self._parent.frame_info(int(self._indices[idx]))

    def filter_split(self, name: str) -> FilteredView:
        """Return a FilteredView restricted to the named predefined split."""
        from apairo.core.profiled_dataset import ProfiledDataset, _apply_lst_filter

        ds = self._parent
        while ds is not None and not isinstance(ds, ProfiledDataset):
            ds = getattr(ds, "_parent", None)
        if ds is None:
            raise AttributeError("No ProfiledDataset found in parent chain.")
        return _apply_lst_filter(self, ds._lst_frame_filter(name))

    def __len__(self) -> int:
        return len(self._indices)

    def _load(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._parent[int(self._indices[idx])]

    def __repr__(self) -> str:
        return f"FilteredView(n={len(self)}, parent={self._parent.__class__.__name__})"
