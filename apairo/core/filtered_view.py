from __future__ import annotations

from typing import TYPE_CHECKING

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

        view = ds.filter("trav_gt", lambda gt: (gt == 1).sum() >= 50)
        len(view)       # number of frames that passed the filter
        view[0]         # first kept frame (transforms applied)
        view.transform("lidar", Normalize())   # chaining works
    """

    def __init__(self, parent: AbstractDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def _load(self, idx: int) -> "Sample":
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._parent._load(self._indices[idx])

    def __repr__(self) -> str:
        return f"FilteredView(n={len(self)}, parent={self._parent.__class__.__name__})"
