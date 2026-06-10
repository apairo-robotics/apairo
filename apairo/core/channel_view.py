from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from apairo.core.abstract_dataset import AbstractDataset

if TYPE_CHECKING:
    from apairo.core.sample import Sample


class ChannelView(AbstractDataset):
    """A view over a dataset restricted to a subset of channels.

    Created by :meth:`~apairo.core.abstract_dataset.AbstractDataset.select`.
    Calls ``parent[idx]`` (parent transforms applied) then projects to the
    requested keys.  Supports full chaining: ``.transform()``, ``.filter()``,
    ``.cache()``, ``.join()``.

    The primary use case is targeting what goes into a cache::

        ds.transform("ground_height_csf", expensive_smooth)
        ds_prior = ds.select(["ground_height_csf"]).cache()

    Args:
        parent: The underlying dataset.
        keys: Channel names to keep.  A ``KeyError`` is raised at access time
            if a key is absent from the parent sample.
    """

    def __init__(self, parent: AbstractDataset, keys: list[str]) -> None:
        self._parent = parent
        self._keys = list(keys)

    def __len__(self) -> int:
        return len(self._parent)

    @property
    def frame_sequence_ids(self) -> "np.ndarray":
        return self._parent.frame_sequence_ids

    @property
    def frame_stems(self) -> "np.ndarray":
        return self._parent.frame_stems

    def _load(self, idx: int) -> "Sample":
        from apairo.core.sample import Sample
        sample = self._parent[idx]
        return Sample(
            data={k: sample.data[k] for k in self._keys},
            timestamp=sample.timestamp,
        )

    def __repr__(self) -> str:
        return f"ChannelView(keys={self._keys}, parent={self._parent.__class__.__name__})"
