from __future__ import annotations

import copy
import logging

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample

logger = logging.getLogger(__name__)


class CachedDataset(AbstractDataset):
    """An in-memory cache of a dataset.

    Materialises every sample at construction time by iterating the parent
    once.  Subsequent accesses are served from RAM with no I/O.

    Created by :meth:`~apairo.core.abstract_dataset.AbstractDataset.cache`.
    Supports full chaining: ``.transform()``, ``.filter()``, ``.join()``.

    .. warning::
        All samples are loaded into RAM.  Only use on datasets that fit in
        memory — typically after a ``.filter()`` or ``.select()`` that has
        already reduced the volume.

    Args:
        parent: Dataset to materialise.

    Example::

        # Cache only the expensive channel — transforms applied before storing
        ds.transform("lidar", expensive_ground_prior, output="ground_prior")
        ds_prior = ds.select(["ground_prior"]).cache()

        # Reuse across different training runs — no re-read, no re-compute
        ds_v1 = base.join(ds_prior).transform(augment_v1)
        ds_v2 = base.join(ds_prior).transform(augment_v2)
    """

    def __init__(self, parent: AbstractDataset) -> None:
        n = len(parent)
        logger.info("Caching %d samples from %s...", n, parent.__class__.__name__)
        self._cache: list[Sample] = [parent[i] for i in range(n)]
        logger.info("Cache ready (%d samples).", n)

        self._keys = list(self._cache[0].data.keys()) if self._cache else []
        self._synchronous = parent.is_synchronous

    @property
    def is_synchronous(self) -> bool:
        return self._synchronous

    def __len__(self) -> int:
        return len(self._cache)

    def _load(self, idx: int) -> Sample:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        s = self._cache[idx]
        return Sample(data={k: copy.copy(v) for k, v in s.data.items()}, timestamp=s.timestamp)

    def cache(self) -> "AbstractDataset":
        logger.warning(
            "cache() called on an already-cached dataset — "
            "this duplicates the data in RAM. Cache before branching."
        )
        return CachedDataset(self)

    def __repr__(self) -> str:
        return f"CachedDataset(n={len(self)}, keys={self._keys})"
