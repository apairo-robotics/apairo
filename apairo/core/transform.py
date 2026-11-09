from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class Compose:
    """Chain multiple callables into a single transform.

    Chain ``transform()`` calls to build a dataset; use ``Compose`` to name an
    operation made of several callables, so it can be reused as one step,
    inside a dataset or on a bare array::

        clean_lidar = Compose([RangeFilter(max=50), Normalize()])
        ds = ds.transform("lidar", clean_lidar)
        points = clean_lidar(raw_points)
    """

    def __init__(self, transforms: Iterable[Callable]) -> None:
        self._transforms = list(transforms)

    @property
    def transforms(self) -> list[Callable]:
        """The parts, in order -- read by :meth:`AbstractDataset.inference`,
        which removes the training-only ones."""
        return list(self._transforms)

    def __call__(self, x: Any) -> Any:
        for fn in self._transforms:
            x = fn(x)
        return x

    def __repr__(self) -> str:
        names = [getattr(fn, "__name__", type(fn).__name__) for fn in self._transforms]
        return f"Compose([{', '.join(names)}])"
