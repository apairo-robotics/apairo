from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class Compose:
    """Chain multiple callables into a single transform.

    Example::

        ds.transform("lidar", Compose([RangeFilter(max=50), Normalize()]))
    """

    def __init__(self, transforms: Iterable[Callable]) -> None:
        self._transforms = list(transforms)

    def __call__(self, x: Any) -> Any:
        for fn in self._transforms:
            x = fn(x)
        return x

    def __repr__(self) -> str:
        names = [getattr(fn, "__name__", type(fn).__name__) for fn in self._transforms]
        return f"Compose([{', '.join(names)}])"
