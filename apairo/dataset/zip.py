from __future__ import annotations

from typing import Literal

from apairo.core import AbstractDataset
from apairo.core.sample import Sample


class ZipDataset(AbstractDataset):
    """Merges channels from multiple datasets of the same length.

    The dual of :class:`~apairo.dataset.concat.ConcatDataset`: where
    ``ConcatDataset`` concatenates along the **frame** axis,
    ``ZipDataset`` merges along the **channel** axis.
    ``zip_ds[i].data`` is the union of ``ds[i].data`` for each parent.

    Transforms registered on each parent are applied before merging.

    Args:
        datasets: Two or more datasets of identical length.
        on_collision: What to do when two parents declare the same key.
            ``"raise"`` (default) raises at construction time.
            ``"last"`` lets the last dataset win silently.

    Example::

        ds_base  = Rellis3DDataset(root, keys=["lidar", "trav_gt"])
        ds_prior = Rellis3DDataset(root, keys=["ground_height"])
        combined = ZipDataset(ds_base, ds_prior)
        combined[0].data  # {"lidar": ..., "trav_gt": ..., "ground_height": ...}

        # Or via the fluent API:
        combined = ds_base.join(ds_prior)

    Raises:
        ValueError: If fewer than two datasets are given, or lengths differ.
        KeyError: If ``on_collision="raise"`` and two datasets share a key.
    """

    def __init__(
        self,
        *datasets: AbstractDataset,
        on_collision: Literal["raise", "last"] = "raise",
    ) -> None:
        if len(datasets) < 2:
            raise ValueError("ZipDataset requires at least two datasets.")

        lengths = [len(ds) for ds in datasets]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All datasets must have the same length. Got: {lengths}"
            )

        if on_collision not in ("raise", "last"):
            raise ValueError(
                f"on_collision must be 'raise' or 'last', got {on_collision!r}"
            )

        if on_collision == "raise":
            seen: dict[str, int] = {}
            for i, ds in enumerate(datasets):
                for key in ds.keys:
                    if key in seen:
                        raise KeyError(
                            f"Key {key!r} appears in dataset {seen[key]} and dataset {i}. "
                            f"Use on_collision='last' to allow overwriting."
                        )
                    seen[key] = i

        self._datasets = list(datasets)
        self._on_collision = on_collision
        self._keys = self._merged_keys()

    def _merged_keys(self) -> list[str]:
        seen: set[str] = set()
        keys: list[str] = []
        source = reversed(self._datasets) if self._on_collision == "last" else self._datasets
        for ds in source:
            for key in ds.keys:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        return keys if self._on_collision != "last" else list(reversed(keys))

    def __len__(self) -> int:
        return len(self._datasets[0])

    @property
    def is_synchronous(self) -> bool:
        return all(ds.is_synchronous for ds in self._datasets)

    def _load(self, idx: int) -> Sample:
        merged: dict = {}
        timestamp = None
        for ds in self._datasets:
            sample = ds[idx]
            merged.update(sample.data)
            if timestamp is None:
                timestamp = sample.timestamp
        return Sample(data=merged, timestamp=timestamp)

    def __repr__(self) -> str:
        names = ", ".join(ds.__class__.__name__ for ds in self._datasets)
        return f"ZipDataset(n={len(self)}, datasets=[{names}])"
