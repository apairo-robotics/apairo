from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from apairo.core.abstract_dataset import AbstractDataset

if TYPE_CHECKING:
    from apairo.core.sample import Sample


class SynchronizedView(AbstractDataset):
    """A synchronous view over an asynchronous dataset.

    Created by :meth:`~apairo.core.abstract_dataset.AbstractDataset.synchronize`.
    Index ``i`` returns a complete sample built around the *i*-th frame of the
    reference channel: every other channel contributes the event matched by
    timestamp.  The matching is a pure index computation (one
    ``np.searchsorted`` per channel at construction time) -- no data is read
    until access.

    Because the view is synchronous (``timestamps`` is ``None``), the full
    chaining API applies: ``.filter()``, ``.select()``, ``.cache()``,
    ``.join()``, and map-style PyTorch ``DataLoader`` with shuffling.

    .. note::
        The view reads channel data directly from the parent's loaders --
        transforms registered on the *parent* are not applied (they were
        written for single-event samples).  Register transforms on the view::

            ds_sync = ds.synchronize().transform("velodyne_0", RangeFilter(50))

    Args:
        parent: Asynchronous dataset exposing per-channel ``timestamps`` and
            ``loaders``.
        reference: Channel providing the clock.  ``None`` -> the channel with
            the lowest frequency.
        method: ``"latest"`` -- last event with ``t <= t_ref`` (zero-order
            hold); ``"nearest"`` -- event closest in time to ``t_ref``.
        tolerance: Maximum ``|t - t_ref|`` in seconds.  Reference frames where
            any channel has no event within tolerance are dropped.

    Example::

        ds = TartanKittiDataset(seq, keys=["velodyne_0", "image_left", "cmd"])
        ds_sync = ds.synchronize(reference="velodyne_0", tolerance=0.05)

        len(ds_sync)            # reference frames kept
        s = ds_sync[0]
        s.data.keys()           # all three channels
        s.timestamp             # timestamp of the reference frame
    """

    timestamps = None  # the view itself is synchronous

    def __init__(
        self,
        parent: AbstractDataset,
        reference: str | None = None,
        method: str = "latest",
        tolerance: float | None = None,
    ) -> None:
        parent_ts = getattr(parent, "timestamps", None)
        if not isinstance(parent_ts, dict) or not parent_ts:
            raise ValueError(
                f"synchronize() requires an asynchronous dataset with "
                f"per-channel timestamps; {parent.__class__.__name__} is "
                f"already synchronous."
            )
        if method not in ("latest", "nearest"):
            raise ValueError(
                f"method must be 'latest' or 'nearest', got {method!r}"
            )

        keys = list(parent.keys)
        if reference is None:
            reference = self._lowest_frequency_key(parent_ts, keys)
        elif reference not in keys:
            raise KeyError(
                f"Reference channel {reference!r} not in dataset keys {keys}."
            )

        ref_ts = np.asarray(parent_ts[reference], dtype=float)
        valid = np.ones(len(ref_ts), dtype=bool)
        index_map: dict[str, np.ndarray] = {}

        for key in keys:
            ts = np.asarray(parent_ts[key], dtype=float)
            right = np.searchsorted(ts, ref_ts, side="right")
            latest = right - 1  # last event with t <= t_ref; -1 when none yet

            if method == "latest":
                idx = latest
                valid &= latest >= 0
            else:  # nearest
                prev = np.clip(latest, 0, len(ts) - 1)
                nxt = np.clip(right, 0, len(ts) - 1)
                idx = np.where(
                    np.abs(ts[prev] - ref_ts) <= np.abs(ts[nxt] - ref_ts),
                    prev,
                    nxt,
                )

            idx = np.clip(idx, 0, len(ts) - 1)
            if tolerance is not None:
                valid &= np.abs(ts[idx] - ref_ts) <= tolerance
            index_map[key] = idx

        keep = np.where(valid)[0]
        self._parent = parent
        self._reference = reference
        self._method = method
        self._tolerance = tolerance
        self._ref_timestamps = ref_ts[keep]
        self._index_map = {
            k: v[keep].astype(np.intp) for k, v in index_map.items()
        }
        self._keys = keys

    @staticmethod
    def _lowest_frequency_key(timestamps: dict, keys: list[str]) -> str:
        from apairo.utils.timestamps import get_reference_timestamps
        return get_reference_timestamps({k: timestamps[k] for k in keys})

    @property
    def reference(self) -> str:
        """Channel providing the clock for this view."""
        return self._reference

    @property
    def reference_timestamps(self) -> np.ndarray:
        """Timestamp of each frame in the view (reference channel's clock)."""
        return self._ref_timestamps

    @property
    def frame_indices(self) -> dict[str, np.ndarray]:
        """Per-channel event indices backing each frame.

        ``frame_indices[key][i]`` is the index into ``parent.loaders[key]``
        used to build frame ``i``.
        """
        return self._index_map

    def time_offsets(self, key: str) -> np.ndarray:
        """Signed ``t_event - t_ref`` per frame for *key*, in seconds."""
        ts = np.asarray(self._parent.timestamps[key], dtype=float)
        return ts[self._index_map[key]] - self._ref_timestamps

    def __len__(self) -> int:
        return len(self._ref_timestamps)

    def _load(self, idx: int) -> "Sample":
        from apairo.core.sample import Sample
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        data = {
            key: self._parent.loaders[key][int(self._index_map[key][idx])]
            for key in self._keys
        }
        return Sample(data=data, timestamp=float(self._ref_timestamps[idx]))

    def __repr__(self) -> str:
        return (
            f"SynchronizedView(n={len(self)}, reference={self._reference!r}, "
            f"method={self._method!r}, keys={self._keys})"
        )
