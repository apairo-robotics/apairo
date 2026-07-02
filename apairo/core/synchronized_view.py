from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Literal, Union

import numpy as np

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.interpolator import Interpolator

if TYPE_CHECKING:
    from apairo.core.sample import Sample

# Matching strategy: (channel_timestamps, reference_timestamps) -> one event
# index per reference tick; a negative index marks "no match" (frame dropped).
SyncMethod = Callable[[np.ndarray, np.ndarray], np.ndarray]

# What a single channel can be synchronized with: a built-in matching mode,
# a custom matching callable, or a value-level interpolator.
# "latest" is a deprecated alias for "previous".
ChannelStrategy = Union[
    Literal["previous", "next", "nearest", "latest"], SyncMethod, Interpolator
]


class SynchronizedView(AbstractDataset):
    """A synchronous view over an asynchronous dataset.

    Created by :meth:`~apairo.core.abstract_dataset.AbstractDataset.synchronize`.
    Index ``i`` returns a complete sample built around the *i*-th tick of the
    reference clock: every channel contributes either an existing event
    matched by timestamp, or a value synthesized at the tick by an
    :class:`~apairo.core.interpolator.Interpolator`.  The matching is a pure
    index computation (one ``np.searchsorted`` per channel at construction
    time) -- no data is read until access.

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
        reference: The clock to resample onto.  Three forms:

            * channel name -- that channel's timestamps drive the view;
            * ``None`` -- the lowest-frequency channel is used;
            * **array of timestamps** -- an external clock.  Enables
              fixed-rate resampling (``np.arange(t0, t1, 1/hz)``) or
              distance-based resampling (see
              :func:`~apairo.utils.timestamps.clock_from_distance`).
        method: Strategy applied to every channel, or a dict mapping channel
            names to per-channel strategies (unlisted channels default to
            ``"previous"``).  A strategy is one of:

            * ``"previous"`` -- last event with ``t <= t_ref`` (zero-order
              hold, never looks into the future; ``"latest"`` is a
              deprecated alias);
            * ``"next"`` -- first event with ``t >= t_ref``;
            * ``"nearest"`` -- event closest in time to ``t_ref``, either
              side (ties favour the earlier event);
            * a **callable** ``(channel_ts, ref_ts) -> indices`` returning,
              for each reference tick, the event index to use (negative = no
              match, the frame is dropped);
            * an :class:`~apairo.core.interpolator.Interpolator` -- the value
              is synthesized at ``t_ref`` from the two bracketing events
              (continuous signals only: poses, IMU, commands).
        tolerance: Maximum ``|t - t_ref|`` in seconds.  For interpolated
            channels, *both* bracketing events must lie within tolerance.
            Reference ticks where any channel has no match are dropped.

    Example::

        from apairo_transform.interp import Se3Interp

        ds = TartanKittiDataset(seq, keys=["velodyne_0", "gicp_poses"])
        ds_sync = ds.synchronize(
            reference="velodyne_0",
            method={"gicp_poses": Se3Interp()},   # velodyne_0 -> "previous"
            tolerance=0.05,
        )

        s = ds_sync[0]
        s.data["gicp_poses"]    # pose interpolated at s.timestamp
    """

    timestamps = None  # the view itself is synchronous

    def __init__(
        self,
        parent: AbstractDataset,
        reference: str | np.ndarray | None = None,
        method: ChannelStrategy | dict[str, ChannelStrategy] = "previous",
        tolerance: float | None = None,
    ) -> None:
        parent_ts = getattr(parent, "timestamps", None)
        if not isinstance(parent_ts, dict) or not parent_ts:
            raise ValueError(
                f"synchronize() requires an asynchronous dataset with "
                f"per-channel timestamps; {parent.__class__.__name__} is "
                f"already synchronous."
            )

        keys = list(parent.keys)
        strategies = self._resolve_strategies(method, keys)
        ref_name, ref_ts = self._resolve_clock(reference, parent_ts, keys)

        valid = np.ones(len(ref_ts), dtype=bool)
        index_map: dict[str, np.ndarray] = {}
        channel_ts: dict[str, np.ndarray] = {}

        for key in keys:
            ts = np.asarray(parent_ts[key], dtype=float)
            channel_ts[key] = ts
            idx, ok = self._match(strategies[key], ts, ref_ts, tolerance)
            valid &= ok
            index_map[key] = idx

        keep = np.where(valid)[0]
        self._parent = parent
        self._reference = ref_name
        self._method = method
        self._strategies = strategies
        self._tolerance = tolerance
        self._ref_timestamps = ref_ts[keep]
        self._index_map = {
            k: v[keep].astype(np.intp) for k, v in index_map.items()
        }
        self._channel_ts = channel_ts
        self._keys = keys

    # ------------------------------------------------------------- resolution

    @staticmethod
    def _resolve_strategies(method, keys: list[str]) -> dict[str, ChannelStrategy]:
        """Normalize *method* into one validated strategy per channel."""
        if isinstance(method, (set, frozenset)):
            raise TypeError(
                f"method must be a single strategy or a dict {{channel: strategy}}, "
                f"not a set -- did you write {{a, b}} instead of {{a: b}}? "
                f"Got {method!r}."
            )
        if isinstance(method, dict):
            unknown = set(method) - set(keys)
            if unknown:
                raise KeyError(
                    f"method maps unknown channels {sorted(unknown)}; "
                    f"dataset keys are {keys}."
                )
            strategies = {k: method.get(k, "previous") for k in keys}
        else:
            strategies = {k: method for k in keys}

        if any(s == "latest" for s in strategies.values()):
            warnings.warn(
                "method='latest' is deprecated, use 'previous' (same "
                "semantics: last event with t <= t_ref).",
                DeprecationWarning,
                stacklevel=4,  # _resolve_strategies <- __init__ <- synchronize()
            )
            strategies = {
                k: ("previous" if s == "latest" else s)
                for k, s in strategies.items()
            }

        for key, strat in strategies.items():
            if isinstance(strat, Interpolator) or callable(strat):
                continue
            if strat not in ("previous", "next", "nearest"):
                raise ValueError(
                    f"Strategy for {key!r} must be 'previous', 'next', "
                    f"'nearest', a callable (channel_ts, ref_ts) -> indices, "
                    f"or an Interpolator, got {strat!r}"
                )
        return strategies

    @staticmethod
    def _resolve_clock(reference, parent_ts: dict, keys: list[str]):
        """Resolve *reference* into ``(name_or_None, timestamp_array)``.

        Accepts a channel name, ``None`` (lowest-frequency channel), or an
        explicit array of timestamps — an external clock (e.g. fixed-rate
        ticks or distance-based ticks from odometry).
        """
        if reference is None:
            from apairo.utils.timestamps import get_reference_timestamps
            name = get_reference_timestamps({k: parent_ts[k] for k in keys})
            return name, np.asarray(parent_ts[name], dtype=float)

        if isinstance(reference, str):
            if reference not in keys:
                raise KeyError(
                    f"Reference channel {reference!r} not in dataset keys {keys}."
                )
            return reference, np.asarray(parent_ts[reference], dtype=float)

        ref_ts = np.asarray(reference, dtype=float)
        if ref_ts.ndim != 1 or len(ref_ts) == 0:
            raise ValueError(
                f"An external clock must be a non-empty 1-D array of "
                f"timestamps, got shape {ref_ts.shape}."
            )
        if np.any(np.diff(ref_ts) < 0):
            raise ValueError("External clock timestamps must be ascending.")
        return None, ref_ts

    # --------------------------------------------------------------- matching

    @staticmethod
    def _match(
        strat: ChannelStrategy,
        ts: np.ndarray,
        ref_ts: np.ndarray,
        tolerance: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute event indices and per-tick validity for one channel.

        Returns ``(idx, valid)`` where ``idx`` has shape ``(N,)`` for matching
        strategies or ``(N, 2)`` (bracketing pair) for interpolators.
        """
        right = np.searchsorted(ts, ref_ts, side="right")
        latest = right - 1  # last event with t <= t_ref; -1 when none yet

        if isinstance(strat, Interpolator):
            i0 = np.clip(latest, 0, len(ts) - 1)
            i1 = np.clip(right, 0, len(ts) - 1)
            # exact matches collapse to a single index -- the stored value is
            # returned directly, the interpolator is never called
            i1 = np.where(ts[i0] == ref_ts, i0, i1)
            # bracketed: an event at or before the tick, and one at or after
            valid = (latest >= 0) & (ts[i1] >= ref_ts)
            if tolerance is not None:
                valid &= np.maximum(ref_ts - ts[i0], ts[i1] - ref_ts) <= tolerance
            return np.stack([i0, i1], axis=1), valid

        if callable(strat):
            idx = np.asarray(strat(ts, ref_ts))
            if idx.shape != ref_ts.shape:
                raise ValueError(
                    f"Custom method returned shape {idx.shape}, expected "
                    f"{ref_ts.shape} (one index per reference tick; "
                    f"negative = no match)."
                )
            valid = (idx >= 0) & (idx < len(ts))
        elif strat == "previous":
            idx = latest
            valid = latest >= 0
        elif strat == "next":
            # first event with t >= t_ref; len(ts) when none remains
            idx = np.searchsorted(ts, ref_ts, side="left")
            valid = idx < len(ts)
        else:  # nearest -- either side; ties favour the earlier event
            prev = np.clip(latest, 0, len(ts) - 1)
            nxt = np.clip(right, 0, len(ts) - 1)
            idx = np.where(
                np.abs(ts[prev] - ref_ts) <= np.abs(ts[nxt] - ref_ts),
                prev,
                nxt,
            )
            valid = np.ones(len(ref_ts), dtype=bool)

        idx = np.clip(idx, 0, len(ts) - 1)
        if tolerance is not None:
            valid = valid & (np.abs(ts[idx] - ref_ts) <= tolerance)
        return idx, valid

    # ------------------------------------------------------------- properties

    @property
    def reference(self) -> str | None:
        """Channel providing the clock, or ``None`` for an external clock."""
        return self._reference

    @property
    def reference_timestamps(self) -> np.ndarray:
        """Timestamp of each frame in the view (reference clock)."""
        return self._ref_timestamps

    @property
    def frame_indices(self) -> dict[str, np.ndarray]:
        """Per-channel event indices backing each frame.

        Shape ``(n,)`` for matched channels (the event used), ``(n, 2)`` for
        interpolated channels (the bracketing pair).
        """
        return self._index_map

    def time_offsets(self, key: str) -> np.ndarray:
        """Signed ``t_event - t_ref`` per frame for *key*, in seconds.

        Interpolated channels return zeros: their values are synthesized at
        the reference instant.
        """
        if isinstance(self._strategies[key], Interpolator):
            return np.zeros(len(self), dtype=float)
        ts = self._channel_ts[key]
        return ts[self._index_map[key]] - self._ref_timestamps

    # ----------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self._ref_timestamps)

    def _load(self, idx: int) -> "Sample":
        from apairo.core.sample import Sample
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        t_ref = float(self._ref_timestamps[idx])
        data = {}
        for key in self._keys:
            strat = self._strategies[key]
            if isinstance(strat, Interpolator):
                i0, i1 = (int(i) for i in self._index_map[key][idx])
                v0 = self._parent.loaders[key][i0]
                if i0 == i1:  # exact match -- no synthesis needed
                    data[key] = v0
                else:
                    ts = self._channel_ts[key]
                    data[key] = strat(
                        t_ref,
                        float(ts[i0]), v0,
                        float(ts[i1]), self._parent.loaders[key][i1],
                    )
            else:
                data[key] = self._parent.loaders[key][int(self._index_map[key][idx])]
        return Sample(data=data, timestamp=t_ref)

    def __repr__(self) -> str:
        ref = self._reference if self._reference is not None else "<external clock>"
        if isinstance(self._method, dict):
            method = "per-channel"
        else:
            method = getattr(self._method, "__name__", self._method)
        return (
            f"SynchronizedView(n={len(self)}, reference={ref!r}, "
            f"method={method!r}, keys={self._keys})"
        )
